import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer_scratch import CausalTransformer, MultiTaskEncoder
from retrieval import RetrievalModule
import os
import math
import random

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        
        for group in self.optimizer.param_groups:
            group['lr'] = lr

def compute_exp_loss(logits, targets, inputs, exp_token_id, criterion):
    """Only compute loss on tokens after <exp> token"""
    batch_size = logits.shape[0]
    vocab_size = logits.shape[-1]
    total_loss = 0
    count = 0
    
    for i in range(batch_size):
        exp_positions = (inputs[i] == exp_token_id).nonzero()
        if len(exp_positions) == 0:
            continue
        exp_pos = exp_positions[0].item()
        
        if exp_pos < logits.shape[1] - 1:
            pred = logits[i, exp_pos:-1, :] 
            tgt = targets[i, exp_pos+1:]
            
            if pred.size(0) > 0:
                # Safety check: target indices must be in range [0, vocab_size-1]
                if (tgt >= vocab_size).any() or (tgt < 0).any():
                    print(f"  Out of range target found in batch item {i}!")
                    continue
                    
                item_loss = criterion(pred, tgt)
                if torch.isnan(item_loss):
                    print(f"  NaN Loss in batch item {i}! Pred range: {pred.min().item():.4f} to {pred.max().item():.4f}")
                    continue
                    
                total_loss += item_loss
                count += 1
    
    return total_loss / max(count, 1)

class RAGDataset(Dataset):
    def __init__(self, data_split, vocab, retriever, encoder, device, k=1, max_len=256, mode='train'):
        self.X, self.Ys, self.Yc, self.texts, self.summs = data_split
        self.vocab = vocab
        self.retriever = retriever
        self.encoder = encoder
        self.device = device
        self.k = k
        self.max_len = max_len
        self.mode = mode
        
        # Pre-calculate embeddings and context for speed
        self.contexts = []
        print("Pre-retrieving contexts...")
        self.encoder.eval()
        with torch.no_grad():
            for i in range(len(self.X)):
                if i % 5000 == 0: print(f"Retrieved {i}/{len(self.X)}")
                query_tensor = self.X[i].unsqueeze(0).to(device)
                _, _, query_emb = self.encoder(query_tensor)
                top_k = self.retriever.retrieve(query_emb.cpu(), k=self.k)
                context = " ".join([res['text'] for res in top_k])
                self.contexts.append(context)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Format: <sos> <ret> context <rev> review <snt> snt <cat> cat <exp> summary <eos>
        review = self.texts[idx]
        summary = self.summs[idx]
        
        # Context Dropout (20% chance to have no context during training)
        context = self.contexts[idx]
        if self.mode == 'train' and random.random() < 0.2:
            context = ""
        
        snt = str(self.Ys[idx].item())
        cat = str(self.Yc[idx].item())
        
        full_text = f"<ret> {context} <rev> {review} <snt> {snt} <cat> {cat} <exp> {summary}"
        tokens = full_text.split()
        indices = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
        
        # Add <sos> and <eos>
        indices = [self.vocab['<sos>']] + indices + [self.vocab['<eos>']]
        
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        
        target = indices[1:] + [self.vocab['<pad>']]
        input_seq = indices
        
        # Padding
        pad_len = self.max_len - len(input_seq)
        input_seq += [self.vocab['<pad>']] * pad_len
        target += [self.vocab['<pad>']] * (self.max_len - len(target))
        
        return torch.tensor(input_seq), torch.tensor(target)

def train_decoder():
    base_path = "/home/minato/Documents/NLP_Assignment_3"
    data = torch.load(os.path.join(base_path, 'data', 'processed_data.pt'))
    vocab = data['vocab']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = MultiTaskEncoder(len(vocab), d_model=256, num_heads=8, d_ff=512, num_layers=4).to(device)
    encoder.load_state_dict(torch.load(os.path.join(base_path, 'models', 'encoder_weights.pt')))
    retriever = RetrievalModule(os.path.join(base_path, 'results', 'train_embeddings.pt'), data['train'][3])
    
    train_ds = RAGDataset(data['train'], vocab, retriever, encoder, device, k=1, mode='train')
    val_ds = RAGDataset(data['val'], vocab, retriever, encoder, device, k=1, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # 6-layer Decoder for Full Marks
    model = CausalTransformer(len(vocab), d_model=256, num_heads=8, d_ff=1024, num_layers=6).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    num_epochs = 40
    warmup_steps = 1000
    total_steps = num_epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)
    
    exp_token_id = vocab['<exp>']
    best_val_loss = float('inf')
    
    print(f"Starting Decoder Training (40 epochs, 6 layers)...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (batch_in, batch_tgt) in enumerate(train_loader):
            batch_in, batch_tgt = batch_in.to(device), batch_tgt.to(device)
            mask = torch.tril(torch.ones(batch_in.size(1), batch_in.size(1))).to(device)
            
            optimizer.zero_grad()
            logits = model(batch_in, mask=mask)
            
            loss = compute_exp_loss(logits, batch_tgt, batch_in, exp_token_id, criterion)
            
            if torch.isnan(loss):
                print(f"NaN Loss at batch {i}!")
                print(f"Logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
                # Check for NaNs in logits
                if torch.isnan(logits).any():
                    print("  NaNs found in logits!")
                # Check for NaNs in model parameters
                for name, p in model.named_parameters():
                    if torch.isnan(p).any():
                        print(f"  NaNs found in parameter: {name}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"  Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_in, b_tgt in val_loader:
                b_in, b_tgt = b_in.to(device), b_tgt.to(device)
                m = torch.tril(torch.ones(b_in.size(1), b_in.size(1))).to(device)
                l = model(b_in, mask=m)
                val_loss += compute_exp_loss(l, b_tgt, b_in, exp_token_id, criterion).item()
        
        avg_val_loss = val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val PPL: {perplexity:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(base_path, 'models', 'decoder_weights.pt'))
            print("  ✅ Best model saved.")

    print("Decoder weights saved.")

if __name__ == "__main__":
    train_decoder()
