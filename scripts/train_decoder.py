import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformer_scratch import CausalTransformer, MultiTaskEncoder
from retrieval import RetrievalModule
import os
import math
import random

class RAGDataset(Dataset):
    def __init__(self, data_split, vocab, retriever, encoder, device, k=1, max_len=150, mode='train'):
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
    
    # Load Encoder for retrieval queries
    encoder = MultiTaskEncoder(len(vocab), d_model=128, num_heads=4, d_ff=512, num_layers=2).to(device)
    encoder.load_state_dict(torch.load(os.path.join(base_path, 'models', 'encoder_weights.pt')))
    
    # Init Retriever
    retriever = RetrievalModule(os.path.join(base_path, 'results', 'train_embeddings.pt'), data['train'][3])
    
    # Create Datasets
    # Using a subset of training data for faster decoder training in this environment if needed
    # But let's try the full 21k (70% of 30k)
    train_ds = RAGDataset(data['train'], vocab, retriever, encoder, device, k=1, mode='train')
    val_ds = RAGDataset(data['val'], vocab, retriever, encoder, device, k=1, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    model = CausalTransformer(len(vocab), d_model=256, num_heads=8, d_ff=1024, num_layers=4).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    num_epochs = 15
    MAX_STEPS = num_epochs * len(train_loader)
    
    def get_lr(step, warmup_steps=200):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    best_val_loss = float('inf')
    
    print(f"Starting Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_in, batch_tgt in train_loader:
            batch_in, batch_tgt = batch_in.to(device), batch_tgt.to(device)
            
            mask = torch.tril(torch.ones(batch_in.size(1), batch_in.size(1))).to(device)
            
            optimizer.zero_grad()
            logits = model(batch_in, mask=mask)
            
            loss = criterion(logits.view(-1, len(vocab)), batch_tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_in, b_tgt in val_loader:
                b_in, b_tgt = b_in.to(device), b_tgt.to(device)
                m = torch.tril(torch.ones(b_in.size(1), b_in.size(1))).to(device)
                l = model(b_in, mask=m)
                val_loss += criterion(l.view(-1, len(vocab)), b_tgt.view(-1)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val PPL: {perplexity:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(base_path, 'models', 'decoder_weights.pt'))
            print("Model improved, weights saved.")

    # Save Decoder Weight
    torch.save(model.state_dict(), os.path.join(base_path, 'models', 'decoder_weights.pt'))
    print("Decoder weights saved.")

if __name__ == "__main__":
    train_decoder()
