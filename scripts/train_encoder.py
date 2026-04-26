import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformer_scratch import MultiTaskEncoder
import os
import matplotlib.pyplot as plt
import math

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

def train():
    base_path = "/home/minato/Documents/NLP_Assignment_3"
    data_path = os.path.join(base_path, 'data', 'processed_data.pt')
    data = torch.load(data_path)
    
    train_X, train_Ys, train_Yc, _, _ = data['train']
    val_X, val_Ys, val_Yc, _, _ = data['val']
    vocab = data['vocab']
    
    batch_size = 64
    train_loader = DataLoader(TensorDataset(train_X, train_Ys, train_Yc), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_Ys, val_Yc), batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MultiTaskEncoder(
        vocab_size=len(vocab),
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    num_epochs = 30
    total_steps = len(train_loader) * num_epochs
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=500, total_steps=total_steps)
    
    train_losses = []
    val_accuracies_s = []
    val_accuracies_c = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Ys, batch_Yc in train_loader:
            batch_X, batch_Ys, batch_Yc = batch_X.to(device), batch_Ys.to(device), batch_Yc.to(device)
            
            optimizer.zero_grad()
            s_logits, c_logits, _ = model(batch_X)
            
            loss_s = criterion(s_logits, batch_Ys)
            loss_c = criterion(c_logits, batch_Yc)
            loss = loss_s + 0.5 * loss_c
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        correct_s = 0
        correct_c = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_Ys, batch_Yc in val_loader:
                batch_X, batch_Ys, batch_Yc = batch_X.to(device), batch_Ys.to(device), batch_Yc.to(device)
                s_logits, c_logits, _ = model(batch_X)
                
                _, pred_s = torch.max(s_logits, 1)
                _, pred_c = torch.max(c_logits, 1)
                
                correct_s += (pred_s == batch_Ys).sum().item()
                correct_c += (pred_c == batch_Yc).sum().item()
                total += batch_Ys.size(0)
        
        acc_s = correct_s / total
        acc_c = correct_c / total
        val_accuracies_s.append(acc_s)
        val_accuracies_c.append(acc_c)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Val Acc S: {acc_s:.4f}, Val Acc C: {acc_c:.4f}")

    # Save model weight
    weights_path = os.path.join(base_path, 'models', 'encoder_weights.pt')
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved to {weights_path}")
    
    # Save training embedding for Part B
    model.eval()
    all_train_embeddings = []
    train_loader_full = DataLoader(TensorDataset(train_X, train_Ys, train_Yc), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_X, _, _ in train_loader_full:
            _, _, pooled = model(batch_X.to(device))
            all_train_embeddings.append(pooled.cpu())
    
    train_embeddings = torch.cat(all_train_embeddings, dim=0)
    emb_path = os.path.join(base_path, 'results', 'train_embeddings.pt')
    torch.save(train_embeddings, emb_path)
    print(f"Embeddings saved to {emb_path}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies_s, label='Sentiment Acc')
    plt.plot(val_accuracies_c, label='Category Acc')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(base_path, 'results', 'encoder_training_curves.png'))
    print("Training curves saved to results/encoder_training_curves.png")

if __name__ == "__main__":
    train()
