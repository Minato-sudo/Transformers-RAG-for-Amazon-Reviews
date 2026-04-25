import torch
import torch.nn as nn
import math
import os
from scripts.transformer_scratch import MultiTaskEncoder, CausalTransformer
from scripts.retrieval import RetrievalModule

def compute_perplexity(decoder, data_split, vocab, retriever, encoder, device, k=1, limit=100):
    decoder.eval()
    encoder.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    X, Ys, Yc, texts, summs = data_split
    total_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(min(limit, len(X))):
            review, summary, snt, cat = texts[i], summs[i], str(Ys[i].item()), str(Yc[i].item())
            
            query_tensor = X[i].unsqueeze(0).to(device)
            _, _, query_emb = encoder(query_tensor)
            top_k = retriever.retrieve(query_emb.cpu(), k=k)
            context = " ".join([r['text'] for r in top_k])
            
            full_text = f"<sos> <ret> {context} <rev> {review} <snt> {snt} <cat> {cat} <exp> {summary} <eos>"
            tokens = full_text.split()
            indices = [vocab.get(t, vocab['<unk>']) for t in tokens]
            if len(indices) < 2: continue
            input_seq = torch.tensor(indices[:-1]).unsqueeze(0).to(device)
            target = torch.tensor(indices[1:]).unsqueeze(0).to(device)
            mask = torch.tril(torch.ones(input_seq.size(1), input_seq.size(1))).to(device)
            logits = decoder(input_seq, mask=mask)
            total_loss += criterion(logits.view(-1, len(vocab)), target.view(-1)).item()
            count += 1
    return math.exp(total_loss / count)

def k_analysis():
    base_path = "/home/minato/Documents/NLP_Assignment_3"
    data = torch.load(os.path.join(base_path, 'data', 'processed_data.pt'))
    vocab = data['vocab']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = MultiTaskEncoder(len(vocab), 128, 4, 512, 2).to(device)
    encoder.load_state_dict(torch.load(os.path.join(base_path, 'models', 'encoder_weights.pt')))
    decoder = CausalTransformer(len(vocab), 256, 8, 1024, 3).to(device)
    decoder.load_state_dict(torch.load(os.path.join(base_path, 'models', 'decoder_weights.pt')))
    retriever = RetrievalModule(os.path.join(base_path, 'results', 'train_embeddings.pt'), data['train'][3])
    
    print("k-Sensitivity Analysis:")
    for k in [1, 3, 5]:
        ppl = compute_perplexity(decoder, data['test'], vocab, retriever, encoder, device, k=k)
        print(f"k={k}: Perplexity = {ppl:.4f}")

if __name__ == "__main__":
    k_analysis()
