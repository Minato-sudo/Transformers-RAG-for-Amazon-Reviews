import torch
import os

class RetrievalModule:
    def __init__(self, train_embeddings_path, train_texts):
        self.train_embeddings = torch.load(train_embeddings_path)
        self.train_texts = train_texts

    def retrieve(self, query_embedding, k=3):
        # query_embedding: (1, d_model) or d_model
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
            
        # Compute cosine similarity
        # (batch, d_model) x (d_model, train_size) -> (batch, train_size)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, self.train_embeddings)
        
        # Get top-k indices
        top_k_scores, top_k_indices = torch.topk(similarities, k)
        
        results = []
        for i in range(k):
            idx = top_k_indices[i].item()
            results.append({
                'text': self.train_texts[idx],
                'score': top_k_scores[i].item()
            })
        return results

def test_retrieval():
    base_path = "/home/minato/Documents/NLP_Assignment_3"
    data = torch.load(os.path.join(base_path, 'data', 'processed_data.pt'))
    emb_path = os.path.join(base_path, 'results', 'train_embeddings.pt')
    
    # Get training texts
    train_texts = data['train'][3]
    
    retriever = RetrievalModule(emb_path, train_texts)
    
    # Take a sample from test set
    test_X, _, _, test_texts = data['test']
    
    # Need encoder to get test embedding
    from transformer_scratch import MultiTaskEncoder
    vocab_size = len(data['vocab'])
    model = MultiTaskEncoder(vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=2)
    model.load_state_dict(torch.load(os.path.join(base_path, 'models', 'encoder_weights.pt')))
    model.eval()
    
    sample_idx = 0
    query_text = test_texts[sample_idx]
    query_tensor = test_X[sample_idx].unsqueeze(0)
    
    with torch.no_grad():
        _, _, query_emb = model(query_tensor)
    
    top_k = retriever.retrieve(query_emb, k=3)
    
    print(f"Query Review: {query_text[:200]}...")
    print("-" * 50)
    for i, res in enumerate(top_k):
        print(f"Result {i+1} (Score: {res['score']:.4f}):")
        print(f"{res['text'][:200]}...")
        print("-" * 30)

if __name__ == "__main__":
    test_retrieval()
