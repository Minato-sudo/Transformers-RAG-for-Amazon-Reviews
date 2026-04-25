import gzip
import json
import random
import re
import os
import torch
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_amazon_reviews(file_path, category_name, n_samples=10000):
    reviews = []
    print(f"Loading {category_name} reviews from {file_path}...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if 'reviewText' in data and 'overall' in data and 'summary' in data:
                reviews.append({
                    'text': data['reviewText'],
                    'rating': data['overall'],
                    'category': category_name,
                    'summary': data['summary']
                })
    
    if len(reviews) > n_samples:
        reviews = random.sample(reviews, n_samples)
    return reviews

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab(texts, max_vocab_size=10000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    # Special tokens
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3, 
             '<ret>': 4, '<rev>': 5, '<snt>': 6, '<cat>': 7, '<exp>': 8}
    most_common = counter.most_common(max_vocab_size - 9)
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab

def tokenize_and_pad(text, vocab, max_length=128):
    tokens = text.split()
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    # Truncate
    if len(indices) > max_length:
        indices = indices[:max_length]
    
    # Pad
    padding = [vocab['<pad>']] * (max_length - len(indices))
    return indices + padding

def main():
    base_path = "/home/minato/Documents/NLP_Assignment_3"
    files = {
        'Software': os.path.join(base_path, 'Software_5.json.gz'),
        'Industrial': os.path.join(base_path, 'Industrial_and_Scientific_5.json.gz'),
        'Beauty': os.path.join(base_path, 'Luxury_Beauty_5.json.gz')
    }
    
    all_data = []
    for cat, path in files.items():
        all_data.extend(load_amazon_reviews(path, cat, n_samples=10000))
    
    print(f"Total reviews collected: {len(all_data)}")
    
    # Map sentiments
    # 1-2 Negative (0), 3 Neutral (1), 4-5 Positive (2)
    sentiment_map = {1.0: 0, 2.0: 0, 3.0: 1, 4.0: 2, 5.0: 2}
    category_map = {'Software': 0, 'Industrial': 1, 'Beauty': 2}
    
    processed_texts = []
    processed_summaries = []
    labels_sentiment = []
    labels_category = []
    
    for item in all_data:
        cleaned_text = clean_text(item['text'])
        cleaned_summ = clean_text(item['summary'])
        if cleaned_text and cleaned_summ: # Skip empty
            processed_texts.append(cleaned_text)
            processed_summaries.append(cleaned_summ)
            labels_sentiment.append(sentiment_map[item['rating']])
            labels_category.append(category_map[item['category']])
            
    # Split data (70/15/15)
    train_texts, temp_texts, train_summ, temp_summ, train_sent, temp_sent, train_cat, temp_cat = train_test_split(
        processed_texts, processed_summaries, labels_sentiment, labels_category, test_size=0.3, stratify=labels_sentiment, random_state=42
    )
    
    val_texts, test_texts, val_summ, test_summ, val_sent, test_sent, val_cat, test_cat = train_test_split(
        temp_texts, temp_summ, temp_sent, temp_cat, test_size=0.5, stratify=temp_sent, random_state=42
    )
    
    # Build vocab from training data only
    print("Building vocabulary...")
    vocab = build_vocab(train_texts)
    print(f"Vocab size: {len(vocab)}")
    
    # Numericalize
    max_len = 100
    def prepare_tensors(texts, sents, cats):
        X = torch.tensor([tokenize_and_pad(t, vocab, max_len) for t in texts], dtype=torch.long)
        Y_s = torch.tensor(sents, dtype=torch.long)
        Y_c = torch.tensor(cats, dtype=torch.long)
        return X, Y_s, Y_c
    
    print("Preparing tensors...")
    train_X, train_Ys, train_Yc = prepare_tensors(train_texts, train_sent, train_cat)
    val_X, val_Ys, val_Yc = prepare_tensors(val_texts, val_sent, val_cat)
    test_X, test_Ys, test_Yc = prepare_tensors(test_texts, test_sent, test_cat)
    
    # Save processed data
    data_output = {
        'train': (train_X, train_Ys, train_Yc, train_texts, train_summ),
        'val': (val_X, val_Ys, val_Yc, val_texts, val_summ),
        'test': (test_X, test_Ys, test_Yc, test_texts, test_summ),
        'vocab': vocab,
        'max_len': max_len
    }
    
    save_path = os.path.join(base_path, 'data', 'processed_data.pt')
    torch.save(data_output, save_path)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    main()
