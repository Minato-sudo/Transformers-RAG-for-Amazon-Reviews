# Transformer-based Review Understanding with RAG-Enhanced Generation

An end-to-end NLP system built from scratch (no `nn.Transformer`) that extracts structured information from Amazon reviews and generates grounded sentiment explanations using Retrieval-Augmented Generation (RAG).

## 🚀 Key Results
- **Sentiment Classification Accuracy**: ~82%
- **Category Classification Accuracy**: ~92%
- **Generation Perplexity (RAG)**: **40.10** (vs. 55.42 without RAG)
- **Dataset**: 30,000+ Amazon Reviews (Software, Industrial & Scientific, Luxury Beauty)

---

## 🏗️ System Architecture

### Part A: Multi-Task Encoder
A custom Transformer Encoder (4 layers, 8 heads, $d_{model}=256$) that performs simultaneous classification of:
1. **Sentiment**: Negative (1-2*), Neutral (3*), and Positive (4-5*).
2. **Product Category**: Software, Industrial, or Beauty.
3. **Embeddings**: Produces a 256-dimensional semantic vector for each review via global average pooling.

### Part B: Retrieval Module
A scale-invariant retrieval system using **Cosine Similarity**. For any query review, it retrieves the most semantically relevant review from the training corpus to provide grounded context for the decoder.

### Part C: Causal Decoder (RAG)
A 6-layer Decoder-only Transformer ($d_{model}=256, d_{ff}=1024$) that generates high-quality summary explanations. 
- **Training**: Uses Masked Explanation Loss (gradients only on generated text).
- **Inference**: Implements Nucleus Sampling (Top-p=0.9), Top-k filtering, and Repetition Penalty for fluent outputs.

---

## 📁 Project Structure
```text
├── data/
│   └── processed_data.pt          # Tokenized and stratified dataset
├── models/
│   ├── encoder_weights.pt         # Trained Multi-Task Encoder
│   └── decoder_weights.pt         # Trained Causal Decoder
├── results/
│   ├── train_embeddings.pt        # Pre-computed training embeddings
│   ├── confusion_matrices.png     # Performance visualizations
│   └── ablation_results.md        # RAG vs. No-RAG comparison
├── scripts/
│   ├── transformer_scratch.py     # Core Architecture (from scratch)
│   ├── data_prep.py               # Preprocessing pipeline
│   ├── train_encoder.py           # Part A Training
│   ├── train_decoder.py           # Part C Training
│   └── ablation_study.py          # Part B/C Evaluation
├── i232548-NLP-Assignment3.ipynb  # Main Project Notebook
├── report.md                      # Detailed Technical Report
└── README.md                      # This file
```

---

## 🛠️ Setup and Usage

### 1. Environment Activation
Activate the virtual environment to ensure all dependencies are available:
```bash
source venv/bin/activate
```

### 2. Dataset Setup
Place the raw `.json.gz` Amazon review files in the root directory and run the data preparation script:
```bash
python scripts/data_prep.py
```

### 3. Training & Evaluation
To reproduce the results, run the scripts in order:
```bash
python scripts/train_encoder.py  # Train Part A
python scripts/train_decoder.py  # Train Part C
python scripts/ablation_study.py # Generate Results
```

### 4. Interactive Notebook
For a full walkthrough with visualizations and qualitative examples, open the Jupyter Notebook:
`i232548-NLP-Assignment3.ipynb`

---

## 🎓 Author
**Student ID**: i232548  
**Course**: CS-4063 Natural Language Processing  
**University**: FAST National University of Computer & Emerging Sciences
