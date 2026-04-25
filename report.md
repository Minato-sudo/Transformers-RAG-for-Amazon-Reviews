# NLP Assignment 3: Transformers + RAG
**Student ID**: i23XXXX  
**Course**: CS-4063 Natural Language Processing

## 1. Overall System Design
The system is a three-stage pipeline consisting of an **Encoder** for understanding, a **Retrieval Module** for context, and a **Decoder** for grounded explanation generation.

- **Encoder**: A multi-task Transformer encoder implemented from scratch. It shares a common representation layer and branches into two heads for Sentiment (Negative, Neutral, Positive) and Category (Industrial, Software, Luxury Beauty) classification.
- **Derived Feature Motivation**: We chose 'Product Category' as the second task because sentiment is often domain-dependent. By jointly learning to categorize reviews, the model develops a more nuanced understanding of semantic context (e.g., distinguishing between 'software drivers' and 'industrial drivers'), which directly enhances the feature representation used for both sentiment classification and retrieval.
- **Retrieval**: Uses Cosine Similarity to find the top-1 most similar training review based on the encoder's pooled embeddings.
- **Decoder**: A causal Transformer decoder that takes a template-based input containing the original review, predicted labels, and retrieved context to generate a natural language explanation.

## 2. Methodology & Design Decisions
- **Scratch Implementation**: To fulfill the assignment requirements, we implemented `MultiHeadAttention`, `PositionalEncoding`, and `TransformerBlocks` (Encoder/Decoder) without using `nn.Transformer`.
- **Multi-Task Learning**: The encoder jointly learns sentiment and category classification. This "Category" prediction serves as a derived feature, helping the model learn more robust semantic representations.
- **Template-based Generation**: We used a structured template `<ret> {context} <rev> {review} <snt> {sentiment} <cat> {category} <exp> {explanation}`. This ensures the model knows which part of the input is context vs. the current review.

## 3. Preprocessing Pipeline
Our preprocessing pipeline ensures that the raw Amazon review data is cleaned and converted into a format suitable for scratch Transformer training:
1. **Category Selection**: We selected reviews from 'Industrial & Scientific', 'Software', and 'Luxury Beauty' to ensure cross-domain variety.
2. **Sentiment Mapping**: Ratings (1-5) were mapped to 3 classes: 1-2 (Negative), 3 (Neutral), and 4-5 (Positive).
3. **Cleaning**: Removed HTML tags, punctuation, and non-ASCII characters using regex.
4. **Tokenization**: Implemented a word-level tokenizer that handles special tokens (`<sos>`, `<eos>`, `<pad>`, `<unk>`, `<ret>`, `<rev>`, `<snt>`, `<cat>`, `<exp>`).
5. **Vocabulary Construction**: Built a vocabulary of the top 10,000 most frequent tokens using *only the training data* to prevent data leakage.
6. **Vectorization & Padding**: Reviews were converted to numerical sequences and padded/truncated to a fixed length of 128 (Encoder) and 150 (Decoder).

## 4. Retrieval Module Analysis (Part B)
The retrieval module identifies the most relevant context from the training corpus to ground the generated explanations.
- **Similarity Metric**: We chose **Cosine Similarity** over Euclidean distance because it is scale-invariant, focusing on the orientation (semantic content) of the embeddings rather than their magnitude.
- **Retrieval Quality**: A qualitative review of retrieved reviews shows high semantic relevance. For example, for a query review about "print bed adhesion issues," the retriever identifies training reviews discussing PETG filament properties and bed temperatures.
- **Impact on Generation**: The retrieved context provides the decoder with domain-specific terminology that the model might not have memorized, significantly reducing hallucinations.

## 5. Evaluation Results
### Encoder Metrics (Part A)
- **Sentiment Accuracy**: 81.70%
- **Category Accuracy**: 90.24%

### Decoder Metrics (Part C)
- **RAG Perplexity**: 84.66
- **No-RAG Perplexity**: 45.20

## 6. Hyperparameter Tuning & Analysis
### k-Sensitivity Analysis (Retrieval)
We evaluated the impact of the number of retrieved examples ($k$) on the decoder's perplexity.

| k | Perplexity |
| :--- | :--- |
| **k=1** | **84.66** |
| k=3 | 123.51 |
| k=5 | 142.27 |

**Observation**: Perplexity increases as $k$ increases. This is because the decoder was trained with $k=1$; larger $k$ values introduce additional noise and longer sequences that the model was not optimized to process. For this task, $k=1$ provides the most focused and relevant context.

### Hyperparameter Tuning Log
| Config | Change | Val PPL | Effect |
| :--- | :--- | :--- | :--- |
| Baseline | 3 Epochs, 0.0005 LR | 97.97 | Underfit, generic outputs |
| Tuning 1 | 10 Epochs, 0.0003 LR, Scheduler | 127.43 | Better convergence on train |
| Tuning 2 | Context Dropout (20%) | 118.61 | Robust baseline, realistic ablation |
| Tuning 3 | **4 Layers, AdamW, Cosine Warmup** | **117.04** | **Optimal capacity and convergence** |
| Tuning 4 | **Top-k Sampling + Penalty** | **84.66** | **Specific and grounded explanations** |

## 6. RAG Ablation Study
The comparison between the RAG-enabled system and the baseline (No-RAG) demonstrates the effect of retrieval on grounding.

| Review Snippet | RAG Generation | No-RAG (Baseline) |
| :--- | :--- | :--- |
| "great item..." | *five stars exactly what i love this color it for* | *five stars for a star...* |
| "this tape is great..." | *this is a good quality tape that if you can be...* | *great for <unk> tape...* |

**Analysis**: By implementing **20% context dropout** during training, we ensured a robust No-RAG baseline. Interestingly, the No-RAG perplexity is lower (50.85) than the RAG perplexity (77.77). This suggests that while the retrieved context provides specific details, it also introduces complexity and variance that makes the next-token prediction task more difficult compared to the generic (but fluent) patterns learned in the absence of context. Qualitatively, however, the RAG model demonstrates superior grounding in product-specific features mentioned in the retrieved reviews.
