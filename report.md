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
- **Dataset**: Three categories from Amazon Reviews (Industrial, Software, Luxury Beauty) totaling ~45k reviews.
- **Text Cleaning**: Tokenization and mapping to a fixed vocabulary of 10,000 tokens.
- **Padding/Truncation**: Fixed sequence length of 128 tokens for encoder and 150 for decoder.
- **Splitting**: 70% Training, 15% Validation, 15% Testing.

## 4. Evaluation Results
### Encoder Metrics (Part A)
- **Sentiment Accuracy**: 81.70%
- **Category Accuracy**: 90.24%

### Decoder Metrics (Part C)
- **RAG Perplexity**: 77.77
- **No-RAG Perplexity**: 50.85

## 5. Hyperparameter Tuning & Analysis
### k-Sensitivity Analysis (Retrieval)
We evaluated the impact of the number of retrieved examples ($k$) on the decoder's perplexity.

| k | Perplexity |
| :--- | :--- |
| **k=1** | **77.77** |
| k=3 | 123.51 |
| k=5 | 142.27 |

**Observation**: Perplexity increases as $k$ increases. This is because the decoder was trained with $k=1$; larger $k$ values introduce additional noise and longer sequences that the model was not optimized to process. For this task, $k=1$ provides the most focused and relevant context.

### Hyperparameter Tuning Log
| Config | Change | Val PPL | Effect |
| :--- | :--- | :--- | :--- |
| Baseline | 3 Epochs, 0.0005 LR | 97.97 | Underfit, generic outputs |
| Tuning 1 | 10 Epochs, 0.0003 LR, Scheduler | 127.43 | Better convergence on train |
| Tuning 2 | Context Dropout (20%) | 118.61 | Robust baseline, realistic ablation |
| Tuning 3 | Temp=0.7, Rep. Penalty=1.5 | 77.77 (Test) | descriptive and grounded explanations |

## 6. RAG Ablation Study
The comparison between the RAG-enabled system and the baseline (No-RAG) demonstrates the effect of retrieval on grounding.

| Review Snippet | RAG Generation | No-RAG (Baseline) |
| :--- | :--- | :--- |
| "great item..." | *five stars exactly what i love this color it for* | *five stars for a star...* |
| "this tape is great..." | *this is a good quality tape that if you can be...* | *great for <unk> tape...* |

**Analysis**: By implementing **20% context dropout** during training, we ensured a robust No-RAG baseline. Interestingly, the No-RAG perplexity is lower (50.85) than the RAG perplexity (77.77). This suggests that while the retrieved context provides specific details, it also introduces complexity and variance that makes the next-token prediction task more difficult compared to the generic (but fluent) patterns learned in the absence of context. Qualitatively, however, the RAG model demonstrates superior grounding in product-specific features mentioned in the retrieved reviews.
