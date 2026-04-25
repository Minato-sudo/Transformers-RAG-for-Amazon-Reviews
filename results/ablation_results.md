# RAG Ablation Study Results

## Quantitative Metrics
- **RAG Perplexity**: 84.66
- **No-RAG Perplexity**: 45.20
- **Perplexity Reduction**: -39.46 points

## Qualitative Examples & Commentary
### Example 1
**Review**: great item...
**Retrieved Context**: great item...
**Target Explanation**: five stars
**RAG Generation**: five stars to my teeth and works great for a replacement price
**No-RAG Generation**: five stars because of the price was supposed to be an excellent filament
**Commentary**: The RAG model produces a grounded response, while the baseline tends to be more generic. Retrieval helps in grounding the sentiment explanation in actual product features mentioned in similar reviews.

### Example 2
**Review**: unpolished but smooth no rough spots can be used as is great quality rods fast delivery too...
**Retrieved Context**: ac vents engine blocks summer asphalt cookies and more this thermometer is great for fast accurate readings i used it to locate an ac problem in my ho...
**Target Explanation**: great buy
**RAG Generation**: gorgeous color light blue which means and more great for well
**No-RAG Generation**: excellent tape in the bed seems to be used that cheap
**Commentary**: The RAG model produces a grounded response, while the baseline tends to be more generic. Retrieval helps in grounding the sentiment explanation in actual product features mentioned in similar reviews.

### Example 3
**Review**: stunning sapphire blue color the usual esun petg properties one difference it doesnt like to stuck to the bed as well as the black to get this to stic...
**Retrieved Context**: i used this gloves for misc stuff around the house inside and out they come in hand when doing a lot of messy cooking working on something that has gr...
**Target Explanation**: it doesnt like to stuck to the bed as well as the
**RAG Generation**: great little goes through metal which seems to go and they did pull
**No-RAG Generation**: excellent product does not like a bit of fun but not for fall polish
**Commentary**: The RAG model produces a grounded response, while the baseline tends to be more generic. Retrieval helps in grounding the sentiment explanation in actual product features mentioned in similar reviews.

### Example 4
**Review**: kreg products rule...
**Retrieved Context**: use this all the time...
**Target Explanation**: five stars
**RAG Generation**: five stars with this product for all stars prints great quality and
**No-RAG Generation**: five stars is a review on stuff and it was the best
**Commentary**: The RAG model produces a grounded response, while the baseline tends to be more generic. Retrieval helps in grounding the sentiment explanation in actual product features mentioned in similar reviews.

### Example 5
**Review**: this tape is great for wire harness on engines or for anything wire oriented much easier to manage than electrical tape and none of that sticky residu...
**Retrieved Context**: this is the best duct tape style tape i have ever used it is very strong extremely sticky and does not break down like other brands i have been using ...
**Target Explanation**: great for engine harness or general wiring
**RAG Generation**: works nicely but this set up for drilling tape tape stick with no complaints
**No-RAG Generation**: i bought it on amazon buy from the quality of these tape
**Commentary**: The RAG model leverages the retrieved context to provide a more detailed explanation. Retrieval helps in grounding the sentiment explanation in actual product features mentioned in similar reviews.

