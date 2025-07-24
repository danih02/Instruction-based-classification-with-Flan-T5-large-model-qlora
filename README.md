# Instruction-based Classification with Flan-T5 Large Model + QLoRA

This project demonstrates instruction-based classification of news articles using the Flan-T5 large model, enhanced with QLoRA for efficient fine-tuning. The workflow covers data exploration, zero-shot and few-shot inference, embedding-based classification, and model training.

## Workflow Overview

1. **Dataset Exploration**
   - Loads the AG News dataset (4 categories: World, Sports, Business, Sci/Tech).
   - Visualizes word count and label distribution.
   - Maps label IDs to names.

2. **Model Setup**
   - Loads Flan-T5 large with quantization (BitsAndBytesConfig).
   - Prepares tokenizer and model for training/inference.

3. **Zero-Shot Classification**
   - Constructs prompts for zero-shot inference.
   - Tokenizes dataset and runs predictions.
   - Evaluates results with a confusion matrix.

4. **Few-Shot Classification**
   - Builds prompts with example articles for selected categories.
   - Tokenizes and predicts with the model.
   - Handles label mapping inconsistencies.

5. **Embedding-Based Classification**
   - Uses SentenceTransformer and Google Gemini embeddings.
   - Trains a logistic regression classifier on embeddings.
   - Evaluates with cosine similarity and confusion matrices.

6. **Model Training**
   - Demonstrates QLoRA setup for efficient fine-tuning.
   - Uses Hugging Face Trainer for supervised training.
   - Computes accuracy and F1 metrics.

## Requirements

- Python 3.8+
- `transformers`, `datasets`, `bitsandbytes`, `sentence-transformers`, `scikit-learn`, `matplotlib`, `seaborn`
- (Optional) Google Colab for Gemini embeddings

## Usage

Open [Instruction_based_classification_with_Flan_T5_large_model_+_qlora.ipynb](Instruction_based_classification_with_Flan_T5_large_model_+_qlora.ipynb) and run the cells sequentially. Adjust dataset size, batch size, and model parameters as needed for your hardware.

## References

- [Flan-T5 Model](https://huggingface.co/google/flan-t5-large)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)

---
For details, see the notebook:
