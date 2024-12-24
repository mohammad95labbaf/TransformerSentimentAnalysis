
# Twitter Sentiment Analysis with Transformer Models

This repository contains a Jupyter Notebook for performing sentiment analysis on a Twitter dataset using various transformer models, including BERT, RoBERTa, DistilBERT, ALBERT, and XLNet. The notebook provides a complete pipeline for loading the data, preprocessing it, training the models, and evaluating their performance.

## Dataset Overview

The dataset consists of tweets related to 32 unique entities, each labeled with one of four sentiment categories:

- **Negative**: Indicates unfavorable sentiment.
- **Positive**: Indicates favorable sentiment.
- **Neutral**: Indicates no strong sentiment.
- **Irrelevant**: Indicates content unrelated to the target entity.

### Data Columns
- **Tweet ID**: Unique identifier for each tweet.
- **Entity**: The subject or topic discussed in the tweet (e.g., Overwatch, PlayStation5).
- **Sentiment**: The sentiment expressed in the tweet (Negative, Positive, Neutral, Irrelevant).
- **Tweet Content**: The actual text of the tweet.

### Data Splits
- **Training Set**: 59,745 tweets used for training the model.
- **Validation Set**: 14,937 tweets used for evaluating model performance.

### Sentiment Distribution
- **Negative**: 30.3% of the dataset.
- **Positive**: 27.5% of the dataset.
- **Neutral**: 24.6% of the dataset.
- **Irrelevant**: 17.5% of the dataset.

## Models Implemented

The notebook uses the following transformer models for sentiment analysis:

- **BERT**: Bidirectional Encoder Representations from Transformers.
- **RoBERTa**: A robustly optimized BERT pretraining approach.
- **DistilBERT**: A smaller, faster version of BERT.
- **ALBERT**: A lite version of BERT with fewer parameters.
- **XLNet**: A generalized autoregressive pretraining model.

## How to Use

### Step 1: Load the Dataset
The notebook loads the dataset containing tweet IDs, entities, sentiments, and tweet content. The dataset is pre-split into training and validation sets.

### Step 2: Preprocess the Data
The notebook includes a preprocessing step that:
- Tokenizes the tweet text using the appropriate tokenizer for each transformer model.
- Converts sentiment labels into numerical format for model training.
- Pads and truncates sequences to a fixed length for consistency.

### Step 3: Train the Model
The notebook allows you to choose from the following transformer models:
- **BERT**
- **RoBERTa**
- **DistilBERT**
- **ALBERT**
- **XLNet**

You can fine-tune the model of your choice on the training set. The notebook provides options to adjust hyperparameters like batch size, learning rate, and number of epochs.

### Step 4: Evaluate the Model
Once trained, the model is evaluated on the validation set. Evaluation metrics include:
- **Accuracy**: Proportion of correct predictions.
- **Precision, Recall, F1-Score**: For each sentiment class (Negative, Positive, Neutral, Irrelevant).
- **Confusion Matrix**: Visualizes the model’s performance by comparing true vs. predicted labels.

### Step 5: Visualize Results
The notebook includes visualizations, including:
- A confusion matrix to understand misclassifications.
- A classification report for detailed performance metrics.

### Step 6: Save the Model
After training, the model can be saved to disk for later use. The notebook provides a function to export the model and tokenizer for inference.

```python
model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_model')
```

## Example Usage

Here’s a simple example of how to run the notebook:

1. Load and preprocess the dataset.
2. Fine-tune a transformer model (e.g., BERT).
3. Evaluate the model and visualize performance metrics.
4. Save the trained model for future use.
