---
layout: project
title: "Sentiment Analysis of YouTube Comments using NLP"
categories: nlp machine-learning text-classification
image: /assets/images/placeholder.svg
technologies: [Python, NLP, Scikit-Learn, VADER, Flair, HuggingFace]
github: https://github.com/Adredes-weslee/Sentiment-Analysis-and-NLP-for-a-Youtube-Video
---

## Project Overview

Built a binary sentiment classifier on 63,000+ YouTube comments using multiple NLP techniques and hybrid labeling approaches. The goal was to accurately identify negative sentiment in user comments to help content creators better understand audience reception.

## Methodology

### Data Collection and Preprocessing

- Extracted 63,000+ comments from YouTube videos using the YouTube API
- Applied rigorous preprocessing including:
  - Text normalization and tokenization
  - Removing stopwords, emojis, and special characters
  - Lemmatization to reduce words to their base forms
  - Handling of misspellings and slang

### Hybrid Labeling Approach

To create reliable ground truth data, I implemented a hybrid labeling approach using multiple sentiment analysis tools:

- **VADER** (Valence Aware Dictionary and Sentiment Reasoner) - Rule-based model
- **Flair** - State-of-the-art NLP framework
- **HuggingFace Transformers** - BERT-based sentiment models

Comments with consistent sentiment scores across multiple models were selected for the training dataset, reducing noise and label uncertainty.

### Feature Engineering

- **Bag of Words** using CountVectorizer
- **TF-IDF** vectorization to account for term importance
- **Word2Vec** embeddings for semantic understanding
- **N-gram analysis** to capture multi-word phrases
- **Part-of-speech (POS) tagging** to extract grammatical patterns

### Model Development

Implemented and compared multiple classification approaches:

- Logistic Regression
- Ensemble methods (Random Forest, XGBoost)
- Model stacking for improved performance

## Results

- Achieved F1-score of 0.655 for negative sentiment using lemmatized input with CountVectorizer
- Successfully handled real-world class imbalance through weighting and sampling techniques
- Provided interpretable results through feature importance and coefficient analysis

## Technical Implementation

```python
# Sample code showing the hybrid labeling approach
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from transformers import pipeline

# Initialize sentiment analyzers
vader = SentimentIntensityAnalyzer()
flair_sentiment = TextClassifier.load('en-sentiment')
huggingface_sentiment = pipeline('sentiment-analysis')

def get_hybrid_label(comment):
    # Get VADER sentiment
    vader_scores = vader.polarity_scores(comment)
    vader_sentiment = 'positive' if vader_scores['compound'] >= 0.05 else 'negative'
    
    # Get Flair sentiment
    flair_sentence = Sentence(comment)
    flair_sentiment.predict(flair_sentence)
    flair_label = flair_sentence.labels[0].value
    
    # Get HuggingFace sentiment
    hf_result = huggingface_sentiment(comment)[0]
    hf_label = hf_result['label'].lower()
    
    # Only return a label if at least 2 models agree
    if (vader_sentiment == flair_label == hf_label) or \
       (vader_sentiment == flair_label) or \
       (vader_sentiment == hf_label) or \
       (flair_label == hf_label):
        # Return the majority label
        labels = [vader_sentiment, flair_label, hf_label]
        return max(set(labels), key=labels.count)
    else:
        # Models disagree significantly
        return None
```

## Challenges and Solutions

- **Class Imbalance**: Applied weighted classes and SMOTE for balanced training
- **Noisy Text**: Developed robust preprocessing pipeline to handle internet language
- **Computational Efficiency**: Optimized vectorization and model training for large dataset

## Technologies Used

- **Python** - Core programming language
- **NLTK & spaCy** - Natural language processing libraries
- **Scikit-learn** - Machine learning framework
- **VADER** - Rule-based sentiment analysis
- **Flair & HuggingFace** - Deep learning NLP libraries
- **Pandas & NumPy** - Data manipulation
