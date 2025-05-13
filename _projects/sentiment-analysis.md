---
layout: project
title: "Sentiment Analysis of YouTube Comments using NLP"
categories: nlp machine-learning text-classification
image: /assets/images/sentiment-analysis.jpg
technologies: [Python, NLP, Scikit-Learn, VADER, Flair, HuggingFace]
github: https://github.com/Adredes-weslee/Sentiment-Analysis-and-NLP-for-a-Youtube-Video
---

## Project Overview

Built a binary sentiment classifier on 63,000+ YouTube comments using multiple NLP techniques and hybrid labeling approaches. The goal was to accurately identify negative sentiment in user comments to help content creators better understand audience reception and engagement patterns.

> Read my detailed blog post: [Building a Robust YouTube Comment Sentiment Analyzer](/nlp/machine-learning/sentiment-analysis/2023/07/10/building-youtube-comment-sentiment-analyzer.html)

## Business Context

Content creators and digital marketers face significant challenges in processing and understanding audience reactions at scale. With thousands of comments on popular videos, manually reviewing sentiment becomes impractical. This project addresses several key needs:

1. **Reputation Management**: Quickly identifying negative sentiment spikes that may indicate PR issues
2. **Content Optimization**: Understanding which video segments or topics generate positive/negative reactions
3. **Community Moderation**: Flagging potentially problematic comments for review
4. **Engagement Analysis**: Correlating sentiment with other engagement metrics (views, likes, shares)
5. **Trend Recognition**: Tracking sentiment changes over time or across video series

## Methodology

### Data Collection and Preprocessing

Using the YouTube Data API, I extracted 63,000+ comments from a diverse set of videos:

```python
from googleapiclient.discovery import build
import pandas as pd
from dateutil import parser

# YouTube API setup
api_key = "YOUR_API_KEY"
youtube = build("youtube", "v3", developerKey=api_key)

def get_video_comments(video_id, max_results=100):
    """Extract comments from a YouTube video."""
    comments = []
    
    # Get initial page of comments
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    
    # Continue fetching comments until no more pages
    while request:
        response = request.execute()
        
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            
            # Extract relevant fields
            comments.append({
                "author": comment["authorDisplayName"],
                "text": comment["textDisplay"],
                "likes": comment["likeCount"],
                "published_at": parser.parse(comment["publishedAt"]),
            })
        
        # Check if more comments exist
        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=max_results,
                pageToken=response["nextPageToken"],
                textFormat="plainText"
            )
        else:
            break
            
    return pd.DataFrame(comments)
```

The preprocessing pipeline included specialized handling for social media text:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

class CommentPreprocessor:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Handle emojis (convert to text or remove)
        text = emoji.demojize(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Handle contractions
        text = self._expand_contractions(text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        # Join tokens back into string
        return ' '.join(tokens)
    
    def _expand_contractions(self, text):
        # Handle common contractions
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            # Add more contractions as needed
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            
        return text
```

### Hybrid Labeling Approach

A key innovation in this project was using multiple models to create a reliable training dataset without manual labeling. This approach combined the strengths of:

#### 1. VADER (Valence Aware Dictionary and Sentiment Reasoner)
A rule-based sentiment lexicon specifically tuned for social media content:

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def vader_sentiment(text):
    """Get VADER sentiment scores."""
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    # Use compound score to determine sentiment
    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'
```

#### 2. Flair NLP Framework
A state-of-the-art NLP framework that uses contextual string embeddings:

```python
from flair.models import TextClassifier
from flair.data import Sentence

def flair_sentiment(text):
    """Get Flair sentiment prediction."""
    # Load sentiment analysis model
    classifier = TextClassifier.load('en-sentiment')
    
    # Create sentence object
    sentence = Sentence(text)
    
    # Predict sentiment
    classifier.predict(sentence)
    
    # Extract label and score
    label = sentence.labels[0]
    
    return {
        'label': label.value,  # 'POSITIVE' or 'NEGATIVE'
        'score': label.score   # Confidence score
    }
```

#### 3. HuggingFace Transformers
BERT-based models with state-of-the-art performance on sentiment tasks:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class BertSentimentClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
    def predict(self, text):
        # Prepare inputs
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = F.softmax(outputs.logits, dim=1)
            
        # For binary sentiment (assumes index 0 = negative, 1 = positive)
        sentiment_score = scores[0][1].item()  # Probability of positive
        
        if sentiment_score > 0.6:
            label = "positive"
        elif sentiment_score < 0.4:
            label = "negative"
        else:
            label = "neutral"
            
        return {
            "label": label,
            "score": sentiment_score
        }
```

#### Consensus-Based Labeling

Comments were only included in the training set if at least two models agreed on the sentiment:

```python
def consensus_labeling(comments):
    """Create consensus-based labels from multiple sentiment models."""
    labeled_data = []
    
    for comment in comments:
        text = comment["text"]
        
        # Get sentiment from each model
        vader_result = vader_sentiment(text)
        flair_result = flair_sentiment(text)
        bert_result = bert_classifier.predict(text)
        
        # Convert labels to common format
        vader_label = vader_result
        flair_label = "positive" if flair_result["label"] == "POSITIVE" else "negative"
        bert_label = bert_result["label"]
        
        # Check for consensus (at least 2 models agree)
        if (vader_label == flair_label) or (vader_label == bert_label) or (flair_label == bert_label):
            # Determine consensus label
            if vader_label == flair_label:
                consensus = vader_label
            elif vader_label == bert_label:
                consensus = vader_label
            else:
                consensus = flair_label
                
            # Only include if consensus is binary (positive/negative)
            if consensus != "neutral":
                labeled_data.append({
                    "text": text,
                    "sentiment": consensus
                })
    
    return pd.DataFrame(labeled_data)
```

### Feature Engineering

Multiple feature extraction methods were implemented and compared:

#### TF-IDF Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),  # Unigrams and bigrams
    sublinear_tf=True    # Apply logarithmic term frequency scaling
)

X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])
```

#### Word Embeddings (Word2Vec)

```python
import gensim
import numpy as np

def train_word2vec(texts, vector_size=100):
    """Train a Word2Vec model from scratch."""
    # Tokenize texts
    tokenized_texts = [text.split() for text in texts]
    
    # Train Word2Vec model
    model = gensim.models.Word2Vec(
        tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=4,
        sg=1  # Skip-gram model
    )
    
    return model

def get_document_embedding(text, model, vector_size=100):
    """Create document embedding by averaging word vectors."""
    tokens = text.split()
    embeddings = []
    
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    
    # If no tokens were found in the model
    if not embeddings:
        return np.zeros(vector_size)
    
    # Average all word vectors
    return np.mean(embeddings, axis=0)
```

#### Feature Importance Analysis

After training, top features were analyzed:

```python
def analyze_feature_importance(model, vectorizer, class_names, top_n=20):
    """Extract and visualize most important features from a linear model."""
    # For binary classification, get feature coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
    else:
        # For ensemble models, use feature_importances_
        coefficients = model.feature_importances_
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Create DataFrame of features and coefficients
    features_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    # Sort by absolute value of coefficient
    features_df['abs_coef'] = features_df['coefficient'].abs()
    features_df = features_df.sort_values('abs_coef', ascending=False)
    
    # Get most positive and negative features
    top_positive = features_df[features_df['coefficient'] > 0].head(top_n)
    top_negative = features_df[features_df['coefficient'] < 0].head(top_n)
    
    return {
        'positive': top_positive,
        'negative': top_negative
    }
```

### Model Development and Comparison

I implemented a variety of classification approaches:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Initialize models
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'random_forest': RandomForestClassifier(class_weight='balanced', n_jobs=-1),
    'gradient_boosting': GradientBoostingClassifier(),
    'linear_svc': LinearSVC(class_weight='balanced', dual=False)
}

# Hyperparameter grids
param_grids = {
    'logistic_regression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    'linear_svc': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Grid search for optimal hyperparameters
    grid_search = GridSearchCV(
        model,
        param_grids[name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    
    # Store results
    results[name] = {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
```

### Model Stacking

To improve performance further, I implemented a stacked ensemble:

```python
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict

# Create base models
base_models = [
    ('logistic', results['logistic_regression']['best_model']),
    ('rf', results['random_forest']['best_model']),
    ('gb', results['gradient_boosting']['best_model'])
]

# Create meta-learner
meta_learner = LogisticRegression()

# Setup stacked model
stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

# Train stacked model
stacked_model.fit(X_train, y_train)

# Evaluate
y_pred_stacked = stacked_model.predict(X_test)
stacked_report = classification_report(y_test, y_pred_stacked, output_dict=True)
```

## Results and Performance

The final stacked model achieved:

| Metric | Score |
|--------|-------|
| Accuracy | 0.87 |
| Precision (Negative) | 0.84 |
| Recall (Negative) | 0.83 |
| F1-Score (Negative) | 0.83 |
| Precision (Positive) | 0.89 |
| Recall (Positive) | 0.90 |
| F1-Score (Positive) | 0.90 |

Performance comparison across models:

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | 0.83 | 0.82 | 0.91 |
| Random Forest | 0.84 | 0.84 | 0.93 |
| Gradient Boosting | 0.85 | 0.85 | 0.94 |
| Linear SVC | 0.82 | 0.82 | 0.90 |
| Stacked Ensemble | 0.87 | 0.87 | 0.95 |

## Sentiment Analysis Insights

The project revealed interesting patterns in YouTube comments:

1. **Time-Dependent Sentiment**: Comments posted shortly after video upload tended to be more positive than later comments
2. **Length-Sentiment Correlation**: Highly negative comments were typically longer (avg 45 words) than positive ones (avg 28 words)
3. **Engagement Correlation**: Videos with higher dislike-to-like ratios had more negative comment sections
4. **Topic Sensitivity**: Certain topics consistently triggered more polarized sentiment
5. **Sentiment Evolution**: Comment sentiment often evolved over the lifetime of a video, with early comments setting the tone

## Applications

The sentiment analysis system was used to:

1. **Create an Automated Dashboard**: Visualizing sentiment trends across videos or channels
2. **Implement Toxic Comment Filtering**: Flagging potentially problematic comments for moderation
3. **Generate Content Recommendations**: Using sentiment to guide future content decisions
4. **Compare Creator Performance**: Benchmarking sentiment patterns across similar creators
5. **Correlate with Video Metrics**: Analyzing how sentiment relates to view count, retention, and other metrics

## Technical Implementation

```
project/
├── data/
│   ├── raw/              # Raw comment data
│   ├── processed/        # Preprocessed comments
│   └── labeled/          # Consensus-labeled data
├── models/
│   ├── vectorizers/      # Fitted vectorizers
│   ├── word2vec/         # Word embedding models
│   └── classifiers/      # Trained classification models
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_consensus_labeling.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_visualization.ipynb
├── src/
│   ├── data/
│   │   ├── collect.py    # YouTube API functions
│   │   └── process.py    # Text preprocessing
│   ├── features/
│   │   ├── build.py      # Feature engineering
│   │   └── embeddings.py # Word embedding functions
│   ├── models/
│   │   ├── train.py      # Model training
│   │   └── evaluate.py   # Evaluation metrics
│   └── visualization/
│       └── dashboard.py  # Plotting functions
├── app/
│   ├── app.py            # Streamlit dashboard
│   └── utils.py          # Helper functions
├── requirements.txt
└── README.md
```

## Future Work

Planned enhancements include:

1. **Multi-Class Sentiment**: Expanding beyond binary classification to capture nuanced emotions
2. **Temporal Analysis**: Analyzing how comment sentiment evolves over time
3. **Cross-Language Support**: Adding multilingual capabilities
4. **Sarcasm Detection**: Implementing specialized models to detect sarcastic comments
5. **Real-Time Processing**: Building a system for live sentiment monitoring during video premieres

## Resources and References

- [Sentiment Analysis in Social Media](https://dl.acm.org/doi/10.1145/3308560.3316502)
- [VADER: A Parsimonious Rule-based Model for Sentiment Analysis](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
- [Flair NLP Framework](https://github.com/flairNLP/flair)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [YouTube Data API Documentation](https://developers.google.com/youtube/v3)
