---
layout: post
title: "Building a YouTube Sentiment Classifier: An NLP Deep Dive into Hybrid Labeling and Model Stacking"
date: 2023-07-10 09:30:00 +0800 # Retaining original date
categories: [nlp, machine-learning, sentiment-analysis]
tags: [youtube, text-classification, python, scikit-learn, nltk, vader, flair, huggingface-transformers, feature-engineering, model-stacking]
author: Wes Lee
feature_image: /assets/images/2023-07-10-building-youtube-comment-sentiment-analyzer.jpg # Or a new, more technical image
---

## Introduction: The Challenge of Understanding YouTube Sentiment at Scale

YouTube comments are a firehose of audience feedback, but their sheer volume and informal nature make manual sentiment analysis impractical. This post details the technical journey of building a robust binary sentiment classifier for YouTube comments, focusing on a hybrid labeling strategy, advanced preprocessing, diverse feature engineering, and model stacking to achieve reliable performance. Our goal was to accurately identify negative sentiment to help content creators manage reputation and optimize content.

> For a higher-level overview of this project's business context, key findings, and applications, please see the [*YouTube Comment Sentiment Analysis: Unlocking Audience Insights for Content Creators* Project Page](/projects/sentiment-analysis/).

## Phase 1: Data Acquisition and Preparation

The foundation of any NLP project is high-quality data.

### 1. Collecting YouTube Comments via API
We started by collecting a large corpus of comments. Using the YouTube Data API v3, we extracted over 63,000 comments (initially over 99,000 including replies, then cleaned) from a diverse set of videos, including one of Justin Bieber's most commented ones.

```python
from googleapiclient.discovery import build
import pandas as pd
from dateutil import parser # For parsing datetime strings

# api_key = "YOUR_YOUTUBE_API_KEY" # Store securely, e.g., in environment variables
# youtube = build("youtube", "v3", developerKey=api_key)

def get_video_comments_and_replies(video_id, youtube_service, max_pages=None):
    """
    Extracts comments and their replies from a YouTube video.
    Args:
        video_id (str): The ID of the YouTube video.
        youtube_service: Initialized YouTube Data API service instance.
        max_pages (int, optional): Maximum number of pages of comments to fetch. Defaults to None (all pages).
    Returns:
        pd.DataFrame: DataFrame containing comment details.
    """
    comments_data = []
    next_page_token = None
    pages_fetched = 0

    while True:
        try:
            request = youtube_service.commentThreads().list(
                part="snippet,replies", # Include replies in the request
                videoId=video_id,
                maxResults=100, # Max allowed by API per page
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()

            for item in response["items"]:
                top_level_comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments_data.append({
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "author": top_level_comment_snippet.get("authorDisplayName"),
                    "text": top_level_comment_snippet.get("textDisplay"),
                    "likes": top_level_comment_snippet.get("likeCount"),
                    "published_at": parser.parse(top_level_comment_snippet.get("publishedAt")) if top_level_comment_snippet.get("publishedAt") else None,
                    "is_reply": False,
                    "parent_id": None
                })

                # Extract replies if they exist
                if item.get("replies"):
                    for reply_item in item["replies"]["comments"]:
                        reply_snippet = reply_item["snippet"]
                        comments_data.append({
                            "comment_id": reply_item["id"],
                            "author": reply_snippet.get("authorDisplayName"),
                            "text": reply_snippet.get("textDisplay"),
                            "likes": reply_snippet.get("likeCount"),
                            "published_at": parser.parse(reply_snippet.get("publishedAt")) if reply_snippet.get("publishedAt") else None,
                            "is_reply": True,
                            "parent_id": reply_snippet.get("parentId")
                        })
            
            next_page_token = response.get("nextPageToken")
            pages_fetched += 1
            if not next_page_token or (max_pages and pages_fetched >= max_pages):
                break
        except Exception as e:
            print(f"An API error occurred: {e}")
            break
            
    df = pd.DataFrame(comments_data)
    # Basic cleaning: drop rows where text is missing, remove duplicates by comment_id
    df.dropna(subset=['text'], inplace=True)
    df.drop_duplicates(subset=['comment_id'], keep='first', inplace=True)
    return df

# Example usage:
# video_to_scrape = "VIDEO_ID_HERE" 
# all_comments_df = get_video_comments_and_replies(video_to_scrape, youtube)
# print(f"Collected {len(all_comments_df)} comments and replies.")
```

### 2. Preprocessing for Noisy Social Media Text
YouTube comments are notoriously informal. A robust preprocessing pipeline is essential.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize # Added for tokenization
import emoji # For handling emojis
import contractions # For expanding contractions (pip install contractions)

# Ensure NLTK resources are downloaded (run once)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True) # For WordNet
# nltk.download('punkt', quiet=True) # For word_tokenize

class YouTubeCommentPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add common YouTube/social media slang to stopwords if needed
        self.stop_words.update(['im', 'u', 'ur', 'r', 'pls', 'plz', 'thx', 'youtu', 'http', 'https'])
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return "" # Handle non-string inputs

        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Expand contractions (e.g., "don't" -> "do not")
        text = contractions.fix(text)
        
        # 3. Handle emojis (convert to text description)
        text = emoji.demojize(text, delimiters=(" :", ": ")) # e.g., 😂 -> :face_with_tears_of_joy:
        
        # 4. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # 5. Remove HTML tags (if any)
        text = re.sub(r'<.*?>', ' ', text)
        
        # 6. Remove user mentions (e.g., @username)
        text = re.sub(r'@\w+', ' ', text)
        
        # 7. Remove special characters and numbers, keep spaces and basic punctuation for context
        text = re.sub(r'[^a-z\s]', ' ', text) # Keep only letters and spaces
        
        # 8. Tokenize
        tokens = word_tokenize(text)
        
        # 9. Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1: # Remove short tokens too
                processed_tokens.append(self.lemmatizer.lemmatize(token))
        
        # 10. Join tokens back into a string
        return ' '.join(processed_tokens)

# Example usage:
# preprocessor = YouTubeCommentPreprocessor()
# sample_comment = "OMG this is sooo FUNNY 😂😂😂 thx 4 sharing @user123! check out my site [www.example.com](https://www.example.com) I can't stop laughing"
# processed_comment = preprocessor.preprocess_text(sample_comment)
# print(f"Original: {sample_comment}")
# print(f"Processed: {processed_comment}")
```
Key steps included lowercasing, contraction expansion, emoji demojization, URL/HTML removal, special character filtering, tokenization, stopword removal, and lemmatization.

## Phase 2: The Hybrid Labeling Strategy - Creating Reliable Training Data

Manually labeling 63,000+ comments is infeasible. We developed a hybrid, consensus-based labeling approach using three distinct sentiment analysis tools:

**1. VADER (Valence Aware Dictionary and sEntiment Reasoner):** A lexicon and rule-based tool optimized for social media text.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon', quiet=True) # Run once

def get_vader_sentiment_label(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']
    if compound_score >= 0.05:
        return 'positive' # Or 1
    elif compound_score <= -0.05:
        return 'negative' # Or 0
    else:
        return 'neutral' # Or handle as needed
```

**2. Flair NLP:** Utilizes contextual string embeddings (pre-trained models).

```python
from flair.models import TextClassifier
from flair.data import Sentence

# Load Flair sentiment classifier (load once globally or in a class __init__)
# flair_classifier = TextClassifier.load('en-sentiment')

def get_flair_sentiment_label(text, classifier):
    if not text.strip(): # Flair needs non-empty string
        return 'neutral'
    sentence = Sentence(text)
    classifier.predict(sentence)
    # Flair returns POSITIVE/NEGATIVE label and a score
    flair_label_obj = sentence.labels[0]
    sentiment = flair_label_obj.value.lower() # 'positive' or 'negative'
    # confidence = flair_label_obj.score
    return sentiment
```

**3. HuggingFace Transformers (DistilBERT):** A pre-trained BERT-based model fine-tuned for sentiment.

```python
from transformers import pipeline

# Load HuggingFace sentiment pipeline (load once globally or in a class __init__)
# hf_sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_huggingface_sentiment_label(text, hf_pipeline):
    if not text.strip():
        return 'neutral'
    results = hf_pipeline(text)
    # HuggingFace returns 'LABEL_1' (POSITIVE) or 'LABEL_0' (NEGATIVE) for some models, or 'POSITIVE'/'NEGATIVE'
    hf_label = results[0]['label'].lower()
    if 'positive' in hf_label or 'label_1' in hf_label: # Adapt based on specific model output
        return 'positive'
    elif 'negative' in hf_label or 'label_0' in hf_label:
        return 'negative'
    return 'neutral'
```

**Consensus Logic:**
A comment was included in the training set for binary classification (positive/negative) only if at least two of the three models agreed on its sentiment, and that consensus was not 'neutral'.

```python
# df['vader_pred'] = df['processed_text'].apply(lambda x: get_vader_sentiment_label(x))
# df['flair_pred'] = df['processed_text'].apply(lambda x: get_flair_sentiment_label(x, flair_classifier))
# df['hf_pred'] = df['processed_text'].apply(lambda x: get_huggingface_sentiment_label(x, hf_sentiment_pipeline))

def determine_consensus_label(row):
    predictions = [row['vader_pred'], row['flair_pred'], row['hf_pred']]
    positive_votes = predictions.count('positive')
    negative_votes = predictions.count('negative')

    if positive_votes >= 2:
        return 'positive'
    elif negative_votes >= 2:
        return 'negative'
    else:
        return 'ambiguous' # Or 'neutral', or None to filter out

# df['consensus_label'] = df.apply(determine_consensus_label, axis=1)
# training_df = df[df['consensus_label'].isin(['positive', 'negative'])].copy()
# training_df['sentiment_numeric'] = training_df['consensus_label'].apply(lambda x: 1 if x == 'positive' else 0)
```
This yielded a more reliable, albeit smaller, dataset for training our custom classifier.

## Phase 3: Feature Engineering - Extracting Signals from Text

We experimented with several feature extraction techniques:

**1. TF-IDF Vectorization:** Captures word importance using Term Frequency-Inverse Document Frequency, including unigrams and bigrams.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vectorizer = TfidfVectorizer(
#     max_features=10000, # Limit feature space
#     min_df=5,           # Ignore terms that appear in less than 5 documents
#     max_df=0.8,         # Ignore terms that appear in more than 80% of documents
#     ngram_range=(1, 2), # Consider both unigrams and bigrams
#     sublinear_tf=True   # Apply logarithmic scaling to term frequency
# )
# X_tfidf = tfidf_vectorizer.fit_transform(training_df['processed_text'])
```

**2. Word Embeddings (Word2Vec):** Trained a Word2Vec model on our corpus to generate dense vector representations for words, then averaged word vectors to get document embeddings.

```python
import gensim
import numpy as np

def train_custom_word2vec(tokenized_texts_list, vector_size=100, window=5, min_count=2, workers=4, sg=1):
    """Trains a Word2Vec model."""
    # model = gensim.models.Word2Vec(
    #     tokenized_texts_list,
    #     vector_size=vector_size,
    #     window=window,
    #     min_count=min_count,
    #     workers=workers,
    #     sg=sg # 1 for skip-gram; 0 for CBOW
    # )
    # return model

def create_document_vector(text, word2vec_model, vector_size=100):
    """Averages word vectors in a document."""
    tokens = text.split()
    # word_vectors = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    # if not word_vectors:
    #     return np.zeros(vector_size)
    # return np.mean(word_vectors, axis=0)

# tokenized_corpus = [text.split() for text in training_df['processed_text']]
# w2v_model = train_custom_word2vec(tokenized_corpus)
# X_word2vec = np.array([create_document_vector(text, w2v_model) for text in training_df['processed_text']])
```

**3. Parts-of-Speech (POS) Features (Conceptual):** While explored, TF-IDF and embeddings were primary. Extracting counts of adjectives or adverbs could also be a feature type.

## Phase 4: Model Development, Comparison, and Stacking

We trained and evaluated several classification models:
-   Logistic Regression
-   Random Forest
-   Gradient Boosting Classifier
-   Linear SVC

Hyperparameters were tuned using `GridSearchCV` with 5-fold cross-validation, optimizing for F1-score.

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# y = training_df['sentiment_numeric']
# X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)


# Example for Logistic Regression
# log_reg_params = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}
# log_reg_grid = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), log_reg_params, cv=5, scoring='f1_weighted', n_jobs=-1)
# log_reg_grid.fit(X_train, y_train)
# best_log_reg = log_reg_grid.best_estimator_
# y_pred_log_reg = best_log_reg.predict(X_test)
# print("Logistic Regression Report:\n", classification_report(y_test, y_pred_log_reg))
```
The original project mentioned Logistic Regression with CountVectorizer (1-3 n-grams, lemmatization) as the best single model, achieving an F1-score (negative class) of 0.656 and overall accuracy of 0.684.

### Model Stacking for Improved Performance
To potentially boost performance, a stacked ensemble model was implemented, combining predictions from base models (Logistic Regression, Random Forest, Gradient Boosting) using a Logistic Regression meta-learner.

```python
from sklearn.ensemble import StackingClassifier

# Assuming best_log_reg, best_rf, best_gb are trained best estimators from GridSearchCV
# base_estimators = [
#     ('logistic', best_log_reg),
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')), # Example RF
#     ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)) # Example GB
# ]
# meta_learner_log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)

# stacked_clf = StackingClassifier(
#     estimators=base_estimators,
#     final_estimator=meta_learner_log_reg,
#     cv=5, # Can use cross-validation for generating base model predictions for meta-learner
#     n_jobs=-1
# )
# stacked_clf.fit(X_train, y_train)
# y_pred_stacked = stacked_clf.predict(X_test)
# print("Stacked Classifier Report:\n", classification_report(y_test, y_pred_stacked))
```
The stacked model achieved an overall accuracy of 0.87 and an F1-score (weighted average) of 0.87, significantly outperforming individual models.

## Analyzing Feature Importance

For linear models like Logistic Regression, examining coefficients helps understand which words/n-grams are most indicative of positive or negative sentiment.
```python
# Assuming 'tfidf_vectorizer' is fitted and 'best_log_reg' is the trained Logistic Regression model
# feature_names = tfidf_vectorizer.get_feature_names_out()
# coefficients = best_log_reg.coef_[0]
# coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
# coef_df = coef_df.sort_values('coefficient', ascending=False)

# print("Top Positive Features:")
# print(coef_df.head(10))
# print("\nTop Negative Features:")
# print(coef_df.tail(10).sort_values('coefficient', ascending=True))
```
This revealed terms like "love," "amazing" for positive, and "hate," "worst" for negative sentiment.

## Technical Lessons Learned

1.  **Hybrid Labeling Value:** Combining multiple off-the-shelf sentiment tools with a consensus mechanism creates surprisingly robust labels for training, mitigating individual model biases.
2.  **Preprocessing is Paramount for Social Media Text:** Standard NLP preprocessing needs to be augmented with specific steps for emojis, slang, URLs, and contractions common in YouTube comments.
3.  **N-grams Capture Context:** Using bigrams and trigrams with TF-IDF often captures more contextual sentiment than unigrams alone.
4.  **Model Stacking Benefits:** Ensemble methods like stacking can effectively combine the strengths of diverse base models to achieve superior performance.
5.  **Error Analysis Drives Improvement:** Understanding common misclassifications (sarcasm, mixed sentiment) points towards areas for future model refinement.

## Conclusion

Building a sentiment analyzer for the noisy and nuanced world of YouTube comments requires a multi-pronged approach. This project demonstrated that by combining a clever hybrid labeling strategy, meticulous text preprocessing, diverse feature engineering, and advanced modeling techniques like stacking, it's possible to create a system that provides valuable insights into audience sentiment. Such tools are indispensable for creators and brands aiming to manage their online reputation and engage more effectively with their audience.

---

*This post details the technical journey of building the YouTube Comment Sentiment Analyzer. For more on the project's business applications and high-level findings, please visit the [project page](/projects/sentiment-analysis/). The source code is available on [GitHub](https://github.com/Adredes-weslee/Sentiment-Analysis-and-NLP-for-a-Youtube-Video).*
