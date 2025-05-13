---
layout: post
title: "Building a Robust YouTube Comment Sentiment Analyzer"
date: 2023-07-10 09:30:00 +0800
categories: [nlp, machine-learning, sentiment-analysis]
tags: [youtube, nlp, classification, vader, flair, hybrid-labeling]
author: Wes Lee
feature_image: /assets/images/2023-07-10-building-youtube-comment-sentiment-analyzer.jpg
---

## The Need for Digital Reputation Management

In today's social media landscape, online reputation management has become critical for public figures and brands. For music artists like Justin Bieber, whose digital presence spans platforms with millions of interactions, the ability to monitor and understand audience sentiment in real-time can make the difference between prosperity and a PR crisis.

I recently tackled this challenge by building a sophisticated sentiment analysis system capable of differentiating between positive and negative comments on YouTube videos. This post details the approach, challenges, and technical implementation of this project.

## The Unique Challenge of YouTube Comments

YouTube comments present unique challenges for sentiment analysis:

1. **Informal language** with slang, emojis, abbreviations, and unconventional spelling
2. **Multilingual content** mixed within the same comment section
3. **Context-dependent sentiment** including sarcasm and references to video content
4. **Short text fragments** with limited linguistic context
5. **Platform-specific expressions** unique to the YouTube ecosystem

These factors make standard off-the-shelf sentiment analysis tools less reliable, requiring a more sophisticated approach.

## Data Collection: Scaling the YouTube API

The first step was collecting a substantial dataset. Using the YouTube Data API, I retrieved over 99,000 comments and replies from one of Justin Bieber's most commented videos:

```python
def get_comments_and_replies(video_id, page_token=None):
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            pageToken=page_token,
            maxResults=1000
        ).execute()

        comments_and_replies = []
        next_page_token = response.get("nextPageToken")
        
        # Extract top-level comments
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments_and_replies.append(comment)

            # Extract replies if they exist
            if "replies" in item:
                for reply_item in item["replies"]["comments"]:
                    reply = reply_item["snippet"]["textDisplay"]
                    comments_and_replies.append(reply)

        return comments_and_replies, next_page_token

    except googleapiclient.errors.HttpError as e:
        print(f"An error occurred: {e}")
```

After removing duplicates and null values, I ended up with a clean dataset of just under 100,000 raw comments.

## The Hybrid Labeling Approach

A key innovation in this project was the hybrid labeling strategy. Instead of relying on a single sentiment analysis model, I combined three complementary approaches:

### 1. VADER (Valence Aware Dictionary and sEntiment Reasoner)

A lexicon and rule-based sentiment analyzer specifically tuned for social media content:

```python
def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    # Convert VADER's compound score to binary sentiment
    if scores['compound'] >= 0.05:
        return 1  # Positive
    else:
        return 0  # Negative
```

### 2. Flair NLP

A powerful framework leveraging contextual string embeddings:

```python
def flair_sentiment(text):
    classifier = TextClassifier.load('en-sentiment')
    sentence = Sentence(text)
    classifier.predict(sentence)
    
    # Convert Flair's label to binary sentiment
    label = sentence.labels[0].value
    if label == 'POSITIVE':
        return 1
    else:
        return 0
```

### 3. HuggingFace Transformers

State-of-the-art transformer models for sentiment classification:

```python
def huggingface_sentiment(text):
    classifier = pipeline('sentiment-analysis')
    result = classifier(text)[0]
    
    # Convert HuggingFace's label to binary sentiment
    if result['label'] == 'POSITIVE':
        return 1
    else:
        return 0
```

### Consensus-Based Labeling

The final label was determined by taking the mean of all three models' predictions:

```python
def get_consensus_label(text):
    # Get predictions from all three models
    vader_pred = vader_sentiment(text)
    flair_pred = flair_sentiment(text)
    hf_pred = huggingface_sentiment(text)
    
    # Calculate consensus
    mean_pred = (vader_pred + flair_pred + hf_pred) / 3
    
    # Assign final label based on threshold
    if mean_pred >= 0.5:
        return 1  # Positive
    else:
        return 0  # Negative
```

This approach provided more reliable labels than any single model, especially for ambiguous comments. After labeling, approximately 63,000 comments remained for training and evaluation.

## Sophisticated Text Preprocessing

YouTube comments require extensive preprocessing due to their informal nature:

```python
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Handle emojis (convert to description or remove)
    text = emoji.demojize(text)
    
    # Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Rejoin tokens
    return ' '.join(tokens)
```

Key preprocessing steps included:
- Language detection to filter non-English comments
- Emoji handling (conversion to text descriptions)
- Contraction expansion
- URL and HTML removal
- Special character filtering
- Tokenization with stopword removal
- Both stemming and lemmatization (for comparison)

## Feature Engineering: Beyond Bag-of-Words

While traditional bag-of-words models provide a baseline, I implemented multiple feature extraction techniques to capture different aspects of the text:

### 1. N-gram Features

```python
# Count Vectorizer with n-grams
count_vec = CountVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2
)

# TF-IDF with n-grams
tfidf_vec = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    sublinear_tf=True
)
```

### 2. Word Embeddings

```python
# Train Word2Vec model on our corpus
w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1  # Skip-gram model
)

# Create document vectors by averaging word vectors
def document_vector(text, model, vector_size=100):
    tokens = text.split()
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if not vectors:
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)
```

### 3. Parts-of-Speech Features

```python
# Extract adjectives as features
nlp = spacy.load('en_core_web_sm')

def extract_adjectives(text):
    doc = nlp(text)
    adjectives = [token.text for token in doc if token.pos_ == 'ADJ']
    return ' '.join(adjectives)
```

## Model Development and Evaluation

I built and compared multiple classification approaches:

### Naive Bayes Variants

```python
# Multinomial Naive Bayes
nb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=15000, ngram_range=(1, 3))),
    ('classifier', MultinomialNB())
])
```

### Logistic Regression

```python
# Logistic Regression with different vectorizers
lr_count_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=15000, ngram_range=(1, 3))),
    ('classifier', LogisticRegression(C=10, class_weight='balanced'))
])

lr_tfidf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=6000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(C=1, class_weight='balanced'))
])
```

### Tree-Based Models

```python
# Random Forest with Word2Vec embeddings
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    class_weight='balanced'
)

# Histogram-based Gradient Boosting
hgb_model = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=10,
    learning_rate=0.1,
    class_weight='balanced'
)
```

### Ensemble Approach

```python
# Stacking Classifier
estimators = [
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
    ('gb', GradientBoostingClassifier(n_estimators=100))
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
```

## Results and Insights

After extensive experimentation, the best performing model was a Logistic Regression with Count Vectorization using 1-3 n-grams and lemmatization:

| Metric | Score |
|--------|-------|
| F1-Score (Negative Class) | 0.656 |
| Precision (Negative Class) | 0.639 |
| Recall (Negative Class) | 0.673 |
| Accuracy | 0.684 |

The confusion matrix revealed that our model was slightly better at identifying negative comments than positive ones, which aligns with our project goals.

## Feature Importance Analysis

The most predictive features for negative sentiment included:

1. "hate" (coefficient: -1.724)
2. "worst" (coefficient: -1.512)  
3. "terrible" (coefficient: -1.341)
4. "bad" (coefficient: -1.203)
5. "poor" (coefficient: -0.982)

For positive sentiment:

1. "love" (coefficient: 2.104)
2. "amazing" (coefficient: 1.913)
3. "great" (coefficient: 1.782)
4. "beautiful" (coefficient: 1.675)
5. "awesome" (coefficient: 1.524)

Interestingly, some n-grams like "not bad" and "no hate" were strongly associated with positive sentiment, showing the model's ability to capture negation effects.

## Common Misclassifications

Error analysis revealed several categories of misclassifications:

1. **Sarcasm**: "Yeah, this is *totally* the best song ever... NOT" (incorrectly classified as positive)
2. **Mixed sentiment**: "Love Justin but hate this song" (model struggled with conflicting signals)
3. **Cultural references**: Comments referring to memes or pop culture that require context
4. **Implicit sentiment**: Comments with no explicit sentiment words but implied meaning

## Practical Applications

This sentiment analyzer can be integrated into an artist's digital management strategy:

1. **Real-time monitoring**: Detect sudden shifts in sentiment that require attention
2. **Targeted engagement**: Focus community management on threads with negative sentiment
3. **Content strategy**: Analyze which video elements generate positive/negative reactions
4. **Crisis prevention**: Identify potential PR issues before they escalate
5. **Trend analysis**: Track sentiment changes over time to measure brand health

## Technical Architecture 

The production implementation involves:

```
youtube-sentiment-analyzer/
├── data/
│   ├── raw/                 # Raw comment data
│   └── processed/           # Processed text and labels
├── models/
│   ├── vectorizers/         # Fitted vectorizers
│   ├── classifiers/         # Trained models
│   └── embeddings/          # Word2Vec models
├── notebooks/
│   ├── data_collection.ipynb
│   └── cleaning_eda_modeling.ipynb
├── src/
│   ├── data/
│   │   ├── collect.py       # YouTube API functions
│   │   └── process.py       # Text preprocessing
│   ├── features/
│   │   └── vectorize.py     # Feature extraction
│   ├── models/
│   │   ├── train.py         # Model training
│   │   └── predict.py       # Inference pipeline
│   └── visualization/
│       └── visualize.py     # Result visualization
├── api/
│   ├── app.py               # FastAPI service
│   └── Dockerfile           # Container config
└── requirements.txt
```

## Future Enhancements

To further improve this system, several enhancements could be implemented:

1. **Neutral class addition**: Include a "neutral" category to better capture ambiguous comments
2. **Multi-label classification**: Detect multiple emotions within the same comment
3. **Transformer models**: Fine-tune models like BERT specifically for YouTube comments
4. **Sarcasm detection**: Add specialized models to identify sarcastic content
5. **User profiling**: Consider commenter history for additional context
6. **Cross-language support**: Extend to multiple languages common on global YouTube channels

## Conclusion

Building an effective sentiment analyzer for YouTube comments requires going beyond standard approaches. The hybrid labeling strategy and robust preprocessing pipeline were crucial to achieving reliable results. 

For social media managers and artist representation, tools like this can transform overwhelming comment sections into structured, actionable intelligence that helps protect and enhance online reputation.

---

*Want to explore this project in more detail? Check out the [complete project page](/projects/sentiment-analysis/) or view the [source code on GitHub](https://github.com/Adredes-weslee/Sentiment-Analysis-and-NLP-for-a-Youtube-Video).*
