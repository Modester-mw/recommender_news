import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
df = pd.read_json('/Users/foundation/Desktop/Courses Docs/recommender/News_Category_Dataset_v3.json', lines=True)

# Remove unnecessary columns
df = df.drop(['link', 'authors'], axis=1)

# Remove any duplicate rows
new_df = df.drop_duplicates()
new_df.dropna(how='any', axis=0)

smaller_df = new_df[['headline', 'category', 'short_description']]

new_data = smaller_df.sample(n=1000, random_state=42)

# Preprocess the text data
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Skip empty rows
    if not text:
        return ''
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation marks
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text


new_data['processed_headline'] = new_data['headline'].apply(preprocess_text)
new_data['processed_description'] = new_data['short_description'].apply(preprocess_text)

# Extract features from the text data
vectorizer = TfidfVectorizer()
headline_features = vectorizer.fit_transform(new_data['processed_headline'])
description_features = vectorizer.fit_transform(new_data['processed_description'])

# Calculate the similarity between the articles based on the features
headline_similarity = cosine_similarity(headline_features)
description_similarity = cosine_similarity(description_features)
overall_similarity = (headline_similarity + description_similarity) / 2

def get_similar_articles(article_content, num_articles=5):
    # Preprocess the input article content
    processed_content = preprocess_text(article_content)
    # Extract features from the processed content
    content_features = vectorizer.transform([processed_content])
    # Calculate the similarity between the input article and all other articles based on the features
    similarity_scores = cosine_similarity(content_features, overall_similarity)[0]
    # Get the indices of the most similar articles
    similar_article_indices = similarity_scores.argsort()[::-1][1:num_articles+1]
    # Return the DataFrame rows for the similar articles
    similar_articles = new_data.iloc[similar_article_indices, [0, 1, 2]].reset_index(drop=True)
    return similar_articles


# Get the index of the news article you want to find similar articles for
# article_index = 1

# Call the get_similar_articles function to get the similar articles
# similar_articles = get_similar_articles(article_index, num_articles=5)

# Print the resulting DataFrame
# print(similar_articles)