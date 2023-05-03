

from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender import new_data, get_similar_articles, preprocess_text


app = Flask(__name__, template_folder='templates')

# Create an instance of the vectorizer and fit it on the content column of your dataset
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(new_data['processed_headline'])

@app.route('/')
def index():
    # Get the latest 10 news articles
    articles = new_data.tail(20)

    # Render the homepage template with the article data
    return render_template('index.html', articles=articles.to_dict('records'))

# Define the article route
@app.route('/article', methods=['POST'])
def display_article():
    data = request.json
    content = data['content']
    # Set a minimum length threshold for the input article
    min_length = 5
    # Check if the input article meets the minimum length requirement
    if len(content) < min_length:
        return jsonify({'error': f'Article content is too short. Please enter an article with at least {min_length} characters.'})
    num_articles = data.get('num_articles', 5)
    # Preprocess the article content
    processed_content = preprocess_text(content)
    # Extract features from the article content
    content_features = vectorizer.transform([processed_content])
    # Calculate the similarity between the article and all the other articles based on the features
    similarity_scores = cosine_similarity(content_features, vectorizer.transform(new_data['processed_headline']))[0]
    # Get the indices of the most similar articles
    similar_article_indices = similarity_scores.argsort()[::-1][1:num_articles+1]
    # Return the DataFrame rows for the similar articles
    similar_articles = new_data.iloc[similar_article_indices, [0, 1, 2]].reset_index(drop=True)
    return jsonify(similar_articles.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, render_template, request
# import recommender
# from recommender import df, get_similar_articles

# app = Flask(__name__, template_folder='templates')

# @app.route('/')
# def homepage():
#     # Get the latest 10 news articles
#     articles = df.tail(20)

#     # Render the homepage template with the article data
#     return render_template('article.html', articles=articles.to_dict('records'))




# @app.route('/article/<int:article_id>')
# def display_article(article_id):
#     # Subtract 1 from the article ID to get the correct index
#     index = article_id - 1
#     # Get the news article with the specified ID
#     article = df.iloc[index]
#     # Get the similar articles for the news article
#     similar_articles = get_similar_articles(article_id, num_articles=5)
#     # Render the article template with the article and similar articles data
#     return render_template('article.html', article=article.to_dict(), similar_articles=similar_articles.to_dict('records'))


# if __name__ == '__main__':
#     app.run(debug=True)
