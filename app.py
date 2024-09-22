from flask import Flask, request, render_template
import pandas as pd
import pickle
import requests
import os


app = Flask(__name__)


#download svd_model from s3 for heroku b/c file to large for git
def download_svd_model():
    url = 'https://svdmodel.s3.us-east-2.amazonaws.com/svd_model.pkl'
    response = requests.get(url)
    
    with open("/tmp/svd_model.pkl", "wb") as file:
        file.write(response.content)
    with open("/tmp/svd_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = download_svd_model()
df = pd.read_csv('joined_dataset.csv')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    
    #get first book that contains user input, ignore case
    book_title = request.form['book_title']
    book_data = df[df['Book-Title'].str.contains(book_title, case=False, na=False)].head(1)

    if book_data.empty:
        return render_template('index.html', message="Book not found!")

    #use nonexistent userid for predictions with surprise
    user_id = 276723
    
    
    book_isbn = book_data['ISBN'].values[0]
    recommendations = get_book_recommendations(user_id, book_isbn)

    return render_template('recommendations.html', books=recommendations)

#use model to predict ratings based on other users ratings
#calc confidence by combining popularity and percentage of high ratings 
def svd_ratings_and_confidence(user_id, unique_books, similar_users, df, model):
    
    #get book ratings for similar user
    book_ratings = df[df['User-ID'].isin(similar_users)].groupby('ISBN').size()
    max_ratings = book_ratings.max()
    high_ratings_count = df[(df['User-ID'].isin(similar_users)) & (df['Book-Rating'] > 7)].groupby('ISBN').size()

    #predict rating and calc confidence
    books_predictions_confidence = []
    for book in unique_books:
        prediction = model.predict(user_id, book[0]).est  
        
        num_ratings = book_ratings.get(book[0], 1) 
        
        confidence = (num_ratings / max_ratings) * 100 
        books_predictions_confidence.append((book[1], book[2], prediction, confidence))

    return sorted(books_predictions_confidence, key=lambda x: x[2], reverse=True)
    

def get_book_recommendations(user_id, book_isbn):
    
    #get books rated highly by users who rated book_isbn highly
    similar_users = df[(df['ISBN'] == book_isbn) & (df['Book-Rating'] >= 8)]['User-ID'].unique()

    #get top 10 books from each user
    top_books = []
    for user in similar_users:
        user_books = df[df['User-ID'] == user].sort_values(by='Book-Rating', ascending=False).head(10)
        top_books.extend(user_books[['ISBN', 'Book-Title', 'Book-Author']].values.tolist())
    
    #remove duplicate books using isbn
    unique_books = remove_duplicate_books(top_books)
    
    #use svd to predict rating and determine confidence
    sorted_books = svd_ratings_and_confidence(user_id, unique_books, similar_users, df, model)

    #keep top 5
    top_books = sorted_books[:5]

    #use list of dicts so recommendations.html can iterate through items
    book_list = [{'Book-Title': title, 'Book-Author': author, 'Predicted-Rating': round(pred, 2), 'Confidence': round(conf, 2)} 
                 for title, author, pred, conf in top_books]

    return book_list

def remove_duplicate_books(books):
    seen_isbns = set()
    unique_books = []
    for book in books:
        if book[0] not in seen_isbns:
            seen_isbns.add(book[0])
            unique_books.append(book)
    return unique_books


if __name__ == '__main__':
    app.run(debug=True)

