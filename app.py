from flask import Flask, request, render_template
import pandas as pd
import os
import boto3
import pickle

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')

def download_svd_model():
    s3 = boto3.client('s3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION)
    
    bucket_name = 'svdmodel'  
    file_key = 'svd_model.pkl'
    download_path = '/tmp/svd_model.pkl'

    s3.download_file(bucket_name, file_key, download_path)

    with open(download_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = download_svd_model()

df = pd.read_csv('joined_dataset.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    #get book that contains title input
    book_title = request.form['book_title']
    book_data = df[df['Book-Title'].str.contains(book_title, case=False, na=False)].head(1)

    if book_data.empty:
        return render_template('index.html', message="Book not found!")

    #set userid to nonexistent user for predictions
    user_id = 276723
    
    
    book_isbn = book_data['ISBN'].values[0]
    recommendations = get_book_recommendations(user_id, book_isbn)

    return render_template('recommendations.html', books=recommendations)

def get_book_recommendations(user_id, book_isbn):
    
    
    #get books rated highly by users who rated book_isbn
    high_rated_users = df[(df['ISBN'] == book_isbn) & (df['Book-Rating'] >= 8)]['User-ID'].unique()

    #get top 10 books from each user
    top_books = []
    for high_user in high_rated_users:
        user_books = df[df['User-ID'] == high_user]
        top_user_books = user_books.sort_values(by='Book-Rating', ascending=False).head(10)
        top_books.extend(top_user_books[['ISBN', 'Book-Title', 'Book-Author']].values.tolist())
    
    #remove duplicate books using isbn
    seen_isbns = set()
    unique_books = []
    for book in top_books:
        if book[0] not in seen_isbns:
            seen_isbns.add(book[0])
            unique_books.append(book)
            
    #predict rating for books
    books_with_predictions = []
    for book in unique_books:
        prediction = model.predict(user_id, book[0]).est  
        books_with_predictions.append((book[1], book[2], prediction))

    sorted_books = sorted(books_with_predictions, key=lambda x: x[2], reverse=True)

    #keep top 5
    top_books = sorted_books[:5]

    #use list of dicts so HTML can access items
    book_list = [{'Book-Title': title, 'Book-Author': author} for title, author, _ in top_books]

    return book_list

if __name__ == '__main__':
    app.run(debug=True)



