from flask import Flask, request, render_template
import pandas as pd
import pickle
import requests
import os


app = Flask(__name__)


#download svd_model from s3 for heroku b/c file too large for git
def download_svd_model():
    model_path = "/tmp/svd_model.pkl"
    
    # Check if the model file already exists
    if not os.path.exists(model_path):
        url = 'https://svdmodel.s3.us-east-2.amazonaws.com/svd_model.pkl'
        response = requests.get(url)
        
        with open(model_path, "wb") as file:
            file.write(response.content)

    # Load the model
    with open(model_path, "rb") as file:
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
def svd_ratings(user_id, unique_books, model):
    
    #predict rating
    books_predictions = []
    for book in unique_books:
        prediction = model.predict(user_id, book[0]).est  

        books_predictions.append((book[1], book[2], prediction))

    return sorted(books_predictions, key=lambda x: x[2], reverse=True)
    

def get_book_recommendations(user_id, book_isbn):
    
    #get books rated highly by users who rated book_isbn highly
    similar_users = df[(df['ISBN'] == book_isbn) & (df['Book-Rating'] >= 8)]['User-ID'].unique()

    #get top 10 books from each user
    top_books = []
    for user in similar_users:
        user_books = df[df['User-ID'] == user].sort_values(by='Book-Rating', ascending=False).head(10)
        top_books.extend(user_books[['ISBN', 'Book-Title', 'Book-Author']].values.tolist())
    
    #remove duplicate books using isbn and title
    unique_books = remove_duplicate_books(top_books)
    
    #use svd to predict rating and determine confidence
    sorted_books = svd_ratings(user_id, unique_books, model)

    #keep top 5
    top_books = sorted_books[:5]

    #use list of dicts so recommendations.html can iterate through items
    book_list = [{'Book-Title': title, 'Book-Author': author, 'Predicted-Rating': round(pred, 2)} 
                 for title, author, pred, conf in top_books]

    return book_list

def remove_duplicate_books(top_books):
    seen_isbns = set()
    seen_titles = set()
    unique_books = []
    
    for book in top_books:
        isbn, title = book[0], book[1]
        
        if isbn not in seen_isbns and title not in seen_titles:
            seen_isbns.add(isbn)
            seen_titles.add(title)
            unique_books.append(book)
    
    return unique_books



if __name__ == '__main__':
    app.run(debug=True)

