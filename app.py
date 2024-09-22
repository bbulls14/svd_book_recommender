from flask import Flask, request, render_template
import pandas as pd
import pickle


def load_svd_model():    
    with open('svd_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_svd_model()

df = pd.read_csv('joined_dataset.csv')

app = Flask(__name__)

item_rating_counts = df.groupby('ISBN')['Book-Rating'].count().to_dict()
max_ratings = max(item_rating_counts.values()) 

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

    book_ratings = df[df['User-ID'].isin(high_rated_users)].groupby('ISBN').size()
    max_ratings_in_subset = book_ratings.max()

    #predict rating for books
    books_with_predictions = []
    for book in unique_books:
        prediction = model.predict(user_id, book[0]).est  
        
        num_ratings = book_ratings.get(book[0], 1)  # Get number of ratings or default to 1 if not found
        confidence = (num_ratings / max_ratings_in_subset) * 100  # Confidence as a percentage
        books_with_predictions.append((book[1], book[2], prediction, confidence))

    sorted_books = sorted(books_with_predictions, key=lambda x: x[2], reverse=True)

    #keep top 5
    top_books = sorted_books[:5]

    #use list of dicts so HTML can access items
    book_list = [{'Book-Title': title, 'Book-Author': author, 'Predicted-Rating': round(pred, 2), 'Confidence': round(conf, 2)} 
                 for title, author, pred, conf in top_books]

    return book_list

if __name__ == '__main__':
    app.run(debug=True)

