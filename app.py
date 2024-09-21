from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load the trained model
with open('svd_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for recommendations
books_df = pd.read_csv('joined_dataset.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the book title from the user input
    book_title = request.form['book_title']

    # Find the ISBN of the book with partial match in title (case-insensitive)
    book_data = books_df[books_df['Book-Title'].str.contains(book_title, case=False, na=False)].head(1)

    if book_data.empty:
        return render_template('index.html', message="Book not found!")

    book_isbn = book_data['ISBN'].values[0]


    # Get recommendations for similar users who liked the book
    recommendations = get_book_recommendations(book_isbn)

    print(recommendations)
    
    return render_template('recommendations.html', books = recommendations)

def get_book_recommendations(book_isbn):
    # Get users who rated the input book
    similar_users = books_df[books_df['ISBN'] == book_isbn]['User-ID']

    # Find other books these users have rated highly
    recommended_books = books_df[books_df['User-ID'].isin(similar_users)]

    # Filter books with ratings above 8
    high_rated_books = recommended_books[recommended_books['Book-Rating'] > 8]

    # Count the number of users who rated each book above 8 (popularity)
    book_popularity = high_rated_books.groupby('ISBN').size().reset_index(name='num_ratings_above_8')

    # Merge popularity with original book data to include titles and ratings
    merged_books = pd.merge(high_rated_books, book_popularity, on='ISBN')

    # Sort by popularity (number of ratings above 8) and then by book rating
    top_books = merged_books.sort_values(by=['num_ratings_above_8', 'Book-Rating'], ascending=[False, False])

    filtered_books = top_books[top_books['ISBN'] != book_isbn]

    # Drop duplicate titles to ensure only unique titles are recommended
    unique_books = filtered_books.drop_duplicates(subset='Book-Title')

    # Convert DataFrame to list of dictionaries, ensuring only unique titles
    book_list = unique_books[['Book-Title', 'Book-Author']].head(5).to_dict(orient='records')

    return book_list


if __name__ == '__main__':
    app.run(debug=True)

