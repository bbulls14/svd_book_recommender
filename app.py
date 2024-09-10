from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    genre = request.form['genre']
    # In a future step, use your ML model to generate a recommendation
    recommendation = f"Books recommended in the genre: {genre}"
    return render_template('home.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
