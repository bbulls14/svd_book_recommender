***Due to costs for Heroku and Amazon S3, website is inactive***

✅ Project Overview

- A web-based book recommendation system using Singular Value Decomposition (SVD).
- Users input the last book they read, and the system suggests similar books based on user rating patterns.
- The project was built using a Kaggle dataset of book ratings.

✅ Data and Model

- Used a Kaggle dataset containing ISBN, book titles, authors, and user ratings.
  - conducted EDA and cleaned dataset
- Trained an SVD model using the [Surprise](https://surpriselib.com/) library for collaborative filtering.
  - evaluating accuracy did not display any improvement of model compared to random ratings
- The trained model is stored on AWS and downloaded upon opening the website.

✅ Web Application

- Built using Flask as the backend framework.
- User inputs last book read, Output is a list of 5 book recommendations based on user rating patterns

✅ Deployment

- Deployed on Heroku.
  - chosen for cost and ease of deployment
- Configured to download the trained SVD model from Amazon S3 upon startup.

✅ Technology Stack

- Python for backend logic and model integration.
- Surprise library for implementing SVD.
- Flask for web development.
- Heroku for deployment.
- Amazon S3 for storing the trained model.
