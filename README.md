This project implements a movie rating prediction system using the MovieLens 20M dataset and metadata from TMDB.
The goal is to predict how a user will rate a movie based on:

the user’s historical ratings

movie metadata (genres, cast, crew, etc.)

learned user and movie embeddings using Probabilistic Matrix Factorization (PMF)

The repository contains the full implementation, trained model, processed metadata, and a Streamlit application for demonstration.

1. Project Overview

The system is built around a PMF-based model that learns low-dimensional representations of users and movies.
To improve accuracy, the movie embeddings are extended using processed metadata from TMDB.
User preference vectors are generated based on their past ratings and the metadata of the movies they have interacted with.

The final model achieves strong performance and can predict ratings directly without retraining.

2. Repository Contents
project/
│
├── app.py                 # Streamlit demo application
├── train.py               # Model training script  
├── model.py               # PMF model implementation
├── data_loader.py         # Data loading and feature preparation
├── tmdb_preprocess.py     # TMDB metadata processing script
├── predict.py             # Single-prediction utility
│
├── pmf_model.pth          # Trained model (included)
├── requirements.txt
│
└── data/
    ├── user_features.npy
    └── tmdb/
        ├── tmdb_features.npy
        ├── tmdb_index_map.json
        └── tmdb_feature_columns.json


The demo application (app.py) uses the already-trained model and processed metadata.
No training or dataset download is required to run the app.

3. How to Run the Demo
Step 1 — Create and activate a virtual environment

(Windows)

python -m venv venv
venv\Scripts\activate

Step 2 — Install dependencies
pip install -r requirements.txt

Step 3 — Start the Streamlit application
streamlit run app.py


This will launch a browser interface where you can select a user and movie and see a predicted rating.

4. Model Training (optional)

Training the model requires the MovieLens 20M dataset and TMDB metadata.
If you wish to retrain the model:

Download datasets manually (links below).


Place them in the appropriate folders.

Run:

python tmdb_preprocess.py
python train.py


This step is optional—the trained model is already included.

5. Dataset Notice

This repository does not include the raw MovieLens 20M or TMDB datasets due to license restrictions and GitHub size limits.

To retrain the model, download the datasets from:

MovieLens 20M
https://grouplens.org/datasets/movielens/20m/

TMDB 5000 Movies & Credits
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

The repository includes all processed metadata files (tmdb_features.npy, JSON maps, and user_features.npy) so the model and demo run without requiring the original datasets.

6. Notes

The trained PMF model (pmf_model.pth) is included so the app runs immediately.

Processed metadata is included under data/tmdb/.

Large dataset folders are intentionally excluded from the repository.