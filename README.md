# Movie Recommender System (PMF + Movie Metadata)

A complete recommendation system that predicts how a user will rate a movie using **Probabilistic Matrix Factorization (PMF)** and movie metadata from the **TMDB** dataset.  
This project uses the **MovieLens 20M** dataset for ratings and TMDB metadata for movie-level features such as genres, cast, and crew.

The project includes fully implemented training, preprocessing, rating prediction, metadata extraction, and a Streamlit web application for interactive use.

---


## Project Overview

This project implements a movie recommendation system based on **Probabilistic Matrix Factorization**, enriched with descriptive movie metadata such as:

- Genres  
- Cast and crew  
- Keywords  
- Overview text  

The system learns:

- **User embeddings** based on their rating behavior  
- **Movie embeddings** based on ratings and metadata  
- A predicted rating from the dot product of latent factors  

The model is trained on millions of real user ratings from MovieLens, providing a strong collaborative-filtering baseline enhanced by rich movie features.

---

## Dataset Information

The project uses two datasets:

### **1. MovieLens 20M (GroupLens)**

Required files:

| File | Purpose |
|------|---------|
| `ratings.csv` | ~20M explicit user ratings |
| `movies.csv` | Movie titles + genres |
| (others optional) | Not required for this project |

Download:  
https://grouplens.org/datasets/movielens/20m/

---

### **2. TMDB 5000 Movie Dataset (Kaggle)**

Required files:

| File | Purpose |
|------|---------|
| `tmdb_5000_movies.csv` | Genres, keywords, overview text |
| `tmdb_5000_credits.csv` | Cast and crew metadata |

Download:  
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## Project Structure

Your folder should look like this **after downloading the datasets**:

movie-recommender/
│
├── app.py
├── train.py
├── predict.py
├── model.py
├── data_loader.py
├── tmdb_preprocess.py
├── demo.ipynb
│
├── pmf_model.pth
├── requirements.txt
├── README.md
│
└── data/
├── ml-20m/
│ ├── ratings.csv
│ ├── movies.csv
│ └── (other MovieLens files, optional)
│
└── tmdb/
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
│
├── tmdb_features.npy
├── tmdb_feature_columns.json
└── tmdb_index_map.json


If the raw datasets are not placed in this structure, training and preprocessing will not run.

---

## Features

### Data Processing

- Mapping MovieLens movie IDs to TMDB IDs  
- Cleaning and merging metadata  
- Parsing JSON fields for cast/crew/genres  
- Multi-hot encoding of categorical metadata  
- Extraction of keywords, genres, production info  

### Model Features

- Probabilistic Matrix Factorization  
- Learnable user and movie embeddings  
- Metadata projection network  
- Adam optimizer with regularization  
- Ratings clamped between 0.5 and 5.0  

### Outputs

- Trained PMF model (`pmf_model.pth`)  
- TMDB feature matrix (`tmdb_features.npy`)  
- Movie index mapping (`tmdb_index_map.json`)  

### Streamlit Web App

- Select user  
- Select movie  
- Get predicted rating  
- Clean, minimal interface  

---

## Installation

Install project dependencies:

pip install -r requirements.txt

Dependencies include:

- torch  
- pandas  
- numpy  
- scikit-learn  
- tqdm  
- streamlit  

---

## Dataset Setup

After downloading the datasets, place them exactly as follows:

data/ml-20m/ratings.csv
data/ml-20m/movies.csv
data/tmdb/tmdb_5000_movies.csv
data/tmdb/tmdb_5000_credits.csv

Do **not** rename these files.

---

## Run TMDB Preprocessing

Before training, generate processed metadata:

python tmdb_preprocess.py


This produces:

- `tmdb_features.npy`  
- `tmdb_index_map.json`  
- `tmdb_feature_columns.json`

These files are required for training and prediction.

---

## Training the Model

Run:

python train.py

This will:

1. Load MovieLens ratings  
2. Load TMDB metadata  
3. Create user and movie embeddings  
4. Train the PMF model  
5. Save weights to `pmf_model.pth`  

---

## Using the Predictor

Example usage:

```python
from predict import predict_rating

rating = predict_rating(
    user_id=1,
    movie_title="Toy Story (1995)",
    use_metadata=True
)

print(rating)

The function:

Loads metadata

Loads trained model

Maps user and movie IDs

Returns the predicted rating

## Streamlit Application
Launch the interactive interface:

streamlit run app.py


Features include:

User dropdown selection

Movie dropdown selection

Real-time predicted rating

Visual display of rating output

## Model Architecture
The PMF model includes:

User Embedding Matrix
Learns latent preferences.

Movie Embedding Matrix
Learns latent movie properties.

Metadata Neural Network
Projects TMDB metadata into embedding space.

Dot-Product Predictor
Computes:
pred_rating = dot(user_vector, movie_vector) + global_bias
Loss Function
Mean Squared Error (MSE) with weight decay regularization.
