# Movie Recommender System (PMF + Movie Metadata)

A complete recommendation system that predicts how a user will rate a movie using **Probabilistic Matrix Factorization (PMF)** and movie metadata from the **TMDB** dataset.  
This project uses the **MovieLens 20M** dataset for ratings and TMDB metadata for movie-level features such as genres, cast, and crew.

The project includes fully implemented training, preprocessing, rating prediction, metadata extraction, and a Streamlit web application for interactive use.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Run TMDB Preprocessing](#run-tmdb-preprocessing)
- [Training the Model](#training-the-model)
- [Using the Predictor](#using-the-predictor)
- [Streamlit Application](#streamlit-application)
- [Model Architecture](#model-architecture)

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
- A predicted rating using the dot product of latent factors  

The model is trained on millions of ratings from MovieLens, providing a strong collaborative-filtering baseline enhanced by rich metadata.

---

## Dataset Information

### **1. MovieLens 20M (GroupLens)**

| File | Purpose |
|------|---------|
| `ratings.csv` | ~20M explicit user ratings |
| `movies.csv` | Movie titles + genres |
| (others optional) | Not required |

Download:  
https://grouplens.org/datasets/movielens/20m/

---

### **2. TMDB 5000 Movie Dataset (Kaggle)**

| File | Purpose |
|------|---------|
| `tmdb_5000_movies.csv` | Genres, keywords, overview |
| `tmdb_5000_credits.csv` | Cast and crew |

Download:  
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## Project Structure

```
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
    │   ├── ratings.csv
    │   ├── movies.csv
    │
    └── tmdb/
        ├── tmdb_5000_movies.csv
        ├── tmdb_5000_credits.csv
        ├── tmdb_features.npy
        ├── tmdb_feature_columns.json
        ├── tmdb_index_map.json
   

```

---

## Features

### Data Processing

- Map MovieLens IDs to TMDB IDs  
- Parse JSON fields (genres, cast, crew)  
- Merge metadata  
- Multi-hot encode categorical features  

### Model Features

- Probabilistic Matrix Factorization  
- Learnable user + movie embeddings  
- Metadata projection neural network  
- Adam optimizer + weight decay  
- Ratings clamped to `[0.5, 5.0]`

### Outputs

- `pmf_model.pth`  
- `tmdb_features.npy`  
- `tmdb_index_map.json`

### Streamlit App

- Select user  
- Select movie  
- Get predicted rating instantly  

---

## Installation

```
pip install -r requirements.txt
```

Includes:

- torch  
- pandas  
- numpy  
- sklearn  
- tqdm  
- streamlit  

---

## Dataset Setup

Place files exactly as:

```
data/ml-20m/ratings.csv
data/ml-20m/movies.csv
data/tmdb/tmdb_5000_movies.csv
data/tmdb/tmdb_5000_credits.csv
```

---

## Run TMDB Preprocessing

```
python tmdb_preprocess.py
```

Outputs:

- `tmdb_features.npy`  
- `tmdb_index_map.json`  
- `tmdb_feature_columns.json`

---

## Training the Model

```
python train.py
```

This will:

1. Load MovieLens ratings  
2. Load TMDB metadata  
3. Create user/movie embeddings  
4. Train PMF model  
5. Save `pmf_model.pth`  

---

## Using the Predictor

Example:

```python
from predict import predict_rating

rating = predict_rating(
    user_id=1,
    movie_title="Toy Story (1995)",
    use_metadata=True
)

print(rating)
```

Returns predicted rating based on metadata + user/movie embeddings.

---

## Streamlit Application

Run:

```
streamlit run app.py
```

Features:

- User dropdown  
- Movie dropdown  
- Real-time rating output  
- Clean UI  

---

## Model Architecture

The PMF model includes:

### User Embedding Matrix  
Represents user preferences.

### Movie Embedding Matrix  
Learns movie properties.

### Metadata Neural Network  
Projects TMDB metadata into embedding space.

### Dot-Product Predictor  

```
pred_rating = dot(user_vector, movie_vector) + global_bias
```

### Loss Function  
Mean Squared Error (MSE) with weight decay.


