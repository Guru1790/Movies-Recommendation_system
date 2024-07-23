import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1, -1)
    neighbours = kNN.kneighbors(movie_vec, return_distance=show_distance)
    
    # Ensure there are enough neighbors
    n_neighbors = len(neighbours[1][0])
    k = min(k, n_neighbors)  # Adjust k if there are fewer neighbors

    for i in range(1, k):  # skip the first one as it is the movie itself
        n = neighbours[1][0][i]
        neighbour_ids.append(movie_inv_mapper[n])
    return neighbour_ids

movie_titles = dict(zip(movies['movieId'], movies['title']))

st.title("Movie Recommendation System")

st.sidebar.header("User Input")

user_id = st.sidebar.number_input("Enter User ID", min_value=1, value=1)
k = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=20, value=10)

if st.sidebar.button("Recommend Movies"):
    df1 = ratings[ratings['userId'] == user_id]

    if df1.empty:
        st.write(f"User with ID {user_id} does not exist.")
    else:
        movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
        similar_ids = find_similar_movies(movie_id, X, k)
        movie_title = movie_titles.get(movie_id, "Movie not found")

        if movie_title == "Movie not found":
            st.write(f"Movie with ID {movie_id} not found.")
        else:
            st.write(f"Since you watched {movie_title}, you might also like:")
            for i in similar_ids:
                st.write(movie_titles.get(i, "Movie not found"))

st.sidebar.markdown("## Explore Movie Ratings")
movie_id = st.sidebar.number_input("Enter Movie ID", min_value=1, value=1)

if st.sidebar.button("Find Similar Movies"):
    if movie_id not in movie_mapper:
        st.write(f"Movie with ID {movie_id} does not exist.")
    else:
        similar_ids = find_similar_movies(movie_id, X, k)
        movie_title = movie_titles.get(movie_id, "Movie not found")

        if movie_title == "Movie not found":
            st.write(f"Movie with ID {movie_id} not found.")
        else:
            st.write(f"Movies similar to {movie_title}:")
            for i in similar_ids:
                st.write(movie_titles.get(i, "Movie not found"))
