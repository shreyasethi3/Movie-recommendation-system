import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    """
    Simple item-item collaborative filtering recommender.
    Builds a user-item matrix and computes item-item cosine similarities.
    """
    def __init__(self, movies_csv: str, ratings_csv: str, use_tf: bool = False):
        self.movies = pd.read_csv(movies_csv)  # expected: movieId, title, genres
        self.ratings = pd.read_csv(ratings_csv)  # expected: userId, movieId, rating
        self.use_tf = use_tf
        self._prep()

    def _prep(self):
        self.ratings['rating'] = pd.to_numeric(self.ratings['rating'], errors='coerce').fillna(0)
        self.user_item = self.ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
        self.movie_ids = list(self.user_item.columns)
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}
        self.idx_to_movie = {i: m for m, i in self.movie_to_idx.items()}
        if len(self.movie_ids) >= 2:
            self.item_sim = cosine_similarity(self.user_item.T)
        else:
            self.item_sim = np.array([[]])

    def recommend_for_user(self, user_id: int, topn: int = 10):
        if user_id not in self.user_item.index:
            raise ValueError(f"Unknown user id: {user_id}")
        user_ratings = self.user_item.loc[user_id].values
        scores = self.item_sim.dot(user_ratings)
        scores[user_ratings > 0] = -np.inf  # exclude already rated
        order = np.argsort(scores)[::-1][:topn]
        movie_ids = [self.movie_ids[i] for i in order]
        return self.movies.set_index("movieId").loc[movie_ids].reset_index()

    def similar_items(self, movie_id: int, topn: int = 6):
        if movie_id not in self.movie_to_idx:
            raise ValueError(f"Unknown movie id: {movie_id}")
        idx = self.movie_to_idx[movie_id]
        sims = self.item_sim[idx]
        order = np.argsort(sims)[::-1]
        top_idx = [i for i in order if i != idx][:topn]
        top_movie_ids = [self.movie_ids[i] for i in top_idx]
        return self.movies.set_index("movieId").loc[top_movie_ids].reset_index()
