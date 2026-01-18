import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

class NMFRecommender:
    """
    Lightweight NMF-based recommender using scikit-learn.
    """
    def __init__(self, movies_csv: str, ratings_csv: str, n_components: int = 15):
        self.movies = pd.read_csv(movies_csv)
        self.ratings = pd.read_csv(ratings_csv)
        self.n_components = n_components
        self._prep()

    def _prep(self):
        self.ratings['rating'] = pd.to_numeric(self.ratings['rating'], errors='coerce').fillna(0)
        self.user_item = self.ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
        self.movie_ids = list(self.user_item.columns)
        self.model = NMF(
            n_components=min(self.n_components, max(2, min(self.user_item.shape) - 1)),
            init='nndsvda',
            random_state=42,
            max_iter=500
        )
        self.W = self.model.fit_transform(self.user_item.values)
        self.H = self.model.components_

    def recommend_for_user(self, user_id: int, topn: int = 10):
        if user_id not in self.user_item.index:
            raise ValueError(f"Unknown user id: {user_id}")
        user_idx = list(self.user_item.index).index(user_id)
        user_pref = self.W[user_idx].dot(self.H)
        rated_mask = self.user_item.iloc[user_idx].values > 0
        user_pref[rated_mask] = -np.inf  # exclude already rated
        order = np.argsort(user_pref)[::-1][:topn]
        movie_ids = [self.movie_ids[i] for i in order]
        return self.movies.set_index("movieId").loc[movie_ids].reset_index()
