import os
import numpy as np
import pandas as pd

from models.collaborative import build_collab_model, build_mappings as collab_maps
from models.content import build_content_model, build_genre_matrix
from models.translator import Translator

class ModelLoader:
    def __init__(self, data_dir='data', out_dir='outputs'):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.models = {}

    def _normalize_ratings(self, df):
        df.columns = [c.lower() for c in df.columns]
        rename = {}
        for old, new in [('userid','user_id'), ('bookid','book_id'), ('movieid','movie_id')]:
            if old in df.columns:
                rename[old] = new
        return df.rename(columns=rename) if rename else df

    def _normalize_items(self, df):
        df.columns = [c.lower() for c in df.columns]
        if 'genre' in df.columns and 'genres' not in df.columns:
            df = df.rename(columns={'genre': 'genres'})
        return df

    def load_all(self):
        # 1) Ratings
        br_path = os.path.join(self.data_dir, 'book_ratings.csv')
        mr_path = os.path.join(self.data_dir, 'movie_ratings.csv')
        if not (os.path.isfile(br_path) and os.path.isfile(mr_path)):
            return  # не выгружено из БД ещё — ничего не грузим

        self.df_br = self._normalize_ratings(pd.read_csv(br_path))
        self.df_mr = self._normalize_ratings(pd.read_csv(mr_path))

        # 2) Items
        books_path = os.path.join(self.data_dir, 'books.csv')
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        self.books_df = self._normalize_items(pd.read_csv(books_path, low_memory=False))
        self.movies_df = self._normalize_items(pd.read_csv(movies_path, low_memory=False))

        # 3) Collab mappings
        self.u2b, self.b2idx = collab_maps(self.df_br, 'book_id')
        self.u2m, self.m2idx = collab_maps(self.df_mr, 'movie_id')

        # 4) Collaborative
        for key, (u2i, wfile, builder) in {
            'collab_book': ( (self.u2b, 'collab_book_weights.h5', lambda: build_collab_model(len(self.u2b), len(self.b2idx))) ),
            'collab_movie': ( (self.u2m, 'collab_movie_weights.h5', lambda: build_collab_model(len(self.u2m), len(self.m2idx))) )
        }.items():
            u2i_map, fname, model_fn = u2i, wfile, builder
            path = os.path.join(self.out_dir, fname)
            if os.path.isfile(path):
                m = model_fn()
                m.load_weights(path)
                self.models[key] = m

        # 5) Content + genres
        # Content для книг:
        self.b_mat, _ = build_genre_matrix(
            self.books_df,
            self.b2idx,
            id_col='book_id',    # ← СЕЙЧАС КОРРЕКТНО
            genres_col='genres'
        )

        # Content для фильмов:
        self.m_mat, _ = build_genre_matrix(
            self.movies_df,
            self.m2idx,
            id_col='movie_id',   # ← ТАК ЖЕ
            genres_col='genres'
        )


        # 6) Embeddings + translators
        b_embp = os.path.join(self.out_dir, 'book_embeddings.npy')
        m_embp = os.path.join(self.out_dir, 'movie_embeddings.npy')
        if os.path.isfile(b_embp) and os.path.isfile(m_embp):
            self.book_embs = np.load(b_embp)
            self.movie_embs = np.load(m_embp)

            b2m_w = os.path.join(self.out_dir, 'book2movie_weights.h5')
            if os.path.isfile(b2m_w):
                t1 = Translator(self.book_embs.shape[1])
                t1.load_weights(b2m_w)
                self.models['book2movie'] = t1

            m2b_w = os.path.join(self.out_dir, 'movie2book_weights.h5')
            if os.path.isfile(m2b_w):
                t2 = Translator(self.movie_embs.shape[1])
                t2.load_weights(m2b_w)
                self.models['movie2book'] = t2
