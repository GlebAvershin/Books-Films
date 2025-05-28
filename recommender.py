import numpy as np
import pandas as pd
import tensorflow as tf

from models.collaborative import build_collab_model, build_mappings as collab_maps
from models.content import load_items, build_genre_matrix, build_content_model
from models.translator import Translator

def get_interacted(df, item2idx, user_id):
    col = 'book_id' if 'book_id' in df.columns else 'movie_id'
    return df[df.user_id == user_id][col].map(item2idx).dropna().astype(int).tolist()

def predict_cross(src_embs, tgt_embs, translator, interacted_ids, top_k=10):
    if not interacted_ids:
        return []
    avg = src_embs[interacted_ids].mean(axis=0, keepdims=True)
    proj = translator(avg).numpy()  # shape: (1, D)
    sims = proj @ tgt_embs.T        # cosine sim можно позже
    top = np.argsort(sims[0])[-top_k:][::-1]
    return top.tolist()  # возвращаем индексы — будет превращено в ID выше

def predict_collab(model, user2idx, item2idx, user_id, top_k=10):
    if user_id not in user2idx:
        return []
    uid = user2idx[user_id]
    all_items = np.arange(len(item2idx))
    users = np.full_like(all_items, uid)
    preds = model.predict([users, all_items], verbose=0)
    top = np.argsort(preds[:, 0])[-top_k:][::-1]
    return [list(item2idx.keys())[i] for i in top]

def predict_content(model, mat, ratings_df, item2idx, user_id, top_k=10):
    if user_id not in ratings_df.user_id.values:
        return []
    col = 'book_id' if 'book_id' in ratings_df.columns else 'movie_id'
    items = ratings_df[ratings_df.user_id == user_id][col].map(item2idx).dropna().astype(int).tolist()
    if not items:
        return []
    avg_feat = mat[items].mean(axis=0, keepdims=True)
    preds = model.predict(np.repeat(avg_feat, len(item2idx), axis=0), verbose=0).flatten()
    top = np.argsort(preds)[-top_k:][::-1]
    return [list(item2idx.keys())[i] for i in top]

def get_all_recommendations(loader, user_id: int, domain: str, top_k: int = 10):
    # collaborative
    if domain == 'books':
        collab = predict_collab(loader.models['collab_book'], loader.u2b, loader.b2idx, user_id, top_k)
        cont = predict_content(loader.models['content_book'], loader.b_mat, loader.df_br, loader.b2idx, user_id, top_k)
        movie_rec = predict_cross(loader.book_embs, loader.movie_embs, loader.models['book2movie'],
                                  get_interacted(loader.df_br, loader.b2idx, user_id), top_k)
        return collab + cont + movie_rec

    elif domain == 'movies':
        collab = predict_collab(loader.models['collab_movie'], loader.u2m, loader.m2idx, user_id, top_k)
        cont = predict_content(loader.models['content_movie'], loader.m_mat, loader.df_mr, loader.m2idx, user_id, top_k)
        book_rec = predict_cross(loader.movie_embs, loader.book_embs, loader.models['movie2book'],
                                 get_interacted(loader.df_mr, loader.m2idx, user_id), top_k)
        return collab + cont + book_rec
