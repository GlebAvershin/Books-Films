import pandas as pd
import numpy as np
import tensorflow as tf

def load_items(path, id_col, genres_col):
    return pd.read_csv(path)

def load_ratings(path):
    return pd.read_csv(path)

def build_mappings(ratings, item_col):
    users = ratings.user_id.unique()
    items = ratings[item_col].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {itm: i for i, itm in enumerate(items)}
    return user2idx, item2idx

def build_genre_matrix(items_df, item2idx, id_col, genres_col):
    """
    items_df: DataFrame с колонками id_col и genres_col
    item2idx: маппинг оригинального ID -> индекс
    """
    # Собираем все жанры
    all_genres = set()
    for _, row in items_df.iterrows():
        genres = row[genres_col]
        # на случай пустых или NaN
        if pd.isna(genres):
            continue
        for g in genres.split('|'):
            all_genres.add(g)
    g2idx = {g: i for i, g in enumerate(sorted(all_genres))}

    # Матрица items × genres
    mat = np.zeros((len(item2idx), len(g2idx)), dtype=np.float32)
    for _, row in items_df.iterrows():
        item_id = row[id_col]
        if item_id not in item2idx:
            continue
        idx = item2idx[item_id]
        genres = row[genres_col]
        if pd.isna(genres):
            continue
        for g in genres.split('|'):
            mat[idx, g2idx[g]] = 1.0

    return mat, g2idx

def make_content_dataset(content_mat, ratings, item2idx, item_col):
    idxs = ratings[item_col].map(item2idx).dropna().astype(int).values
    feats = content_mat[idxs]
    y = ratings.rating.values.astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((feats, y)).shuffle(10000).batch(64)

def build_content_model(input_dim, hidden_dim=64):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(inputs)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, out)
    model.compile(optimizer='adam', loss='mse')
    return model
