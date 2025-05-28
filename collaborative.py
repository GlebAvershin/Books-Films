import pandas as pd
import numpy as np
import tensorflow as tf

# Collaborative Filtering for Books and Movies

def load_ratings(path):
    return pd.read_csv(path)

def build_mappings(ratings, item_col):
    users = ratings.user_id.unique()
    items = ratings[item_col].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {itm: i for i, itm in enumerate(items)}
    return user2idx, item2idx

def make_collab_dataset(ratings, user2idx, item2idx, item_col):
    u = ratings.user_id.map(user2idx).values.astype(np.int32)
    i = ratings[item_col].map(item2idx).values.astype(np.int32)
    y = ratings.rating.values.astype(np.float32)
    return tf.data.Dataset.from_tensor_slices(((u, i), y)).shuffle(10000).batch(64)

def build_collab_model(num_users, num_items, embedding_dim=64):
    user_in = tf.keras.Input(shape=(), name='user')
    item_in = tf.keras.Input(shape=(), name='item')
    u_emb = tf.keras.layers.Embedding(num_users, embedding_dim)(user_in)
    i_emb = tf.keras.layers.Embedding(num_items, embedding_dim)(item_in)
    dot = tf.keras.layers.Dot(axes=1)([u_emb, i_emb])
    out = tf.keras.layers.Flatten()(dot)
    model = tf.keras.Model([user_in, item_in], out)
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    # Train books collab
    book_r = load_ratings('data/book_ratings.csv')
    u2b, b2idx = build_mappings(book_r, 'book_id')
    b_ds = make_collab_dataset(book_r, u2b, b2idx, 'book_id')
    b_model = build_collab_model(len(u2b), len(b2idx))
    b_model.fit(b_ds, epochs=5)
    b_model.save_weights('outputs/collab_book_weights.h5')

    # Train movies collab
    movie_r = load_ratings('data/movie_ratings.csv')
    u2m, m2idx = build_mappings(movie_r, 'movie_id')
    m_ds = make_collab_dataset(movie_r, u2m, m2idx, 'movie_id')
    m_model = build_collab_model(len(u2m), len(m2idx))
    m_model.fit(m_ds, epochs=5)
    m_model.save_weights('outputs/collab_movie_weights.h5')