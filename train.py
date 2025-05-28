import os
import numpy as np
import pandas as pd
from models import collaborative, content, skipgram, translator
from models.skipgram import build_skipgram_dataset, build_skipgram_model
import tensorflow as tf

def train_all_models(data_dir='data', out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)

    # Collaborative
    df_b = collaborative.load_ratings(f"{data_dir}/book_ratings.csv")
    u2b, b2idx = collaborative.build_mappings(df_b, 'book_id')
    ds_b = collaborative.make_collab_dataset(df_b, u2b, b2idx, 'book_id')
    model_b = collaborative.build_collab_model(len(u2b), len(b2idx))
    model_b.fit(ds_b, epochs=5)
    model_b.save_weights(f"{out_dir}/collab_book_weights.h5")

    df_m = collaborative.load_ratings(f"{data_dir}/movie_ratings.csv")
    u2m, m2idx = collaborative.build_mappings(df_m, 'movie_id')
    ds_m = collaborative.make_collab_dataset(df_m, u2m, m2idx, 'movie_id')
    model_m = collaborative.build_collab_model(len(u2m), len(m2idx))
    model_m.fit(ds_m, epochs=5)
    model_m.save_weights(f"{out_dir}/collab_movie_weights.h5")

    # Content
    books = content.load_items(f"{data_dir}/books.csv", 'ID', 'genres')
    b_mat, _ = content.build_genre_matrix(books, b2idx, 'ID', 'genres')
    b_ds = content.make_content_dataset(b_mat, df_b, b2idx, 'book_id')
    cont_b = content.build_content_model(b_mat.shape[1])
    cont_b.fit(b_ds, epochs=5)
    cont_b.save_weights(f"{out_dir}/content_book_weights.h5")

    movies = content.load_items(f"{data_dir}/movies.csv", 'ID', 'genres')
    m_mat, _ = content.build_genre_matrix(movies, m2idx, 'ID', 'genres')
    m_ds = content.make_content_dataset(m_mat, df_m, m2idx, 'movie_id')
    cont_m = content.build_content_model(m_mat.shape[1])
    cont_m.fit(m_ds, epochs=5)
    cont_m.save_weights(f"{out_dir}/content_movie_weights.h5")

    # SkipGram
    df_b_sg = skipgram.load_ratings(f"{data_dir}/book_ratings.csv", 'book_id')
    map_b = skipgram.build_mappings(df_b_sg)
    ds_sg_b = build_skipgram_dataset(df_b_sg, map_b)
    sg_b = build_skipgram_model(len(map_b))
    sg_b.fit(ds_sg_b, epochs=5)
    np.save(f"{out_dir}/book_embeddings.npy", sg_b.get_layer('embedding').get_weights()[0])

    df_m_sg = skipgram.load_ratings(f"{data_dir}/movie_ratings.csv", 'movie_id')
    map_m = skipgram.build_mappings(df_m_sg)
    ds_sg_m = build_skipgram_dataset(df_m_sg, map_m)
    sg_m = build_skipgram_model(len(map_m))
    sg_m.fit(ds_sg_m, epochs=5)
    np.save(f"{out_dir}/movie_embeddings.npy", sg_m.get_layer('embedding').get_weights()[0])

    # Translator
    book_embs = np.load(f"{out_dir}/book_embeddings.npy")
    movie_embs = np.load(f"{out_dir}/movie_embeddings.npy")
    dim = book_embs.shape[1]

    b2m = translator.Translator(dim)
    b2m.compile(optimizer='adam', loss='mse')
    b2m.fit(book_embs, movie_embs, epochs=5, batch_size=64)
    b2m.save_weights(f"{out_dir}/book2movie_weights.h5")

    m2b = translator.Translator(dim)
    m2b.compile(optimizer='adam', loss='mse')
    m2b.fit(movie_embs, book_embs, epochs=5, batch_size=64)
    m2b.save_weights(f"{out_dir}/movie2book_weights.h5")
