import pandas as pd
import numpy as np
import tensorflow as tf

# SkipGram for Books and Movies

def load_ratings(path, col):
    df = pd.read_csv(path).rename(columns={col: 'item_id'})
    return df

def build_mappings(ratings):
    return {itm: i for i, itm in enumerate(ratings.item_id.unique())}

def build_skipgram_dataset(ratings, item2idx, window=2, neg_samples=1, batch_size=64):
    pairs = []
    for u in ratings.user_id.unique():
        seq = ratings[ratings.user_id == u].item_id.map(item2idx).tolist()
        for idx, target in enumerate(seq):
            contexts = seq[max(0, idx-window): idx] + seq[idx+1: idx+1+window]
            for ctx in contexts:
                pairs.append((target, ctx, 1))
                for _ in range(neg_samples):
                    neg = np.random.choice(list(item2idx.values()))
                    pairs.append((target, neg, 0))
    arr = np.array(pairs)
    return tf.data.Dataset.from_tensor_slices((arr[:,0], arr[:,1], arr[:,2])).shuffle(100000).batch(batch_size)

def build_skipgram_model(vocab_size, embedding_dim=64):
    in_t = tf.keras.Input(shape=(), name='target')
    in_c = tf.keras.Input(shape=(), name='context')
    emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    t_emb = emb(in_t)
    c_emb = emb(in_c)
    dot = tf.keras.layers.Dot(axes=1)([t_emb, c_emb])
    out = tf.keras.layers.Flatten()(dot)
    model = tf.keras.Model([in_t, in_c], out)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    return model

if __name__ == '__main__':
    # Books SkipGram
    b_r = load_ratings('data/book_ratings.csv', 'book_id')
    b_map = build_mappings(b_r)
    b_ds = build_skipgram_dataset(b_r, b_map)
    b_model = build_skipgram_model(len(b_map))
    b_model.fit(b_ds, epochs=5)
    np.save('outputs/book_embeddings.npy', b_model.get_layer('embedding').get_weights()[0])

    # Movies SkipGram
    m_r = load_ratings('data/movie_ratings.csv', 'movie_id')
    m_map = build_mappings(m_r)
    m_ds = build_skipgram_dataset(m_r, m_map)
    m_model = build_skipgram_model(len(m_map))
    m_model.fit(m_ds, epochs=5)
    np.save('outputs/movie_embeddings.npy', m_model.get_layer('embedding').get_weights()[0])