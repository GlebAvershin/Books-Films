import numpy as np
import tensorflow as tf

class Translator(tf.keras.Model):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.d2 = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        return self.d2(self.d1(x))

if __name__ == '__main__':
    book_embs = np.load('outputs/book_embeddings.npy')
    movie_embs = np.load('outputs/movie_embeddings.npy')

    dim = book_embs.shape[1]
    b2m = Translator(dim)
    m2b = Translator(dim)
    b2m.compile(optimizer='adam', loss='mse')
    m2b.compile(optimizer='adam', loss='mse')

    b2m.fit(book_embs, movie_embs, epochs=10, batch_size=64)
    m2b.fit(movie_embs, book_embs, epochs=10, batch_size=64)

    b2m.save_weights('outputs/book2movie_weights.h5')
    m2b.save_weights('outputs/movie2book_weights.h5')