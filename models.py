from datetime import datetime
from app import db, login_manager
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index = True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    favorite_books = db.relationship('FavoriteBook', back_populates='user', cascade='all, delete-orphan')
    favorite_movies = db.relationship('FavoriteMovie', back_populates='user', cascade='all, delete-orphan')
    ratings_books = db.relationship('RatingBook', back_populates='user', cascade='all, delete-orphan')
    ratings_movies = db.relationship('RatingMovie', back_populates='user', cascade='all, delete-orphan')
    role = db.Column(db.String(20), default='user', nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ISBN = db.Column(db.String(32), unique=True, nullable=False)
    bookTitle = db.Column(db.String(512), nullable=False)
    bookAuthor = db.Column(db.String(256), nullable=False)
    yearOfPublication = db.Column(db.Integer)
    publisher = db.Column(db.String(256))
    genres = db.Column(db.String(256))
    favorites = db.relationship('FavoriteBook', back_populates='book', cascade='all, delete-orphan')
    ratings = db.relationship('RatingBook', back_populates='book', cascade='all, delete-orphan')

class FavoriteBook(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'book_id', name='uix_user_book'),)
    user = db.relationship('User', back_populates='favorite_books')
    book = db.relationship('Book', back_populates='favorites')

class RatingBook(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('user_id', 'book_id', name='uix_user_book_rating'),)
    user = db.relationship('User', back_populates='ratings_books')
    book = db.relationship('Book', back_populates='ratings')

class Movie(db.Model):
    movieId = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    genres = db.Column(db.String(255))
    favorites = db.relationship('FavoriteMovie', back_populates='movie', cascade='all, delete-orphan')
    ratings = db.relationship('RatingMovie', back_populates='movie', cascade='all, delete-orphan')

class FavoriteMovie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.movieId'), nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'movie_id', name='uix_user_movie'),)
    user = db.relationship('User', back_populates='favorite_movies')
    movie = db.relationship('Movie', back_populates='favorites')

class RatingMovie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.movieId'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint('user_id', 'movie_id', name='uix_user_movie_rating'),)
    user = db.relationship('User', back_populates='ratings_movies')
    movie = db.relationship('Movie', back_populates='ratings')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))