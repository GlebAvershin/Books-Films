from flask import Flask, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config


db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    login_manager.init_app(app)

    # Blueprints
    from auth.views import auth_bp
    from books.views import books_bp
    from movies.views import movies_bp
    from recommendations.views import rec_bp
    from admin.views import admin_bp, book_admin_bp, movie_admin_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(books_bp, url_prefix='/books')
    app.register_blueprint(movies_bp, url_prefix='/movies')
    app.register_blueprint(rec_bp, url_prefix='/recommend')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(book_admin_bp, url_prefix='/admin/books')
    app.register_blueprint(movie_admin_bp, url_prefix='/admin/movies')

    @app.route('/')
    def index():
        return render_template('index.html')

    return app