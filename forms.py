from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Optional, NumberRange

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class SearchBookForm(FlaskForm):
    q = StringField('Название', validators=[Optional()])
    author = StringField('Автор', validators=[Optional()])
    publisher = StringField('Издательство', validators=[Optional()])
    year = IntegerField('Год публикации', validators=[Optional(), NumberRange(min=0, max=datetime.now().year)])
    submit = SubmitField('Поиск')

class SearchMovieForm(FlaskForm):
    q = StringField('Название', validators=[Optional()])
    genres = StringField('Жанры', validators=[Optional()])
    submit = SubmitField('Поиск')

class RatingForm(FlaskForm):
    rating = IntegerField('Оценка (1–10)', validators=[DataRequired(), NumberRange(min=1, max=10)])
    submit = SubmitField('Отправить')

class EmptyForm(FlaskForm):
    submit = SubmitField('OK')

class SearchForm(FlaskForm):
    q = StringField('Поиск', validators=[Optional()])
    genres = StringField('Жанры', validators=[Optional()])
    submit = SubmitField('Найти')

class BookForm(FlaskForm):
    ISBN = StringField('ISBN', validators=[DataRequired()])
    bookTitle = StringField('Название', validators=[DataRequired()])
    bookAuthor = StringField('Автор', validators=[DataRequired()])
    yearOfPublication = IntegerField('Год', validators=[Optional(), NumberRange(min=0, max=datetime.now().year)])
    publisher = StringField('Издательство', validators=[Optional()])
    submit = SubmitField('Сохранить')

class MovieForm(FlaskForm):
    title = StringField('Название', validators=[DataRequired()])
    genres = StringField('Жанры', validators=[Optional()])
    submit = SubmitField('Сохранить')