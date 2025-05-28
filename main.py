# app/main.py
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from app.schemas import UserRequest, RecommendationResponse
from app.train import train_all_models
from app.loader import ModelLoader
from app.recommender import get_all_recommendations
from random import randint


app = FastAPI()
loader = ModelLoader()
#loader.load_all()


def flatten_recs(recs_dict: dict, domain: str) -> list[int]:
    """Берем recs_dict, выбираем нужные ключи по domain, складываем в плоский список без дублей."""
    order = []
    if domain == "books":
        keys = ["book_collaborative", "book_content", "movie2book"]
    else:  # movies
        keys = ["movie_collaborative", "movie_content", "book2movie"]
    for k in keys:
        for item in recs_dict.get(k, []):
            if item not in order:
                order.append(item)
    print(order)
    return order


@app.post("/recommend/{domain}", response_model=RecommendationResponse)
def recommend(domain: str, req: UserRequest):
    if domain not in ["books", "movies"]:
        raise HTTPException(400, "Invalid domain")

    user_str = str(req.user_id)

    needed = {
        "books": ["collab_book", "content_book", "movie2book"],
        "movies": ["collab_movie", "content_movie", "book2movie"]
    }
    miss = [m for m in needed[domain] if m not in loader.models]
    if miss:
        raise HTTPException(
            503,
            f"Models not ready: {miss}. Call /retrain first."
        )

    raw = get_all_recommendations(loader, req.user_id, domain)
    flat = flatten_recs(raw, domain)
    return RecommendationResponse(recommendations=flat)


@app.post("/retrain")
def retrain():
    train_all_models()
    loader.load_all()
    return {"status": "retrained and loaded"}
