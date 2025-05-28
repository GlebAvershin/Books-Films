from pydantic import BaseModel
from typing import List

class UserRequest(BaseModel):
    user_id: int

class RecommendationResponse(BaseModel):
    recommendations: List[int]
