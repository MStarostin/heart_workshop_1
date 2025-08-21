from typing import List

from pydantic import BaseModel, Field


class Song(BaseModel):
    track_name: str = Field(..., example="Time")
    artists: str = Field(..., example="Pink Floyd")
    album_name: str = Field(..., example="The Dark Side of the Moon")


class PredictionsResponse(BaseModel):
    requested_track: str = Field(..., example="Time")
    recommendations: List[Song]
