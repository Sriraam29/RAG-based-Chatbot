# Updated pydantic_models.py
# Update ModelName to match your Ollama models (e.g., llama3 instead of llama2)

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    MISTRAL = "mistral"
    LLAMA3 = "llama2"  # Updated to llama3 as in notebook
    # Add others if needed


class QueryInput(BaseModel):
    question: str
    session_id: str | None = Field(default=None)
    model: ModelName = Field(default=ModelName.LLAMA3)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int