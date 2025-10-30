from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DocumentResponse(BaseModel):
    document_ids: List[str]
    message: str

class QuestionRequest(BaseModel):
    document_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class ExtractionResponse(BaseModel):
    document_id: str
    data: Dict[str, Any]

class AuditResponse(BaseModel):
    document_id: str
    audit: Dict[str, Any]