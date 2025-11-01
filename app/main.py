from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
import asyncio

from app.model import DocumentResponse, QuestionRequest, AnswerResponse, ExtractionResponse, AuditResponse
from app.services import document_service, qa_service, audit_service

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Contract Intelligence API",
    description="AI-powered contract analysis and Q&A",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 

@app.get("/")
async def root():
    return {"message": "Contract Intelligence API", "version": "1.0.0"}

@app.post("/ingest", response_model=DocumentResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files provided")
    
    document_ids = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(400, "Only PDF files supported")
        
        try:
            content = await file.read()
            doc_id = document_service.process_pdf(content, file.filename)
            document_ids.append(doc_id)
        except Exception as e:
            raise HTTPException(500, f"Failed to process {file.filename}: {str(e)}")
    
    return DocumentResponse(
        document_ids=document_ids,
        message=f"Successfully processed {len(files)} documents"
    )

@app.post("/extract", response_model=ExtractionResponse)
async def extract_data(document_id: str):
    try:
        extracted_data = qa_service.extract_data(document_id)
        return ExtractionResponse(document_id=document_id, data=extracted_data)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        result = qa_service.ask_question(request.document_id, request.question)
        return AnswerResponse(**result)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Question failed: {str(e)}")

@app.post("/audit", response_model=AuditResponse)
async def audit_contract(document_id: str):
    try:
        audit_result = audit_service.audit_contract(document_id)
        return AuditResponse(document_id=document_id, audit=audit_result)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Audit failed: {str(e)}")

@app.get("/ask/stream")
async def ask_question_stream(document_id: str, question: str):
    try:
        result = qa_service.ask_question(document_id, question)
        
        async def generate():
            for word in result["answer"].split():
                yield f"data: {word} \n\n"
                await asyncio.sleep(0.05)
        
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

@app.get("/documents")
async def list_documents():
    return document_service.document_metadata

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
