import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate

from app.services.document_service import document_service

class AuditService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.index_name = "contract-intelligence-v2"

    def audit_contract(self, document_id: str) -> Dict[str, Any]:
        if not document_service.document_exists(document_id):
            raise ValueError("Document not found")
        
        # Get document content
        vectorstore = Pinecone.from_existing_index(
            self.index_name, self.embeddings, namespace=document_id
        )
        docs = vectorstore.similarity_search("", k=15)
        full_text = "\n".join([doc.page_content for doc in docs])
        
        # Audit contract
        prompt = ChatPromptTemplate.from_template("""
        Audit this contract for risks and return JSON with:
        - findings: risk findings with severity
        - overall_risk: low/medium/high
        - recommendations: suggestions

        Contract: {text}
        Return ONLY valid JSON.
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"text": full_text[:3000]})
        
        # Parse JSON
        try:
            content = response.content
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                return json.loads(json_str)
        except:
            pass
        
        return {"error": "Could not parse audit results"}

audit_service = AuditService()