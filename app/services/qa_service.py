import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

from app.services.document_service import document_service

class QAService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.index_name = "contract-intelligence-v2"

    def extract_data(self, document_id: str) -> Dict[str, Any]:
        if not document_service.document_exists(document_id):
            raise ValueError("Document not found")
        
        # Get document from Pinecone
        vectorstore = Pinecone.from_existing_index(
            self.index_name, self.embeddings, namespace=document_id
        )
        docs = vectorstore.similarity_search("", k=10)
        full_text = "\n".join([doc.page_content for doc in docs])

        # Extract structured data
        prompt = ChatPromptTemplate.from_template("""
        Extract contract information as JSON:
        - parties: list of parties
        - effective_date: effective date
        - term: contract term
        - governing_law: governing law
        - payment_terms: payment terms
        - termination: termination clauses
        - liability_cap: liability cap

        Contract: {text}
        Return ONLY valid JSON.
        """)

        chain = prompt | self.llm
        response = chain.invoke({"text": full_text[:4000]})

        # Parse JSON
        try:
            content = response.content
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                return json.loads(json_str)
        except:
            pass
        
        return {"error": "Could not extract structured data"}
    
    def ask_question(self, document_id: str, question: str) -> Dict[str, Any]:
        if not document_service.document_exists(document_id):
            raise ValueError("Document not found")
        
        # Get vector store
        vectorstore = Pinecone.from_existing_index(
            self.index_name, self.embeddings, namespace=document_id
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                } for doc in result["source_documents"]
            ]
        }

qa_service = QAService()