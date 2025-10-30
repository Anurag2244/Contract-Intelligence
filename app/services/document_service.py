import uuid
import os
from typing import Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import time

from app.config import settings

class DocumentService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.index_name = "contract-intelligence-v2"
        self.document_metadata = {}
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Initialize Pinecone with new SDK
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name in existing_indexes:
                # Check dimension
                index_info = self.pc.describe_index(self.index_name)
                if index_info.dimension != 1536:
                    print(f"Index has wrong dimension ({index_info.dimension}). Deleting...")
                    self.pc.delete_index(self.index_name)
                    existing_indexes.remove(self.index_name)
            
            if self.index_name not in existing_indexes:
                print(f"Creating index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                time.sleep(10)
                print(f"Index '{self.index_name}' created successfully")
            else:
                print(f"Using existing index '{self.index_name}'")
                
        except Exception as e:
            print(f"Pinecone initialization failed: {e}")
            raise

    def process_pdf(self, file_content: bytes, filename: str) -> str:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}.pdf")

        with open(file_path, "wb") as f:
            f.write(file_content)

        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            print(f"Loaded {len(documents)} pages from {filename}")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")
            
            if not chunks:
                raise Exception("No content could be extracted from PDF")

            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            print(f"Storing in Pinecone with namespace: {doc_id}")

            # Store in Pinecone with namespace
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace=doc_id
            )
            
            # Verify storage
            time.sleep(2)  # Wait for indexing
            test_results = vectorstore.similarity_search("contract agreement", k=1)
            print(f"Verification: Found {len(test_results)} chunks in vectorstore")

            # Save metadata
            self.document_metadata[doc_id] = {
                "filename": filename,
                "chunks": len(chunks),
                "pages": len(documents),
                "status": "processed",
                "namespace": doc_id
            }

            print(f"Successfully processed {filename} (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            raise
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        return self.document_metadata.get(document_id)
    
    def get_all_documents(self) -> Dict[str, Any]:
        return self.document_metadata
    
    def document_exists(self, document_id: str) -> bool:
        return document_id in self.document_metadata
    
# Create singleton instance
document_service = DocumentService()