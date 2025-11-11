"""
RAG Pipeline - FREE Version
Uses HuggingFace + ChromaDB + Ollama (No OpenAI needed)
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from document_processor import DocumentProcessor
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests


class FreeRAGPipeline:
    """
    Complete RAG pipeline using FREE alternatives.
    No OpenAI API key required!
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize FREE RAG Pipeline."""
        self.config = self._load_config(config_path)
        self._initialize_components()
        logger.info("FREE RAG Pipeline initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration for FREE system."""
        return {
            'document_processing': {
                'chunking': {
                    'chunk_size': 1000,
                    'chunk_overlap': 200
                }
            },
            'embeddings': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2'
            },
            'vector_store': {
                'persist_directory': './data/chroma_db_free',
                'collection_name': 'rag_documents'
            },
            'retrieval': {
                'top_k': 5
            },
            'generation': {
                'model': 'phi3:mini',
                'ollama_url': 'http://localhost:11434'
            }
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Document Processor
        doc_config = self.config['document_processing']['chunking']
        self.document_processor = DocumentProcessor(
            chunk_size=doc_config['chunk_size'],
            chunk_overlap=doc_config['chunk_overlap']
        )
        logger.info("✓ Document processor initialized")
        
        # Embedding Model (FREE - HuggingFace)
        emb_config = self.config['embeddings']
        logger.info(f"Loading embedding model: {emb_config['model']}...")
        self.embedding_model = SentenceTransformer(emb_config['model'])
        logger.info("✓ Embedding model loaded")
        
        # Vector Store (FREE - ChromaDB)
        vs_config = self.config['vector_store']
        self.chroma_client = chromadb.PersistentClient(
            path=vs_config['persist_directory']
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=vs_config['collection_name']
        )
        logger.info("✓ Vector store initialized")
        
        # Ollama settings
        gen_config = self.config['generation']
        self.ollama_model = gen_config['model']
        self.ollama_url = gen_config['ollama_url']
        logger.info(f"✓ Ollama configured: {self.ollama_model}")
    
    def ingest_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Ingesting document: {document_path}")
        
        try:
            chunks = self.document_processor.process_document(
                document_path,
                metadata=metadata
            )
            
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            chunk_metadata = [chunk['metadata'] for chunk in chunks]
            chunk_ids = [f"{Path(document_path).stem}_chunk_{i}" for i in range(len(chunks))]
            
            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=chunk_metadata,
                ids=chunk_ids
            )
            
            return {
                'status': 'success',
                'document': document_path,
                'chunks_processed': len(chunks),
                'embeddings_generated': len(embeddings)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return {'status': 'error', 'document': document_path, 'error': str(e)}
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        logger.info(f"Processing query: {question}")
        
        try:
            k = top_k or self.config['retrieval']['top_k']
            query_embedding = self.embedding_model.encode([question])
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k
            )
            
            if not results['documents'][0]:
                return {'question': question, 'answer': "No relevant documents found.", 'sources': [], 'num_sources': 0}
            
            context = "\n\n".join([
                f"[Document {i+1}]\n{doc}"
                for i, doc in enumerate(results['documents'][0])
            ])
            
            answer = self._generate_answer(question, context)
            
            response = {
                'question': question,
                'answer': answer,
                'num_sources': len(results['documents'][0])
            }
            
            if return_sources:
                response['sources'] = [
                    {
                        'text': doc[:200] + '...' if len(doc) > 200 else doc,
                        'metadata': meta
                    }
                    for doc, meta in zip(results['documents'][0], results['metadatas'][0])
                ]
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {'question': question, 'answer': f"Error: {str(e)}", 'error': str(e)}
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate a **concise and clean** answer (instead of dumping resume text).
        """
        prompt = f"""
You are a helpful assistant. Answer the user's question based ONLY on the context.

Context:
{context}

Question:
{question}

Your answer must follow these rules:
- Respond in 1–4 clear sentences.
- Summarize the key information.
- DO NOT copy long text chunks.
- DO NOT include phone numbers, emails, or IDs.
- If the answer is not in the context, say: "I couldn't find that information in the documents."

Answer:
""".strip()

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'vector_store': 'chromadb',
            'embedding_model': self.config['embeddings']['model'],
            'generation_model': self.ollama_model,
            'total_documents': self.collection.count()
        }
    
    def clear_database(self):
        logger.warning("Clearing vector store...")
        self.chroma_client.delete_collection(self.collection.name)
        self.collection = self.chroma_client.create_collection(self.collection.name)
        logger.info("Vector store cleared")


if __name__ == "__main__":
    pipeline = FreeRAGPipeline()
    print("FREE RAG Pipeline ready!")
    print(pipeline.get_statistics())
