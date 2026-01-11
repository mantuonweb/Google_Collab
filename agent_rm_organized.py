#!/usr/bin/env python
"""
Resume Management System - Organized Version
Handles resume ingestion, search, and management using LangChain and FAISS
"""

import os
import shutil
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables FIRST before any other imports
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader


class Config:
    """Configuration settings for the resume management system"""
    
    # Directory paths
    RESUMES_DIR = "./resumes"
    RESUME_DB_DIR = "./resume_db"
    
    # Model settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0
    
    # Text splitting settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Search settings
    SEARCH_K = 5
    
    @classmethod
    def setup(cls):
        """Initialize directories and environment"""
        # Ensure .env is loaded
        load_dotenv()
        
        # Create directories if they don't exist
        os.makedirs(cls.RESUMES_DIR, exist_ok=True)
        os.makedirs(cls.RESUME_DB_DIR, exist_ok=True)
        
        # Verify API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found. Please create a .env file with your API key."
            )
        
        print("✓ Setup complete")
        print(f"✓ Resumes folder: {cls.RESUMES_DIR}")
        print(f"✓ Database folder: {cls.RESUME_DB_DIR}")


class ResumeManager:
    """Core resume management functionality"""
    
    def __init__(self):
        """Initialize the resume manager with embeddings and LLM"""
        # Ensure environment is set up
        if not os.getenv("OPENAI_API_KEY"):
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
    
    def ingest_resumes(self) -> str:
        """
        Load resumes from ./resumes folder and add to vector database
        
        Returns:
            str: Status message about the ingestion process
        """
        print("📥 Ingesting resumes...")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                Config.RESUMES_DIR, 
                glob="**/*.txt", 
                loader_cls=TextLoader
            )
            txt_docs = txt_loader.load()
        except Exception as e:
            print(f"Warning loading text files: {e}")
            txt_docs = []
        
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                Config.RESUMES_DIR, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
        except Exception as e:
            print(f"Warning loading PDF files: {e}")
            pdf_docs = []
        
        all_docs = txt_docs + pdf_docs
        
        if not all_docs:
            print("❌ No resumes found in ./resumes folder")
            return "No resumes found in ./resumes folder"
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, 
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(all_docs)
        
        # Check if FAISS index file exists
        db_file_exists = os.path.exists(f"{Config.RESUME_DB_DIR}/index.faiss")
        
        if db_file_exists:
            # Load existing and add new documents
            vectorstore = FAISS.load_local(
                Config.RESUME_DB_DIR, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(chunks)
            print(f"✓ Added {len(chunks)} chunks from {len(all_docs)} resumes")
            result = f"Added {len(chunks)} chunks from {len(all_docs)} resumes to existing database"
        else:
            # Create new vector store
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            print(f"✓ Created new database with {len(chunks)} chunks from {len(all_docs)} resumes")
            result = f"Created new database with {len(chunks)} chunks from {len(all_docs)} resumes"
        
        vectorstore.save_local(Config.RESUME_DB_DIR)
        print("✓ Database saved successfully")
        return result
    
    def list_resumes(self) -> str:
        """
        List all resumes stored in vector database
        
        Returns:
            str: Formatted list of resume filenames
        """
        print("📋 Listing resumes...")
        
        if not os.path.exists(f"{Config.RESUME_DB_DIR}/index.faiss"):
            print("❌ No database found. Please ingest resumes first.")
            return "No database found. Please ingest resumes first."
        
        vectorstore = FAISS.load_local(
            Config.RESUME_DB_DIR, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Get all documents
        all_docs = vectorstore.docstore._dict
        
        # Extract unique sources
        sources = set()
        for doc in all_docs.values():
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.add(os.path.basename(doc.metadata['source']))
        
        result = f"Found {len(sources)} resumes in database:\n"
        for i, source in enumerate(sorted(sources), 1):
            result += f"{i}. {source}\n"
            print(f" {i}. {source}")
        
        return result
    
    def search_resumes(self, skills: str) -> str:
        """
        Search resumes by skills and return best matches
        
        Args:
            skills: Comma-separated string of required skills
            
        Returns:
            str: LLM-generated summary of best matching candidates
        """
        print(f"🔍 Searching for candidates with skills: {skills}")
        
        if not os.path.exists(f"{Config.RESUME_DB_DIR}/index.faiss"):
            print("❌ No database found. Please ingest resumes first.")
            return "No database found. Please ingest resumes first."
        
        vectorstore = FAISS.load_local(
            Config.RESUME_DB_DIR, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Search for relevant resume chunks
        retriever = vectorstore.as_retriever(search_kwargs={"k": Config.SEARCH_K})
        docs = retriever.invoke(skills)
        
        # Create context from retrieved documents
        context = "\n\n".join([
            f"Resume {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
        
        # Create prompt for LLM
        prompt = f"""You are a recruiter assistant. Based on the following resume excerpts, identify and rank the best candidates for the required skills.

Required Skills: {skills}

Resume Excerpts:
{context}

Please provide a quick summary for the top 3 best matching candidates. For each candidate, include their relevant skills, why they are a good fit, and a matching percentage. The response should be concise.

Answer:"""
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        
        print("\n" + "="*60)
        print("🎯 SEARCH RESULTS")
        print("="*60)
        print(response.content)
        print("="*60)
        
        return response.content
    
    def clear_resumes(self) -> str:
        """
        Clear all resumes from vector database
        
        Returns:
            str: Status message
        """
        print("🗑️ Clearing resume database...")
        
        if os.path.exists(Config.RESUME_DB_DIR):
            shutil.rmtree(Config.RESUME_DB_DIR)
            os.makedirs(Config.RESUME_DB_DIR, exist_ok=True)
            print("✓ Database cleared successfully")
            return "Database cleared successfully"
        else:
            print("❌ No database found")
            return "No database found"


# Example usage and testing
if __name__ == "__main__":
    # Setup
    Config.setup()
    
    # Initialize manager
    manager = ResumeManager()
    
    # Example operations
    print("\n" + "="*60)
    print("RESUME MANAGEMENT SYSTEM - DEMO")
    print("="*60)
    
    # Uncomment to test:
    # manager.ingest_resumes()
    # manager.list_resumes()
    # manager.search_resumes("Python, Machine Learning")
