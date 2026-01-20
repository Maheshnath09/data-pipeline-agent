"""
Knowledge Base Indexer - Indexes markdown documents into ChromaDB vector store.
Run this script to populate the RAG knowledge base.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def parse_markdown_sections(content: str) -> List[Tuple[str, str, str]]:
    """
    Parse markdown content into sections.
    
    Returns:
        List of (section_title, category, content) tuples
    """
    sections = []
    current_h1 = "General"
    current_h2 = ""
    current_content = []
    
    lines = content.split('\n')
    
    for line in lines:
        # H1 header
        if line.startswith('# '):
            # Save previous section
            if current_content:
                section_text = '\n'.join(current_content).strip()
                if section_text:
                    title = current_h2 or current_h1
                    sections.append((title, current_h1.lower().replace(' ', '_'), section_text))
            current_h1 = line[2:].strip()
            current_h2 = ""
            current_content = []
        
        # H2 header
        elif line.startswith('## '):
            # Save previous section
            if current_content:
                section_text = '\n'.join(current_content).strip()
                if section_text:
                    title = current_h2 or current_h1
                    sections.append((title, current_h1.lower().replace(' ', '_'), section_text))
            current_h2 = line[3:].strip()
            current_content = []
        
        # H3 header - include as content with title
        elif line.startswith('### '):
            current_content.append(line)
        
        # Regular content
        else:
            current_content.append(line)
    
    # Don't forget last section
    if current_content:
        section_text = '\n'.join(current_content).strip()
        if section_text:
            title = current_h2 or current_h1
            sections.append((title, current_h1.lower().replace(' ', '_'), section_text))
    
    return sections


def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into smaller chunks for better embedding quality.
    """
    # Split by double newlines first (paragraphs)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        if current_size + para_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


def index_knowledge_base(
    knowledge_base_dir: str = None,
    persist_directory: str = None
):
    """
    Index all markdown files in the knowledge base directory.
    """
    from rag.vector_store import VectorStore
    
    # Default paths
    if knowledge_base_dir is None:
        knowledge_base_dir = Path(__file__).parent / "knowledge_base"
    else:
        knowledge_base_dir = Path(knowledge_base_dir)
    
    if persist_directory is None:
        persist_directory = Path(__file__).parent.parent / "chroma_db"
    
    # Initialize vector store
    vector_store = VectorStore(
        persist_directory=str(persist_directory),
        collection_name="data_pipeline_knowledge"
    )
    
    # Reset existing data
    print("[Indexer] Resetting vector store...")
    vector_store.reset()
    
    # Find all markdown files
    md_files = list(knowledge_base_dir.glob("*.md"))
    print(f"[Indexer] Found {len(md_files)} markdown files")
    
    total_docs = 0
    
    for md_file in md_files:
        print(f"[Indexer] Processing: {md_file.name}")
        
        content = md_file.read_text(encoding='utf-8')
        sections = parse_markdown_sections(content)
        
        file_category = md_file.stem.lower()
        
        documents = []
        metadatas = []
        ids = []
        
        for i, (title, category, text) in enumerate(sections):
            # Chunk the text
            chunks = chunk_text(text, max_chunk_size=500)
            
            for j, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Skip very short chunks
                    continue
                
                doc_id = f"{file_category}_{i}_{j}"
                documents.append(chunk)
                metadatas.append({
                    "source": md_file.name,
                    "category": category,
                    "title": title,
                    "file_category": file_category
                })
                ids.append(doc_id)
        
        if documents:
            vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            total_docs += len(documents)
    
    print(f"[Indexer] Complete! Indexed {total_docs} document chunks")
    print(f"[Indexer] Vector store location: {persist_directory}")
    
    return total_docs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    index_knowledge_base()
