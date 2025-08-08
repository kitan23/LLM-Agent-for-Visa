import os
import pandas as pd
from pathlib import Path
import pypdf  # Modern replacement for PyPDF2
import fitz  # pymupdf - better PDF extraction
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyMuPDF (better than PyPDF2)"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def extract_text_from_pdf_pypdf(pdf_path):
    """Fallback method using pypdf"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path} with pypdf: {e}")
        return ""

def process_pdfs_to_parquet(pdf_directory, output_path, chunk_size=400, chunk_overlap=100):
    """
    Extract text from all PDFs in directory and save as parquet
    
    Args:
        pdf_directory: Path to directory containing PDF files
        output_path: Path where to save the parquet file
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
    """
    pdf_directory = Path(pdf_directory)
    pdf_files = list(pdf_directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return
    
    all_documents = []
    
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        
        # Try PyMuPDF first, fallback to pypdf
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            text = extract_text_from_pdf_pypdf(pdf_file)
        
        if text.strip():
            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_file.name,
                    "file_path": str(pdf_file),
                    "file_size": pdf_file.stat().st_size
                }
            )
            all_documents.append(doc)
        else:
            print(f"Warning: No text extracted from {pdf_file.name}")
    
    if not all_documents:
        print("No documents to process!")
        return
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(all_documents)
    
    # Convert to DataFrame
    data = []
    for i, doc in enumerate(splits):
        data.append({
            "chunk_id": i,
            "passage": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "file_path": doc.metadata.get("file_path", ""),
            "file_size": doc.metadata.get("file_size", 0),
            "chunk_length": len(doc.page_content)
        })
    
    df = pd.DataFrame(data)
    df['created_timestamp'] = pd.Timestamp.now(tz='UTC')
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} chunks from {len(all_documents)} documents to {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    # Configuration
    PDF_DIRECTORY = "examples"  # Directory containing PDF files
    OUTPUT_PATH = "data/processed/documents.parquet"
    
    # Process PDFs
    process_pdfs_to_parquet(PDF_DIRECTORY, OUTPUT_PATH)
    
    # Show sample of the data
    df = pd.read_parquet(OUTPUT_PATH)
    print("\nSample data:")
    print(df.head(2))
    print(f"\nUnique sources: {df['source'].unique()}") 