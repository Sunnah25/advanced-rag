import sys
import re
sys.stdout.reconfigure(encoding="utf-8")
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.schema import Document

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]




def smart_chunk(text, chunk_size=3, overlap=1):
    sentences = split_into_sentences(text)
    chunks = []
    
    i = 0
    while i < len(sentences):
        # Take chunk_size sentences
        chunk = sentences[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        # Move forward by chunk_size - overlap
        i += chunk_size - overlap
    
    return chunks





def smart_chunk_document(file_path, chunk_size=3, overlap=1):
    # Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Combine all pages into one text
    full_text = ' '.join([page.page_content for page in pages])
    
    # Smart chunk
    chunks = smart_chunk(full_text, chunk_size, overlap)
    
    # Convert to LangChain Document objects
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append(Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": file_path}
        ))
    
    return documents

# Test with CV
docs = smart_chunk_document(r"AICV9SEP.pdf")
print(f"Total chunks: {len(docs)}")
print(f"\nFirst chunk:")
print(docs[0].page_content)
print(f"\nSecond chunk:")
print(docs[1].page_content)
print(f"\nThird chunk:")
print(docs[2].page_content)
print(f"\nFourth chunk:")
print(docs[3].page_content)