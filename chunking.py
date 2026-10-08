import re


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

text = """Messi scored a brilliant hat-trick against Real Madrid. 
It was the Champions League final last night. 
The crowd went wild at the Bernabeu. 
Argentina's captain was unstoppable."""




sentences = split_into_sentences(text)
for i, s in enumerate(sentences):
    print(f"Sentences {i+1}: {s}")


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

# Test it
chunks = smart_chunk(text)
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)