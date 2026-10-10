from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np


#sample chunks
chunks = [
    "Messi is the greatest footballer of all time.",
    "Messi scored 4 goals in the 2022 World Cup final.",
    "Argentina won the 2022 world cup defeating France on penalties.",
    "Cristiano Ronaldo has won 5 Ballon d'Or awards.",
    "The Champions League final was held at the Bernabue"
]

# BM5 needs tokenized input - split each chunk into words
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
# Create BM25 index
bm25 = BM25Okapi(tokenized_chunks)


# Semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)


def hybrid_search(query, top_k=2, alpha=0.5):
    # alpha controls balance: 0.0 = pure keyword, 1.0 = pure semantic

    #BM25 scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    #Semantic scores
    query_embedding = model.encode([query])
    semantic_scores = np.dot(chunk_embeddings, query_embedding.T).flatten()


    #Normanlise both to 0-1 range
    bm25_norm = bm25_scores/(bm25_scores.max() + 1e-9)
    semantic_norm = semantic_scores/(semantic_scores.max() + 1e-9)

    #Combine
    combined = alpha * semantic_norm + (1-alpha) + bm25_norm


    #Get top_k results
    top_indices = combined.argsort()[::-1][:top_k]

    return [(chunks[i], combined[i]) for i in top_indices]








# Test
query = "Argentina captain trophy"
print(f"Query: '{query}'\n")

print("Keyword only (alpha=0.0)")
for chunk, score in hybrid_search(query, alpha=0.0):
    print(f"  {score:.3f}: {chunk}")

print("\nSemantic only (alpha=1.0)")
for chunk, score in hybrid_search(query, alpha=1.0):
    print(f"  {score:.3f}: {chunk}")

print("\nHybrid (alpha=0.5)")
for chunk, score in hybrid_search(query, alpha=0.5):
    print(f"  {score:.3f}: {chunk}")