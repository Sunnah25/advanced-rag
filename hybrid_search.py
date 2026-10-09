from rank_bm25 import BM25Okapi


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
print(tokenized_chunks[0])

# Create BM25 index
bm25 = BM25Okapi(tokenized_chunks)


#Search
query = "2022 World Cup goals"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)



print(f"\nQuery: '{query}'")
for i, (chunk, score) in enumerate(zip(chunks, scores)):
    print(f"Score {score:.3f}: {chunk}")