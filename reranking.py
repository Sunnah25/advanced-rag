from sentence_transformers import CrossEncoder

# Load cross-encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

chunks = [
    "Messi is the greatest footballer of all time",
    "Messi scored 4 goals in the 2022 World Cup final",
    "Argentina won the 2022 World Cup defeating France on penalties",
    "Cristiano Ronaldo has won five Ballon d'Or awards",
    "The Champions League final was held at the Bernabeu"
]


query = "How many goals did Messi score in the World Cup?"

#Cross-encoder needs (query, chunk) pairs
pairs = [[query, chunk] for chunk in chunks]

#Score each pair
scores = model.predict(pairs)


# Sort by score descending
ranked = sorted(zip(scores, chunks), reverse=True)

print(f"Query: {query}\n")
for score, chunk in ranked:
    print(f"{score:.2f} : {chunk}")