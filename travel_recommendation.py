import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load and preprocess data
df = pd.read_csv('Hotel_Reviews.csv')
df = df[df['Positive_Review'].notna() & df['Negative_Review'].notna()]  # Remove rows with NaN values in reviews
data = df.sample(700).to_dict('records')  # Sample 700 records

# Encode data using Sentence Transformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Qdrant (running on another container)
qdrant = QdrantClient("http://qdrant:6333")

# Create a collection to store reviews
qdrant.recreate_collection(
    collection_name="travel_recommendations",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)

# Prepare data points
points = [
    models.PointStruct(id=idx, vector=encoder.encode(doc["Positive_Review"] + " " + doc["Negative_Review"]).tolist(), payload=doc)
    for idx, doc in enumerate(data)
]

# Upload data points using the correct method
qdrant.upsert(
    collection_name="travel_recommendations",
    points=points
)

# Query
user_prompt = "I'm looking for a hotel in Amsterdam with great service and beautiful surroundings."
query_vector = encoder.encode(user_prompt).tolist()
hits = qdrant.search(
    collection_name="travel_recommendations",
    query_vector=query_vector,
    limit=3
)

search_results = [hit.payload for hit in hits]
for hit in hits:
    print(hit.payload, "score:", hit.score)

# Connect to local LLM with OpenAI interface
client = OpenAI(
    base_url="http://host.docker.internal:8080/v1",  # Use Docker gateway
    api_key="sk-no-key-required"
)

completion = client.chat.completions.create(
    model="LLaMA_CPP",
    messages=[
        {"role": "system", "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."},
        {"role": "user", "content": "I'm looking for a hotel in Amsterdam with great service and beautiful surroundings."},
        {"role": "assistant", "content": str(search_results)}
    ]
)
print(completion.choices[0].message)
