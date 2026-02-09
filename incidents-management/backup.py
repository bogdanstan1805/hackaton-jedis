import pandas as pd
from langchain_ollama import ChatOllama
from langchain.tools import tool
import numpy as np

df = pd.read_csv("incidents.csv")

# Combine embedding columns into a single vector
embedding_columns = [col for col in df.columns if col.startswith("emb_")]
df["embeddings"] = df[embedding_columns].apply(lambda row: row.values, axis=1)

# Example: Calculate similarity between a query embedding and the embeddings in the CSV
def calculate_similarity(query_embedding, csv_embeddings):
    # Normalize the embeddings
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    csv_embeddings = np.array([emb / np.linalg.norm(emb) for emb in csv_embeddings])

    # Compute cosine similarity
    similarities = np.dot(csv_embeddings, query_embedding)
    return similarities

# Example usage
query_embedding = np.array([0.1] * 768)  # Replace with your 768-dimensional query embedding
similarities = calculate_similarity(query_embedding, df["embeddings"].tolist())

# Add similarities to the DataFrame and sort by relevance
df["similarity"] = similarities
df_sorted = df.sort_values(by="similarity", ascending=False)

# Display the top 5 most similar rows
print(df_sorted.head(5))
@tool
def get_incident_by_id(number: str) -> dict:
    """Return an incident by its ID which is specified in number column."""
    row = df[df["number"].astype(str) == number]
    if row.empty:
        return {"error": "incident not found"}
    return row.to_dict(orient="records")[0]

@tool
def keyword_search(query: str) -> dict:
    """Search the CSV rows containing the keyword."""
    result = df[df.apply(
        lambda row: row.astype(str).str.contains(query, case=False).any(),
        axis=1
    )]
    return result.head(10).to_dict(orient="records")

@tool
def list_columns(_: str = "") -> list:
    """List all columns available in the CSV."""
    return df.columns.tolist()

def find_similar_incidents(query_embedding: list) -> list:
    """Find incidents most similar to the given query embedding."""
    import numpy as np

    # Convert query embedding to a numpy array
    query_embedding = np.array(query_embedding)

    # Normalize the embeddings
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    csv_embeddings = np.array([emb / np.linalg.norm(emb) for emb in df["embeddings"]])

    # Compute cosine similarity
    similarities = np.dot(csv_embeddings, query_embedding)

    # Add similarities to the DataFrame and sort by relevance
    df["similarity"] = similarities
    df_sorted = df.sort_values(by="similarity", ascending=False)

    # Return the top 5 most similar rows
    return df_sorted.head(5).to_dict(orient="records")

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0
).bind_tools([get_incident_by_id, keyword_search, list_columns, find_similar_incidents])

base_messages = [
    (
        "system",
        "You are an incident analysis assistant. "
        "You MUST use the provided tools to answer questions about incidents. "
        "Return a tool call instead of answering directly."
    )
]

def main():
    print("Assistant ready.\n")

    while True:
        user_input = input("You: ")

        messages = base_messages + [("human", user_input)]

        # DO NOT USE .stream() – tool calling will fail.
        response = llm.invoke(messages)
        print("Assistant:", response)


if __name__ == "__main__":
    main()