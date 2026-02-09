"""
INCIDENT ANALYST AGENT (NO FAISS, NO EMBEDDINGS)
Uses CSV directly via Pandas.
"""

import pandas as pd
from langchain_ollama import ChatOllama
from langchain.tools import tool


# ---------------------------------------------------------
# Load CSV only once
# ---------------------------------------------------------
df = pd.read_csv("incidents.csv")


# ---------------------------------------------------------
# Tools that directly use the CSV
# ---------------------------------------------------------

@tool
def get_incident_by_id(incident_id: str):
    """Return an incident by its ID."""
    row = df[df["number"].astype(str) == incident_id]
    if row.empty:
        return {"error": "incident not found"}
    return row.to_dict(orient="records")[0]


@tool
def keyword_search(query: str):
    """Search the CSV rows containing the keyword."""
    result = df[df.apply(
        lambda row: row.astype(str).str.contains(query, case=False).any(),
        axis=1
    )]
    return result.head(10).to_dict(orient="records")


@tool
def list_columns(_: str = ""):
    """List all columns available in the CSV."""
    return df.columns.tolist()


# ---------------------------------------------------------
# LLM with tools
# ---------------------------------------------------------

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0,
    streaming=True,
).bind_tools([get_incident_by_id, keyword_search, list_columns])


# ---------------------------------------------------------
# System instructions ONCE
# ---------------------------------------------------------

base_messages = [
    (
        "system",
        "You are an incident analysis assistant. "
        "Use the provided tools to look up information from the CSV. "
        "Do NOT try to analyze the CSV directly."
    )
]


# ---------------------------------------------------------
# Chat loop
# ---------------------------------------------------------

def main():
    print("Assistant ready. Using CSV directly (no FAISS, no embeddings).\n")

    while True:
        user_input = input("You: ")

        messages = base_messages + [("human", user_input)]

        print("Assistant:", end=" ", flush=True)
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()