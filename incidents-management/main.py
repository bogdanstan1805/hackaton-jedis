import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ollama import Client
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.prompts import ChatPromptTemplate

# ============================================================
# LOAD CSV & PREPARE EMBEDDINGS
# ============================================================

df = pd.read_csv("incidents.csv")

# Identify all embedding columns automatically
emb_cols = [c for c in df.columns if c.startswith("emb_")]
EMB_DIM = len(emb_cols)
print(f"Detected {EMB_DIM} embedding dimensions.")

# Convert embeddings to matrix
emb_matrix = df[emb_cols].to_numpy(dtype=np.float32)


# ============================================================
# OLLAMA CLIENT FOR QUERY EMBEDDINGS (nomic-embed-text)
# ============================================================

client = Client()
EMBED_MODEL = "nomic-embed-text"  # must match your original model

def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    vec = np.array(resp["embedding"], dtype=np.float32)
    if vec.shape[0] != EMB_DIM:
        raise ValueError(f"Embedding dim mismatch: query={vec.shape[0]}, csv={EMB_DIM}")
    return vec.reshape(1, -1)


# ============================================================
# TOOLS
# ============================================================

@tool
def get_incident_by_id(incident_number: str) -> dict:
    """Return a single incident by its 'number' column."""
    row = df[df["number"].astype(str) == incident_number]
    if row.empty:
        return {"error": f"Incident {incident_number} not found"}
    return row.iloc[0].to_dict()


@tool
def semantic_search(query: str, k: int = 5) -> list:
    """Return the top-k semantically related incidents using embeddings."""
    qvec = embed_query(query)
    sims = cosine_similarity(qvec, emb_matrix)[0]

    top_idx = np.argsort(-sims)[:k]

    output = []
    for i in top_idx:
        item = {
            "number": str(df.iloc[i]["number"]),
            "short_description": df.iloc[i].get("short_description", ""),
            "assignment_group": df.iloc[i].get("assignment_group", ""),
            "priority": int(df.iloc[i].get("priority", -1)),
            "similarity": float(sims[i]),
            "description": df.iloc[i].get("description", "")
        }
        output.append(item)

    return output


@tool
def list_columns(_: str = "") -> list:
    """Return list of CSV columns."""
    return df.columns.tolist()


tools = [get_incident_by_id, semantic_search, list_columns]


# ============================================================
# LLM WITH AGENT (NO STREAMING — REQUIRED FOR gpt-oss:20b)
# ============================================================

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI Incident Analysis Assistant. "
         "Use the available tools ONLY to answer questions about incidents. "
         "Use semantic_search for natural-language queries. "
         "Use get_incident_by_id when the user provides an incident number. "
         "Do NOT hallucinate incident data."
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("\nAI Incident Assistant ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        result = agent_executor.invoke({"input": user_input})
        print("\nAssistant:", result["output"], "\n")


if __name__ == "__main__":
    main()