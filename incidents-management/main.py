import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ollama import Client

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.prompts import ChatPromptTemplate

from flask import Flask, request, jsonify

app = Flask(__name__)

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv("incidents.csv")
emb_cols = [c for c in df.columns if c.startswith("emb_")]
EMB_DIM = len(emb_cols)
emb_matrix = df[emb_cols].to_numpy(dtype=np.float32)


# ============================================================
# EMBEDDING CLIENT
# ============================================================

client = Client()
EMBED_MODEL = "nomic-embed-text"


def embed_query(text: str) -> np.ndarray:
    resp = client.embeddings(model=EMBED_MODEL, prompt=text)
    vec = np.array(resp["embedding"], dtype=np.float32)
    if vec.shape[0] != EMB_DIM:
        raise ValueError("Embedding dimension mismatch")
    return vec.reshape(1, -1)


# ============================================================
# TOOLS
# ============================================================

@tool
def get_incident_by_id(incident_number: str) -> dict:
    """Return an incident based on its 'number' column."""
    row = df[df["number"].astype(str) == incident_number]
    if row.empty:
        return {"error": f"Incident {incident_number} not found"}
    return row.iloc[0].to_dict()


@tool
def semantic_search(query: str, k: int = 5) -> list:
    """Semantic search over incident embeddings."""
    qvec = embed_query(query)
    sims = cosine_similarity(qvec, emb_matrix)[0]
    top_idx = np.argsort(-sims)[:k]

    return [
        {
            "number": str(df.iloc[i]["number"]),
            "short_description": df.iloc[i].get("short_description", ""),
            "priority": int(df.iloc[i].get("priority", -1)),
            "assignment_group": df.iloc[i].get("assignment_group", ""),
            "similarity": float(sims[i]),
        }
        for i in top_idx
    ]


@tool
def list_columns(_: str = "") -> list:
    """List CSV columns."""
    return df.columns.tolist()


tools = [get_incident_by_id, semantic_search, list_columns]


# ============================================================
# LLM + AGENT WITH MEMORY
# ============================================================

llm = ChatOllama(model="gpt-oss:20b", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are an AI Incident Analysis Assistant.\n"
         "You always use tools to fetch incident data.\n"
         "You remember prior conversation context.\n"
         "If the user refers to 'that incident', you infer it from chat history.\n"
         "Never hallucinate incident details."
         "If you find any incident, give the user url http:127.0.0.1:5000/incident/numberOfIncident just one link"
         "Be a little bit less verbose, just a short analysis and that should be it, or print multiple incidents if they are similar"
         "Use emojis when it's the case and format the text in a beautiful way to be displayed in UI"
         "If you find an incident, ADD some incidents in the response that are similar with it"
         "Please calculate confidence score and show at the end of the output based on the similarity score you have"
        ),
        ("human", "{chat_history}\nUser: {input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

history = ""  # text format works best for gpt-oss models

@app.route('/chat', methods=['POST'])
def chat():
    global history

    try:
        # Get JSON data from the request
        data = request.get_json()
        user_input = data.get('input', '')

        print(history)

        if not user_input:
            return jsonify({"error": "Input is required"}), 400

        # Process the input using the model
        result = executor.invoke({
            "input": user_input,
            "chat_history": history  # Optional chat history
        })

        # Extract the model's response
        response = result.get("output", "No response generated")

        history += f"\nUser: {user_input}\nAssistant: {response}\n"

        # Return the response as JSON
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)