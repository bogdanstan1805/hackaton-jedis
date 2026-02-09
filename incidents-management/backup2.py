import pandas as pd
import numpy as np
import requests
from langchain_ollama import ChatOllama

EMBEDDINGS_CSV = "incidents_backup.csv"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:1b"
OLLAMA_URL = "http://localhost:11434"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5

df = pd.read_csv(EMBEDDINGS_CSV)

emb_cols = [c for c in df.columns if c.startswith("emb_")]
data_cols = [c for c in df.columns if not c.startswith("emb_")]

embeddings_matrix = df[emb_cols].values.astype(np.float32)
norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1
embeddings_matrix_norm = embeddings_matrix / norms

df_data = df[data_cols]
conversation_history = []


def get_query_embedding(text: str) -> np.ndarray:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)


def retrieve(query: str) -> pd.DataFrame:
    q_emb = get_query_embedding(query)
    q_norm = np.linalg.norm(q_emb)
    if q_norm > 0:
        q_emb = q_emb / q_norm

    scores = embeddings_matrix_norm @ q_emb
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = df_data.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]
    return results[results["similarity_score"] >= SIMILARITY_THRESHOLD]


def format_context(results: pd.DataFrame) -> str:
    parts = []
    for _, row in results.iterrows():
        inc = row.get("number", "UNKNOWN")
        score = row["similarity_score"]
        desc = row.get("short_description", "")
        cause = row.get("caused_by", row.get("close_notes", ""))
        resolution = row.get("work_notes", "")
        parts.append(
            f"Incident: {inc} ({score:.0%} match)\n"
            f"Description: {desc}\n"
            f"Root cause: {cause}\n"
            f"Resolution: {resolution}"
        )
    return "\n\n".join(parts)


llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    streaming=True,
)

SYSTEM_PROMPT = (
    "You are an incident analysis assistant. "
    "You have access to a database of past incidents for reference.\n\n"
    "IMPORTANT RULES:\n"
    "1. If the user's description is vague or lacks details (no error code, no specific endpoint, "
    "no specific service mentioned), you MUST ask clarifying questions FIRST. "
    "For example ask: What error code or HTTP status are you seeing? Which endpoint or service is affected? "
    "When did it start? Is it intermittent or constant?\n"
    "2. Only AFTER you have enough details, search the similar incidents provided in context "
    "and present matches with incident number, match percentage, root cause, and resolution.\n"
    "3. Never guess or assume details the user hasn't provided.\n"
    "4. Be concise and actionable."
)


def build_messages(user_input: str, context: str | None) -> list:
    messages = [("system", SYSTEM_PROMPT)]

    for role, content in conversation_history:
        messages.append((role, content))

    if context:
        messages.append((
            "human",
            f"[Similar past incidents for reference - use only when you have enough details "
            f"from the user to match]\n{context}\n\nUser: {user_input}"
        ))
    else:
        messages.append(("human", user_input))

    return messages


def main():
    print(f"RAG Incident Analyst ready. top-{TOP_K}, >{SIMILARITY_THRESHOLD:.0%} threshold.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        results = retrieve(user_input)
        context = format_context(results) if not results.empty else None

        messages = build_messages(user_input, context)

        print("Assistant:", end=" ", flush=True)
        response_text = ""
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
            response_text += chunk.content
        print("\n")

        conversation_history.append(("human", user_input))
        conversation_history.append(("assistant", response_text))

        if len(conversation_history) > 20:
            conversation_history[:] = conversation_history[-20:]


if __name__ == "__main__":
    main()