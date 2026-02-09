import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import AIMessage

from langchain_core.prompts import PromptTemplate


# Function to generate a creative short description
def generate_short_description(query):
    # Define the prompt template
    template = """
    You are an AI assistant tasked with creating a summarized and creative short description for an incident report.
    Based on the following query, generate a concise and engaging short description:

    Query: {query}

    Short Description:
    """
    prompt = PromptTemplate(input_variables=["query"], template=template)

    # Initialize the language model (replace with your preferred model)
    llm = ChatOllama(model="llama3")  # Ensure you have access to this model

    # Generate the short description
    short_description = llm.invoke(prompt.format(query=query))
    return short_description.text.strip()


# Function to create a new incident
def create_new_incident(file_path, query):
    short_description = generate_short_description(query)
    comments = "Generated incident based on user query."

    # Append the new incident to the CSV file
    append_to_csv(file_path, short_description, comments)
    print(f"New incident created with short description: {short_description}")

# Load incidents from CSV
def load_incidents(file_path):
    df = pd.read_csv(file_path)
    df['combined_text'] = df['short_description'] + " " + df['comments']
    return df

def find_match_with_ollama(query, df, model):
    # Initialize Ollama embeddings
    ollama = OllamaEmbeddings(model=model)

    # Generate query embedding
    query_embedding = ollama.embed_query(query)

    # Generate embeddings for the dataset
    df["embedding"] = df["combined_text"].apply(lambda x: ollama.embed_query(x))

    # Calculate similarities
    similarities = df["embedding"].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
    max_similarity = similarities.max()

    print(max_similarity)

    if max_similarity > 0.7:  # Threshold for a useful match
        best_match_index = similarities.idxmax()
        return df.iloc[best_match_index][['number', 'short_description', 'comments']], max_similarity
    return None, max_similarity


# Append a new incident to the CSV file
def append_to_csv(file_path, short_description, comments):
    df = pd.read_csv(file_path)
    new_row = {
        'number': f'NEW_INC_{len(df) + 1}',
        'short_description': short_description,
        'comments': comments
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(file_path, index=False)


@tool
def initialize_search():
    """
    Search for a incident that is similar to the user query.
    If a match is found, return the incident details.
    If no match is found, ask the user for more information and try again up to three times.
    If still no match is found after three attempts, create a new incident with the provided information.
    Args:
        Incident number:
        Short Description:
        TraceId:
    """
    return True

# Main function
def main():
    file_path = 'incidents.csv'  # Path to your CSV file
    df = load_incidents(file_path)

    llm = ChatOllama(
        model="gpt-oss:20b",
        validate_model_on_init=True,
        temperature=0,
    ).bind_tools([initialize_search])

    while True:
        prompt = input()
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        result = llm.invoke(
            messages
        )
        print(result.content)
        # if isinstance(result, AIMessage) and result.tool_calls:
        #     print(result.tool_calls)


    # print("Welcome! I’m here to help you find or add incidents.")
    # while True:
    #     query = input("\nWhat’s your query? (Type 'exit' to quit): ")
    #     if query.lower() == 'exit':
    #         print("Goodbye!")
    #         break
    #
    #     tries = 0
    #     while tries < 3:
    #         match, similarity = find_match_with_ollama(query, df, "llama3")
    #
    #         if match is not None:
    #             print(f"\nMatch found!")
    #             print(f"Incident Number: {match['number']}")
    #             print(f"Description: {match['short_description']}")
    #             print(f"Comments: {match['comments']}")
    #             print(f"Similarity: {similarity:.2f}")
    #             break
    #         else:
    #             print("\nNo match found.")
    #             additional_info = input("Could you provide more details (e.g., traceId, additional context)? ")
    #             if not additional_info.strip():
    #                 print("No additional information provided. Unable to find a match.")
    #                 break
    #             query += " " + additional_info
    #             tries += 1
    #
    #     if tries == 3:
    #         print("\nNo match found after three tries. Creating a new incident...")
    #         incident_data = {
    #             "short_description": create_new_incident(file_path, query),
    #             "comments": "No match found after three attempts."
    #         }
    #         response = requests.post("http://localhost:5000/api/create", json=incident_data)
    #         if response.status_code == 201:
    #             print("Incident successfully created on the server.")
    #         else:
    #             print(f"Failed to create incident. Server responded with status code {response.status_code}.")


if __name__ == '__main__':
    main()