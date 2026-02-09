# def find_match_with_ollama(query, df, model):
#     # Initialize Ollama embeddings
#     ollama = OllamaEmbeddings(model=model)
#
#     # Generate query embedding
#     query_embedding = ollama.embed_query(query)
#
#     # Generate embeddings for the dataset
#     df["embedding"] = df["combined_text"].apply(lambda x: ollama.embed_query(x))
#
#     # Calculate similarities
#     similarities = df["embedding"].apply(lambda x: cosine_similarity([query_embedding], [x])[0][0])
#     max_similarity = similarities.max()
#
#     print(max_similarity)
#
#     if max_similarity > 0.7:  # Threshold for a useful match
#         best_match_index = similarities.idxmax()
#         return df.iloc[best_match_index][['number', 'short_description', 'comments']], max_similarity
#     return None, max_similarity




# def create_new_incident(file_path, query):
#     short_description = generate_short_description(query)
#     comments = "Generated incident based on user query."
#
#     # Append the new incident to the CSV file
#     append_to_csv(file_path, short_description, comments)
#     print(f"New incident created with short description: {short_description}")