from langchain_community.llms import LlamaCpp  # Ensure you have the llama_cpp_py package installed

# Assuming other necessary imports are already there
import pickle

# Initialize LlamaCpp model
llm = LlamaCpp(
    model_path="llama-2-7b-chat.GGUF.q4_0.bin",  # Path to the model file
    n_gpu_layers=40,  # Number of GPU layers (adjust based on available GPUs)
    n_batch=512,  # Batch size for model processing
    verbose=False,
      n_ctx = 1500  # Enable detailed logging for debugging
)

def generate_answer_with_llamacpp(query, context_file, max_chunk_size=4096, max_context_length=4096):
    # Load the context from the pickle file
    with open(context_file, 'rb') as file:
        relevant_sources = pickle.load(file)

    # Combine the context from relevant sources
    context = []
    for source in relevant_sources:
        source_text = source.get('text', '')
        source_chunks = [source_text[i:i+max_chunk_size] for i in range(0, len(source_text), max_chunk_size)]
        context.extend(source_chunks)

    # Truncate the context if the total length exceeds max_context_length
    truncated_context = ' '.join(context)[:max_context_length]

    # Concatenate the query and truncated context to form the prompt
    prompt = f"Query: {query}\nContext: {truncated_context}\nAnswer:"

    # Use LlamaCpp to generate the answer directly, ensuring prompt is a list
    answer = llm.generate([prompt], max_tokens=2000, temperature=0.2, top_p=0.9)[0]

    return answer.strip()

# Specify the path to the pickle file containing the relevant sources
context_file = 'Filtered Chunks.pkl'

query = 'Tell me the history of the Arsenal football club by the decade from their inception in 1886. Include the trophies they won each season who the managers were and who are the legends that have played for Arsenal football club. Also include the history at Highbury and at the Emirates stadium'

# Generate the answer using LlamaCpp
answer = generate_answer_with_llamacpp(query, context_file)

# Print the generated answer
print(answer)
