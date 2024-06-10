import pickle
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vector_store(data_file):
    '''Create a vector store from a pickle file'''
    # load data from pickle file
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # interpret information in the data
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    for item in data:
        text = item.get('text', '')  # Assuming the text content is in a 'text' key
        texts.extend(splitter.split_text(text))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'mps'})

    # create the vector store database
    db = FAISS.from_texts(texts, embeddings)
    return db

def load_llm():
    # Adjust GPU usage based on your hardware
    llm = LlamaCpp(
        model_path="llama-2-7b-chat.GGUF.q4_0.bin",  # Path to the model file
        n_gpu_layers=40,  # Number of GPU layers (adjust based on available GPUs)
        n_batch=512,  # Batch size for model processing
        verbose=False,
        max_tokens=4096,
        n_ctx = 5000  # Increase the value to allow longer outputs
    )
    return llm

def create_prompt_template():
    # prepare the template we will use when prompting the AI
    template = """Use the provided context to answer the user's question. If you don't know the answer, respond with "I do not know".
Context: {context}
Question: {question}
Answer: """
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question']
    )
    return prompt

def create_chain():
    db = create_vector_store('/Users/rohit1208/Desktop/oleole/FinDem5/data/Filtered Chunks.pkl')
    llm = load_llm()
    prompt = create_prompt_template()
    retriever = db.as_retriever(search_kwargs={'k': 63})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=False, chain_type_kwargs={'prompt': prompt})
    return chain

def query_doc(chain, question):
    return chain({'query': question})['result']

def main():
    chain = create_chain()
    print("Chatbot for PDF files initialized, ready to query...")
    while True:
        question = input("> ")
        answer = query_doc(chain, question)
        print(': ', answer, '\n')

main()