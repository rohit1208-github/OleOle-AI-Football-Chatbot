import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import html
from html import escape
from preprocessing import load_config, parse_args
from utils import Query
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.cache import SQLiteCache
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.globals import set_llm_cache
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

app = Flask(__name__)

def setup_memory():
    Path("memory").mkdir(parents=True, exist_ok=True)
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )
    set_llm_cache(SQLiteCache(database_path="memory/cache.db"))

    return memory

def import_db(config: dict):
    embeddings = HuggingFaceEmbeddings(
        cache_folder="./model",
        model_name=config["embeddings_model"],
    )
    vectordb = Chroma(
        persist_directory=config["data_dir"], embedding_function=embeddings
    )

    return vectordb

def create_chain(config: dict):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    memory = setup_memory()
    vectordb = import_db(config)
    model = ChatOllama(
        cache=True,
        callback_manager=callback_manager,
        model=config["model"],
        repeat_penalty=config["settings"]["repeat_penalty"],
        temperature=config["settings"]["temperature"],
        top_k=config["settings"]["top_k"],
        top_p=config["settings"]["top_p"],
    )
    chain = ConversationalRetrievalChain.from_llm(
        chain_type="stuff",
        llm=model,
        memory=memory,
        retriever=vectordb.as_retriever(
            search_kwargs={"k": int(config["settings"]["num_sources"])}
        ),
        return_source_documents=True,
    )

    return chain

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    prompt = request.form['prompt']
    config = load_config()
    print(config["model"])
    settings = ["num_sources", "temperature", "repeat_penalty", "top_k", "top_p"]
    for setting in settings:
        if setting in ["temperature", "repeat_penalty", "top_p"]:
            config["settings"][setting] = float(request.form.get(setting, config["settings"][setting]))
        else:
            config["settings"][setting] = int(request.form.get(setting, config["settings"][setting]))
    chain = create_chain(config)
    res = chain(prompt)
    answer = res["answer"]
    source_documents = res["source_documents"]
    sources = [doc.metadata["source"] for doc in source_documents]
    if sources:
        answer += f"\nSources: {', '.join(sources)}"
    else:
        answer += "\nNo sources found"

    # Split the answer into lines and apply HTML formatting
    formatted_answer = "<br>".join(html.escape(line) for line in answer.split("\n"))

    return jsonify(answer=formatted_answer)

@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

if __name__ == '__main__':
    app.run(debug=True)