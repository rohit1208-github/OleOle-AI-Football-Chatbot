# -*- coding: utf-8 -*-
"""CRAG_v3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Id64WKVoKPkMV-Bfc9Wd_1T1kcOKbEYu
"""

import os
import requests
import pandas as pd
from time import perf_counter as timer
import tqdm as tqdm
pdf_path = "sodapdf-converted.pdf"

import fitz
from tqdm.auto import tqdm

def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number - 0,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4,
                                "text": text})
    return pages_and_texts

from spacy.lang.en import English
nlp = English()
nlp.add_pipe("sentencizer")

pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
for item in tqdm(pages_and_texts):
    item["sentences"] = list(nlp(item["text"]).sents)
    item["sentences"] = [str(sentence) for sentence in item["sentences"]]
    item["page_sentence_count_spacy"] = len(item["sentences"])

num_sentence_chunk_size = 1
def split_list(input_list: list,
               slice_size: int) -> list[list[str]]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]
for item in tqdm(pages_and_texts):
    item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                         slice_size=num_sentence_chunk_size)
    item["num_chunks"] = len(item["sentence_chunks"])

import re
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentence_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
        chunk_dict["sentence_chunk"] = joined_sentence_chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4
        pages_and_chunks.append(chunk_dict)
len(pages_and_chunks)

import pandas as pd
df = pd.DataFrame(pages_and_chunks)

min_token_length = 15
pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer(model_name_or_path="sentence-transformers/gtr-t5-large",
                                      device="cuda")

# Commented out IPython magic to ensure Python compatibility.
# %%time
# embedding_model.to("cuda")
# for item in tqdm(pages_and_chunks_over_min_token_len):
#     item["embedding"] = embedding_model.encode(item["sentence_chunk"])

text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_token_len]

# Commented out IPython magic to ensure Python compatibility.
# %%time
# text_chunk_embeddings = embedding_model.encode(text_chunks,
#                                                batch_size=32,
#                                                convert_to_tensor=True)
# text_chunk_embeddings

text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
embeddings_df_save_path = "text_chunks_and_embeddings_df.csv"
text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False, escapechar='\\')

text_chunks_and_embedding_df_load = pd.read_csv(embeddings_df_save_path)

import random
import torch
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
text_chunks_and_embedding_df = pd.read_csv("text_chunks_and_embeddings_df.csv")
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
embeddings.shape

from sentence_transformers import util

import textwrap
def print_wrapped(text, wrap_length=15):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=13,  # Update the default value to 15
                                print_time: bool=True):
    query_embedding = model.encode(query,
                                   convert_to_tensor=True)
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)

    return scores, indices

query = 'Tell me the history of the Arsenal football club by the decade from their inception in 1886. Include the trophies they won each season who the managers were and who are the legends that have played for Arsenal football club. Also include the history at Highbury and at the Emirates stadium'

NUM_OF_CHUCKNS = 100

scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings,
                                              n_resources_to_return=NUM_OF_CHUCKNS)
context_items = [pages_and_chunks[i] for i in indices]
for i, item in enumerate(context_items):
    item["score"] = scores[i].cpu().item()
print(f"Number of sources used: {len(context_items)}")

context_with_scores = []
for item in context_items:
        context_with_scores.append(f"[Score: {item['score']:.4f}, Page: {item['page_number']}] {item['sentence_chunk']}")
context = " \n ".join(context_with_scores)
context_with_scores

# SCORE_THRESHOLD = 0.6939

def get_relevant_chunks(query):
    scores, indices = retrieve_relevant_resources(query=query,
                                            embeddings=embeddings,
                                            n_resources_to_return=NUM_OF_CHUCKNS)
    # for i, item in enumerate(context_items):
    #     item["score"] = scores[i].cpu().item()
    context_items = [pages_and_chunks[i] for i in indices]
    # top_contexts = [{"text":e["sentence_chunk"], "page":e['page_number']} for e in context_items]
    top_contexts = []
    for i, item in enumerate(context_items):
        # score = scores[i].cpu().item()
        # if score > SCORE_THRESHOLD:
        top_contexts.append({"text":item["sentence_chunk"], "page":item['page_number']} )
    return top_contexts

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pydantic import BaseModel, Field

# Data model
# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""
#     binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(load_in_8bit=False)
device_map = {"": 0}

gradingModel = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    use_auth_token="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"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def structured_llm_grader(question, document):
    # system = """You are a grader assessing relevance of a retrieved document to a user question.
    # If the document contains a sentence or keyword or semantic meaning related to the question abut Arsenal, grade it as relevant. If not related to the question about Arsenal, grade as non relevant.
    # Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    system = """You are a grader assessing the relevance of a retrieved document to a user question. The questions will be related to football, its players, their journey, or lifestyle. The chunks you are verifying are data about a team, player, or league scraped from the internet.
                If the document contains information that directly answers the question or provides important context related to the question, grade it as relevant by responding with "yes". If the document does not contain any information that would help answer the question or is only tangentially related, grade it as non-relevant by responding with "no"."""

    prompt = f"{system}\n\nRetrieved document:\n{document}\nUser question: {question}\nResponse:"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(gradingModel.device)
    output = gradingModel.generate(input_ids, max_new_tokens=404, num_return_sequences=1, early_stopping=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Response:" in response:
        response = response[(response.find("Response:")+len("Response:")):]
    if "\n" in response:
        response = response[:response.find("\n")]
    if "yes" in response and "no" not in response:
        return True
    elif "no" in response ans "yes" not in response:
        return False
    else:
        return True

def get_filtered_chunks(query):
    relevant_chunks = get_relevant_chunks(query)
    filtered_chunks = [chunk for chunk in tqdm(relevant_chunks) if structured_llm_grader(query, chunk['text'])]
    print(f"Number of filtered chunks are: {len(filtered_chunks)}")
    return filtered_chunks

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # LlamaConfig
)
from langchain.llms import HuggingFacePipeline

def get_llm():
    # Define the model name and retrieve the necessary token for authentication.
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    token = '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'

    # Configure the model for quantization to reduce memory usage.
    bnb_config = BitsAndBytesConfig(load_in_8bit=False)
    device_map = {"": 0}

    # Load the model and tokenizer from Hugging Face with the specified configurations.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        use_auth_token=token#,
        # config= LlamaConfig(
        #     tie_word_embeddings=True,
        #     max_position_embeddings=4096
        # )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

    # Create a pipeline for text generation using the loaded model and tokenizer.
    llama_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=llama_pipeline, model_kwargs={'temperature':0.7})

    return llm

llm = get_llm()

def format_docs(docs):
    return "\n\n".join([doc.text for doc in docs])

query = 'Tell me the history of the Arsenal football club by the decade from their inception in 1886. Include the trophies they won each season who the managers were and who are the legends that have played for Arsenal football club. Also include the history at Highbury and at the Emirates stadium'

from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.schema import StrOutputParser
from langchain import PromptTemplate
from operator import itemgetter

docs = get_filtered_chunks(query)

# Define a template for the prompt to be used with the large language model.
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Given the provided context about Arsenal FC, generate a comprehensive and accurate history of the club, strictly based on the information present in the context. Do not include any information or dates that are not directly supported by the provided context.

Your answer should be in the form of well-structured passages, each focusing on a specific era or aspect of the club's history. Use clear and concise language to provide a factual narrative that captures the key events, personalities, and achievements of each period.

Aim to cover the following aspects in your answer, but only if they are explicitly mentioned in the context:

1. The founding and early years of the club
2. Notable eras under influential managers
3. Significant successes and rebuilding periods
4. Major achievements and resurgences in later decades
5. Impact of key managers on the club's development
6. Transformative moments and legendary players
7. Recent history, challenges, and future aspirations

Structure your answer in a chronological and coherent manner, using smooth transitions between different eras. Use subheadings or bold text to clearly demarcate each passage.

Focus on the most significant and accurate information provided in the context. Include specific dates, years, and facts only if they are explicitly stated in the given context. Avoid making assumptions or including information that is not directly supported.

Emphasize accuracy and stick strictly to the information provided in the context.

Include specific dates, years, and facts only if they are explicitly stated for any time during the specific time period.

Answer: {context}

Question: {question}

Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableMap(
    {"documents": docs, "question": RunnablePassthrough()}
) | {
    "page No.": lambda input: [doc.page for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

response = rag_chain_with_source.invoke(query)