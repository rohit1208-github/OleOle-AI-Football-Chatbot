# -*- coding: utf-8 -*-
"""MainRAGMistral 7B.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AluGVpBJk9WKMmTe0a11SGtfgHu1px0I
"""

!pip install -q -U torch datasets transformers tensorflow langchain playwright html2text sentence_transformers faiss-cpu
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 trl==0.4.7

!playwright install
!playwright install-deps

!wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
!dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
!apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
!apt-get update
!apt-get install cuda-11-1
!export CUDA_PATH=/usr/local/cuda-11.1
!nvcc --version

!pip install pypdf

!export BNB_CUDA_VERSION=111
!export LD_LIBRARY_PATH=$JD_LIBRARY_PATH:/usr/local/cuda-11.1

!pip install --quiet bitsandbytes
!python -m bitsandbytes

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain

model_name='mistralai/Mistral-7B-Instruct-v0.2'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

use_4bit = True

bnb_4bit_compute_dtype = "float16"

bnb_4bit_quant_type = "nf4"

use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model))

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=3500,
    top_p = 0.95,
    top_k = 5,
    do_sample=True,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

from langchain.document_loaders import PyPDFLoader

documents = PyPDFLoader("/content/arsenal full.pdf").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,
                                              chunk_overlap=250)
chunked_documents = text_splitter.split_documents(documents)

db = FAISS.from_documents(chunked_documents,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/gtr-t5-large'))

retriever = db.as_retriever(search_kwargs={'k': 13})

prompt_template = """
### [INST] Instruction: Answer the question based on your Arsenal football club history knowledge. Here is context to help:

{context}

### QUESTION:
{question} [/INST]
 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
question = "Tell me all about arsenal decade wise."
result = rag_chain.invoke(question)

result['context']

print(result['text'])