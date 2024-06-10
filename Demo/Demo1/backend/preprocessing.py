import sys
from collections import namedtuple
from typing import Any
import argparse
import yaml
import torch
from tqdm.contrib.concurrent import process_map
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

Document = namedtuple("Document", ["page_content", "metadata"])
if not torch.cuda.is_available():
    torch.set_num_threads(torch.get_num_threads() * 2)
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

def parse_args(config: dict, args: list):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(args)
    return config

def load_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: File config.yaml not found.")
        sys.exit(1)
    except yaml.YAMLError as err:
        print(f"Error reading YAML file: {err}")
        sys.exit(1)

    return data

def rename_duplicates(documents: [Document]):
    document_counts = {}
    for idx, doc in enumerate(documents):
        doc_source = doc.metadata["source"]
        count = document_counts.get(doc_source, 0) + 1
        document_counts[doc_source] = count
        documents[idx].metadata["source"] = (
            doc_source if count == 1 else f"{doc_source}_{count - 1}"
        )

    return documents

def load_document(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    return [Document(doc.page_content, {"source": doc.metadata["source"]}) for doc in loader.load()]

class CustomTextSplitter(RecursiveCharacterTextSplitter):

    def __init__(self, **kwargs: Any) -> None:
        separators = [r"\w(=){3}\n", r"\w(=){2}\n", r"\n\n", r"\n", r"\s"]
        super().__init__(separators=separators, keep_separator=False, **kwargs)

def load_documents(config: dict):
    documents = sum(
        process_map(
            load_document,
            [f"{config['source']}/{file}" for file in config["mediawikis"]],
            desc="Loading Documents",
            max_workers=torch.get_num_threads(),
        ),
        [],
    )
    splitter = CustomTextSplitter(
        add_start_index=True,
        chunk_size=1000,
        is_separator_regex=True,
    )
    documents = sum(
        process_map(
            splitter.split_documents,
            [[doc] for doc in documents],
            chunksize=1,
            desc="Splitting Documents",
            max_workers=torch.get_num_threads(),
        ),
        [],
    )
    documents = rename_duplicates(documents)

    return documents

if __name__ == "__main__":
    config = load_config()
    config = parse_args(config, sys.argv[1:])
    documents = load_documents(config)
    print(f"Embedding {len(documents)} Documents, this may take a while.")
    embeddings = HuggingFaceEmbeddings(
        cache_folder="./model",
        model_name=config["embeddings_model"],
        show_progress=True,
    )
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=config["data_dir"],
    )
    vectordb.persist()