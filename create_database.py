import json
import os
import pandas as pd
from utils import *
import pandas as pd 
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

with open('keys.json', 'r') as file:
    api_keys = json.load(file)

os.environ["GROQ_API_KEY"] = api_keys["GROQ_API_KEY"]
os.environ["JINA_API_KEY"] = api_keys["JINA_API_KEY"]
os.environ["GOOGLE_CSE_ID"] = api_keys["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = api_keys["GOOGLE_API_KEY"]
os.environ["HUGGINGFACE_TOKEN"] = api_keys["HUGGINGFACE_TOKEN"]

embeddings = CustomJinaEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=200,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ]
)


# original code for creating smaller scaled database
# texts = list(pd.read_csv("data/fineweb_edu_2024_10_subset_30k.csv")["text"])

# documents = text_splitter.create_documents(texts = texts)

# persist_directory = 'fineweb_db_new_30k'

# vectordb = Chroma.from_documents(documents=documents,
#                                  embedding=embeddings,
#                                  persist_directory=persist_directory)

# creating larger scaled database, use split docs to overcome chroma database batch limit
texts = list(pd.read_csv("data/fineweb_edu_2024_10_subset_100k.csv")["text"])

documents = text_splitter.create_documents(texts = texts)

persist_directory = 'fineweb_db_new_100k'

def split_list(input_list, chunk_size):
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i + chunk_size]
        
split_docs_chunked = split_list(documents, 41000)

for split_docs_chunk in split_docs_chunked:
    vectordb = Chroma.from_documents(
        documents=split_docs_chunk,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectordb.persist()
