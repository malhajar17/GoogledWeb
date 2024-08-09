import json
import os
import pandas as pd
from utils import *
import pandas as pd 
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

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


texts = list(pd.read_csv("data/fineweb_edu_2024_10_subset.csv")["text"])

documents = text_splitter.create_documents(texts = texts)

persist_directory = 'fineweb_db_new'

vectordb = Chroma.from_documents(documents=documents,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)