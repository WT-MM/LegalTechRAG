import os
from dotenv import load_dotenv
from pprint import pprint

import pandas as pd

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import google.generativeai as genai

from IPython.display import Markdown

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)
print(api_key)

from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    num_pages = len(pdf_reader.pages)

    page_offset = 7
    text = ""

    for page in range(page_offset, num_pages):
        text += pdf_reader.pages[page].extract_text()

    return text


text = extract_text_from_pdf('../pdfs/DACA-toolkit.pdf')
print(text)

def clean_extracted_text(text):
    cleaned_text = ""

    for i, line in enumerate(text.split('\n')):
        if len(line) > 10 and i > 70:
            cleaned_text += line + '\n'

    cleaned_text = cleaned_text.replace('.', '')
    cleaned_text = cleaned_text.replace('~', '')
    cleaned_text = cleaned_text.replace('©', '')
    cleaned_text = cleaned_text.replace('_', '')
    cleaned_text = cleaned_text.replace(';:;', '')
    return cleaned_text

cleaned_text = clean_extracted_text(text)

from langchain.text_splitter import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents([cleaned_text])
pprint(texts[0].page_content)


documents = []

for chunk in texts:
    documents.append(chunk.page_content)

pprint(documents[0])

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        # for better results, try to provide a title for each input if the corpus is covering a lot of domains
        title = "DACA Toolkit"

        return genai.embed_content(
            model=model,
            content=input,
            task_type="retrieval_document",
            title=title)["embedding"]

import time
from tqdm import tqdm

def create_chroma_db(documents, name):
    chroma_client = chromadb.PersistentClient(path="../database/")

    db = chroma_client.get_or_create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction())

    initiali_size = db.count()
    for i, d in tqdm(enumerate(documents), total=len(documents), desc="Creating Chroma DB"):
        db.add(
            documents=d,
            ids=str(i + initiali_size)
        )
        time.sleep(0.5)
    return db


def get_chroma_db(name):
    chroma_client = chromadb.PersistentClient(path="../database/")
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

db = create_chroma_db(documents, "sme_db")
print(db.count())

