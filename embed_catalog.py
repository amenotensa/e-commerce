# embed_catalog_lc.py
import json, pickle
from pathlib import Path
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

CATALOG_FILE = "catalog.json"
INDEX_FILE   = "catalog_faiss.pkl"

def main():
    items = json.load(open(CATALOG_FILE, "r"))
    texts = [f"{it['name']}. {it['desc']}" for it in items]
    metadata = [{"id": it["id"]} for it in items]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vstore = FAISS.from_texts(texts, embeddings, metadatas=metadata)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(vstore, f)

if __name__ == "__main__":
    main()