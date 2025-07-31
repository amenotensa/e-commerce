# embed_catalog_lc.py
import json, pickle
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

items = json.load(open("catalog.json"))
texts = [f"{it['name']}. {it['desc']}" for it in items]
metas = [{"id": it["id"]} for it in items]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vstore = FAISS.from_texts(texts, embeddings, metadatas=metas)

with open("catalog_faiss.pkl", "wb") as f:
    pickle.dump(vstore, f)