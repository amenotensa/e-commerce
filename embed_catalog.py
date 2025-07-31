# embed_catalog.py  —— 生成 catalog_faiss.pkl
import json, pickle, os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()                           # 读取 .env 中的 OPENAI_API_KEY
CATALOG = "catalog.json"
OUTPUT  = "catalog_faiss.pkl"

def main():
    with open(CATALOG, "r") as f:
        items = json.load(f)

    texts = [f"{it['name']}. {it['desc']}" for it in items]
    metadatas = [{"id": it["id"]} for it in items]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    with open(OUTPUT, "wb") as f:
        pickle.dump(vstore, f)
    print(f"✅ 向量库已生成 → {OUTPUT} ({os.path.getsize(OUTPUT)} bytes)")

if __name__ == "__main__":
    main()