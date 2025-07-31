# embed_catalog.py
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

CATALOG = "catalog.json"
OUT_DIR = "faiss_store"           # ← 用文件夹而不是 .pkl

def main():
    items = json.load(open(CATALOG))
    texts = [f"{it['name']}. {it['desc']}" for it in items]
    metas = [{"id": it["id"]} for it in items]

    vstore = FAISS.from_texts(
        texts,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        metadatas=metas,
    )

    vstore.save_local(OUT_DIR)    # ← 关键改动
    print("✅ 向量库已保存到", OUT_DIR)

if __name__ == "__main__":
    main()
