# product_tool.py
import os, json
from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

BASE = os.path.dirname(__file__)
STORE_DIR = os.path.join(BASE, "faiss_store")
CATALOG   = os.path.join(BASE, "catalog.json")

def _build_vectorstore():
    """若本地无索引或损坏，则重新生成"""
    items = json.load(open(CATALOG))
    texts = [f"{it['name']}. {it['desc']}" for it in items]
    metas = [{"id": it["id"]} for it in items]
    v = FAISS.from_texts(
        texts,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        metadatas=metas,
    )
    v.save_local(STORE_DIR)
    return v

# 先尝试 load_local
try:
    _VSTORE = FAISS.load_local(STORE_DIR,
                               OpenAIEmbeddings(model="text-embedding-3-small"))
except Exception as e:
    print("⚠️ 无法加载向量库，正在重新生成 →", e)
    _VSTORE = _build_vectorstore()

# ------- 工具主体 --------
previous_results: List[Document] = []

@tool
def recommend_products(query: str) -> str:
    """用户说“推荐××”时调用，返回 JSON 数组（每项含 id）"""
    docs = _VSTORE.similarity_search(query, k=4)
    previous_results.clear()
    previous_results.extend(docs)
    return json.dumps([d.metadata for d in docs], ensure_ascii=False)
