# product_tool.py
import os, pickle, json
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

_PKL = os.path.join(os.path.dirname(__file__), "catalog_faiss.pkl")
_JSON = os.path.join(os.path.dirname(__file__), "catalog.json")

def _build_vectorstore():
    """首次启动 / 文件损坏时自动新建向量库"""
    with open(_JSON, "r") as f:
        items = json.load(f)
    texts = [f"{it['name']}. {it['desc']}" for it in items]
    metas = [{"id": it["id"]} for it in items]
    v = FAISS.from_texts(texts, OpenAIEmbeddings(model="text-embedding-3-small"), metadatas=metas)
    with open(_PKL, "wb") as f:
        pickle.dump(v, f)
    return v

# ① 尝试读取
try:
    _VSTORE = pickle.load(open(_PKL, "rb"))
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    print("⚠️ catalog_faiss.pkl 不存在或损坏，正在重新生成…")
    _VSTORE = _build_vectorstore()

# —— 下方保持你的 @tool 逻辑 —— #
previous_results = []

@tool
def recommend_products(query: str) -> str:
    docs = _VSTORE.similarity_search(query, k=4)
    previous_results.clear()
    previous_results.extend(docs)
    return json.dumps([d.metadata for d in docs], ensure_ascii=False)