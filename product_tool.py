# product_tool.py
import pickle, json
from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document

_VS_PATH = "catalog_faiss.pkl"
_VSTORE  = pickle.load(open(_VS_PATH, "rb"))  # ⸺一次加载即可

def _format_result(docs: List[Document], top_k=4) -> str:
    """把向量检索结果转成 LLM 可读 JSON 字符串"""
    metas = [doc.metadata for doc in docs[:top_k]]
    return json.dumps(metas, ensure_ascii=False)

@tool
def recommend_products(query: str) -> str:
    """当用户要求“推荐 + 关键词”时调用。返回 JSON 数组，字段：id。"""
    docs = _VSTORE.similarity_search(query, k=4)
    return _format_result(docs)