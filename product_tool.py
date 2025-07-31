# product_tool.py
import pickle, json
from typing import List
from langchain_core.tools import tool
from langchain_core.documents import Document

# 加载向量库
_VSTORE = pickle.load(open("catalog_faiss.pkl", "rb"))

# 保存上次推荐结果
previous_results = []

def _format_result(docs: List[Document], top_k=4) -> str:
    """将推荐商品的 metadata 变成 JSON 字符串"""
    metas = [doc.metadata for doc in docs[:top_k]]
    return json.dumps(metas, ensure_ascii=False)

@tool
def recommend_products(query: str) -> str:
    """用户说‘推荐xxx’时调用，返回商品ID列表"""
    docs = _VSTORE.similarity_search(query, k=4)
    previous_results.clear()
    previous_results.extend(docs)
    return _format_result(docs)