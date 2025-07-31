# agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from product_tool import recommend_products, previous_results
from dotenv import load_dotenv

load_dotenv()

# 提示词：指导模型在推荐后记住商品
SYSTEM_PROMPT = """
你是一个电商助手。
用户说“推荐+关键词”时，你会调用推荐工具，并记住这些商品。

如果用户接下来问“哪个更适合……”，你要从上一次推荐的商品中进行判断，结合商品描述给出推荐理由。
工具返回的是 id 列表，但你可以使用 catalog.json 中的内容来丰富回答。
"""

# LLM 模型
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# 对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 构造 Agent
agent_executor = initialize_agent(
    tools=[recommend_products],
    llm=llm,
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    memory=memory,
    verbose=True,
    system_message=SYSTEM_PROMPT
)

# 封装：生成上次推荐商品的信息作为上下文
def format_previous_context():
    if not previous_results:
        return ""
    return "\n".join(
        f"{doc.metadata['id']}: {doc.page_content}" for doc in previous_results
    )

# 对外暴露一个统一接口（供 app.py 调用）
def ask_agent(user_input: str) -> str:
    context_info = format_previous_context()
    full_input = f"{user_input}\n\n[上次推荐商品]\n{context_info}" if context_info else user_input
    result = agent_executor.invoke({"input": full_input})
    return result["output"]