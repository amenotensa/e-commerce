# agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from product_tool import recommend_products

SYSTEM_PROMPT = """
你是一名电商助手。只有当用户输入中包含“推荐”二字时，
才调用 `recommend_products` 工具，其余情况正常对话。
工具返回的是 id 列表，你要用 catalog.json 中的信息生成中文推荐理由。
"""

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent_executor = initialize_agent(
    tools=[recommend_products],
    llm=llm,
    agent=AgentType.OPENAI_MULTI_FUNCTIONS,  # 多工具、函数调用式
    verbose=True,
    memory=memory,
    system_message=SYSTEM_PROMPT
)