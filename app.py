# app.py
import streamlit as st
from agent import ask_agent

st.set_page_config(page_title="商品推荐助手", page_icon="🛍️")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史
for role, msg in st.session_state.messages:
    st.chat_message(role).write(msg)

# 等待用户输入
user_input = st.chat_input("请问需要推荐什么商品？")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append(("user", user_input))

    # 获取 AI 回答
    answer = ask_agent(user_input)
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append(("assistant", answer))