from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
import gradio as gr
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.api_key= os.getenv("PINECONE_API_KEY")

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# 初始化聊天模型和聊天记忆
llm = ChatOpenAI(model_name="gpt-4")
buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# 设置聊天提示模板
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""利用所提供的上下文信息，尽可能真实地回答问题，如果答案不在下面的文字中，请说 '我不知道'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# 初始化聊天链
conversation = ConversationChain(memory=buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# 设置网页配置
st.set_page_config(page_title='关于Sam Altman的聊天机器人', layout='wide')

# 定义边栏导航
with st.sidebar:
    choose = st.radio('选择页面', ['网站介绍', '技术解析'], format_func=lambda x: '祝钰博的网站 - ' + x)

if choose == '网站介绍':
    st.title('欢迎来到祝钰博的AI问答网站')
    st.write('这个网站用于回答和 Sam Altman 相关问题，请按照说明使用，如果有问题请联系我')
elif choose == '技术解析':
    st.title('技术详解')
    st.write('聊天助手基于LLM, Langchain, Pinecone组成的检索增强技术，主要善于回答和 Sam Altman 相关问题')

# 初始化会话状态
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["您好，有什么可以帮您？"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# 聊天模型初始化
llm = ChatOpenAI(model_name="gpt-4")
# 设置聊天记忆
if 'buffer_memory' not in st.session_state:
    st.session_state['buffer_memory'] = ConversationBufferWindowMemory(k=3, return_messages=True)

# 设置聊天提示模板
system_msg_template = SystemMessagePromptTemplate.from_template(template="""利用所提供的上下文信息，尽可能真实地回答问题，
如果答案不在下面的文字中，请说 '我不知道'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state['buffer_memory'], prompt=prompt_template, llm=llm, verbose=True)

# 聊天历史和输入框布局
with st.container():
    with st.form("user_input_form"):
        query = st.text_input("请输入您的问题:", key="input")
        submit_button = st.form_submit_button("发送")
        if submit_button and query:
            with st.spinner("正在生成回复..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(conversation_string, query)
                st.subheader("我猜您是想问:")
                st.write(refined_query)
                context = find_match(refined_query)
                response = conversation.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
                st.session_state.requests.append(query)
                st.session_state.responses.append(response)

# 聊天历史展示
response_container = st.container()
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            # 显示机器人回复
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image(r"C:\Users\zhuyu\OneDrive\桌面\download.jpg", width=150)  # 假设您有一张名为 sam_altman_avatar.jpg 的图像文件
            with col2:
                message(st.session_state['responses'][i], key=str(i))
            # 显示用户回复
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
