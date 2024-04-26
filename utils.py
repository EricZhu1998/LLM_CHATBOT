from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec, Index
import pinecone
import openai
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
import os

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key= os.getenv("PINECONE_API_KEY")

# 配置嵌入模型名称
model_name = 'text-embedding-ada-002'
# 初始化 OpenAI 嵌入处理
embeddings = OpenAIEmbeddings(
model = model_name)

pc = Pinecone()

index_name = "langchain-chatbot-zyb"

# 查看索引是否存在， 如若不存在则创造一个新索引
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2')
    )

index = pc.Index(index_name)


def find_match(input):
    """
    生成输入文本的嵌入向量并查询索引以找到最匹配的条目。

    Args:
    input (str): 用户输入的查询文本。

    Returns:
    str: 匹配到的前两个条目的文本。
    """

    # 使用 embeddings 创建输入的向量
    input_em = openai.Embedding.create(
        model="text-embedding-ada-002",  # 使用嵌入模型 text-embedding-ada-002
        input=input
    )['data'][0]['embedding']

    # 使用关键字参数来调用 query 方法
    result = index.query(vector=input_em, top_k=2, include_metadata=True)

    # 返回前两个最佳匹配项的元数据中的文本
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']


def query_refiner(conversation, query):
    """
        使用 GPT-4 模型基于对话历史和用户查询优化查询。

        Args:
        conversation (str): 到目前为止的对话历史。
        query (str): 用户的原始查询。

        Returns:
        str: 优化后的查询。
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "您即将使用的是自动优化查询系统，请根据对话历史和用户查询提出最合适的问题。"},
            {"role": "user", "content": f"对话记录: \n{conversation}\n\n用户查询: {query}"}
        ],
        max_tokens=256
    )
    return response['choices'][0]['message']['content']


chat_history = []

def get_conversation_string():
    """
    从会话状态中构建并返回整个对话的字符串表示形式。

    Returns:
    str: 表示整个对话的字符串。
    """

    # 构建会话字符串用于显示
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string