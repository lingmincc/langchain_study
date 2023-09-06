import gradio as gr
import random
import time

from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import ChatGLM

def createData():

    with open("real_estate_sales_data.txt", encoding="utf-8") as f:
        real_estate_sales = f.read()
    text_splitter = CharacterTextSplitter(
        separator=r'\d+\.',
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
    )
    docs = text_splitter.create_documents([real_estate_sales])
    query = "小区吵不吵"
    db = FAISS.from_documents(docs, OpenAIEmbeddings( openai_api_key="sk-sKkBrVzEQ3uWNFbY6eDfT3BlbkFJRme6ek057pqqmCsP"))
    #db = FAISS.from_documents(docs,
                             # HuggingFaceEmbeddings())
    answer_list = db.similarity_search(query)
    for ans in answer_list:
        print(ans.page_content + "\n")
    db.save_local("real_estates_sale")

    # topK_retriever = db.as_retriever(search_kwargs={"k": 3})
    # docs = topK_retriever.get_relevant_documents(query)
    # for doc in docs:
    #     print(doc.page_content + "\n")
    #
    # docs = topK_retriever.get_relevant_documents("你们有没有1000万的豪宅啊？")
    # for doc in docs:
    #     print(doc.page_content + "\n")
    #
    # retriever = db.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"score_threshold": 0.8}
    # )
    #
    # docs = retriever.get_relevant_documents(query)
    # for doc in docs:
    #     print(doc.page_content + "\n")

def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):

    embeddings =  OpenAIEmbeddings( openai_api_key="sk-sKkBrVzEQ3uWNFbY6eDfT3BlbkFJRme6ek057pqqmCsP")
    #embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local(vector_store_dir, embeddings)

    # llm = ChatGLM(
    #     endpoint_url="http://aibox.bigdata-hub.cn:8000/ai",
    #     max_token=80000,
    #     history=[],
    #     top_p=0.7,
    #     temperature=0,
    #     model_kwargs={"sample_model_args": False}
    # )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="sk-sKkBrVzEQ3uWNFbY6eDfT3BlbkFJRme6ek057pqqmCsP")


    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.1}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产|电器｜教育｜家装销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600,label="聊天对话框"),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
    #createData()