from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

vectorstore = FAISS.from_texts(
["Nitish worked at exa.ai"], embedding = OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
{"context":retriever, "question": RunnablePassthrough()}
| prompt | model | StrOutputParser()
)

print(chain.invoke("Where Nitish worked?"))
