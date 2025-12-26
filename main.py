#!/usr/bin/env python
# coding: utf-8

# In[18]:


# imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import os
from dotenv import load_dotenv
load_dotenv()


# In[10]:


# Data ingestion, through wikipedia
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Deep_learning")
docs = loader.load()
docs


# In[11]:


# text splitting the docs for better embedding and context retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)
split_docs


# In[12]:


# embedding the documents using huggingface embeddings (open source free)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding


# In[13]:


# creating chroma db from the split docs
vectorstore = Chroma.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever()
retriever


# In[14]:


# defining a llm
llm = ChatGroq(
    model = "groq/compound-mini",
    api_key = os.getenv("GROQ_API_KEY"),
)
llm


# In[ ]:


# creating a prompt template for context with chat history
system_prompt = """
    You are an AI assistant that ONLY answers questions about deep learning and closely related neural-network topics (e.g., backpropagation, CNNs, RNNs, transformers, optimization methods, regularization, training tricks, etc.).
    You are connected to an external knowledge base. On every question, you receive a piece of text called CONTEXT. This context comes from a Deep Learning reference article and is the ONLY source of information you are allowed to use.

    Your rules:

    1. Use ONLY the CONTEXT plus the ongoing conversation to answer. Ignore any knowledge from your pre-training that is not supported by the CONTEXT.
    2. If the user asks about something outside deep learning (for example: recipes, travel, personal life advice, unrelated programming questions, general world knowledge, etc.), respond with EXACTLY: “I can only answer questions about deep learning based on the provided context.”
    4. If the CONTEXT does not contain enough information to answer a deep-learning question, say: “I don’t know based on the provided context.”
    5. Never invent facts that are not clearly supported by the CONTEXT.
    6. Default to brief explanations (1 to 3 sentences). Only give longer, more detailed explanations if the user explicitly asks for them (for example, “explain in detail” or “give a longer explanation”).
    7. When you answer, you may refer to the CONTEXT implicitly (e.g., “according to the article” or “the context says…”), but you must never contradict it.

    CONTEXT:
    {context}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
prompt_template


# In[21]:


#@@ building the rag_chain
# using the create_stuff_documents_chain and create_retrieval_chain utility functions
# what this does is that it combines a list of documents into a single prompt and feeds it to the llm
# pipeline: retriever -> stuff into prompt -> llm -> output_parseer (so you get the final string output)
document_chain = create_stuff_documents_chain(llm, prompt_template)
document_chain


# In[ ]:


def chatbot():
    chat_history = []

    print("GenAI chatbot is ready! You can now ask questions about Deep Learning.")
    user_input = input("\nUser:\n")
    while user_input.lower() not in ["exit", "quit"]:
        response = document_chain.invoke({
            "context": retriever.invoke(user_input),
            "history": chat_history,
            "input": user_input
        })

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response)
        ])

        print(f"\nAI:\n{response}\n")
        user_input = input("User:\n")

if __name__ == "__main__":
    chatbot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




