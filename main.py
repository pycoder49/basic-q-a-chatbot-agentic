#!/usr/bin/env python
# coding: utf-8

# In[33]:


# imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

import os
from dotenv import load_dotenv
load_dotenv()


# In[24]:


# Data ingestion, through wikipedia
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Deep_learning")
docs = loader.load()
docs


# In[25]:


# text splitting the docs for better embedding and context retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)
split_docs


# In[26]:


# embedding the documents using huggingface embeddings (open source free)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embedding


# In[27]:


# creating chroma db from the split docs
vectorstore = Chroma.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever()
retriever


# In[28]:


# defining a llm
llm = ChatGroq(
    model = "groq/compound-mini",
    api_key = os.getenv("GROQ_API_KEY"),
)
llm


# In[34]:


# creating a history aware retriever
retriever_system_prompt = """
You are a query rewriter for a Deep Learning question-answering system.
Your ONLY job is to take the conversation so far and the user's latest message, and produce a single, clear standalone search query that can be used to look up relevant passages in a Deep Learning reference article.

The assistant downstream will answer the question; you must NOT answer it yourself.
You only rewrite the query.

Use these rules:
1. Incorporate context from the chat history.
2. If the latest user message uses pronouns or vague references (e.g., “it”, “that”, “the first one”), rewrite it so that it explicitly mentions the Deep Learning concept being discussed (e.g., “backpropagation”, “convolutional neural networks”, “transformers”, “SGD”, etc.).
3. Focus on Deep Learning and closely related neural-network topics only.
4. If the user asks about something clearly unrelated to Deep Learning (for example: recipes, travel, sports, general life advice), output exactly the string:
   NO_DEEP_LEARNING_QUERY and nothing else.
5. Do not include chit-chat or extra wording.
6. The output should be a short, precise query that would work well for semantic search over a Deep Learning article.
7. Do NOT include phrases like “User asked:” or “The query is:”. Just output the query text itself.
8. Never mention that you are rewriting a query.
9. The downstream system only wants the final rewritten query.

You will receive:
1. hat_history: the prior conversation between the user and the assistant
2. input: the user's latest message

Using the chat_history and input, write a single standalone Deep Learning search query, or NO_DEEP_LEARNING_QUERY if the request is off-topic.
"""

history_retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", retriever_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# here the llm redefines the user query based on chat history before passing it to the retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, history_retriever_prompt)


# In[35]:


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


# In[30]:


#@@ building the rag_chain
# using the create_stuff_documents_chain and create_retrieval_chain utility functions
# what this does is that it combines a list of documents into a single prompt and feeds it to the llm
# pipeline: retriever -> stuff into prompt -> llm -> output_parseer (so you get the final string output)
document_chain = create_stuff_documents_chain(llm, prompt_template)
document_chain


# In[ ]:


# creating a retreival chain
rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)
rag_chain


# In[ ]:


def chatbot():
    chat_history = []

    print("GenAI chatbot is ready! You can now ask questions about Deep Learning.")
    user_input = input("\nUser:\n")
    while user_input.lower() not in ["exit", "quit"]:
        response = rag_chain.invoke({
            "history": chat_history,
            "input": user_input
        })
        answer = response["answer"]

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ])

        print(f"\nAI:\n{answer}\n")
        user_input = input("User:\n")

if __name__ == "__main__":
    chatbot()


# In[ ]:




