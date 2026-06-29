"""
Project: Docverse
Author: Tanishq 
Version: 1.0
Description: AI-powered PDF chatbot built with Streamlit, LangChain, Chroma, and Groq.
Date: June 2026
"""
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ---------------- Streamlit UI ------------- #
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "Chat 1": [
            {
                "role": "assistant",
                "content": """
 Hello! I'm **DocVerse AI**.

You can ask me questions about the PDF. I can help you:

• Summarize the uploaded PDF

• Explain concepts

• Answer questions

• Extract key information

"""
            }
        ]
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"


# ---------------- SESSION STATE -------------- #

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "Chat 1": []
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"


# ---------------- SIDEBAR ------------- #

with st.sidebar:
    st.image("logo.png", width=220)

    st.title("Chats")

    # Display chat list
    selected_chat = st.radio(
        "Previous Chats",
        options=list(st.session_state.chat_sessions.keys()),
        index=list(st.session_state.chat_sessions.keys()).index(
            st.session_state.current_chat
        )
    )

    st.session_state.current_chat = selected_chat

    st.divider()

    if st.button("➕ New Chat"):

        new_chat_name = (
            f"Chat {len(st.session_state.chat_sessions)+1}"
        )

        st.session_state.chat_sessions[new_chat_name] = []

        st.session_state.current_chat = new_chat_name

        st.rerun()
    st.divider()

    st.markdown("""
    ---
    **Docverse v1.0**

    Developed by **Tanishq**

    📧 tanishqdeep2005@gmail.com
                
    🔗 [GitHub](https://github.com/Tanishqdeep2005)  
                
    🚀 [Repo Link](https://github.com/Tanishqdeep2005/DocVerse/) 
                
    💼 [LinkedIn](https://www.linkedin.com/in/tanishqdeep/)
    """)


st.markdown("""
<h1 style='text-align:center;'>Docverse</h1>
<p style='text-align:center; color:#9CA3AF; font-size:20px;'>
Your AI-powered document assistant
</p>
""", unsafe_allow_html=True)
st.write("Upload a PDF to begin.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)



from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from dotenv import load_dotenv

load_dotenv()

# ---------------- Main Logic ---------------- #

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    st.success("PDF uploaded successfully! Now you can ask about it")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()


    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Create retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=1
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are Docverse, an AI document assistant.

    Answer ONLY from the provided context.

    Rules:
    - Use proper Markdown formatting.
    - Leave a blank line between sections.
    - Use bullet points sometimes.
    - Use headings only when necessary.
    - Keep answers concise and readable.
    - Never return text as one paragraph.


    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """)
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
        )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
        )
    
    # ---------------- CHAT HISTORY ---------------- #

    messages = st.session_state.chat_sessions[
        st.session_state.current_chat
    ]

    # Display previous messages
    for message in messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ---------------- CHAT INPUT ---------------- #

    user_prompt = st.chat_input(
        "Ask something about your PDF..."
    )

    # ---------------- HANDLE USER MESSAGE ---------------- #

    if user_prompt:

        # Save user message
        messages.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = retrieval_chain.invoke(
                    {
                        "input": user_prompt
                    }
                )

                answer = response["answer"]

                st.markdown(answer)

        # Save assistant response
        messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

