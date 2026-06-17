"""
RAG Chatbot - PDF par question-answer karnara chatbot
--------------------------------------------------------
Hi project khalil flow follow karta:
1. User PDF upload karto
2. PDF chunks madhe todla jato
3. Pratyek chunk cha embedding (vector) banवला jato
4. Embeddings ChromaDB (vector database) madhe save hotat
5. User question type karto -> tyacha hi embedding banते
6. Sarvat relevant chunks retrieve hotat
7. Te chunks + question LLM la pass kele jatat -> answer milte

Run karaycha command (terminal madhe):
    streamlit run app.py

Required environment variable (Groq वापरत असाल तर):
    GROQ_API_KEY=your_key_here
"""

import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF वर प्रश्न विचारा (RAG Chatbot)")
st.caption("PDF upload kara, mag tyat lihilelya goshtinwar prashna vicharaa. (Groq cha free, fast LLM vaparla aahe)")

# ---------------------------------------------------------
# SIDEBAR — API KEY INPUT
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API key", type="password", help="console.groq.com var jaun free key banva (credit card lagat nahi)")
    st.markdown("---")
    st.markdown(
        "**Tip:** embeddings free, local model वापरतो (Hugging Face). "
        "Groq cha LLM pan free aahe — credit card lagat nahi."
    )

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# ---------------------------------------------------------
# SESSION STATE — store vectorstore across reruns
# ---------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# STEP 1 + 2 + 3: UPLOAD -> LOAD -> CHUNK -> EMBED -> STORE
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Tumcha PDF ithe upload kara", type=["pdf"])

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("PDF process karat ahe... (pahilyanda thoda time lagel)"):
        # Temp file la save kara, kaaran PyPDFLoader la path pahije
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # PDF load kara
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Chunks madhe tod -- 1000 characters per chunk, 200 overlap
        # (overlap thevla karan context tutaycha nahi paragraph cha)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        # Free, local embedding model (no API key needed for this step)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ChromaDB madhe store kara (in-memory, sopa, local)
        vectorstore = Chroma.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore
        st.success(f"PDF process zala! ({len(chunks)} chunks banle)")

# ---------------------------------------------------------
# STEP 4 + 5: RETRIEVAL + LLM ANSWER
# ---------------------------------------------------------
if st.session_state.vectorstore is not None:
    if not api_key:
        st.warning("Answer generate karaylaa sidebar madhe Groq API key takaa (free aahe, console.groq.com var ja).")
    else:
        question = st.chat_input("Tumcha prashna ithe lihaa...")

        # Custom prompt -- LLM la sangतो ki document var based answer de
        prompt_template = """Khalil context vaparun question cha answer dya.
Jar answer context madhe nasel tar "Mala ya document madhe he uttar sapadla nahi" asa sang.
Context madhech rahaa, swatachya knowledge cha use karu naka.

Context: {context}

Question: {question}

Answer:"""
        custom_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        # Groq cha free, fast LLM -- Llama 3.3 70B model vaparla
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})

        # Display previous chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.spinner("Vichaar karat ahe..."):
                # STEP 4: Retrieval -- sarvat relevant chunks shodhayche
                retrieved_docs = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # STEP 5: Generation -- context + question LLM la dyaycha
                final_prompt = custom_prompt.format(context=context_text, question=question)
                response = llm.invoke(final_prompt)
                answer = response.content

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.write(answer)

                # Source chunks dakhav -- transparency sathi chaan asta
                with st.expander("📖 Konte sources vaparle"):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**Source {i}** (page {doc.metadata.get('page', '?')})")
                        st.text(doc.page_content[:300] + "...")

else:
    st.info("Suruwat karaycha sathi var PDF upload kara.")
