import streamlit as st
import faiss
import os
from io import BytesIO
from docx import Document
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint
from secret_api_keys import huggingface_api_key

def process_input(input_type, input_data):
    '''Process different types of input data and returns a vector store.'''
    loader = None
    if input_type == "Link":
        loader = WebBaseLoader(input_data)
        documents = loader.load()
    elif input_type == "PDF":
        if isinstance(input_data, BytesIO):
            pdf_reader = PdfReader(input_data)
        elif isinstance(input_data, UploadedFile):
            pdf_reader = PdfReader(BytesIO(input_data.read()))
        else:
            raise ValueError("Invalid input type for PDF.")
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        documents = text

    elif input_type == "DOCX":
        if isinstance(input_data, BytesIO):
            doc = Document(input_data)
        elif isinstance(input_data, UploadedFile):
            doc = Document(BytesIO(input_data.read()))
        else:
            raise ValueError("Invalid input type for DOCX.")
        text = "\n".join([para.text for para in doc.paragraphs])
        documents = text

    elif input_type == "Text":
        if isinstance(input_data, str):
            documents = input_data
        else:
            raise ValueError("Invalid input type for Text.")
        
    elif input_type == "TXT":
        if isinstance(input_data, BytesIO):
            text = input_data.read().decode('utf-8')
        elif isinstance(input_data, UploadedFile):
            text = str(input_data.read().decode('utf-8'))
        else:
            raise ValueError("Invalid input type for TXT.")
        documents = text

    else:
        raise ValueError("Unsupported input type.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    if input_type == "Link":
        texts = text_splitter.split_documents(documents)
        texts = [str(doc.page_content) for doc in texts]
    else:
        texts = text_splitter.split_text(documents)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    sample_embedding = np.array(hf_embeddings.embed_query("sample text"))
    dimension = sample_embedding.shape[0]
    index = faiss.IndexFlatL2(dimension)

    vector_store = FAISS(
        index=index,
        embedding_function=hf_embeddings.embed_query,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_texts(texts)
    return vector_store


def answer_question(vectorstore, query):
    '''Answer a question using the vector store.'''
    llm = HuggingFaceEndpoint(
        repo_id='tiiuae/falcon-7b-instruct',
        huggingfacehub_api_token=huggingface_api_key,
        temperature=0.6
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    answer = qa.invoke({"query": query})
    return answer


def main():
    st.title("Advanced RAG-QnA Application")
    input_type = st.selectbox("Select Input Type", ["Link", "PDF", "DOCX", "Text", "TXT"])
    if input_type == "Link":
        number_input = st.number_input("Enter Number of Links", min_value=1, max_value=20, step=1)
        input_data = []
        for i in range(number_input):
            url = st.sidebar.text_input(f"Link {i + 1}")
            if url:
                input_data.append(url)
    elif input_type == "PDF":
        input_data = st.file_uploader("Upload PDF files", type=["pdf"])
    elif input_type == "DOCX":  
        input_data = st.file_uploader("Upload DOCX files", type=["docx", "doc"])
    elif input_type == "Text":
        input_data = st.text_area("Enter Text")
    elif input_type == "TXT":
        input_data = st.file_uploader("Upload TXT files", type=["txt"])
    if st.button("Proceed"):
        vectorstore = process_input(input_type, input_data)
        st.session_state["vectorstore"] = vectorstore
    if "vectorstore" in st.session_state:
        query = st.text_input("Ask a question")
        if st.button("Submit"):
            answer = answer_question(st.session_state["vectorstore"], query)
            st.write(answer)


if __name__ == "__main__":
    main()