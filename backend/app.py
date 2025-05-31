from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import random
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from docx import Document

import pinecone

# Step 1: Azure OpenAI Configuration
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://bbazuresc-openai.openai.azure.com"


# Step 3: Load and Split the PDF
def load_and_split_pdf(pdf_folder):
    print("Loading PDF...")
    all_docs = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs = loader.load()
            all_docs.extend(docs)    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# Step 4: Initialize Pinecone and Store Vectors
def create_pinecone_index(index_name, chunks):
    print("Initializing Pinecone...")
    pinecone_api_key = "pcsk_59Fkmz_6C6SzLkugDbFq7XVTvZ5PTk47SKSVfoSeceCZ49m9YUg7hLZVLDZVpepe8ShhEB"
    pinecone_env = "your_pinecone_environment_here"  # Example: "gcp-starter"

    pc = Pinecone(
        api_key=pinecone_api_key)
    


    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(index_name)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint='https://bbazuresc-openai.openai.azure.com',
        azure_deployment='text-embedding-ada-002',
        openai_api_version='2023-05-15',
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )



    text_field = "text"  # the metadata field that contains our text

    # initialize the vector store object
    vector_store = PineconeVectorStore(index, embeddings, text_key="text")
    vector_store.add_documents(chunks)


    print("Vectors stored in Pinecone.")
    return vector_store

# Step 4: Define Custom Prompt Template
def get_custom_prompt_template():
    custom_prompt = """
    You are a helpful and knowledgeable assistant. Respond to the user's questions in **first person** as if you are the one providing advice or solutions. Be clear, concise, and actionable.Use the provided context to answer the user's question.If the user asks for troubleshooting advice or next steps, give practical, step-by-step instructions. If you don't know the answer, say "I don't know. RCA heading gives RCA of the problem, Resolution Steps explains how the issue was fixed"

    Context: {context}
    Question: {question}

    Helpful Answer:
    """
    return PromptTemplate(input_variables=["context", "question"], template=custom_prompt)

# Step 5: Initialize Azure OpenAI QA Chain
def create_qa_chain(vector_store):
    print("Initializing QA chain...")


    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  # Correct endpoint parameter
        deployment_name="GPT3_5",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-09-15-preview" 
    )
    custom_prompt_template = get_custom_prompt_template()

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": custom_prompt_template}
    )
    return qa_chain

# Load the .docx file
def get_all_questions(folder):
    # Read all the paragraphs
    full_text = []
    for filename in os.listdir(folder):
        if filename.endswith(".docx"):
            doc = Document(folder + "/" + filename)
            for para in doc.paragraphs:
                full_text.append(para.text)
    return full_text

def ask_question(qa_chain, query, chat_history=[]):
    print(f"\nQuestion: {query}")
    response = qa_chain({"question": query, "chat_history": chat_history})
    return response['answer']


# Main Function
def main():
    # Configurations
    local_pdf_folder = "Insurance PDFs"
    pinecone_index_name = "chatbot"

    # Step 1: Process PDF
    chunks = load_and_split_pdf(local_pdf_folder)

    # Step 2: Store Vectors in Pinecone
    vector_store = create_pinecone_index(pinecone_index_name, chunks)

    # Step 3: QA Chain Initialization
    qa_chain = create_qa_chain(vector_store)

    chat_history = []
    questions = get_all_questions(local_pdf_folder)

    for question in questions:
        answer = ask_question(qa_chain, question, chat_history)
        chat_history.append((question, answer))
    return chat_history

# Simulated function to replace actual RAG logic
def get_answer_from_rag(query: str,answers) -> str:
    for key in answers:
        question = key[0]
        answer = key[1]
        if query==question:
            return answer
    return "I don't Know"

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
question_answer = main()
@app.post("/query", response_model=QueryResponse)
async def query(data: QueryRequest):
    answer = get_answer_from_rag(data.question,question_answer)
    return {"answer": answer}
