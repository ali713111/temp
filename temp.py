import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import getpass
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import sentence_transformers
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from datetime import datetime
import gdown
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore  # ✅ Correct
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import tiktoken

# Load environment variables
load_dotenv()
###############################################################setting openai ai api##################################################################
file_id = "1ug8pf1M1tes-CJMhS_sso372tvC4RQv8"
output_file = "open_ai_key.txt"

# https://docs.google.com/spreadsheets/d/1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g/edit?gid=0#gid=0
sheet_id = '1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g' # replace with your sheet's ID
url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df=pd.read_csv(url)
# st.write(df)

# def download_db():
#     url = f"https://drive.google.com/uc?id={file_id}"
#     gdown.download(url, output_file, quiet=False)
#     return output_file
# k=""
# with open(download_db(),'r') as f:
#     f=f.read()
#     # st.write(f)
#     k=f
os.environ["OPENAI_API_KEY"] = df.keys()[0]
#####################################################################################################################################################
# # Load all PDFs in a directory
# pdf_folder = "database"
# loader = DirectoryLoader(pdf_folder, glob="*.pdf", loader_cls=PyPDFLoader)

# # Load documents
# documents = loader.load()

# st.write(f"Loaded {len(documents)} documents from the directory")

# text_splitter = TokenTextSplitter(encoding_name='o200k_base', chunk_size=100, chunk_overlap=20)
# texts = text_splitter.split_documents(documents)
# st.write(texts)
# Assuming you have OpenAI API key set up in your environment
embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()

########################################################################### Loading the vector db ###########################################################
# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load metadata
with open("faiss_metadata.pkl", "rb") as f:
    docstore_data = pickle.load(f)

# ✅ Fix: Wrap the docstore dictionary inside InMemoryDocstore
docstore = InMemoryDocstore(docstore_data)

# Load index-to-docstore mapping
with open("faiss_index_to_docstore.pkl", "rb") as f:
    index_to_docstore_id = pickle.load(f)

# ✅ Fix: Ensure FAISS is initialized with proper embeddings
vector_store = FAISS(
    index=index,
    embedding_function=embeddings,  # ✅ Ensure embeddings are passed correctly
    docstore=docstore,  # ✅ Wrap docstore properly
    index_to_docstore_id=index_to_docstore_id
)
# Set up retriever
retriever = vector_store.as_retriever()

##########################################################################setting groq api ###############################################################

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm_llama3 = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

##############################################################setting opne ai llm ##################################################################

llm_openai = ChatOpenAI(model="gpt-4o-mini")
###########################################################setting RAG document formatting ##############################################################


prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
##############################################################setting prompts ###########################################################################################

def get_feedback(feedback):
    return f"""
    ### What improvements would you like to see in the chatbot?  
    - {feedback["What improvements would you like to see in the chatbot?"]}

    ### What challenges did you face while using the chatbot for academic support, and how do you think these could be addressed?  
    - {feedback["What challenges did you face while using the chatbot for academic support, and how do you think these could be addressed?"]}

    ### Did you face any issues while using the chatbot? If yes, please explain.  
    - {feedback["Did you face any issues while using the chatbot? If yes, please explain."]}

    ### Summary of Responses:

    - **Improvements:** Improvements Here
    - **Challenges:** Challenges Here
    - **Issues:** Issues Here
    """

# Define a custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant answering user queries based on the provided context.
    
    Context:
    {context}
    
    Question:
    {question}

    You can add additional information from your knowledgebase as well.
    Provide a clear and concise response.
    """
    )
############################################################################################################################################################


st.title("Cyber Warrior AI")
selections=st.sidebar.selectbox("☰ Menu", ["Home","AI Assistant", "Feedback"])


query=""
# tokens={}
if selections=="Home":
    st.markdown("""The School Student Assistant Chatbot is an AI-powered virtual assistant designed to help students with their academic and school-related queries. It provides instant responses to common questions, assists with homework, shares important school updates, and offers guidance on schedules, subjects, and extracurricular activities.  
     Key Features:  
    ✅ Homework Assistance – Provides explanations and study resources.  
    ✅ Timetable & Schedule Support – Helps students check class schedules.  
    ✅ School Announcements & Notices – Delivers updates on events and policies.  
    ✅ Subject Guidance – Answers subject-related queries.  
    ✅ Interactive – Allows students to communicate via text.  """)
    
    
if selections=="AI Assistant":
    query=st.text_input("Write Query Here")
    if st.button("Submit") and query!="":
        rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm_openai
        )
        st.subheader("OpenAI GPT Response")
        res=rag_chain.invoke(query)
        st.write(res.content)
        # st.write(res.response_metadata['token_usage']['total_tokens'])
        # tokens["open_ai"]=res.response_metadata['token_usage']['total_tokens']

    
        # # performing a similarity search to fetch the most relevant context
        st.write("")
        st.write("")
        st.write("")
    
        






    
