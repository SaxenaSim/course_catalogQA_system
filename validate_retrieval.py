from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
#from create_retriever import LangChainPDFProcessor as lg

def Validation():
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory="resources/embeddings/",embedding_function=embedding)
    #obj = lg()
    retrieval = db.as_retriever(search_kwargs={"k":6})
    #db = Chroma(persist_directory="resources/embeddings\\",embedding_function=embedding)
    course_name = "Build Data Pipelines with Delta Live Tables"
    search_query = f"Get me the chunks having course '{course_name}' with particular course's Duration ,Course description, Prerequisites ,Learning objectives and also include the nearby chunks as well."
    #docs = lg.create_retriever.invoke(search_query,include_previous=2)
    docs = retrieval.invoke(search_query,include_previous=2)
    #docs = retrieval.invoke("Get me all the chunks containing information about the course 'Azure Databricks Workspace Administration Fundamentals' and contains all the information about this course like - 1. Duration 2. Course description 3. Prerequisites 4. Learning objectives")
    print(len(docs))
    print(docs)
    
Docs = Validation()

