import json
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import JsonOutputParser


class LangChainPDFProcessor:
    def __init__(self):
        self.pdf_file = "data/course-catalog1.pdf"
        self.data = None
        self.chunks = None
        self.vectordb = None
        self.retrieval = None

    def load_pdf(self):
        loader = PyPDFLoader(self.pdf_file)
        self.data = loader.load()

    def split_text(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, add_start_index=True)
        self.chunks = text_splitter.split_documents(self.data)

    def create_embeddings(self):
        persist_dir = "resources/embeddings"
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(documents=self.chunks, embedding=embeddings, persist_directory=persist_dir)

    def create_retriever(self):
        self.retrieval = self.vectordb.as_retriever(search_kwargs={"k": 6})

    def process(self):
        self.load_pdf()
        self.split_text()
        self.create_embeddings()
        self.create_retriever()


def main():
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    pdf_processor = LangChainPDFProcessor()
    pdf_processor.process()


if __name__ == "__main__":
    main()






