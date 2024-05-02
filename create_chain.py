from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os, json
from validate_retrieval import Docs
from langchain_core.output_parsers import JsonOutputParser 


def create_chain():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    embedding = OpenAIEmbeddings()
    db = Chroma(persist_directory="resources/embeddings/",embedding_function=embedding)

    template = """Your task is to provide details about a specific course mentioned in the question.
    Don't try to make up any answer , if don't no the answer specify that you don't know the answer. 
    Extract the following information from the provided context:
    - Course Name: [Course Name]
    - Duration: [Duration of the course in hours that is mentioned in the context]
    - {question}: [Answer to the question including all the points mentioned in the context]

    Your output should be a JSON object with the following keys and their respective values:
    1. Course Name
    2. Duration
    3. {question}

    Context:
    {context}
    """
    #{format_instructions}
    parser = JsonOutputParser()
    prompt = PromptTemplate(template=template,input_variables=["question","context"])
    

    qa = RetrievalQA.from_chain_type(
        llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    question = "What are the prerequisites for the course Build Data Pipelines with Delta Live Tables?"
    result = qa({"query":question,"context":Docs})
    # print(type(result))
    # print(result)
    print(result['result'])
    # print(type(result['result']))
    
    # response_schemas = [
    # ResponseSchema(name="Course Name", description="name of the course"),
    # ResponseSchema(name="Duration", description="duration of the course"),
    # ResponseSchema(name="answer", description="answerto the user query"),
#]
    #output_parser = StructuredOutputParser(response_schemas)
    #format_instructions = output_parser.get_format_instructions()
    
    # parser = JsonOutputParser()
    
    # # Parse the output using the JsonOutputParser
    # json_result = parser.parse(result['result'])
    
    # # Convert the parsed result to a JSON string for printing
    # json_output = json.dumps(result)
    # print(json_output)
    # print(type(json_output))
    result_json = json.dumps(result['result'])
    print(type(result_json))

    # Save result to a JSON file
    # with open('result.json', 'w') as f:
    #     f.write(result_json)
create_chain()
