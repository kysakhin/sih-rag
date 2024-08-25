import os 
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader


#first we create a function that takes the file input and creates a vector db 
#this function will be later called in interface file 
#depending on the file extension we will use the required loader
#the file is then split into chunks using textsplitter
# a vector db is then made using FAISS
"""def get_vectordb(file:str):
    filename, file_extension = os.path.splitext(file) 
    embeddings = HuggingFaceEmbeddings()

    faiss_index_path = f"faiss_index_{filename}"

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path = file)
    elif file_extension == ".txt":
        loader = TextLoader(file_path = file)
    else:
        print("file extension not supported")
        return None
    
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 0
    )

    docs = text_splitter.split_documents(documents)


    db = FAISS.from_documents(docs,embeddings)
    #db.save(faiss_index_path)
    return db
    """
def get_vectordb(file: str):
    try:
        # Check if the file exists
        if not os.path.exists(file):
            print(f"File {file} does not exist.")
            return None

        filename, file_extension = os.path.splitext(file)
        embeddings = HuggingFaceEmbeddings()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path=file)
        elif file_extension == ".txt":
            loader = TextLoader(file_path=file)
        else:
            print("File extension not supported")
            return None

        try:
            print(f"Loading document: {file}")
            documents = loader.load()
            print("Document loaded successfully")
        except Exception as e:
            print(f"Error loading document: {e}")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=0
        )

        docs = text_splitter.split_documents(documents)

        db = FAISS.from_documents(docs, embeddings)
        # db.save(faiss_index_path)
        return db

    except Exception as e:
        print(f"An error occurred in get_vectordb: {e}")
        return None
####################################################

#now we create the main function that basically runs the llms
#it takes in the key,db and question parameters (also called later in interfac)
#we define the model using huggingface hub 

#a prompt is designed and then the prompt template is used 

def run_llm(key,db,query:str)-> str:
    llm = HuggingFaceHub(
        repo_id = "google/flan-t5-base",
        model_kwargs = {"temperature" : 0.8, "max_length": 512}

    )

    prompt_temp = """
    You are an AI assistant that helps the user by providing relevant information about the document given to you.
    Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1.strictly answer the question based on the given document only, no external questions must be answered
    2. if an external question is asked which is not related to given document reply with "Information not in given document"
    3. if any general knowledge question is asked to you, like the name of an animal or a country reply with "Information not in given document" 
    4. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    5. If you find the answer, write the answer with five sentences maximum.
    6. Make sure that the answer you are giving is related to the document
    7. double check the information provided to you and answer accordingly 
    

    {context}

    Question: {question}

    Helpful Answer:
    
    """

    #now we create a retrival qa to get the info and make sure it returns source document as well
    Prompt = PromptTemplate(template= prompt_temp , input_variables=["context","question"])
    retrival = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={"prompt": Prompt}

    )

    answer =   retrival.invoke({"query":query})
    return answer
