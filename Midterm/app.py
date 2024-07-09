import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable.config import RunnableConfig

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ---- GLOBAL DECLARATIONS ---- #

#Read text/tables from the pdf
loader = PyPDFLoader("data/Airbnb_10k.pdf", extract_images=True)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 50
)

documents = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

qdrant_vector_store = Qdrant.from_documents(
    documents,
    embeddings,
    metadata_payload_key="payload",
    location=":memory:",
    collection_name="Airbnb_10k"
)

retriever = qdrant_vector_store.as_retriever()

template = """Answer all questions from the user. Any questions that have accompanying context should be answered with the context. If the context does not pertain to the question, or you cannot find the answer within the context, politely tell the user that you do not know the answer to the question and guide them toward a similar question that you may be able to answer with the context:

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

primary_qa_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
          
@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Airbnb 10-k Bot'.
    """
    rename_dict = {
        "Assistant" : "Airbnb 10-k Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    retrieval_augmented_qa_chain = (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
    )
    cl.user_session.set("retrieval_augmented_qa_chain", retrieval_augmented_qa_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    retrieval_augmented_qa_chain = cl.user_session.get("retrieval_augmented_qa_chain")

    msg = cl.Message(content="")

    async for chunk in retrieval_augmented_qa_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        for key in chunk: #ignore this because it throws an 'Object of type Document is not JSON serializable' error
            if key == 'context':
                continue
            else:
                #This is also to prevent an "Object of type Document is not JSON serializable" error
                data = chunk.get(key)
                await msg.stream_token(data.content)

    await msg.send()