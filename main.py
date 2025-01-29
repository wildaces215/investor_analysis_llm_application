from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from langchain_chroma import Chroma

loader = PyPDFLoader("investors_report.pdf")

data = loader.load()

chunk_size = 24
chunk_overlap = 3

#Character split since it is only one document
splitter = CharacterTextSplitter(
    separator='.',
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

docs = splitter.split_documents(data)

#llm of choice
llm=OllamaLLM(model="llama.3.2")

embeddings_func = OllamaEmbeddings(
    model="llama3.2"
)

vectorstore = Chroma.from_documents(
    docs,
    embedding=embeddings_func,
    persist_directory='test/'
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":2}
)
message = """Lets pretend your an investor and your looking at the earnings report of this company.
What are the strengths and weaknesses of the company in this quarterly investors report?
"""
prompt_template = ChatPromptTemplate.from_messages([("human",message)])

rag_chain = ({"guidelines":retriever,"copy":RunnablePassthrough()}
              | prompt_template
              | llm
              )

resp=rag_chain.invoke("Please look at the document and see if it is a company worth investing in")
print(resp)




