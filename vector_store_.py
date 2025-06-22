


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Set your embedding model explicitly
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Step 2: Load and split documents
# loader = TextLoader("docs/knowledge.txt", encoding="utf-8")
loader = TextLoader("docs/laws.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# Step 3: Create vectorstore and save it
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")



