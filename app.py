from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st


import base64


def generate_answer(query):
    # Load embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    # Use Ollama to load Mistral model
    llm = Ollama(model="mistral")

    # Set up RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(f"Answer this question in a friendly and simple way for children: {query}")




def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


img_base64 = get_base64_image("bg.jpg")

# llm = Llama(
#     model_path="models/mistral-7b-instruct-v0.1.Q4_K_S.gguf",
#     n_ctx=2048,
#     n_threads=4,  # Adjust based on CPU
#     verbose=False
# )



# def generate_answer(query):
#     # embeddings = HuggingFaceEmbeddings()
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     # vectorstore = FAISS.load_local("faiss_index", embeddings)
#     vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     retriever = vectorstore.as_retriever()

#     llm = LlamaCpp(
#         model_path="models/mistral-7b-instruct-v0.1.Q4_K_S.gguf",  # update path if needed
#         n_ctx=2048,
#         temperature=0.7,
#         verbose=True
#     )

#     qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#     return qa_chain.run(f"Answer this question in a friendly and simple way for children: {query}")
#     # qa_chain.run(query)



# import streamlit as st
 # Assuming your function is here

# Set page config
st.set_page_config(page_title="KuttiZBot - Child Law Bot", page_icon="🧒", layout="centered")
# st.markdown("👋 Hi! I’m **KuttiZBot**, your child rights buddy. You can ask me anything about your rights, safety, or help.")

# Background image with CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .chat-box {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }}
     label[for="user_input"] {{
        color: white;  /* Deep Purple */
        font-weight: bold;
        font-size: 18px;
    }}
    .input-box {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Section
st.markdown("<h1 style='text-align: center; color: #ffffff;'>🤖 KuttiZBot</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #f9f9f9;'>"👋 Hi! I’m **KuttiZBot**, your child rights buddy. You can ask me anything about your rights, safety, or help. 👦👧"</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #f9f9f9;'>👋 Hi! I’m <strong>KuttiZBot</strong>, your child rights buddy. You can ask me anything about your rights, safety, or help. 👦👧</h3>", unsafe_allow_html=True)

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
with st.container():
    query = st.text_input("Ask me anything:", key="user_input",  help="e.g. What are my rights at school?", placeholder="Type your question here...")

if query:
    with st.spinner("Thinking..."):
        response = generate_answer(query)
        st.session_state.history.append((query, response))

# Display conversation
for q, a in reversed(st.session_state.history):
    st.markdown(f"<div class='chat-box'><strong>You:</strong> {q}<br><strong>KuttiZBot:</strong> {a}</div>", unsafe_allow_html=True)

# st.title("🧠 Local RAG Chatbot")

# if "history" not in st.session_state:
#     st.session_state.history = []

# query = st.text_input("Ask me anything:")

# if query:
#     with st.spinner("Thinking..."):
#         response = generate_answer(query)
#         st.session_state.history.append((query, response))

# for q, a in reversed(st.session_state.history):
#     st.markdown(f"**You**: {q}")
#     st.markdown(f"**Bot**: {a}")