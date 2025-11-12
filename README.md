
# 🤖 KuttiZBot – A Child Rights Chatbot
An Llm-based Conversational Bot For Legal Education &amp; Protection Of Children
KuttiZBot is a locally running conversational AI chatbot designed to help **children understand their rights, laws, and safety information**. It uses Retrieval-Augmented Generation (RAG) to answer queries based on documents provided about laws for children.

Built with **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Streamlit** UI — all powered by **Mistral 7B** running locally via **LlamaCpp**.

---

## 🧠 Project Goals

- Provide a safe, friendly chatbot for kids to ask questions about child laws and rights.
- Run locally (offline-compatible) to ensure data privacy.
- Use RAG (Retrieval-Augmented Generation) to give accurate, document-based answers.

---

## ⚙️ Tools & Technologies

| Tool                | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| [LangChain](https://www.langchain.com/) | Framework to manage chains, LLMs, and document retrieval.       |
| [FAISS](https://github.com/facebookresearch/faiss)         | Vector store for fast similarity search.                        |
| [HuggingFace Sentence Transformers](https://www.sbert.net/) | Text embedding model.                                           |
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) | Lightweight C++ framework to run LLMs like Llama 2 or Mistral locally on CPU/GPU. |
| [Streamlit](https://streamlit.io/)      | Interactive and simple Python web UI.                          |
| Mistral 7B via Ollama                   | Lightweight open-weight instruction-tuned LLM.                 |

---

## 🗂️ Project Structure
```bash
KuttiZBot/
│
├── app.py # Streamlit frontend & chatbot logic
├── vector_store_.py # Vector DB creation script (embedding + indexing)
├── docs/
│ └── laws.txt # Your knowledge base (can be updated anytime)
├── faiss_index/ # Saved FAISS index
├── models/ # Used earlier, now replaced by Ollama
├── bg.jpg # Background image for app
├── requirements.txt # All dependencies
└── README.md # You're reading it!


```

---

## 🚀 Setup Instructions

### 1. 🔧 Install Dependencies

**Make sure you're using Python 3.10+. Create and activate a virtual environment:**


python -m venv venv

source venv/bin/activate  # or venv\Scripts\activate on Windows


**Then install the required packages:**

pip install -r requirements.txt


### 2. Set Up Llama.cpp & Mistral Model
**Download the Mistral-7B-Instruct GGUF model** and place it in the `models/` folder:

1. Visit [Hugging Face Mistral Models](https://huggingface.co/models?search=mistral+gguf) and download a GGUF quantized model (e.g., `mistral-7b-instruct-v0.1.Q4_K_S.gguf`).

2. Save the model in the `models/` directory:


### 3. 🧠 Build the Vector Index
**Place your laws in a .txt file in docs/laws.txt.**

Then run:


python vector_store_.py

This will:


Split your text


Embed the chunks using HuggingFace


Save a FAISS index to faiss_index/


### 4. 🧒 Run the KuttiZBot App

streamlit run app.py

It will open localhost:8501 with a friendly UI where children can ask legal questions.


### 🎨 UI Features
- **Custom background image (bg.jpg)**

- **Friendly greetings and explanations**

- **Styled input and output blocks using Streamlit markdown + HTML**

- **Session history preserved while the app is open**

### ✅ Example Questions

What is child abuse?

What are my rights in school?

Can a child be arrested?

What should I do if someone is hurting me?


### 📈 Future Improvements
Add speech-to-text and text-to-speech for younger children.


Include multi-language support (e.g., Hindi, Malayalam).


Mobile UI optimization.


Upgrade to larger or quantized models for better performance.


Add content filters to keep responses age-appropriate.


### 📚 References
UNICEF Child Rights Resources


National Commission for Protection of Child Rights – India


LangChain Docs


Ollama Docs


Mistral Model


**🤝 Contributing**

**If you'd like to contribute improvements, feel free to fork the repo and open a PR. Let's make legal education accessible for kids! ❤️**


### 📄 License
MIT License – free to use and modify.




Let me know if you’d like:
- A matching **PowerPoint presentation**
- UML or system architecture **diagram**
- Or a **PDF version** of this `README`

Happy coding with KuttiZBot! 🧒💬
