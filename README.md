## 🚀 Live Demo

[Try the Live Application](https://rag-muti-pdf-o33f4f6gwjexvvwm2hsjnt.streamlit.app)

# RAG Multi-PDF

A simple **Multi-PDF Question Answering System** using **Retrieval-Augmented Generation (RAG)**.

### Tech Stack

* Python
* LangChain
* Sentence Transformers
* Pinecone
* Groq LLM
* Streamlit

### How It Works

```text
Multiple PDFs
     ↓
Text Splitting
     ↓
Embeddings
     ↓
Pinecone
     ↓
Retriever
     ↓
LLM
     ↓
Answer
```



### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Create a `.env` file and add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```
