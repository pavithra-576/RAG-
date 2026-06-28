# 🔍 RAG — Retrieval-Augmented Generation

> Grounding LLMs with real-world knowledge through dynamic retrieval.

---

## 📖 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture pattern that enhances Large Language Models (LLMs) by providing them with relevant, up-to-date information retrieved from an external knowledge base — before generating a response.

Instead of relying solely on what an LLM learned during training, RAG dynamically fetches the right context at query time and passes it into the prompt.

---

## ❓ Why RAG?

| Problem with Plain LLMs | How RAG Solves It |
|---|---|
| Knowledge cutoff (stale data) | Retrieves current documents at query time |
| Hallucinations (made-up facts) | Grounds answers in retrieved source text |
| Can't access private/internal data | Connects to your own knowledge base |
| No citations or traceability | Retrieved chunks serve as verifiable sources |
| Context window limits for large corpora | Only fetches the most relevant chunks |

---

## 🏗️ RAG Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Query Embedding   │  ← Convert query to vector
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   Vector Store      │  ← Similarity search over document embeddings
│  (FAISS / Chroma /  │
│   Pinecone etc.)    │
└─────────────────────┘
    │
    ▼ Top-K relevant chunks
┌─────────────────────┐
│   Context Assembly  │  ← Combine retrieved chunks + original query
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│       LLM           │  ← Generate answer grounded in context
└─────────────────────┘
    │
    ▼
Final Answer (with sources)
```

---

## 🔄 RAG Pipeline — Step by Step

### Phase 1: Indexing (Offline)

```
Raw Documents
    │
    ▼ Document Loaders (PDF, URL, DB, etc.)
    │
    ▼ Text Splitting (chunking)
    │
    ▼ Embedding Model (text → vectors)
    │
    ▼ Vector Store (store + index vectors)
```

### Phase 2: Retrieval + Generation (Online)

```
User Query
    │
    ▼ Embed query
    │
    ▼ Similarity search in vector store
    │
    ▼ Retrieve Top-K chunks
    │
    ▼ Inject into prompt
    │
    ▼ LLM generates grounded response
```

---

## 💻 Implementation with LangChain

### Step 1 — Load Documents

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# From PDF
loader = PyPDFLoader("knowledge_base.pdf")
docs = loader.load()

# From a website
loader = WebBaseLoader("https://docs.example.com")
docs = loader.load()
```

### Step 2 — Split into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")
```

### Step 3 — Embed & Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Persist to disk
vectorstore.save_local("faiss_index")

# Reload later
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

### Step 4 — Build Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",   # or "mmr" for diversity
    search_kwargs={"k": 5}      # retrieve top 5 chunks
)

# Test retrieval
results = retriever.invoke("What is the refund policy?")
for doc in results:
    print(doc.page_content[:200])
```

### Step 5 — RAG Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the return window for purchases?")
print(answer)
```

---

## 🧪 RAG with Sources (Citations)

```python
from langchain.chains import RetrievalQAWithSourcesChain

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever
)

result = chain.invoke({"question": "What are the payment options?"})
print("Answer:", result["answer"])
print("Sources:", result["sources"])
```

---

## 🗃️ Vector Store Options

| Store | Best For | Notes |
|---|---|---|
| **FAISS** | Local / prototyping | Fast, in-memory, no server needed |
| **Chroma** | Local / small scale | Persistent, easy setup |
| **Pinecone** | Production / cloud | Fully managed, scalable |
| **Weaviate** | Hybrid search | Supports semantic + keyword |
| **Qdrant** | High performance | Open-source + cloud |
| **pgvector** | PostgreSQL users | Embed vector search in existing DB |
| **Milvus** | Large-scale | Enterprise-grade vector DB |
| **Redis** | Low-latency apps | Vector search with caching |

---

## 🧠 Embedding Models

| Model | Provider | Dimension | Notes |
|---|---|---|---|
| `text-embedding-3-small` | OpenAI | 1536 | Fast, cheap, great quality |
| `text-embedding-3-large` | OpenAI | 3072 | Best OpenAI quality |
| `embed-english-v3.0` | Cohere | 1024 | Strong for English |
| `all-MiniLM-L6-v2` | HuggingFace | 384 | Free, local, lightweight |
| `bge-large-en-v1.5` | HuggingFace | 1024 | Best open-source quality |
| `nomic-embed-text` | Nomic / Ollama | 768 | Free, runs locally |

---

## 🔧 Advanced RAG Techniques

### 🔀 Hybrid Search (Semantic + Keyword)
Combine vector similarity with BM25 keyword search for better recall:

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5

ensemble = EnsembleRetriever(
    retrievers=[bm25, vectorstore.as_retriever(search_kwargs={"k": 5})],
    weights=[0.4, 0.6]
)
```

### 🔁 Multi-Query Retrieval
Generate multiple query variants to improve recall:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

### 🧹 Contextual Compression
Filter retrieved chunks to only the relevant parts:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 🔗 Parent Document Retriever
Store small chunks for retrieval but return larger parent chunks for context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

store = InMemoryStore()
parent_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000),
)
parent_retriever.add_documents(docs)
```

---

## 📊 RAG Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Context Precision** | Are retrieved chunks relevant to the query? |
| **Context Recall** | Did retrieval capture all relevant information? |
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer actually address the question? |

### Evaluate with RAGAS

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(results)
```

---

## 🔁 RAG vs Fine-Tuning

| | RAG | Fine-Tuning |
|---|---|---|
| **Knowledge update** | Real-time | Requires retraining |
| **Private data** | ✅ Easy | Requires careful handling |
| **Cost** | Low (retrieval costs) | High (GPU compute) |
| **Hallucination control** | Strong | Moderate |
| **Custom behavior/style** | Limited | Strong |
| **Best for** | Dynamic, factual Q&A | Style, tone, domain tasks |

---

## 🏆 Best Practices

- ✅ Use **chunk overlap** (50–100 tokens) to avoid losing context at boundaries
- ✅ Choose **chunk size** based on your embedding model's token limit (typically 256–512 tokens)
- ✅ Store **metadata** (source, page number, date) alongside chunks for citations
- ✅ Use **hybrid search** for better precision + recall
- ✅ Apply **re-ranking** (e.g., Cohere Rerank) after retrieval for better top-K quality
- ✅ Always **evaluate** your RAG pipeline with RAGAS or LangSmith
- ✅ Add a **fallback** ("I don't know") when retrieved context is insufficient
- ❌ Don't use chunks that are too large — they reduce retrieval precision
- ❌ Don't skip metadata — you'll lose traceability for citations

---

## 📦 Full Stack RAG Example

```
┌─────────────────────────────────────────────┐
│              RAG Application Stack           │
├──────────────┬──────────────────────────────┤
│ Frontend     │ React / Streamlit / Gradio    │
│ Backend      │ FastAPI + LangServe           │
│ Orchestration│ LangChain / LangGraph         │
│ LLM          │ GPT-4o / Claude / Llama3      │
│ Embeddings   │ OpenAI / BGE / Nomic          │
│ Vector Store │ Pinecone / Chroma / pgvector  │
│ Document DB  │ S3 / GCS / Local FS           │
│ Evaluation   │ RAGAS / LangSmith             │
└──────────────┴──────────────────────────────┘
```

---

## 📄 License

This document is freely available for educational and commercial use.

---

## 📚 Resources

- 📘 [LangChain RAG Docs](https://docs.langchain.com/docs/use-cases/question_answering)
- 📊 [RAGAS Evaluation Framework](https://docs.ragas.io)
- 🧪 [LangSmith (Tracing)](https://smith.langchain.com)
- 📖 [Original RAG Paper — Lewis et al. 2020](https://arxiv.org/abs/2005.11401)
- 🎓 [RAG Survey Paper](https://arxiv.org/abs/2312.10997)
- 🐙 [LangChain GitHub](https://github.com/langchain-ai/langchain)
