# 📄 RAG-Powered PDF Chatbot with Chainlit & DeepSeek

This project is a lightweight, locally hosted **Retrieval-Augmented Generation (RAG)** application that allows users to **chat with PDF documents**, including rich data like **tables**, using an intuitive interface built with **Chainlit**.

Powered by **DeepSeek** models and enhanced with **Qdrant** as the vector store, this system enables intelligent semantic search and response generation based on your document's content.

---

## 💡 Project Highlights

- **📚 Retrieval-Augmented Generation (RAG)**  
  Combines traditional retrieval with large language models (LLMs) to ground the output in external knowledge—your PDF files.

- **🔗 Chainlit UI**  
  Provides an elegant, real-time chat interface where users can interact directly with their PDF content. Minimal setup, maximum interactivity.

- **🧠 DeepSeek Integration**  
  Utilizes DeepSeek's LLMs to generate intelligent responses. The model interprets context and extracts relevant information even from complex document structures.

- **📊 Structured Data Handling**  
  Goes beyond plain text! The system can accurately extract and reason about **tables, lists, and other structured elements** within PDFs—crucial for reports, scientific papers, and financial documents.

- **📥 Local-First & Privacy-Respecting**  
  All data processing and retrieval happens **locally**. Nothing is sent to third-party servers, making this solution ideal for sensitive or proprietary documents.

- **⚡ Fast & Efficient with Qdrant**  
  Uses **Qdrant**, an open-source vector similarity engine, for efficient semantic search and retrieval. Hosted locally via Docker for simplicity and speed.

---

## 🛠️ Tech Stack

| Component       | Description                                 |
|----------------|---------------------------------------------|
| **Chainlit**    | Interactive UI for RAG applications         |
| **DeepSeek**    | LLM for context-aware response generation   |
| **Qdrant**      | Vector store for embedding-based retrieval  |
| **PDF Parser**  | Extracts text and table data from documents |
| **Docker**      | Runs Qdrant locally                         |
| **uv**          | Fast Python dependency manager & runner     |

---

## 🎯 Goal

The objective of this project is to **empower users to extract and interact with knowledge buried in PDFs**, without relying on cloud services or exposing private data.