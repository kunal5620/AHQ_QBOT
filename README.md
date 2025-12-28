# AHQ_QBOT
LLM Based Custom Knowledge Based Chatbot for Anand House of Quality 🤖

A hybrid Large Language Model (LLM) chatbot designed to deliver accurate, context-aware, and domain-specific responses by combining semantic search and generative AI. The system integrates NVIDIA’s embedding model for efficient knowledge retrieval with Meta’s generative model for natural language response generation.

🚀 Project Overview

This project implements a custom knowledge base chatbot using a two-stage hybrid architecture:

Semantic Search & Context Retrieval
NVIDIA’s 7.85B parameter embedding model converts knowledge base content into vector representations, enabling efficient semantic search and retrieval of the most relevant context.

Answer Generation
Meta’s 3.2B parameter generative model uses the retrieved context to generate coherent, accurate, and human-like responses.

The chatbot is designed to ensure high relevance, accuracy, and fluency, making it suitable for domain-specific question answering systems.

🧠 Architecture Overview
User Query
    ↓
NVIDIA Embedding Model (7.85B)
    ↓
Semantic Search (Vector Database)
    ↓
Relevant Context Retrieved
    ↓
Meta Generative Model (3.2B)
    ↓
Natural Language Response

✨ Key Features

Hybrid LLM Architecture combining retrieval and generation

Semantic Search using high-dimensional embeddings

Custom Knowledge Base Integration for domain-specific accuracy

Flask Backend for API handling and model orchestration

Responsive Frontend built with HTML, CSS, and JavaScript

Scalable Design suitable for expanding knowledge bases

🛠️ Tech Stack
Models

Meta LLM (3.2B parameters) – Response generation

NVIDIA Embedding Model (7.85B parameters) – Semantic search & context retrieval

Backend: Python, Flask – API development and request handling

Frontend: HTML, CSS, JavaScript

AI & NLP: Vector embeddings, Semantic search, Retrieval-Augmented Generation (RAG)



