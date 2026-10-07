# Enterprise Document Q&A System

A full-stack **Retrieval-Augmented Generation (RAG)** system built with **Java, Spring Boot, LangChain4j, and Google Gemini AI** that allows users to upload PDF documents and ask natural language questions about their contents.

## 🏗️ Architecture

```
User uploads PDF
       ↓
[INGESTION PIPELINE]
  Apache PDFBox → Extract text
       ↓
  DocumentSplitters.recursive(2500, 400) → Split into chunks
       ↓
  AllMiniLmL6V2EmbeddingModel (local, JVM) → Convert chunks to 384-dim vectors
       ↓
  InMemoryEmbeddingStore → Store vectors in RAM

User asks a question
       ↓
[RETRIEVAL + GENERATION PIPELINE]
  AllMiniLmL6V2 → Convert question to vector
       ↓
  EmbeddingStoreContentRetriever (Top-8, minScore=0.4) → Semantic similarity search
       ↓
  LangChain4j @AiService → Build prompt (SystemMessage + chunks + question)
       ↓
  Google Gemini 3.5-flash-lite API → Synthesize answer across all sections
       ↓
  JSON response to frontend
```

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Java 17+ / Spring Boot 3 | Backend REST API |
| LangChain4j 1.0.0-beta5 | AI orchestration framework |
| Google Gemini API | Large Language Model (answer generation) |
| AllMiniLmL6V2 (ONNX) | Local embedding model (no API cost) |
| InMemoryEmbeddingStore | In-process vector database |
| Apache PDFBox | PDF text extraction |
| HTML/CSS/JS | Frontend UI (dark glassmorphism design) |

## 🚀 Getting Started

### Prerequisites
- Java 17+
- Maven 3.8+
- Google AI Studio API Key ([get one free](https://aistudio.google.com/app/apikey))

### Configuration
Edit `src/main/resources/application.yml`:
```yaml
gemini:
  api-key: YOUR_API_KEY_HERE
  model: gemini-3.5-flash-lite
```

### Run
```bash
./mvnw spring-boot:run
```
Open [http://localhost:8080](http://localhost:8080)

## 🔧 RAG Tuning (Key Improvements)

This system was iteratively improved to handle **multi-hop retrieval** — questions that require combining information from multiple sections of a document (e.g., Results + Discussion):

| Parameter | Before | After | Reason |
|---|---|---|---|
| Chunk size | 1000 chars | 2500 chars | Captures more cross-section context |
| Chunk overlap | 150 chars | 400 chars | Prevents context loss at section boundaries |
| Top-K retrieval | 3 | 8 | Retrieves from multiple paper sections simultaneously |
| Min similarity score | 0.6 | 0.4 | Doesn't filter out semantically weaker Discussion chunks |
| Max output tokens | 1024 | 2048 | Allows Gemini to write complete synthesized answers |
| System prompt | Generic | Multi-section synthesis instructions | Forces Gemini to combine Results + Discussion |

## 📁 Project Structure

```
src/main/java/com/example/enterprisedocqa/
├── EnterpriseDocQaApplication.java   # Spring Boot entry point
├── config/
│   └── RagConfiguration.java         # LLM, Embeddings, VectorStore, Retriever beans
├── controller/
│   └── QaController.java             # REST endpoints: /upload and /chat
└── service/
    ├── DocumentService.java           # RAG ingestion pipeline (parse→chunk→embed→store)
    └── DocumentAssistant.java         # LangChain4j @AiService (retrieval→generation)

src/main/resources/
├── application.yml                   # App configuration (API key, model, file size limits)
└── static/
    └── index.html                    # Frontend UI
```

## 🔌 REST API

### Upload Document
```
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: <PDF file>
```
Response: `{"message": "File uploaded and processed successfully"}`

### Ask a Question
```
POST /api/v1/chat
Content-Type: application/json

{"question": "What are the main findings of the paper?"}
```
Response: `{"answer": "According to the Results section..."}`
