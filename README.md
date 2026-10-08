# Enterprise Document Q&A System

A full-stack **Retrieval-Augmented Generation (RAG)** system built with **Java, Spring Boot, LangChain4j, and Google Gemini AI** that allows users to upload PDF documents and ask natural language questions about their contents. 


---

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
  Qdrant Vector Database (persistent, port 6333) → Store vectors + metadata

User asks a question
       ↓
[RETRIEVAL + GENERATION PIPELINE]
  AllMiniLmL6V2 → Convert question to 384-dim vector
       ↓
  QdrantEmbeddingStore → Cosine similarity search (Top-8, minScore=0.4)
       ↓
  LangChain4j @AiService → Build prompt (SystemMessage + chunks + question)
       ↓
  Google Gemini Flash API → Synthesize answer across all sections
       ↓
  JSON response to frontend
```

---

## 🛠️ Technologies

| Technology | Version | Purpose |
|---|---|---|
| Java | 17+ | Backend language |
| Spring Boot | 3.x | REST API framework |
| LangChain4j | 1.0.0-beta5 | AI orchestration framework |
| Google Gemini API | gemini-3.5-flash-lite | LLM for answer generation |
| AllMiniLmL6V2 (ONNX) | bundled | Local embedding model (no API cost, 384-dim) |
| **Qdrant** | **latest** | **Persistent vector database** |
| Apache PDFBox | 3.x | PDF text extraction |
| Docker / Docker Compose | any | Run Qdrant container |
| HTML / CSS / JS | — | Frontend UI (dark glassmorphism design) |

---

## 🚀 Getting Started

### Prerequisites
- Java 17+
- Maven 3.8+
- Docker & Docker Compose (to run Qdrant)
- Google AI Studio API Key → [get one free](https://aistudio.google.com/app/apikey)

---

### Step 1 — Start Qdrant

**Using Docker Compose (recommended):**
```bash
docker-compose up -d
```
This starts Qdrant with persistent storage in `./qdrant_storage/`.

**Manually verify Qdrant is running:**
```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

Qdrant dashboard is available at: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

---

### Step 2 — Configure API Key

Edit `src/main/resources/application.yml`:
```yaml
gemini:
  api-key: YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE
  model: gemini-3.5-flash-lite

qdrant:
  host: localhost
  port: 6334          # gRPC port
  collection-name: documents
  dimension: 384      # AllMiniLmL6V2 output dimension
```

> ⚠️ **Never commit your real API key.** Use environment variables or a secrets manager in production.

---

### Step 3 — Run the Application

```bash
./mvnw spring-boot:run
```

Open [http://localhost:8080](http://localhost:8080)

---

## 📦 Qdrant — Vector Database

This application uses **Qdrant** as its vector database — a high-performance, disk-backed store built for similarity search at scale.

| Feature | Detail |
|---|---|
| Storage | Disk-backed — vectors persist across restarts |
| Scalability | Production-ready, supports millions of vectors |
| Dashboard | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) |
| Setup | Via `docker-compose.yml` included in the project |
| Auto-collection | Created automatically on first startup (384-dim, cosine) |

> The `documents` collection is auto-created on first run with **cosine distance** and **384 dimensions** to match the AllMiniLmL6V2 embedding model. No manual setup needed.

---

## 🔧 RAG Configuration & Tuning

The system is tuned to handle **multi-hop retrieval** — questions that require combining information from multiple sections of a document (e.g., Results + Discussion):

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 2500 chars | Captures more cross-section context |
| Chunk overlap | 400 chars | Prevents context loss at section boundaries |
| Top-K retrieval | 8 | Retrieves from multiple document sections simultaneously |
| Min similarity score | 0.4 | Includes semantically related but indirectly-worded chunks |
| Max output tokens | 2048 | Allows Gemini to write complete synthesized answers |
| System prompt | Multi-section synthesis | Guides Gemini to combine Results + Discussion sections |
| Vector store | Qdrant (disk-backed) | Persistent storage, survives application restarts |

---

## 📁 Project Structure

```
enterprise-doc-qa/
├── docker-compose.yml                          # Start Qdrant with one command
├── src/
│   └── main/
│       ├── java/com/example/enterprisedocqa/
│       │   ├── EnterpriseDocQaApplication.java   # Spring Boot entry point
│       │   ├── config/
│       │   │   └── RagConfiguration.java         # LLM, Embeddings, Qdrant, Retriever beans
│       │   ├── controller/
│       │   │   └── QaController.java             # REST endpoints: /upload and /chat
│       │   └── service/
│       │       ├── DocumentService.java           # RAG ingestion: parse→chunk→embed→qdrant
│       │       └── DocumentAssistant.java         # LangChain4j @AiService (retrieval→generation)
│       └── resources/
│           ├── application.yml                   # App config (API key, Qdrant, model, limits)
│           └── static/
│               └── index.html                    # Frontend UI
├── pom.xml                                       # Dependencies incl. langchain4j-qdrant
└── README.md
```

---

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

---

## 🗂️ How RAG Works (Conceptually)

1. **Ingest**: User uploads a PDF → text extracted → split into overlapping chunks → each chunk converted to a 384-dimensional vector by AllMiniLmL6V2 → stored in Qdrant.
2. **Retrieve**: User asks a question → question converted to a 384-dim vector → Qdrant performs cosine similarity search → top-8 most relevant chunks returned.
3. **Generate**: The retrieved chunks + the question are assembled into a prompt → sent to Gemini API → Gemini synthesizes a comprehensive answer → returned to the user.

---

## 🐳 Docker Compose Reference

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"   # REST API + Dashboard
      - "6334:6334"   # gRPC (used by LangChain4j)
    volumes:
      - ./qdrant_storage:/qdrant/storage   # Persistent storage
```

---

## 📚 Key Dependencies (`pom.xml`)

```xml
<!-- LangChain4j core -->
<dependency>
  <groupId>dev.langchain4j</groupId>
  <artifactId>langchain4j-spring-boot-starter</artifactId>
</dependency>

<!-- Google Gemini -->
<dependency>
  <groupId>dev.langchain4j</groupId>
  <artifactId>langchain4j-google-ai-gemini-spring-boot-starter</artifactId>
</dependency>

<!-- Local embeddings (AllMiniLmL6V2, runs in JVM) -->
<dependency>
  <groupId>dev.langchain4j</groupId>
  <artifactId>langchain4j-embeddings-all-minilm-l6-v2</artifactId>
</dependency>

<!-- Qdrant vector store -->
<dependency>
  <groupId>dev.langchain4j</groupId>
  <artifactId>langchain4j-qdrant</artifactId>
</dependency>
```

---

## 🧑‍💻 Built With

- [LangChain4j](https://github.com/langchain4j/langchain4j) — Java AI framework
- [Qdrant](https://qdrant.tech/) — High-performance vector database
- [Google AI Studio](https://aistudio.google.com/) — Gemini API
- [Spring Boot](https://spring.io/projects/spring-boot) — Backend framework
