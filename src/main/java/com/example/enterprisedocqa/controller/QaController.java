package com.example.enterprisedocqa.controller;

import com.example.enterprisedocqa.service.DocumentAssistant;
import com.example.enterprisedocqa.service.DocumentService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * ============================================================
 * QaController — REST API Controller
 * ============================================================
 *
 * This controller exposes two HTTP REST endpoints that power the frontend UI:
 *
 *  1. POST /api/v1/documents/upload  → Triggers the RAG ingestion pipeline
 *  2. POST /api/v1/chat             → Triggers the RAG retrieval + generation pipeline
 *
 * @RestController = @Controller + @ResponseBody
 *   Tells Spring this class handles HTTP requests and returns JSON responses.
 *   Every method automatically serializes its return value to JSON.
 *
 * @RequestMapping("/api/v1")
 *   All endpoints in this controller are prefixed with /api/v1 for versioning.
 *   This is best practice — if you ever break the API, you release /api/v2 without
 *   breaking existing clients using /api/v1.
 */
@RestController
@RequestMapping("/api/v1")
public class QaController {

    /**
     * DocumentService handles the INGESTION pipeline:
     *   PDF → Parse → Chunk → Embed → Store in Vector DB
     */
    private final DocumentService documentService;

    /**
     * DocumentAssistant is the LangChain4j @AiService proxy that handles RETRIEVAL + GENERATION:
     *   Question → Vector Search → Retrieved Chunks → Gemini LLM → Answer
     */
    private final DocumentAssistant documentAssistant;

    /**
     * Constructor injection (preferred over @Autowired field injection).
     * Spring automatically provides the DocumentService and DocumentAssistant beans.
     * Using constructor injection makes the class easier to unit-test (you can pass mocks).
     */
    public QaController(DocumentService documentService, DocumentAssistant documentAssistant) {
        this.documentService = documentService;
        this.documentAssistant = documentAssistant;
    }

    // -----------------------------------------------------------------------
    // ENDPOINT 1: PDF Upload — triggers RAG Ingestion Pipeline
    // -----------------------------------------------------------------------
    /**
     * POST /api/v1/documents/upload
     *
     * Accepts a multipart PDF file upload, validates it, and passes it to
     * DocumentService for the full ingestion pipeline (parse → chunk → embed → store).
     *
     * Called by the frontend when the user clicks "Ingest Document".
     *
     * @param file The uploaded file from the multipart form (form field name: "file")
     * @return 200 OK with {"message": "File uploaded and processed successfully"}
     *         400 BAD REQUEST if the file is empty or not a PDF
     *         500 INTERNAL SERVER ERROR if an exception occurs during processing
     *
     * Frontend JS call:
     *   const formData = new FormData();
     *   formData.append("file", fileInput.files[0]);
     *   fetch('/api/v1/documents/upload', { method: 'POST', body: formData });
     */
    @PostMapping("/documents/upload")
    public ResponseEntity<Map<String, String>> uploadDocument(@RequestParam("file") MultipartFile file) {

        // Validate: reject empty files before wasting any processing
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "File is empty"));
        }

        // Validate: only PDF files are supported (Apache PDFBox can only parse PDFs)
        if (!file.getOriginalFilename().toLowerCase().endsWith(".pdf")) {
            return ResponseEntity.badRequest().body(Map.of("error", "Only PDF files are supported"));
        }

        // Delegate to DocumentService which runs the full RAG ingestion pipeline:
        // PDF → ApachePdfBoxDocumentParser → DocumentSplitters.recursive(2500,400) → AllMiniLmL6V2 → QdrantEmbeddingStore
        documentService.ingestPdf(file);

        // Return a simple JSON success message — the frontend shows this to the user
        return ResponseEntity.ok(Map.of("message", "File uploaded and processed successfully"));
    }

    // -----------------------------------------------------------------------
    // ENDPOINT 2: Chat — triggers RAG Retrieval + Generation Pipeline
    // -----------------------------------------------------------------------
    /**
     * POST /api/v1/chat
     *
     * Accepts a JSON body with the user's question and returns an AI-generated answer
     * by triggering the full RAG retrieval + generation pipeline through DocumentAssistant.
     *
     * The pipeline triggered by documentAssistant.answer(question):
     *  1. Convert question to embedding vector (local AllMiniLmL6V2)
     *  2. Search QdrantEmbeddingStore for top-8 most similar document chunks (minScore≥0.4)
     *  3. Build the LLM prompt: @SystemMessage + retrieved chunks + user question
     *  4. Send prompt to Google Gemini API
     *  5. Return Gemini's synthesized answer as a String
     *
     * @param request A JSON object containing the "question" field.
     *                Uses Java record (ChatRequest) for concise immutable deserialization.
     * @return 200 OK with {"answer": "<AI generated response>"}
     *         500 INTERNAL SERVER ERROR if the LLM call fails
     *
     * Frontend JS call:
     *   fetch('/api/v1/chat', {
     *     method: 'POST',
     *     headers: { 'Content-Type': 'application/json' },
     *     body: JSON.stringify({ question: "What causes the higher FN rate?" })
     *   });
     *
     * Example Request Body:
     *   { "question": "Why does the proposed model have a lower recall than YOLOv11?" }
     *
     * Example Response:
     *   { "answer": "According to the Discussion section, the proposed model has a lower
     *                recall because ... (synthesized multi-section answer)" }
     */
    @PostMapping("/chat")
    public ResponseEntity<Map<String, String>> chat(@RequestBody ChatRequest request) {
        // documentAssistant.answer() triggers the entire RAG pipeline:
        // LangChain4j's @AiService proxy handles retrieval + prompt assembly + LLM call
        String answer = documentAssistant.answer(request.question());

        // Wrap the answer in a JSON map — Spring Boot auto-serializes this to JSON
        return ResponseEntity.ok(Map.of("answer", answer));
    }

    // -----------------------------------------------------------------------
    // DTO: ChatRequest Record
    // -----------------------------------------------------------------------
    /**
     * ChatRequest is a Java Record — a concise, immutable data class introduced in Java 16.
     * It replaces a traditional POJO with getter, constructor, equals, hashCode, and toString.
     *
     * Spring Boot's @RequestBody deserializes the incoming JSON body:
     *   { "question": "What is the FN rate?" }
     * into a ChatRequest object where request.question() = "What is the FN rate?"
     *
     * Using a record instead of a raw String parameter is better practice because:
     *  - It makes the API contract explicit and self-documenting
     *  - Easier to extend later (add "sessionId", "language", etc.)
     */
    public record ChatRequest(String question) {}
}
