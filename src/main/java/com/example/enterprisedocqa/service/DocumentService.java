package com.example.enterprisedocqa.service;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.parser.apache.pdfbox.ApachePdfBoxDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;

/**
 * ============================================================
 * DocumentService — RAG Ingestion Pipeline
 * ============================================================
 *
 * This service handles the INGESTION phase of the RAG pipeline.
 * It is called when the user uploads a PDF via POST /api/v1/documents/upload.
 *
 * The ingestion pipeline has 3 steps:
 *
 *  STEP 1 — PARSE
 *    Apache PDFBox reads the raw bytes of the PDF and extracts all text content.
 *    The result is a single LangChain4j Document object containing the full text.
 *
 *  STEP 2 — CHUNK (Split)
 *    The full document text is split into smaller overlapping segments ("chunks").
 *    We use a recursive character splitter with:
 *      - chunkSize = 2500 chars (INCREASED from 1000 to capture more context per chunk)
 *      - overlap   = 400  chars (INCREASED from 150 to prevent context loss at boundaries)
 *
 *    WHY chunk at all?
 *      LLMs have a limited context window. Sending an entire 20-page PDF would
 *      exceed the token limit and waste cost. Chunking lets us send only the 8
 *      most relevant pieces of text.
 *
 *    WHY larger chunks + more overlap?
 *      Before tuning (1000/150): Results and Discussion sections were split into
 *      separate, isolated chunks → the RAG couldn't answer multi-hop questions
 *      (e.g., "what are the causes AND consequences" from two different sections).
 *      After tuning (2500/400): Fewer splits between sections, heavy overlap means
 *      context from one section "bleeds" into the next chunk → multi-hop retrieval works.
 *
 *  STEP 3 — EMBED & STORE
 *    Each chunk is converted to a 384-dimensional embedding vector by the local
 *    AllMiniLmL6V2 model. The vector + original text are then stored together
 *    in the InMemoryEmbeddingStore (our vector database).
 *
 * @Service marks this as a Spring-managed service component.
 */
@Service
public class DocumentService {

    // SLF4J logger — logs INFO/ERROR messages to the Spring Boot console
    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);

    // Injected by Spring via constructor injection (preferred over @Autowired)
    // EmbeddingStore: our in-memory vector database
    private final EmbeddingStore<TextSegment> embeddingStore;

    // EmbeddingModel: local all-MiniLM-L6-v2 model that converts text → vectors
    private final EmbeddingModel embeddingModel;

    /**
     * Constructor injection: Spring auto-wires the EmbeddingStore and EmbeddingModel
     * beans that are defined in RagConfiguration.java.
     */
    public DocumentService(EmbeddingStore<TextSegment> embeddingStore, EmbeddingModel embeddingModel) {
        this.embeddingStore = embeddingStore;
        this.embeddingModel = embeddingModel;
    }

    /**
     * Main ingestion method — processes an uploaded PDF through the full RAG pipeline.
     *
     * @param file The PDF file uploaded by the user via the REST API.
     */
    public void ingestPdf(MultipartFile file) {
        // Use try-with-resources to ensure the InputStream is always closed after processing
        try (InputStream inputStream = file.getInputStream()) {
            log.info("Starting ingestion for file: {}", file.getOriginalFilename());

            // ----------------------------------------------------------------
            // STEP 1: PARSE — Extract raw text from the PDF
            // ----------------------------------------------------------------
            // ApachePdfBoxDocumentParser uses the Apache PDFBox library to:
            //  - Read the PDF binary structure
            //  - Extract all text content from every page
            //  - Return a LangChain4j Document object with the full text as a String
            DocumentParser parser = new ApachePdfBoxDocumentParser();
            Document document = parser.parse(inputStream);

            // Attach metadata to the document before splitting.
            // This metadata is stored alongside each chunk in the vector store
            // and can later be used to cite the source file name in the answer.
            document.metadata().put("filename", file.getOriginalFilename());

            // ----------------------------------------------------------------
            // STEP 2 + 3: CHUNK → EMBED → STORE (all done by EmbeddingStoreIngestor)
            // ----------------------------------------------------------------
            // EmbeddingStoreIngestor is LangChain4j's all-in-one ingestion component.
            // It chains: Splitter → EmbeddingModel → EmbeddingStore automatically.
            EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()

                    // DocumentSplitters.recursive() is a smart splitter that:
                    //  - First tries to split on paragraph boundaries (\n\n)
                    //  - Then on sentence boundaries (. ! ?)
                    //  - Then on word boundaries (spaces)
                    //  - Falls back to character-level splits as a last resort
                    // This preserves semantic units (sentences, paragraphs) as much as possible.
                    //
                    // chunkSize = 2500 characters (~380 tokens):
                    //   INCREASED from 1000 chars.
                    //   Larger chunks mean each chunk carries more context, reducing the chance
                    //   that related sentences (e.g., a Result number and its Discussion explanation)
                    //   end up in completely separate, unretrieved chunks.
                    //
                    // overlap = 400 characters:
                    //   INCREASED from 150 chars.
                    //   Overlap means the last 400 chars of chunk N are repeated as the
                    //   first 400 chars of chunk N+1. This is critical at section boundaries:
                    //   if the Discussion section starts near the end of a chunk, its intro
                    //   is still captured in the next chunk's beginning via overlap.
                    .documentSplitter(DocumentSplitters.recursive(
                            2500, // chunkSize: INCREASED from 1000 → 2500 for multi-section context
                            400   // overlap:   INCREASED from 150 → 400 to preserve section boundaries
                    ))

                    // The embedding model converts each text chunk into a 384-dim vector.
                    // This is the AllMiniLmL6V2EmbeddingModel running locally in the JVM.
                    .embeddingModel(embeddingModel)

                    // The embedding store saves each (vector, text chunk) pair.
                    // This is the InMemoryEmbeddingStore (our in-RAM vector database).
                    .embeddingStore(embeddingStore)

                    .build();

            // Trigger the full pipeline: parse → split → embed → store
            ingestor.ingest(document);

            log.info("Successfully ingested file: {}", file.getOriginalFilename());

        } catch (Exception e) {
            // Log the full stack trace and re-throw as an unchecked exception
            // so Spring Boot returns a 500 Internal Server Error to the client
            log.error("Failed to process document: {}", file.getOriginalFilename(), e);
            throw new RuntimeException("Document processing failed", e);
        }
    }
}
