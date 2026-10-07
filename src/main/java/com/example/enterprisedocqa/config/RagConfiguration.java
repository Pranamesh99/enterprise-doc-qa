package com.example.enterprisedocqa.config;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * ============================================================
 * RAG Configuration — Central AI/LLM Setup
 * ============================================================
 *
 * This class is the heart of the RAG system.
 * It defines 4 Spring Beans that wire the entire AI pipeline together:
 *
 *  1. GoogleAiGeminiChatModel  → The Large Language Model (LLM) that generates answers
 *  2. EmbeddingModel           → Converts text to numerical vectors locally (no API call)
 *  3. EmbeddingStore           → In-memory vector database that stores document chunks
 *  4. ContentRetriever         → Performs semantic similarity search to find relevant chunks
 *
 * These 4 beans are auto-detected by LangChain4j's @AiService and wired into
 * DocumentAssistant automatically — no manual wiring needed!
 *
 * @Configuration tells Spring this class contains @Bean definitions.
 */
@Configuration
public class RagConfiguration {

    /**
     * Injected from application.yml → gemini.api-key
     * This is your Google AI Studio API key used to authenticate with the Gemini API.
     * NEVER hardcode API keys — always use environment variables or config files.
     */
    @Value("${gemini.api-key}")
    private String apiKey;

    /**
     * Injected from application.yml → gemini.model
     * Default fallback value is "gemini-3.5-flash-lite" if the property is not set.
     *
     * WHY gemini-3.5-flash-lite?
     *  - gemini-1.5-flash and gemini-2.5-flash are deprecated for new API keys.
     *  - gemini-3.8-flash was returning 503 (high demand) during testing.
     *  - gemini-3.5-flash-lite is fast, cost-effective, and returned correct answers.
     */
    @Value("${gemini.model:gemini-3.5-flash-lite}")
    private String modelName;

    // -----------------------------------------------------------------------
    // BEAN 1: LLM — Google Gemini Chat Model
    // -----------------------------------------------------------------------
    /**
     * Configures the Google Gemini LLM that generates the final answer.
     *
     * LangChain4j calls this model in the final "Generate" step of RAG:
     *   [User Question] + [Retrieved Chunks] → Gemini → [Final Answer]
     *
     * Parameters:
     *  - apiKey          : Your Google AI Studio API key
     *  - modelName       : Which Gemini model version to use (from application.yml)
     *  - temperature(0.2): Controls randomness/creativity of the output.
     *                      0.0 = fully deterministic, 1.0 = creative.
     *                      We use 0.2 (low) so Gemini stays factual and doesn't hallucinate.
     *  - maxOutputTokens : Maximum length of the generated answer.
     *                      INCREASED from 1024 → 2048 so Gemini can synthesize
     *                      complete, multi-section answers (important for multi-hop RAG).
     */
    @Bean
    public GoogleAiGeminiChatModel chatLanguageModel() {
        return GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName(modelName)
                .temperature(0.2)       // Low temperature = factual, not creative
                .maxOutputTokens(2048)  // INCREASED from 1024 to allow full synthesis answers
                .build();
    }

    // -----------------------------------------------------------------------
    // BEAN 2: Embedding Model — Local Text-to-Vector Converter
    // -----------------------------------------------------------------------
    /**
     * Configures the local embedding model: all-MiniLM-L6-v2.
     *
     * What is an Embedding?
     *   An embedding is a dense numerical vector (array of 384 floating-point numbers)
     *   that represents the semantic meaning of a piece of text.
     *   Texts that are semantically similar will have vectors that are close together
     *   in 384-dimensional space (measured by cosine similarity).
     *
     * Why run it LOCALLY (not via API)?
     *  1. COST: No API charges for embedding thousands of document chunks.
     *  2. PRIVACY: Sensitive enterprise documents are never sent over the network.
     *  3. LATENCY: In-JVM inference is instant (no network round-trip).
     *
     * This model is downloaded as a Maven dependency and runs entirely inside the JVM
     * using ONNX Runtime (Open Neural Network Exchange) — no Python or GPU needed.
     *
     * Used in TWO places:
     *  - During INGEST: to convert each document chunk into a vector and store it.
     *  - During QUERY:  to convert the user's question into a vector for similarity search.
     */
    @Bean
    public EmbeddingModel embeddingModel() {
        return new AllMiniLmL6V2EmbeddingModel();
    }

    // -----------------------------------------------------------------------
    // BEAN 3: Vector Store — In-Memory Embedding Database
    // -----------------------------------------------------------------------
    /**
     * Configures the vector store: InMemoryEmbeddingStore.
     *
     * What is a Vector Store?
     *   A vector store is a database optimized to store and search high-dimensional vectors.
     *   Instead of SQL queries (WHERE name = 'X'), it answers: "Which stored vectors are
     *   most similar to this query vector?"
     *
     * InMemoryEmbeddingStore:
     *  - Stores all document chunk vectors in RAM (no external DB like Qdrant or Pinecone needed).
     *  - Perfect for demos, prototypes, and small-to-medium document sets.
     *  - Data is LOST when the application restarts (not persistent).
     *
     * For production scale-out, this can be swapped with:
     *  - Qdrant (Docker-based, persistent)
     *  - Pinecone (cloud-managed)
     *  - pgvector (PostgreSQL extension)
     *  LangChain4j supports all of these with the same interface — just change this bean!
     */
    @Bean
    public EmbeddingStore<dev.langchain4j.data.segment.TextSegment> embeddingStore() {
        return new InMemoryEmbeddingStore<>();
    }

    // -----------------------------------------------------------------------
    // BEAN 4: Content Retriever — Semantic Search Engine
    // -----------------------------------------------------------------------
    /**
     * Configures the content retriever that performs semantic similarity search.
     *
     * This is the "Retrieval" step of RAG:
     *   1. User asks a question → EmbeddingModel converts it to a query vector.
     *   2. ContentRetriever searches the EmbeddingStore for the closest chunk vectors.
     *   3. The top-K most similar chunks are returned as context for the LLM.
     *
     * Key parameters (TUNED during debugging for better multi-hop retrieval):
     *
     *  - maxResults(8):
     *    The number of top-K chunks to retrieve from the vector store.
     *    INCREASED from 3 → 8.
     *    WHY: With only 3 chunks, we only got the Results section.
     *         Increasing to 8 allows retrieval from multiple sections
     *         (Results, Discussion, Conclusion) simultaneously — enabling multi-hop answers.
     *
     *  - minScore(0.4):
     *    Minimum cosine similarity score for a chunk to be included.
     *    Score range: 0.0 (no similarity) to 1.0 (identical).
     *    LOWERED from 0.6 → 0.4.
     *    WHY: The Discussion section discusses consequences and causes in different words
     *         than the Results section. With 0.6, these semantically weaker but relevant
     *         chunks were being filtered out. Lowering to 0.4 allows them through.
     *
     * Result of tuning: System went from single-section retrieval to full multi-hop
     * retrieval across Results + Discussion + Introduction sections simultaneously.
     */
    @Bean
    public ContentRetriever contentRetriever(
            EmbeddingStore<dev.langchain4j.data.segment.TextSegment> embeddingStore,
            EmbeddingModel embeddingModel) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(8)    // INCREASED from 3→8: retrieve from multiple paper sections
                .minScore(0.4)    // LOWERED from 0.6→0.4: don't filter out Discussion chunks
                .build();
    }
}
