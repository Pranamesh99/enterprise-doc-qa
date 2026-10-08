package com.example.enterprisedocqa.service;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.spring.AiService;

/**
 * ============================================================
 * DocumentAssistant — LangChain4j AI Service Interface
 * ============================================================
 *
 * This interface is the core of the "Generate" step in the RAG pipeline.
 * LangChain4j automatically generates a full implementation of this interface
 * at startup via the @AiService annotation — we write ZERO implementation code.
 *
 * What @AiService does automatically:
 *  1. Detects the GoogleAiGeminiChatModel bean (our LLM)
 *  2. Detects the ContentRetriever bean (our semantic search engine)
 *  3. When answer() is called:
 *     a. Converts the user's question to a vector using EmbeddingModel
 *     b. Runs similarity search via ContentRetriever → retrieves top-8 relevant chunks
 *     c. Assembles the @SystemMessage + retrieved chunks + @UserMessage into a single prompt
 *     d. Sends the prompt to Gemini → gets the answer back as a String
 *
 * RAG Flow triggered by calling answer(userMessage):
 *
 *   [User Question]
 *        ↓
 *   EmbeddingModel (local) converts question → 384-dim vector
 *        ↓
 *   ContentRetriever searches QdrantEmbeddingStore for top-8 similar chunks
 *        ↓
 *   LangChain4j builds prompt:
 *       ┌─ @SystemMessage (instructions to Gemini)
 *       ├─ Retrieved chunk 1 (e.g., from Results section)
 *       ├─ Retrieved chunk 2 (e.g., from Discussion section)
 *       ├─ ...up to 8 chunks...
 *       └─ @UserMessage (the user's question)
 *        ↓
 *   Google Gemini API generates a synthesized answer
 *        ↓
 *   answer() returns the answer as a plain String
 */
@AiService
public interface DocumentAssistant {

    /**
     * @SystemMessage defines the "system prompt" — the instructions that tell Gemini
     * HOW to behave and how to process the retrieved chunks.
     *
     * This was a KEY improvement made during RAG debugging:
     *
     * BEFORE (basic system prompt):
     *   "Answer based on the provided context."
     *   Problem: Gemini only answered from the most similar chunk (Results section)
     *   and ignored chunks from the Discussion section → incomplete multi-hop answers.
     *
     * AFTER (improved synthesis prompt):
     *   Explicitly instructs Gemini to:
     *    1. READ all chunks, not just the first/most similar one.
     *    2. IDENTIFY which chunk has numbers (Results) vs explanations (Discussion).
     *    3. COMBINE them into one complete synthesized answer.
     *    4. CITE which section each piece of information came from.
     *
     * This instruction-engineering at the prompt level complemented our retrieval improvements
     * (higher Top-K, lower minScore) to achieve true multi-hop RAG.
     *
     * Interview talking point:
     *   "I improved the RAG system at two levels simultaneously:
     *    - RETRIEVAL level: tuned chunk size, overlap, Top-K, and minScore.
     *    - GENERATION level: engineered the system prompt to force cross-section synthesis."
     */
    @SystemMessage({
        // Identity: tell Gemini what role it plays
        "You are an expert enterprise document analyst and research assistant.",

        // Explain the context structure: tell Gemini it will receive multiple chunks from different sections
        "You will receive a user question and multiple retrieved text chunks from one or more sections of a document " +
        "(e.g., Abstract, Introduction, Results, Discussion, Conclusion).",

        // CRITICAL: force synthesis across ALL chunks — this was the main fix for multi-hop retrieval
        "CRITICAL INSTRUCTION: You must SYNTHESIZE information from ALL provided chunks to form a complete answer. " +
        "Do NOT rely only on the first or most similar chunk. Different chunks may come from different sections of the paper " +
        "(e.g., numerical results from Results, explanations from Discussion). " +
        "A complete answer MUST combine all relevant pieces across sections.",

        // Step-by-step reasoning process — forces structured thinking
        "Follow this reasoning process step by step:",
        "1. Read ALL the provided context chunks carefully.",
        "2. Identify which chunks contain quantitative results (numbers, tables, metrics).",
        "3. Identify which chunks contain explanations, causes, or consequences (usually in Discussion or Conclusion).",
        "4. Combine both to produce one complete, well-structured answer.",

        // Honesty instruction: don't hallucinate missing information
        "If the answer to any part of the question is not present anywhere in the provided context, explicitly state: " +
        "'This specific detail is not available in the retrieved document sections.'",

        // Citation instruction: cite sections for credibility and traceability
        "Always mention which section of the document your information comes from (e.g., 'According to the Results section...', " +
        "'The Discussion section explains...').",

        // Formatting instruction: structured output for readability
        "Format multi-part answers with clear paragraphs or numbered points for readability."
    })
    /**
     * The main question-answering method.
     *
     * @param userMessage The user's natural language question from the frontend.
     *                    Annotated with @UserMessage so LangChain4j knows this is the
     *                    human turn of the conversation to send to the LLM.
     * @return The AI-generated answer as a plain String, synthesized from the
     *         retrieved document chunks and the Gemini LLM.
     */
    String answer(@UserMessage String userMessage);
}
