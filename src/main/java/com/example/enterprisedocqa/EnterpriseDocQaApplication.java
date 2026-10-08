package com.example.enterprisedocqa;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * ============================================================
 * Enterprise Document Q&A System - Main Entry Point
 * ============================================================
 *
 * This is the main Spring Boot application class.
 *
 * Technologies used in this project:
 *  - Spring Boot 3          : Backend REST API framework
 *  - LangChain4j            : Java AI orchestration library (like LangChain in Python)
 *  - Google Gemini API      : Large Language Model (LLM) for generating answers
 *  - AllMiniLmL6V2          : Local embedding model running inside the JVM (no API call)
 *  - QdrantEmbeddingStore   : Persistent vector database (runs in Docker, survives restarts)
 *  - Apache PDFBox          : PDF text extraction library
 *
 * Architecture Pattern: RAG (Retrieval-Augmented Generation)
 *  Step 1 - INGEST : User uploads PDF → text is extracted → split into chunks
 *                    → chunks converted to vectors → stored in Vector DB
 *  Step 2 - RETRIEVE: User asks question → question converted to vector
 *                    → vector similarity search retrieves top-K relevant chunks
 *  Step 3 - GENERATE: Retrieved chunks + user question sent to Gemini LLM
 *                    → Gemini synthesizes a complete, accurate answer
 *
 * @SpringBootApplication enables:
 *  - @ComponentScan  : Auto-detects all @Service, @Controller, @Configuration classes
 *  - @EnableAutoConfiguration : Automatically configures Spring beans based on classpath
 *  - @Configuration  : Marks this as a configuration source
 */
@SpringBootApplication
public class EnterpriseDocQaApplication {

    public static void main(String[] args) {
        // Boots the embedded Tomcat server on port 8080 and starts the Spring context
        SpringApplication.run(EnterpriseDocQaApplication.class, args);
    }
}
