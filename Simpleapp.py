# Import the libraries
import os
import asyncio
import logging
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def bootstrap_environment():
    """Validates parameters and applies self-healing mechanisms for configuration strings."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("CRITICAL: 'OPENAI_API_KEY' environment variable is missing. Pipeline halted.")

    # Self-healing logic for LangSmith tracing
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY missing. Disabling live cloud tracing dashboards.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

        # If your .env lacks a project name, don't crash. Create one automatically!
        if not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "VS-Code-Advanced-Pipeline"

bootstrap_environment()

# Primary high-performance model configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Fallback model configuration in case of rate limits or service disruptions
fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# =============================================================================
# Agentic RAG engine
# =============================================================================
class AgenticSelfCorrectingRAG:
    """
    A cutting-edge Agentic RAG engine equipped with async non-blocking execution pipelines,
    Maximal Marginal Relevance diversity lookup, and an automated content grading supervisor layer.
    """

    def __init__(self, primary_url: str):
        self.primary_url = primary_url
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.vector_store = None
        self.retriever = None
        self.grade_agent = None
        self.generation_chain = None

    async def initialize_pipeline(self):
        """Asynchronously scrapes target URLs and builds clean dense vector engines."""
        logger.info(f"Connecting network ingestion worker to: {self.primary_url}")

        # Keep Jupyter cell processing fluid by offloading I/O operations to an isolated executor
        loader = WebBaseLoader(
            self.primary_url,
            bs_kwargs={"parse_only": SoupStrainer(["article", "main", "p", "h1", "h2", "h3", "li"])}
        )
        loop = asyncio.get_running_loop()
        raw_documents = await loop.run_in_executor(None, loader.load)

        # Advanced symmetric splitting shortcuts
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )

        processed_chunks = text_splitter.split_documents(raw_documents)
        logger.info(f"Successfully processed and optimized data into {len(processed_chunks)} context nodes.")

        # Instantiate vector metrics in memory
        self.vector_store = await loop.run_in_executor(
            None, lambda: FAISS.from_documents(processed_chunks, self.embeddings)
        )

        # Enforce Maximal Marginal Relevance (MMR) to maximize unique contextual information
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 12}
        )

        # Compile Advanced Operations Chains
        self._compile_supervisor_agent()
        self._compile_generation_chain()
        logger.info("All modular agent chains compiled successfully. Operations engine online.")

    def _compile_supervisor_agent(self):
        """Creates an intelligent grading agent to validate document context quality."""
        grading_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI quality assurance supervisor. Assess if the retrieved document chunk "
                "is related to or could help answer the user query, even partially.\n\n"
                "Return a strict JSON object with a single key 'relevance_score' containing either "
                "'YES' or 'NO' values only. Do not provide any additional text or explanations.\n\n"
                "Retrieved Document:\n{document}\n"
            )),
            ("human", "User Query: {question}")
        ])

        # Modern LCEL pattern connecting straight to a structured JSON output parser
        self.grade_agent = grading_prompt | self.llm | JsonOutputParser()

    def _compile_generation_chain(self):
        """Assembles the final answer generation line using LCEL syntax patterns."""
        system_instructions = (
            "You are a principal enterprise AI systems agent. Answer the question using only "
            "the structured context blocks provided below. If the data engine context does not contain "
            "explicit proof to build an answer, output: 'Insufficient verified data in memory engine.'\n\n"
            "Context Blocks:\n{context}"
        )

        generation_prompt = ChatPromptTemplate.from_messages([
            ("system", system_instructions),
            ("human", "{question}")
        ])

        # Core generation block mapping
        self.generation_chain = generation_prompt | self.llm | StrOutputParser()

    def _format_context(self, verified_docs: List[Document]) -> str:
        """Transforms structural text arrays into clean, reference-cited string structures."""
        return "\n\n---\n\n".join(
            f"[Verified Resource Chunk #{idx+1} | Origin Source: {doc.metadata.get('source')}]\n{doc.page_content}"
            for idx, doc in enumerate(verified_docs)
        )

    async def execute_agentic_workflow(self, user_question: str):
        """
        Executes the agentic self-correction loop. Evaluates context relevance concurrently,
        and drops low-quality text items before streaming out clean answers.
        """
        if not self.retriever or not self.generation_chain:
            raise RuntimeError("The engine's active runtime has not been initialized. Run initialize_pipeline() first.")

        logger.info(f"Routing query parameters through Agentic Filter Matrix...")

        # Fetch candidate document chunks from the vector database
        loop = asyncio.get_running_loop()
        candidate_docs = await loop.run_in_executor(None, lambda: self.retriever.invoke(user_question))

        # Spin up parallel tasks to grade all document pieces concurrently
        grading_tasks = [
            self.grade_agent.ainvoke({"document": doc.page_content, "question": user_question})
            for doc in candidate_docs
        ]
        grading_results = await asyncio.gather(*grading_tasks)

        # Filter and retain only highly relevant documents
        verified_documents = []
        for idx, feedback in enumerate(grading_results):
            if feedback.get("relevance_score", "").upper() == "YES":
                verified_documents.append(candidate_docs[idx])

        logger.info(f"Supervisor matrix complete. Retained ({len(verified_documents)}/{len(candidate_docs)}) high-quality context blocks.")

        # Fallback safeguard: If all document chunks are rejected, halt execution safely
        if not verified_documents:
            print("\n🛑 [Agentic Halt]: No relevant documentation context found in the vector matrix.")
            return

        # Context packaging
        formatted_context_string = self._format_context(verified_documents)

        print(f"\n🖥️ User Prompt: {user_question}")
        print("🤖 Engine Streaming Response: ", end="", flush=True)

        # Stream the multi-token response out in real-time
        async for token in self.generation_chain.astream({
            "context": formatted_context_string,
            "question": user_question
        }):
            print(token, end="", flush=True)
        print("\n" + "="*90)


# =============================================================================
# Execution Initialization Loop
# =============================================================================
async def main():
    # Target documentation URL
    target_documentation_url = "https://en.wikipedia.org/wiki/LangChain"
    # target_documentation_url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"

    # Instantiate and fire up the engine pipeline
    engine = AgenticSelfCorrectingRAG(primary_url=target_documentation_url)
    await engine.initialize_pipeline()

    # Execute Streaming Queries
    # Live Test Prompt targeting complex dataset concepts
    query = "What is LangChain used for?"
    # query = "What are the two usage limits in LangSmith and how do we calculate total traces limit?"


    # Execute the self-correcting generation pipeline
    await engine.execute_agentic_workflow(query)

if __name__ == "__main__":
    asyncio.run(main())
