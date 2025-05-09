"""rag_agentic.py ────────────────────────────────────────────────────────────────
RAG (Re Retrieval‑Augmented Generation) **ReAct Agent** – fully commented (EN)
───────────────────────────────────────────────────────────────────────────────
This script implements an autonomous agent that combines:
  • **ReAct** (Thought → Action → Observation) for step‑by‑step reasoning.
  • **RAG** to pull facts from a local knowledge base via a FAISS vector store.
  • **LangChain** (modular stack ≥ 0.1.21) with an OpenAI language model.

High‑level flow ▶
    1. Load PDFs / TXTs / MDs from a directory (recursively).
    2. Build a FAISS vector index with OpenAI embeddings.
    3. Create two tools:
         • `search_docs`   → semantic retrieval over the vector store.
         • `answer_direct` → echoes the answer when the LLM already knows it.
    4. Assemble a ReAct prompt with a **few‑shot example** that shows the exact
       format expected.
    5. Run the Thought/Action/Observation loop until the agent emits **Final Answer**.

Quick start
───────────
$ pip install "langchain>=0.1.21" langchain-community langchain-openai \
             faiss-cpu tiktoken python-dotenv pypdf
$ echo "OPENAI_API_KEY=sk‑…" > .env
$ python rag_agentic.py --docs ./docs "What is a RAG agent?"
"""
from __future__ import annotations

import argparse                           # Parse CLI arguments
from pathlib import Path                  # Cross‑platform path handling
from typing import List

from dotenv import load_dotenv            # Load env vars from .env file

# ── LangChain (modular stack) ───────────────────────────────────────────────
from langchain_openai import ChatOpenAI   # Official wrapper for OpenAI models
from langchain_community.document_loaders import (
    DirectoryLoader,  # Walks a directory tree and loads matching files
    TextLoader,       # Reads .txt / .md files
    PyPDFLoader,      # Reads .pdf files via pypdf
)
from langchain_community.vectorstores import FAISS        # In‑memory FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory       # Simple memory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# ════════════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════════════

def load_documents(path: Path):
    """Recursively load .txt, .md and .pdf files under *path*.

    Parameters
    ----------
    path : Path
        Root directory containing the knowledge base files.

    Returns
    -------
    list[Document]
        List of LangChain `Document` objects ready for indexing.
    """
    loaders: List[DirectoryLoader] = []
    for ext in (".txt", ".md", ".pdf"):
        pattern = "**/*.pdf" if ext == ".pdf" else f"**/*{ext}"
        loader_cls = PyPDFLoader if ext == ".pdf" else TextLoader
        loaders.append(DirectoryLoader(str(path), glob=pattern, loader_cls=loader_cls))

    docs = []
    for loader in loaders:
        docs.extend(loader.load())  # Each loader returns a list of Document

    if not docs:
        raise RuntimeError(f"No documents found under {path}")
    return docs


def build_vector_store(docs):
    """Embed documents and build an in‑memory FAISS vector store."""
    return FAISS.from_documents(docs, OpenAIEmbeddings())

# ════════════════════════════════════════════════════════════════════════════
# ReAct + RAG agent construction
# ════════════════════════════════════════════════════════════════════════════

def create_agent(vector_store: FAISS) -> AgentExecutor:
    """Return a fully‑configured `AgentExecutor`."""

    # 1⃣  LLM & memory -------------------------------------------------------
    #  • Using `gpt‑4o‑mini` (cheap + moderately obedient).
    #  • Temperature 0 for deterministic output (easier parsing).
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Conversation memory keeps the full chat history between agent and user.
    memory = ConversationBufferMemory(
        memory_key="chat_history",    # Key used internally by AgentExecutor
        return_messages=True,
    )

    # 2⃣  Tools --------------------------------------------------------------
    # 2.1 `search_docs` – semantic retrieval tool wrapping the FAISS index.
    retriever = vector_store.as_retriever()
    rag_tool = create_retriever_tool(
        retriever,
        name="search_docs",
        description="Search the knowledge base for relevant context to answer the question.",
    )

    # 2.2 `answer_direct` – identity function for direct answers.
    def answer_direct(text: str) -> str:
        """Return *text* unchanged – used when the LLM already knows the answer."""
        return text

    direct_tool = Tool(
        name="answer_direct",
        func=answer_direct,
        description="Answer directly when you already know the answer from reasoning or memory.",
    )

    tools = [rag_tool, direct_tool]

    # 3⃣  ReAct prompt with a few‑shot example ------------------------------
    tool_desc = "\n".join(f"{t.name}: {t.description}" for t in tools)
    tool_names = ", ".join(t.name for t in tools)

    # The prompt *must* end with placeholders `{input}` (user question) and
    # `{agent_scratchpad}` (accumulated Thoughts/Actions/Observations).
    system_prompt = f"""
You are **RAG‑Agent**, an expert assistant with access to the following tools:
{tool_desc}

When you use a tool, follow **exactly** this format:
Thought: your reasoning
Action: one of [{tool_names}]
Action Input: the input for the action
Observation: result of the action
(repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: <answer to the user>

# Few‑shot example – shows the format explicitly
User question: How many states are there in the USA?
Thought: I know this directly.
Action: answer_direct
Action Input: 50
Observation: 50
Thought: I now know the final answer.
Final Answer: There are 50 states in the United States.
# End of example
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),           # Instructions + few‑shot
            ("human", "{input}"),                # User question (runtime)
            MessagesPlaceholder("agent_scratchpad"),  # Accumulated TAO steps
        ]
    )

    # 4⃣  Build ReAct agent --------------------------------------------------
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 5⃣  Final executor with guard rails ------------------------------------
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,            # Print full chain to stdout
        max_iterations=6,        # Prevent infinite loops
        handle_parsing_errors=True,  # Retry if LLM leaves the format
    )
    return executor

# ════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    load_dotenv()  # Make OPENAI_API_KEY available for the OpenAI SDK

    parser = argparse.ArgumentParser(description="Run the RAG‑Agent on a question.")
    parser.add_argument("--docs", type=Path, required=True, help="Directory with knowledge‑base docs (.pdf/.txt/.md)")
    parser.add_argument("question", type=str, help="User question to ask the agent")
    args = parser.parse_args()

    # 1. Load documents ------------------------------------------------------
    print("\n[1/3] Loading documents …")
    docs = load_documents(args.docs)
    print(f"Loaded {len(docs)} docs.")

    # 2. Build vector store --------------------------------------------------
    print("[2/3] Building vector store …")
    vs = build_vector_store(docs)

    # 3. Create agent --------------------------------------------------------
    print("[3/3] Creating agent …")
    agent_exec = create_agent(vs)

    # 4. Ask the question ----------------------------------------------------
    print("\n───────── AGENT ANSWER ─────────")
    response = agent_exec.invoke({"input": args.question})
    print(response["output"])

# ── Run when executed directly ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Bye! 🖐️")
