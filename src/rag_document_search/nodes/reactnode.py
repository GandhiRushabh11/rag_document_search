"""Langgraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional
from src.rag_document_search.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
import uuid
import requests
import wikipedia

# or if you use UUID directly in type hints:
from uuid import UUID

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun

import builtins
import uuid

builtins.uuid = uuid
builtins.UUID = uuid.UUID


class RAGNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  ##lazy init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question, retrieved_docs=docs)

    # Build tools

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query: str):
            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No Documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title} \n {d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed vectorstore",
            func=retriever_tool_fn,
        )
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        def wikipedia_tool_fn(query: str) -> str:
            try:
                return wiki.run(query)
            except (
                requests.exceptions.RequestException,
                wikipedia.exceptions.WikipediaException,
            ) as error:
                return f"Wikipedia is temporarily unavailable ({type(error).__name__})."

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wikipedia_tool_fn,
        )
        return [retriever_tool, wikipedia_tool]

    ##Build Agent
    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG Agent. "
            "Perfer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "If Wikipedia is unavailable, answer from your own knowledge and say so. "
            "Return only the final useful answer."
        )

        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:

        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke(
            {"messages": [HumanMessage(content=state.question)]}
        )

        messages = result.get("messages", [])

        answer: Optional[str] = None

        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer,
        )
