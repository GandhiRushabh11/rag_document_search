from collections.abc import Callable

from langgraph.graph import StateGraph, END
from src.rag_document_search.state.rag_state import RAGState

# from src.rag_document_search.nodes.nodes import RAGNodes
from src.rag_document_search.nodes.reactnode import RAGNodes


class GraphBuilder:
    """Builds and manages the langgraph workflow"""

    def __init__(self, retriever, llm):
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):

        # create state graph
        builder = StateGraph(RAGState)

        # add node
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        # add entry point to graph
        builder.set_entry_point("retriever")

        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        self.graph = builder.compile()
        return self.graph

    def run(
        self, question: str, progress_callback: Callable[[str], None] | None = None
    ) -> dict:
        """
        Run the RAG workflow

        Args:
            question: User question

        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()

        initial_state = RAGState(question=question)
        result = initial_state.model_dump()
        for update in self.graph.stream(
            initial_state,
            config={"recursion_limit": 10},
            stream_mode="updates",
        ):
            for node_name, state_update in update.items():
                result.update(state_update)
                if progress_callback is not None:
                    progress_callback(node_name)

        return result
