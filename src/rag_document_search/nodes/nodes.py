"""Langgraph nodes for RAG Workflow"""

from src.rag_document_search.state.rag_state import RAGState


class RAGNodes:
    """ "Contains node functions for RAG Workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question, retrieved_docs=docs)

    def generate_answer(self, state: RAGState):

        # combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # create prompt
        prompt = f"""Answer the question based on the context.
        
        Context:
        {context}
        
        Question:
        {state.question}s
        
        """

        # Generate Answer
        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content,
        )

    def run(self, question: str) -> dict:
        if self.graph is None:
            self.build()
        initial_state = RAGState(question=question)

        return self.graph.invoke(initial_state)
