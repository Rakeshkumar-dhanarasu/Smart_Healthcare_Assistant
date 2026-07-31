from dotenv import load_dotenv

class RAGService:
    def __init__(self):
        from graphrag import GraphRetriever
        from vectorrag import VectorDebriefer
        from langchain_huggingface import HuggingFaceEmbeddings
        from pathlib import Path
        import os

        load_dotenv()
        # ---------------- ENV SETUP ----------------
        self.api_key = os.getenv("GENAI_API_KEY")

        self.embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment variables")

        # ---------------- INIT MODULES ----------------
        self.graph_rag = GraphRetriever(api_key=self.api_key)
        self.vector_rag = VectorDebriefer(api_key=self.api_key,embed_model=self.embed_model)

    # ---------------- MAIN PIPELINE ----------------
    def run_pipeline(self, user_query: str) -> str:
        """
        Full Hybrid RAG pipeline:
        User Query → GraphRAG → VectorRAG → Final Answer
        """

        try:
            # -------- STEP 1: GRAPH RAG --------
            graph_output = self.graph_rag.run(user_query)

            if not graph_output:
                return "Graph retrieval failed."

            # -------- STEP 2: VECTOR RAG --------
            final_response = self.vector_rag.run(
                user_query=user_query,
                graph_output=graph_output
            )

            return final_response

        except Exception as e:
            return f"Error in pipeline: {str(e)}"

    # ---------------- OPTIONAL: DEBUG MODE ----------------
    def debug_pipeline(self, user_query: str) -> dict:
        """
        Returns intermediate outputs (VERY useful for debugging)
        """

        debug_data = {}

        try:
            graph_output = self.graph_rag.run(user_query)
            debug_data["graph_output"] = graph_output

            vector_output = self.vector_rag.run(
                user_query=user_query,
                graph_output=graph_output
            )

            debug_data["final_output"] = vector_output

            return debug_data

        except Exception as e:
            return {"error": str(e)}

    # ---------------- OPTIONAL: EXTRA FUNCTIONS ----------------
    def health_check(self) -> str:
        return "RAG Service is running 🚀"