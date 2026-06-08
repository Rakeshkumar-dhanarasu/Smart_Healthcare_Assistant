class VectorDebriefer:
    def __init__ (self, api_key, stepback_prompt_path, final_answer_prompt_path):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from pydantic import BaseModel,Field
        from typing import List,Optional
        from google import genai
        import os

        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
    
        FAISS_PATH = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "datasets",
            "vector_db",
            "information_enricher_db"
        )    

        self.db = FAISS.load_local(
            folder_path=FAISS_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )

        with open(stepback_prompt_path, "r") as f1:
            self.stepback_prompt_template = f1.read()

        with open(final_answer_prompt_path,"r") as f2:
            self.final_answer_prompt_template  =f2.read()

    def _stepback_questioner(self, graph_res: dict) -> list:
        cypher = graph_res.get("cypher","NA")
        data = graph_res.get("data",[])

        attributes = "\n".join([str(item) for item in data])
        
        prompt = f"""Instruction: {self.stepback_prompt_template} Cypher: {cypher} Relationships: {attributes}"""

        res = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return [q.strip() for q in res.text.split("\n") if q.strip()]
    
    def _retrieve_content(self, questions: list) -> list:
        answers = []

        for q in questions:
            docs = self.db.similarity_search(q,k=3)
            context = "\n".join([doc.page_content for doc in docs])

            answers.append({
                "question" : q,
                "answer": context
            })

        return answers
    
    def _final_answer(self, retrieved_data: list) -> str:
        formatted_context = ""

        for item in retrieved_data:
            formatted_context += f"Question: {item['question']}\n"
            formatted_context += f"Context: {item['answer']}\n\n"

        prompt = (
            self.final_answer_prompt_template +
            "\n\nUse the following information to generate a complete and concise answer:\n\n" +
            formatted_context
        )

        res = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return res.text.strip()
    
    def user_output(self, graph_info:dict) -> str:
        questions = self._stepback_questioner(graph_info)
        if not questions:
            return "Unable to generate follow-up questions from the provided data."

        retrieved_data = self._retrieve_content(questions)
        if not retrieved_data:
            return "No relevant information found in the knowledge base."

        final_response = self._final_answer(retrieved_data)
        return final_response