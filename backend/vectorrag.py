class VectorDebriefer:

    def __init__(self, api_key, embed_model):
        from langchain_community.vectorstores import FAISS
        from google import genai
        from pathlib import Path
        from dotenv import load_dotenv
        import os

        load_dotenv()

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.5-flash"

        BASE_DIR = Path(__file__).resolve().parent

        def get_path(env_var):
            path = os.getenv(env_var)
            return (BASE_DIR / path).resolve()

        FAISS_PATH = get_path("VECTOR_DB_PATH")

        self.db = FAISS.load_local(
            folder_path=FAISS_PATH,
            embeddings=embed_model,
            allow_dangerous_deserialization=True
        )

        # Load external prompt
        prompt_path = get_path("STEPBACK_QUESTIONER_PROMPT_PATH")

        with open(prompt_path, "r") as f:
            self.stepback_template = f.read()

    # ---------------- STEP-BACK ----------------
    def _stepback_questions(self, query, graph_nodes):

        nodes_text = ", ".join(graph_nodes)

        prompt = f"""
            Related entities: {nodes_text}

            Focus:
            - symptoms
            - herbs
            - nutrients
            - usage

            Query:
            {query}
            """

        full_prompt = self.stepback_template + prompt

        res = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt
        )

        return [q.strip() for q in res.text.split("\n") if q.strip()]

    # ---------------- VECTOR RETRIEVAL ----------------
    def _vector_retrieve(self, query):

        docs = self.db.max_marginal_relevance_search(query, k=4)

        return [
            {
                "text": d.page_content,
                "entity": d.metadata.get("entity"),
                "type": d.metadata.get("type")
            }
            for d in docs
        ]

    # ---------------- SUB-QUESTION ANSWERING ----------------
    def _answer_subquestion(self, question, docs):

        context = "\n\n".join([d["text"][:300] for d in docs])

        prompt = f"""
            Answer the question using the given context.

            Rules:
            - Focus only on helpful remedies
            - Keep it short (3-4 lines)
            - No diagnosis

            Context:
            {context}

            Question:
            {question}
            """

        res = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return res.text.strip()

    # ---------------- FINAL SUMMARIZATION ----------------
    def _summarize(self, query, sub_answers):

        combined = "\n\n".join(sub_answers)

        prompt = f"""
            You are a helpful health assistant.

            STRICT RULES:
            1. DO NOT diagnose diseases
            2. DO NOT mention causes
            3. ONLY give natural remedies
            4. Use clean numbered format
            5. No markdown or symbols

            Structure:

            Response:
            <Short summary>

            Suggested Remedies:
            1. ...
            2. ...

            Precautions:
            1. ...
            2. ...

            Important Note:
            This is not a medical diagnosis. Consult a healthcare professional if symptoms persist.

            ---

            User Query:
            {query}

            Collected Insights:
            {combined}
            """

        res = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return res.text.strip()

    # ---------------- MAIN PIPELINE ----------------
    def run(self, user_query: str, graph_output: dict):

        graph_nodes = graph_output.get("nodes", [])

        # Step 1: Generate questions
        questions = self._stepback_questions(user_query, graph_nodes)

        if not questions:
            return "Could not generate follow-up questions."

        sub_answers = []

        # Step 2: Loop pipeline
        for q in questions:
            docs = self._vector_retrieve(q)

            if not docs:
                continue

            ans = self._answer_subquestion(q, docs)
            sub_answers.append(ans)

        if not sub_answers:
            return "No useful information found."

        # Step 3: Final answer
        final = self._summarize(user_query, sub_answers)

        return final