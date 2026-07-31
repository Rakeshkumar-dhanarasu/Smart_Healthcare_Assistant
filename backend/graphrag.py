class GraphRetriever:
    def __init__(self, api_key):
        from neo4j import GraphDatabase
        from google import genai
        import os
        from dotenv import load_dotenv
        from pathlib import Path

        load_dotenv()

        self.driver = GraphDatabase.driver(
            uri=os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASS"))
        )

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.5-flash"

        BASE_DIR = Path(__file__).resolve().parent
        prompt_path = (BASE_DIR / os.getenv("GRAPH_RETRIEVAL_PROMPT_PATH")).resolve()

        with open(prompt_path, "r") as file:
            self.content = file.read()

    def _generate_cypher(self, query):
        prompt = self.content + query

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return (response.text or "").strip()

    def _run_cypher(self, cypher):
        with self.driver.session() as session:
            result = session.run(cypher)
            return [record.data() for record in result]

    def run(self, user_query):
        cypher = self._generate_cypher(user_query)

        if not cypher or "MATCH" not in cypher:
            return ["Invalid Cypher generated"]

        try:
            results = self._run_cypher(cypher)

            extracted = []
            for r in results:
                for value in r.values():
                    extracted.append(str(value))

            return {
                "nodes": list(set(extracted))
            }

        except Exception as e:
            return [f"Graph Error: {str(e)}"]

    def close(self):
        self.driver.close()