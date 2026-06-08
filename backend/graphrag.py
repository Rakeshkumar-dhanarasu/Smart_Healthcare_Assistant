class GraphRetriever:
    def __init__(self, api_key, prompt_path):
        from neo4j import GraphDatabase
        from google import genai
        import os
        from dotenv import load_dotenv

        self.driver = GraphDatabase.driver(uri=os.getenv("NEO4J_URI"),auth=(os.getenv("NEO4J_USER"),os.getenv("NEO4J_PASS")))
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()

    def _generate_cypher(self, query: str) -> str:
        prompt = self.prompt_template + query

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text.strip()

    def _run_cypher(self, cypher: str) -> list:
        with self.driver.session() as session:
            result = session.run(cypher)
            return [record.data() for record in result]

    def get_context(self, query: str) -> dict:
        try:
            cypher = self._generate_cypher(query)
            data = self._run_cypher(cypher)

            return {
                "cypher": cypher,
                "data": data,
                "query": query
            }

        except Exception as e:
            return {
                "error": str(e),
                "data": []
            }