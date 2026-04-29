from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from google import genai
from neo4j import GraphDatabase

import os

# ---------------- CONFIG ----------------
FAISS_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "datasets",
    "vector_database",
    "raptor_index_v1"
)
MODEL = "gemini-2.5-flash-lite"
GENAI_API_KEY = "AIzaSyAWCxPmMaTE_U2LVAHgx5TU2cgCixHz90g"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Goppanmavane@2"

# ---------------- INIT ----------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

db = FAISS.load_local(
    FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

client = genai.Client(api_key=GENAI_API_KEY)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS)
)

# ---------------- STEP-BACK ----------------
def generate_stepback_questions(query):

    prompt = f"""
Break the query into 3-5 smaller questions.

Focus:
- symptoms
- herbs
- nutrients
- causes

Query:
{query}
"""

    res = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return [q.strip() for q in res.text.split("\n") if q.strip()]


# ---------------- GRAPH RETRIEVAL ----------------
def graph_retrieve(tx, user_query):

    cypher = """
    MATCH (h:FoodnHerb)-[:HELPS_WITH]->(s:Symptom)
    WHERE toLower(s.name) CONTAINS toLower($q)
    RETURN DISTINCT h.name LIMIT 10
    """

    result = tx.run(cypher, q=user_query)

    return [r["h.name"] for r in result]


# ---------------- VECTOR RETRIEVAL ----------------
def vector_retrieve(query):

    docs = db.max_marginal_relevance_search(query, k=5)

    return [
        {
            "text": d.page_content,
            "entity": d.metadata.get("entity"),
            "type": d.metadata.get("type")
        }
        for d in docs
    ]


# ---------------- SCORING ----------------
def score_results(graph_results, vector_docs):

    scores = {}

    for herb in graph_results:
        scores[herb] = scores.get(herb, 0) + 3

    for doc in vector_docs:
        entity = doc["entity"]
        if entity:
            scores[entity] = scores.get(entity, 0) + 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked


# ---------------- CONTEXT BUILD ----------------
def build_context(stepback_qs, graph_data, vector_data):

    return f"""
STEP-BACK:
{stepback_qs}

GRAPH:
{graph_data}

VECTOR:
{[v['text'][:200] for v in vector_data]}
"""

# ---------------- FINAL ANSWER ----------------
def generate_answer(query, context):

    prompt = f"""
You are a helpful health assistant.

STRICT RULES (MUST FOLLOW):
1. DO NOT diagnose any disease under any circumstances.
2. DO NOT mention or infer the root cause of symptoms.
3. ONLY provide supportive care and natural remedies.
4. ALWAYS structure the response in clean, readable format.
5. DO NOT use asterisks (*), emojis, or markdown symbols.
6. ALWAYS use numbered lists where applicable.

OUTPUT FORMAT (STRICT):
Response:
<Short 1-2 line summary>

Suggested Remedies:
1. Remedy one
2. Remedy two
3. Remedy three

Precautions:
1. Precaution one
2. Precaution two

Important Note:
This is not a medical diagnosis. Consult a healthcare professional if symptoms persist.

---

Context:
{context}

Query:
{query}
"""

    res = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return res.text


# ================= MAIN FUNCTION =================
def get_answer(query):

    # Step 1: Step-back
    stepback_qs = generate_stepback_questions(query)

    all_graph = []
    all_vector = []

    # Step 2: Retrieve for each sub-question
    with driver.session() as session:
        for q in stepback_qs:

            g = session.execute_read(graph_retrieve, q)
            v = vector_retrieve(q)

            all_graph.extend(g)
            all_vector.extend(v)

    # Step 3: Deduplicate
    all_graph = list(set(all_graph))

    # Step 4: Rank
    ranked = score_results(all_graph, all_vector)

    # Step 5: Build context
    context = build_context(stepback_qs, ranked[:5], all_vector[:5])

    # Step 6: Final answer
    answer = generate_answer(query, context)

    return {
        "answer": answer,
        "ranked_entities": ranked[:5]
    }
