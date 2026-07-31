from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from google import genai
from neo4j import GraphDatabase

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
FAISS_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "datasets",
    "vector_db",
    "information_enricher_db"
)
MODEL = "gemini-2.5-flash-lite"
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

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

# ---------------- CYPHER GENERATOR (DYNAMIC) ----------------
with open("../prompts/graph_retrieval.txt", "r") as file:
    content = file.read()

def generate_cypher(query):
    prompt = content + query
    
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    
    return response.text.strip()

# ---------------- CYPHER PASSER ----------------
def run_cypher(cypher):
    with driver.session() as session:
        result = session.run(cypher)
        return [record.data() for record in result]

# ---------------- GRAPH RETRIEVAL (DYNAMIC) ----------------
def graph_retrieve_dynamic(user_query):

    cypher = generate_cypher(user_query)

    try:
        results = run_cypher(cypher)

        extracted = []
        for r in results:
            for value in r.values():
                if isinstance(value, str):
                    extracted.append(value)

        return list(set(extracted))

    except Exception as e:
        print("❌ Graph Error:", e)
        return []


# ---------------- STEP-BACK ----------------
with open("../prompts/stepback_questioner.txt", "r") as file:
    content2 = file.read()

def generate_stepback_questions(query, graph_nodes):

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

    prompt = content2 + prompt    

    res = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return [q.strip() for q in res.text.split("\n") if q.strip()]


# ---------------- VECTOR RETRIEVAL ----------------
def vector_retrieve(query):

    docs = db.max_marginal_relevance_search(query, k=4)

    return [
        {
            "text": d.page_content,
            "entity": d.metadata.get("entity"),
            "type": d.metadata.get("type")
        }
        for d in docs
    ]


# ---------------- SUB-QUESTION ANSWERING ----------------
def answer_subquestion(question, docs):

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

    res = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return res.text.strip()


# ---------------- FINAL SUMMARIZATION ----------------
def summarize_answers(query, sub_answers):

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

    res = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return res.text


# ---------------- MAIN HYBRID RAG ----------------
def hybrid_rag(query):

    print("\n🔍 USER QUERY:", query)

    # Step 1: Graph Retrieval
    graph_nodes = graph_retrieve_dynamic(query)
    print("\n📊 GRAPH NODES:", graph_nodes)

    # Step 2: Step-back
    stepback_qs = generate_stepback_questions(query, graph_nodes)
    print("\n🧠 STEP-BACK QUESTIONS:")
    for q in stepback_qs:
        print("-", q)

    all_answers = []
    entity_scores = {}

    # Step 3: Process each question
    for q in stepback_qs:

        docs = vector_retrieve(q)

        # Track entity scores
        for d in docs:
            if d["entity"]:
                entity_scores[d["entity"]] = entity_scores.get(d["entity"], 0) + 1

        # Answer sub-question
        ans = answer_subquestion(q, docs)

        print(f"\n➡️ {q}\n{ans}")

        all_answers.append(ans)

    # Step 4: Final summary
    final_answer = summarize_answers(query, all_answers)

    ranked_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "answer": final_answer,
        "graph_nodes": graph_nodes,
        "ranked_entities": ranked_entities[:5]
    }


# ---------------- TEST ----------------
if __name__ == "__main__":
    result = hybrid_rag("What are the medical benefits of Tulsi")

    print("\n🧾 FINAL ANSWER:\n", result["answer"])
    print("\n🏆 TOP ENTITIES:\n", result["ranked_entities"])