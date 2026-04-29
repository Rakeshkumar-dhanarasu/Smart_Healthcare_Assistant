from paddleocr import PaddleOCR
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

# --- Initialize models once ---
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = FAISS.load_local(
    "C:/Users/Rakesh Kumar/VSCode/Capstone/image_classification/allopathy_drugs_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

client = genai.Client(api_key="AIzaSyAWCxPmMaTE_U2LVAHgx5TU2cgCixHz90g")
MODEL = "gemini-2.5-flash-lite"

# --- Define main pipeline function ---
def ocr_rag_pipeline(image_path, query, k=4, allowed_types=None):
    """
    Input:
        image_path : str : path to the medicine image
        query : str : the question/query you want to ask
        k : int : number of filtered results from RAG
        allowed_types : list : types of documents to consider from DB
    Output:
        answer : str : generated answer from RAG + GenAI
    """
    
    # Step 1: OCR
    ocr_result = ocr.predict(image_path)
    extracted_text = " ".join([line[1][0] for line in ocr_result[0]])
    
    # Step 2: Retrieval function
    def retrieve(query, k=k, allowed_types=allowed_types):
        results = db.max_marginal_relevance_search(query, k=10)
        filtered = []
        for r in results:
            t = r.metadata.get("type")
            if allowed_types and t not in allowed_types:
                continue
            filtered.append({
                "text": r.page_content,
                "type": t,
                "entity": r.metadata.get("entity")
            })
            if len(filtered) == k:
                break
        return filtered
    
    docs = retrieve(extracted_text)
    
    # Step 3: Generate answer
    context = "\n\n".join([f"{d['type']} ({d['entity']}): {d['text']}" for d in docs])
    prompt = f"""
Answer the query using the context below.

Rules:
- Describe only about the medicine
- No extra hallucination

Context:
{context}

Query:
{query}
"""
    res = client.models.generate_content(model=MODEL, contents=prompt)
    
    return res.text

# --- Testing ---
image_path = "C:/Users/Rakesh Kumar/VSCode/Capstone/datasets/medical_drug_images/Amoxicillin/cropped_words/word_19.jpg"
query = "What are the uses of this medicine?"
answer = ocr_rag_pipeline(image_path, query, allowed_types=["medicine", "drug"])
print("🧠 Answer:\n", answer)