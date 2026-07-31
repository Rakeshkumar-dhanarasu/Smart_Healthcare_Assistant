import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load .env from root
load_dotenv()

api_key = os.getenv("GENAI_API_KEY")

print("API KEY:", repr(api_key))

genai.configure(api_key=api_key)

print("\nAvailable Models:\n")

for model in genai.list_models():
    print(model.name, "->", model.supported_generation_methods)