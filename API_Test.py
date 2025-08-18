import os, google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

print(genai.GenerativeModel("gemini-1.5-flash").generate_content("ping").candidates[0].content.parts[0].text[:50])
