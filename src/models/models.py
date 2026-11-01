import os
from dotenv import load_dotenv
# pyrefly: ignore[missing-import]
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

embd = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
