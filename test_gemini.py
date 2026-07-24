from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

response = llm.invoke("Say hello in one sentence.")

print(response.content)