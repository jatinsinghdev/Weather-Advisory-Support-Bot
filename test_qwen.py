import os
import traceback
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

class IntentExtraction(BaseModel):
    activity: str = Field(description="The core generalized outdoor activity.")
    location: str = Field(description="The city or location.")

def test_extract():
    llm = ChatGroq(model_name="qwen/qwen3.8-27b")
    structured_llm = llm.with_structured_output(IntentExtraction)
    try:
        response = structured_llm.invoke("I want to go biking in sf")
        print("Success:", response)
    except Exception as e:
        print("Error encountered:")
        traceback.print_exc()

if __name__ == "__main__":
    test_extract()
