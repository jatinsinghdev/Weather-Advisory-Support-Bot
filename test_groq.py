import os
import traceback
from dotenv import load_dotenv
load_dotenv()
from graph import get_llm, IntentExtraction
from langchain_core.messages import SystemMessage, HumanMessage

def test_extract():
    llm = get_llm()
    structured_llm = llm.with_structured_output(IntentExtraction)
    sys_prompt = SystemMessage(content="You are an intent extractor.")
    messages = [sys_prompt, HumanMessage(content="Is it safe to cycle today in Bhopal?")]
    try:
        response = structured_llm.invoke(messages)
        print("Success:", response)
    except Exception as e:
        print("Error encountered:")
        traceback.print_exc()

if __name__ == "__main__":
    test_extract()
