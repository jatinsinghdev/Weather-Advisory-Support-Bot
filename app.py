import streamlit as st
import os
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Weather Advisory Bot", layout="wide")
st.title("Weather-Advisory Support Bot")

with st.sidebar:
    st.header("Settings")
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        st.success("✅ API Key securely loaded from environment")
        api_key = env_key
    else:
        api_key = st.text_input("Groq API Key (If not in .env)", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
    try:
        with open("sops.json", "r") as f:
            sops = json.load(f)
        st.success(f"Active policies loaded: {len(sops)}")
    except Exception:
        st.error("Failed to load sops.json.")
        
    if "weather_drawer" in st.session_state:
        st.header("Live Weather Telemetry")
        st.json(st.session_state.weather_drawer)

if not os.getenv("GROQ_API_KEY"):
    st.warning("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.write(msg.content)

prompt = st.chat_input("Ask about an outdoor activity (e.g., 'Is it safe to cycle in Bhopal today?')")
if prompt:
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.write(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing rules & live weather..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {"messages": [user_msg]}
            
            res = st.session_state.graph.invoke(inputs, config=config)
            
            final_msg = res["messages"][-1].content
            st.write(final_msg)
            st.session_state.messages.append(AIMessage(content=final_msg))
            
            if "weather_data" in res and res["weather_data"]:
                st.session_state.weather_drawer = res["weather_data"]
                st.rerun()
