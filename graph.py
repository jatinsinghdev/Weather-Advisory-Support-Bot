import os
import json
import requests
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent_activity: str
    intent_location: str
    lat: float
    lon: float
    weather_data: dict
    matched_sops: list[dict]
    error_msg: str
    should_fallback: bool

class IntentExtraction(BaseModel):
    activity: str = Field(description="The core generalized outdoor activity the user is asking about (e.g., 'cycling', 'picnic', 'driving', 'hiking'). Leave empty if none detected.")
    location: str = Field(description="The city or location mentioned for the activity. Leave empty if none detected.")

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(model_name="qwen/qwen3.8-27b", temperature=0, groq_api_key=api_key)

def extract_intent(state: AgentState) -> dict:
    llm = get_llm()
    structured_llm = llm.with_structured_output(IntentExtraction)
    sys_prompt = SystemMessage(content="You are an intent extractor. Read the history. Extract the outdoor activity and location. Normalize paraphrased activities (e.g., 'pedal-powered' -> 'cycling').")
    
    try:
        response = structured_llm.invoke([sys_prompt] + state["messages"])
    except Exception as e:
        return {"should_fallback": True, "error_msg": f"System error during intent extraction: {str(e)}"}
    
    loc = response.location if response.location else state.get("intent_location", "")
    act = response.activity if response.activity else state.get("intent_activity", "")
    
    if not loc:
        return {"should_fallback": True, "error_msg": "I need a location to check the weather. Where are you planning to do this?"}
        
    return {"intent_activity": act, "intent_location": loc, "should_fallback": False, "error_msg": ""}

def geocode_location(state: AgentState) -> dict:
    loc = state.get("intent_location")
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={loc}&count=1"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "results" not in data or len(data["results"]) == 0:
            return {"should_fallback": True, "error_msg": f"Unable to retrieve weather data for {loc}. The location could not be found."}
        return {"lat": data["results"][0]["latitude"], "lon": data["results"][0]["longitude"]}
    except Exception:
        return {"should_fallback": True, "error_msg": f"Unable to retrieve weather data for {loc}. Geocoding service error."}

def fetch_weather(state: AgentState) -> dict:
    lat, lon = state.get("lat"), state.get("lon")
    try:
        # Dynamically scan sops.json to know what to request
        with open("sops.json", "r") as f: sops = json.load(f)
        variables = set(["temperature_2m", "wind_speed_10m", "precipitation", "uv_index"])
        for sop in sops:
            for k in sop.get("conditions", {}).keys():
                if k.endswith("_min"): variables.add(k[:-4])
                elif k.endswith("_max"): variables.add(k[:-4])
                elif k.endswith("_eq"): variables.add(k[:-3])
        
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current={','.join(variables)}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "current" not in data:
            return {"should_fallback": True, "error_msg": "Weather data missing from API response."}
        return {"weather_data": data["current"]}
    except Exception:
        return {"should_fallback": True, "error_msg": "Unable to retrieve weather data for the requested location. Weather API failed."}

def match_sops(state: AgentState) -> dict:
    with open("sops.json", "r") as f: sops = json.load(f)
    
    activity = state.get("intent_activity", "").lower()
    last_msg = state["messages"][-1].content.lower() if state["messages"] else ""
    weather = state.get("weather_data", {})
    matched = []
    
    for sop in sops:
        patterns = [p.lower() for p in sop.get("activity_patterns", [])]
        activity_matches = any(p == "*" or p in activity or p in last_msg for p in patterns)
        if not activity_matches: continue
            
        cond = sop.get("conditions", {})
        operator = cond.get("operator", "AND")
        checks = []
        
        for k, v in cond.items():
            if k in ["operator", "fuzzy_composite"]: continue
            if k.endswith("_min") and k[:-4] in weather: checks.append(weather[k[:-4]] >= v)
            elif k.endswith("_max") and k[:-4] in weather: checks.append(weather[k[:-4]] <= v)
            elif k.endswith("_eq") and k[:-3] in weather: checks.append(weather[k[:-3]] == v)
                    
        if not checks and not cond.get("fuzzy_composite"): continue
            
        if cond.get("fuzzy_composite") and not checks: is_match = True
        elif operator == "OR": is_match = any(checks)
        else: is_match = all(checks)
            
        if is_match: matched.append(sop)
            
    severity_order = {"CRITICAL": 4, "HIGH": 3, "MODERATE": 2, "LOW": 1}
    matched.sort(key=lambda x: severity_order.get(x.get("severity", "LOW"), 0), reverse=True)
    return {"matched_sops": matched}

def generate_response(state: AgentState) -> dict:
    llm = get_llm()
    matched_sops, weather = state.get("matched_sops", []), state.get("weather_data", {})
    
    if not matched_sops:
        msg = f"Current weather: {json.dumps(weather)}\n\nWe do not currently have safety guidance for this activity under these conditions."
        return {"messages": [AIMessage(content=msg)]}
        
    sys_msg = SystemMessage(content=(
        "You are a strict Weather-Advisory Bot. Provide safety guidance using ONLY the provided Matched SOPs and exact provided Weather Data.\n\n"
        f"Weather Data: {json.dumps(weather)}\nMatched SOPs: {json.dumps(matched_sops)}\n\n"
        "Never invent safety guidance or thresholds. Report numerical data exactly as received. Always cite the 'citation' field of the applied SOP(s). "
        "List high severity SOPs first."
    ))
    
    res = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [AIMessage(content=res.content)]}

def fallback_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=state["error_msg"])]}

def route_eval(state: AgentState, next_node: str) -> str:
    return "fallback_node" if state.get("should_fallback") else next_node

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("extract_intent_and_location", extract_intent)
    workflow.add_node("geocode_location", geocode_location)
    workflow.add_node("fetch_weather", fetch_weather)
    workflow.add_node("match_sops", match_sops)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("fallback_node", fallback_node)
    
    workflow.add_edge(START, "extract_intent_and_location")
    workflow.add_conditional_edges("extract_intent_and_location", lambda s: route_eval(s, "geocode_location"))
    workflow.add_conditional_edges("geocode_location", lambda s: route_eval(s, "fetch_weather"))
    workflow.add_conditional_edges("fetch_weather", lambda s: route_eval(s, "match_sops"))
    workflow.add_edge("match_sops", "generate_response")
    workflow.add_edge("generate_response", END)
    workflow.add_edge("fallback_node", END)
    
    return workflow.compile(checkpointer=MemorySaver())
