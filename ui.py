import streamlit as st
from main import build_agent,AgentState
from langfuse import observe ,get_client
from dotenv import load_dotenv
import os

load_dotenv()

st.title("AI Assistant")
user_input = st.text_input("Enter your question:")



if user_input:
   with st.spinner(text='Model is thinking...'):

    agent = build_agent()

   initial_state: AgentState = {
        "question": user_input,
        "tables": None,
        "schema": None,
        "sql_query": None,
        "sql_result": None,
        "error": None,
        "retries": 0,
        "max_retries": 3,
        "final_answer":None
   }

   result = agent.invoke(initial_state)
   st.markdown("**Assistant**")
   st.markdown(result["final_answer"])
   get_client().flush()