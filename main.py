from typing import TypedDict, Optional

from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langfuse import observe ,get_client
from dotenv import load_dotenv
import os

load_dotenv()
def get_database(db_path: str = "productiondb.db") -> SQLDatabase:
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")

llm = ChatOllama(model="deepseek-r1:1.5b")
class AgentState(TypedDict):
    question: str
    tables: Optional[str]
    schema: Optional[str]
    sql_query: Optional[str]
    sql_result: Optional[str]
    error: Optional[str]
    retries: int
    max_retries: int
    final_answer: Optional[str]

@observe
def list_tables_node(state: AgentState) -> AgentState:
    db = get_database()
    tables = db.get_usable_table_names()
    return {
        **state,
        "tables": ", ".join(tables),
    }
@observe
def get_schema_node(state: AgentState) -> AgentState:
    db = get_database()
    schema_text = db.get_table_info()
    return {
        **state,
        "schema": schema_text,
    }
@observe(as_type="generation")
def generate_sql_node(state: AgentState) -> AgentState:


    prompt = f"""
You are a careful SQL assistant.

Your job is to write a valid SQLite query based on the user question.

Rules:
- Use proper SQLite syntax.
- Return only the SQL query.
- Use ONLY the tables listed below.
- Use ONLY columns that exist in the schema.
- Do not include markdown.
- Do not include explanation.
- If there was a previous error, fix the SQL using that error.

User question:
{state['question']}

Available tables:
{state['tables']}

Schema:
{state['schema']}

Previous error:
{state['error']}
"""

    response = llm.invoke(prompt)
    sql_query = response.content.strip()

    return {
        **state,
        "sql_query": sql_query,
    }
@observe
def execute_sql_node(state: AgentState) -> AgentState:
    db = get_database()

    try:
        result = db.run(state["sql_query"])
        return {
            **state,
            "sql_result": str(result),
            "error": None,
        }
    except Exception as e:
        return {
            **state,
            "sql_result": None,
            "error": str(e),
            "retries": state["retries"] + 1,
        }

def route_after_execution(state: AgentState) -> str:
    if state["error"] is None:
        return "generate_answer"

    if state["retries"] < state["max_retries"]:
        return "generate_sql"

    return "fail"
@observe(as_type="generation")
def generate_answer_node(state: AgentState) -> AgentState:


    prompt = f"""
You are a helpful data assistant.

Given the user question, the SQL query, and the SQL result,
write a clear natural language answer.

User question:
{state['question']}

SQL used:
{state['sql_query']}

SQL result:
{state['sql_result']}

Write a short, accurate, easy-to-understand response.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content.strip(),
    }

def fail_node(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": (
            f"I could not complete the SQL query after {state['max_retries']} retries. "
            f"Last error: {state['error']}"
        ),
    }

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("list_tables", list_tables_node)
    graph.add_node("get_schema", get_schema_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("fail", fail_node)

    graph.add_edge(START, "list_tables")
    graph.add_edge("list_tables", "get_schema")
    graph.add_edge("get_schema", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")

    graph.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "generate_answer": "generate_answer",
            "generate_sql": "generate_sql",
            "fail": "fail",
        },
    )

    graph.add_edge("generate_answer", END)
    graph.add_edge("fail", END)

    return graph.compile()


def main():

    agent = build_agent()

    question = "give me list of well names?"

    initial_state: AgentState = {
        "question": question,
        "tables": None,
        "schema": None,
        "sql_query": None,
        "sql_result": None,
        "error": None,
        "retries": 0,
        "max_retries": 3,
        "final_answer": None,
    }

    result = agent.invoke(initial_state)

  ##  print("\nQUESTION:")
   ## print(result["question"])
    print("\nSQL QUERY:")
    print(result["sql_query"])
  ##  print("\nSQL RESULT:")
   ## print(result["sql_result"])
  ##  print("\nFINAL ANSWER:")
  ##  print(result["final_answer"])


if __name__ == "__main__":
    main()