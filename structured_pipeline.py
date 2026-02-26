import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def run_structured(file, question):

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    api_key = os.getenv("GROQ_API_KEY")
    model = ChatGroq(
        model_name = "llama-3.3-70b-versatile",
        api_key = api_key
    )

    agent = create_pandas_dataframe_agent(
        model, 
        df, 
        verbose=True, 
        allow_dangerous_code=True
    )

    full_query = (
        f"Question: {question}\n\n"
        "IMPORTANT: You MUST search the entire dataframe. Do not rely on any sample head() you might see. "
        "Use Python to find the answer. Trust only the results of your code execution. "
        "generate a response in a well formatted manner"
        "Ensure all matching entries are included in your final response."
    )

    return agent.run(full_query)