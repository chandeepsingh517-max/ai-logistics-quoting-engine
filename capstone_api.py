import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- BUG FIX FOR CREWAI ---
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

# --- SETUP & GLOBAL VARIABLES ---
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = LLM(model="groq/llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
search_tool = SerperDevTool()

app = FastAPI(title="Logistics Quoting API")

embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
logistics_db = Chroma(persist_directory="./logistics_db", embedding_function=embedding_function)

class FreightEmail(BaseModel):
    email_text: str

@app.post("/api/quote-freight")
def generate_freight_quote(request: FreightEmail):
    
    # 1. PHASE 1: DATA EXTRACTION (Groq)
    print("Extracting email data...")
    system_prompt = """
    You are a precise data extraction API for a logistics company.
    Analyze the inbound email and extract the freight details into strict JSON format.
    CRITICAL RULES: Output ONLY raw JSON. No markdown ticks, no conversational text.
    REQUIRED SCHEMA:
    {
        "origin_city": "city, state",
        "destination_city": "city, state",
        "total_pallets": integer,
        "total_weight_kg": integer,
        "is_fragile": boolean,
        "timeline": "Expedited" or "Standard"
    }
    """
    extraction_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.email_text}
        ],
        temperature=0.0
    )
    extracted_json = extraction_response.choices[0].message.content

    # 2. PHASE 2: POLICY RETRIEVAL (RAG)
    print("Retrieving pricing matrix...")
    docs = logistics_db.similarity_search("freight pricing base rate heavy pallet fragile surcharge", k=1)
    pricing_rules = docs[0].page_content if docs else "No rules found."

    # 3. PHASE 3: CREWAI AUTONOMOUS WORKFORCE
    print("Deploying AI Workforce...")
    
    estimator = Agent(
        role='Senior Freight Estimator',
        goal='Calculate accurate freight costs using company rules and live driving distances.',
        backstory='You are a mathematical genius in logistics. You use the search tool to find driving distances in kilometers, then strictly apply company pricing rules to calculate totals.',
        tools=[search_tool],
        verbose=True,
        llm=llm
    )

    sales_exec = Agent(
        role='Logistics Sales Executive',
        goal='Draft professional quote emails to clients.',
        backstory='You write clean, highly professional B2B sales emails. You take the math from the Estimator and format it nicely for the client.',
        verbose=True,
        llm=llm
    )

    task_calculate = Task(
        description=f"""
        1. Look at this extracted shipment data: {extracted_json}
        2. Use the search tool to find the live driving distance in kilometers between the origin and destination.
        3. Use these exact company rules to calculate the final price in INR: {pricing_rules}
        Break down your math step-by-step so the sales exec can see the fees.
        """,
        expected_output="A step-by-step mathematical breakdown of the final price in INR.",
        agent=estimator
    )

    task_draft_email = Task(
        description="Write a professional email to the client providing the quote. Include the breakdown of costs (Base rate, pallet fees, surcharges if applicable). Be polite and urgent.",
        expected_output="A finalized professional email string ready to send.",
        agent=sales_exec
    )

    crew = Crew(agents=[estimator, sales_exec], tasks=[task_calculate, task_draft_email], verbose=True)
    final_email = crew.kickoff()

    # 4. RETURN TO CLIENT
    return {
        "status": "success", 
        "extracted_data": json.loads(extracted_json), # Converts string to actual JSON object
        "final_quote_email": str(final_email)
    }