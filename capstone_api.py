import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# --- BUG FIX FOR CREWAI ---
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

# --- SETUP & GLOBAL VARIABLES ---
load_dotenv()

# Universal Client & LLM for both Routing and CrewAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llm = LLM(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0
)

app = FastAPI(title="Logistics Autonomous Inbox API")

class IncomingEmail(BaseModel):
    email_text: str

# --- THE PYTHON CALCULATOR TOOL ---
@tool("Freight Calculator")
def calculate_freight(weight_kg: int, distance_km: int, is_fragile: bool, timeline: str) -> str:
    """Use this to calculate ALL freight costs. Pass weight, distance, fragile status, and timeline."""
    base_rate = 0.1 
    total = weight_kg * distance_km * base_rate
    
    if is_fragile:
        total += 1000
    if timeline.lower() == "expedited":
        total += 2000
        
    return f"The final exact cost is ₹{total}"

# --- HELPER FUNCTION: THE ROUTER ---
def classify_intent(email_body: str) -> str:
    """Fast, ultra-cheap LLM call to classify the email intent."""
    print("🚦 Routing Email...")
    system_prompt = (
        "You are an email routing bot. Read the email and output EXACTLY ONE WORD from this list: "
        "'QUOTE' (if they want a price/estimate), "
        "'TRACKING' (if they are asking where their shipment is), "
        "'SUPPORT' (for anything else)."
    )
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": email_body}
        ],
        temperature=0.0,
        max_tokens=10
    )
    
    intent = response.choices[0].message.content.strip().upper()
    return intent

# --- MAIN ENDPOINT ---
@app.post("/api/process-email")
def process_incoming_email(request: IncomingEmail):
    
    # STEP 1: ROUTE THE TRAFFIC
    intent = classify_intent(request.email_text)
    print(f"📩 Intent Detected: {intent}")
    
    # -----------------------------------------
    # ROUTE A: NON-QUOTE EMAILS (Bypass CrewAI to save tokens)
    # -----------------------------------------
    if intent == "TRACKING":
        return {
            "status": "success", 
            "intent": intent, 
            "action": "Triggered Tracking Lookup Webhook",
            "message": "This would trigger a fast DB lookup for the client's tracking number."
        }
        
    elif intent == "SUPPORT":
        return {
            "status": "success", 
            "intent": intent, 
            "action": "Forwarded to Human Agent",
            "message": "This email was flagged for a human to read. AI bypassed."
        }

    # -----------------------------------------
    # ROUTE B: QUOTE EMAILS (Deploy the Heavyweight CrewAI Pipeline)
    # -----------------------------------------
    elif intent == "QUOTE":
        print("⚙️ Deploying Estimator Pipeline...")
        
        # 1. Extract JSON
        extract_prompt = "Extract to strict JSON: origin_city, destination_city, total_pallets, total_weight_kg, is_fragile, timeline. Output raw JSON only."
        extraction_response = openai_client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": extract_prompt},
                {"role": "user", "content": request.email_text}
            ],
            temperature=0.0
        )
        
        raw_content = extraction_response.choices[0].message.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        
        extracted_json = raw_content

        # 2. Deploy CrewAI Workforce
        estimator = Agent(
            role='Estimator',
            goal='Calculate the freight price using the Freight Calculator tool.',
            backstory='You are a precise estimator. You ALWAYS use the Freight Calculator tool.',
            tools=[calculate_freight],
            verbose=False, 
            max_iter=2,    
            llm=llm 
        )

        sales = Agent(
            role='Sales',
            goal='Draft a short quote email using ONLY the exact final price provided.',
            backstory='You are a strict, no-nonsense sales executive. You never invent fees, breakdowns, or numbers. You only quote the single final price given to you by the Estimator.',
            verbose=False, 
            max_iter=2,
            llm=llm 
        )

        task_calc = Task(
            description=f"Data: {extracted_json}. Assume the driving distance is exactly 250 km. MANDATORY: Pass the weight, distance (250), fragile status, and timeline into the 'Freight Calculator' tool.",
            expected_output="The final price.",
            agent=estimator
        )

        task_email = Task(
            description="Write a very short client email providing the final price quote. CRITICAL INSTRUCTION: You MUST use the exact final cost string provided by the Estimator. Do NOT break the cost down. Do NOT invent pallet fees, base rates, or surcharges. Just provide the final total.",
            expected_output="A brief finalized email containing the exact calculated price and no other numbers.",
            agent=sales
        )

        crew = Crew(agents=[estimator, sales], tasks=[task_calc, task_email], verbose=False)
        final_email = crew.kickoff()

        # 3. Return the fully processed quote
        return {
            "status": "success", 
            "intent": intent,
            "extracted_data": json.loads(extracted_json), 
            "final_quote_email": str(final_email)
        }