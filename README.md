# AI Logistics Quoting Engine 🚛⚡

An enterprise-grade API that completely automates B2B freight quoting. By combining structured data extraction, RAG (Retrieval-Augmented Generation), and autonomous AI agents, this system reduces a 45-minute manual quoting process to under 14 seconds.

## 🏗️ The Architecture
1. **Extraction (FastAPI + Groq):** Ingests messy, unstructured client emails and strictly parses out origin, destination, weight, pallet count, and urgency into a clean JSON schema using a zero-temperature LLM.
2. **Policy Retrieval (ChromaDB):** Acts as the company brain, instantly pulling the specific internal pricing matrix, heavy-weight surcharges, and fragile-goods rules from a local vector database.
3. **Live Web Search (Serper API):** Equips the AI with real-time internet access to calculate the exact driving distance between the origin and destination cities.
4. **Autonomous Workforce (CrewAI):** * **The Estimator Agent:** Executes the complex math, calculating distance multipliers, applying pallet fees, and checking logic for weight surcharges.
   * **The Sales Agent:** Takes the Estimator's final math and drafts a highly professional, ready-to-send B2B email to win the bid.

## 💻 Tech Stack
* **Backend:** FastAPI, Python
* **LLM Engine:** Groq (Llama-3.1-8b-instant)
* **Agent Framework:** CrewAI
* **Vector Database:** ChromaDB & LangChain
* **External Tooling:** Serper API (Google Search)

## 🚀 How It Works
Send a POST request to `/api/quote-freight` with a messy client email. The API will autonomously extract the data, calculate the exact route distance, apply company pricing rules, and return a finalized professional sales quote.