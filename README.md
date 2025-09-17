### Clinical Trial & Research Navigator

**Status:** In Progress

**Objective:** To build a Retrieval-Augmented Generation (RAG) system that allows researchers and clinicians to ask natural language questions and receive accurate, context-aware answers based on a vast corpus of biomedical literature and clinical trial data.

**Core Features:**
* Data ingestion from PubMed and ClinicalTrials.gov.
* Advanced text chunking and embedding using domain-specific models (BioBERT).
* Efficient information retrieval using a vector database (Chroma DB).
* Answer generation with source citation using modern LLMs (Google Gemini / Ollama).

**Proposed Tech Stack:**
* **Language:** Python
* **Core Framework:** LangChain
* **Data Sources:** PubMed (via BioPython), ClinicalTrials.gov API
* **Embedding Model:** GIST-BioBERT-v1 via Hugging Face Sentence Transformers
* **Vector Database:** Chroma DB
* **LLM:** Google Gemini API (Free Tier) / Ollama (Llama 3, Phi-3)
* **UI:** Streamlit

**Project Plan:**
* **Phase 1:** Setup and Data Acquisition
* **Phase 2:** Building the Core RAG Pipeline
* **Phase 3:** Building the User Interface
* **Phase 4:** Refinement, Evaluation, and Source Citation
