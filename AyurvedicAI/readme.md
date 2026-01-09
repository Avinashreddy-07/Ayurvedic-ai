# Ayurvedic AI

**Ayurvedic AI** is a Retrieval-Augmented Generation (RAG) application dedicated to sharing the principles of Ayurveda. It is grounded exclusively in the classical texts interpreted by Maharishi Vāgbhaṭa (popularized by Rajiv Dixit Ji), ensuring that answers are authoritative and traceable to specific Sutras.

This project uses **LangGraph** for conversational flow, **Chroma** for vector storage, **Google Gemini** for reasoning through **Open-Router**, and **Streamlit** for the user interface.

## 🎥 Demo

![Ayurvedic AI Demo](assets/demo.gif)

> *The demo above shows the bot answering a query about water consumption and providing a safety disclaimer.*

---

## ✨ Features

- **Authentic Ayurvedic Knowledge:** Answers are strictly grounded in the provided `data.csv` containing Sutras, Meanings, and Analysis.
- **Sutra-Based Citations:** Every response cites the specific Ayurvedic source/Sutra used to generate the advice.
- **Safety Guardrails:** Automatically detects "CRITICAL" advice levels (e.g., contraindications) and appends mandatory medical disclaimers.
- **Conversational Memory:** Maintains chat history during the session, allowing users to have contextual conversations.
- **Modern RAG Pipeline:**
  - **Vector Search:** Uses Hugging-Face `intfloat/e5-base-v2` model and Chroma for semantic retrieval.
  - **Agentic Flow:** Built with **LangGraph** to manage state and tool execution.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini (gemini-2.0-flash)- through **Open-router**
- **Orchestration:** LangChain & LangGraph
- **Vector Database:** Chroma
- **Embeddings:** Hugging Face Embeddings

---

## 📂 Project Structure

```bash
AyurvedicAI/
├── app.py                     # Main Streamlit application entry point
├── config.py                  # Configuration and environment variable management
├── config.toml                # Streamlit configuration
├── graph.py                   # LangGraph state machine and node definitions
├── main.py                    # CLI version of the bot (for testing)
├── prompts.py                 # System prompts and persona definitions
├── retrieval.py               # Vector store retrieval logic (Chroma)
├── tools.py                   # Tool definitions (Ayurvedic Source retrieval)
├── requirements.txt           # Python dependencies
├── chroma_db/                 # Chroma vector database storage
├── data/
│   └── data.csv               # The knowledge base (Sutras and analysis)
├── indexer/
│   └── chroma_indexer.py      # Script to embed and upload data to Chroma
└── assets/
    └── demo.gif               # Demo GIF for showcasing the application
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yadavadarsh55/VagbhataAI.git
cd AYURVEDICAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory with the following keys:

```ini
GOOGLE_API_KEY=your_google_api_key
```

### 4. Initialize the Knowledge Base

Run the indexer script to generate embeddings from `data.csv` and upload them to Chroma.

```bash
python indexer/chroma_indexer.py
```

### 5. Run the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

---

## 🧠 Usage

1. **Ask a Question:** Type a query like *"When should I drink water?"* or *"What are the rules for eating?"*.
2. **View Sources:** The bot will retrieve relevant Sutras from the database and interpret them.
3. **Check Safety:** If your query involves critical health advice, the bot will provide a disclaimer.
4. **History:** Access your previous conversation threads from the sidebar.

---

## ⚠️ Disclaimer

This AI assistant provides information based on Ayurvedic texts. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical conditions.
