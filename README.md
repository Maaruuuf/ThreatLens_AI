[![Live Demo](https://img.shields.io/badge/Live-Demo-green)](https://threatlensai-js8wyfwetjrjya42paybeo.streamlit.app/)

# 🛡️ ThreatLens AI — Cybersecurity RAG Assistant

ThreatLens AI is an intelligent cybersecurity assistant that combines **Machine Learning (IDS)**, **Vector Search (Pinecone)**, and **LLM reasoning (Groq - LLaMA 3.3)** to perform **automated incident analysis**.

It simulates a real-world **SOC (Security Operations Center)** workflow using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🌐 Live Demo

**Experience ThreatLens AI in action:**

👉 **[Launch the App](https://threatlensai-js8wyfwetjrjya42paybeo.streamlit.app/)**

---

###  What you can do in the demo:

*  Generate realistic network intrusion incident packets
*  Run full SOC-level AI analysis reports
*  Retrieve similar historical attack patterns
*  Explore detailed mitigation strategies (MITRE & NIST)
*  Interact via ChatGPT-style multi-session interface


---

##  Features

*  **ML-based Intrusion Detection (XGBoost)**
*  **Dynamic Incident Packet Generation**
*  **Semantic Retrieval using Pinecone**
*  **Mitigation Knowledge from MITRE & NIST**
*  **Confidence-aware Retrieval Strategy**
*  **Cross-encoder Reranking**
*  **LLM-based SOC Analysis Report**
*  **ChatGPT-style UI with Multi-Session Support**

---

## System Architecture

```
Network Traffic
      ↓
ML IDS Model (XGBoost)
      ↓
Prediction + Confidence
      ↓
Incident Packet Generation
      ↓
Embedding (BGE)
      ↓
Pinecone Vector Search
      ↓
Reranking (CrossEncoder)
      ↓
Context Building
      ↓
LLM (LLaMA 3.3 via Groq)
      ↓
SOC Incident Report
```

---

## 📂 Project Structure

```
ThreatLens_AI/
│
├── app.py                          # Streamlit UI
├── requirements.txt
├── README.md
├── .env.example
│
├── utils/
│   ├── model_utils.py              # ML model loading & prediction
│   ├── incident_utils.py           # Incident packet generation
│   ├── embedding_utils.py          # HuggingFace embedding client
│   ├── retrieval_utils.py          # Pinecone search + reranking
│   ├── llm_utils.py                # Groq LLM integration
│   └── results/
│       ├── xgb_model.pkl
│       └── label_mapping.json
│ 

    
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/ThreatLens_AI.git
cd ThreatLens_AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```
hf_token=your_huggingface_token
pinecone_api_key=your_pinecone_key
GROQ_API_KEY=your_groq_key
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## Usage

### Step 1: Generate Incident

```
generate incident
```

### Step 2: Analyze

```
analyze
```

---

## Retrieval Strategy

* Semantic search (vector similarity)
* Metadata filtering (attack type)
* Confidence-aware retrieval
* Cross-encoder reranking

---

## LLM Reasoning

The LLM performs:

* Attack validation
* Pattern comparison
* Feature-level analysis
* Conflict detection
* Mitigation recommendation
* Severity assessment

---

## Example Output

```
SOC INCIDENT REPORT

Predicted Attack: Mirai-Ackflooding
Confidence: 0.34

Final Assessment: Mirai-UDP Flooding (override)

Severity: HIGH

Recommended Mitigations:
- Network Intrusion Prevention (M1031)
- Vulnerability Scanning (RA-5)
```

---

##  Datasets Used

* IoTID20


---

##  Author

**Maruf Islam**

---

## License

This project is for research and educational purposes.
