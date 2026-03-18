import streamlit as st
import os
from dotenv import load_dotenv

from utils.model_utils import *
from utils.incident_utils import *
from utils.embedding_utils import EmbeddingClient
from utils.retrieval_utils import *
from utils.llm_utils import *
import streamlit as st
from sentence_transformers import CrossEncoder

# ------------------------
# Setup
# ------------------------



load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]  # Streamlit Cloud
    except:
        return os.getenv(key)   # Local .env

HF_TOKEN = get_secret("hf_token")
PINECONE_API_KEY = get_secret("pinecone_api_key")
GROQ_API_KEY = get_secret("GROQ_API_KEY")



MODEL_PATH = "utils/results/xgb_model.pkl"
LABEL_MAP_PATH = "utils/results/label_mapping.json"

st.set_page_config(page_title="ThreatLens AI", layout="wide")

# ------------------------
# Styling
# ------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ ThreatLens AI — Cybersecurity Assistant")

# ------------------------
# SESSION INIT (MULTI CHAT)
# ------------------------
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"
    st.session_state.chat_sessions["Chat 1"] = {
        "messages": [],
        "incident_data": None
    }

chat = st.session_state.chat_sessions[st.session_state.current_chat]

# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:
    st.title("🛡️ ThreatLens AI")

    if st.button("➕ New Chat"):
        name = f"Chat {len(st.session_state.chat_sessions)+1}"
        st.session_state.chat_sessions[name] = {
            "messages": [],
            "incident_data": None
        }
        st.session_state.current_chat = name

    st.markdown("---")

    for chat_name in st.session_state.chat_sessions:
        if st.button(chat_name):
            st.session_state.current_chat = chat_name

    st.markdown("---")
    st.caption("SOC AI Assistant")

# ------------------------
# Load system (cached)
# ------------------------
@st.cache_resource
def load_all():
    model = load_model(MODEL_PATH)
    label_to_id, id_to_label = load_label_mapping(LABEL_MAP_PATH)
    df = load_data("maruuf/iotid20_dataset", HF_TOKEN)
    features = get_model_features(model)

    embedder = EmbeddingClient(HF_TOKEN)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    detection_index, mitigation_index = connect_pinecone(
        PINECONE_API_KEY,
        "cybersec-llm-rag",
        "mitigation-vector-db"
    )

    feature_levels = compute_feature_levels(df, features)

    return model, df, features, id_to_label, embedder, reranker, detection_index, mitigation_index, feature_levels


model, df, features, id_to_label, embedder, reranker, detection_index, mitigation_index, feature_levels = load_all()

# ------------------------
# SHOW CHAT HISTORY
# ------------------------
for msg in chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------
# INPUT
# ------------------------
user_input = st.chat_input("Type: 'generate incident' or 'analyze'")

if user_input:

    # Save user message
    chat["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    response = ""
    command = user_input.lower().strip()

    # ------------------------
    # GENERATE INCIDENT
    # ------------------------
    if command in ["generate incident", "generate", "incident"]:

        idx, row = select_random_row(df)

        pred_id, pred_label, confidence = predict_sample(
            model, row, features, id_to_label
        )

        true_label = extract_true_label(row, df)

        incident_packet = build_incident_packet_descriptive(
            df,
            feature_levels,
            features,
            idx,
            pred_label,
            confidence,
            true_label
        )

        incident_data = build_incident_data(
            idx, pred_id, pred_label, confidence,
            true_label, incident_packet, features
        )

        chat["incident_data"] = incident_data

        # 🔥 RENAME CHAT TO ATTACK NAME
        if st.session_state.current_chat.startswith("Chat"):
            new_name = pred_label

            if new_name in st.session_state.chat_sessions:
                new_name = f"{pred_label}_{len(st.session_state.chat_sessions)}"

            st.session_state.chat_sessions[new_name] = chat
            del st.session_state.chat_sessions[st.session_state.current_chat]
            st.session_state.current_chat = new_name

        response = f"""### 📦 Incident Packet
{incident_packet}
"""

    # ------------------------
    # ANALYZE
    # ------------------------
    elif command in ["analyze", "analysis", "report"]:

        if chat["incident_data"] is None:
            response = "⚠️ Please generate an incident first using: `generate incident`"

        else:
            incident_data = chat["incident_data"]

            with st.spinner("🔍 Retrieving and analyzing..."):

                query_embedding = embedder.embed(incident_data["incident_packet"])

                global_matches, filtered_matches = confidence_aware_retrieval(
                    detection_index,
                    query_embedding,
                    incident_data["predicted_label"],
                    incident_data["confidence"]
                )

                merged = merge_and_dedupe_matches(global_matches, filtered_matches)
                diverse = limit_per_dataset(merged)

                top_cases = rerank_matches(
                    incident_data["incident_packet"],
                    diverse,
                    reranker
                )

                mitigation_query = f"""
                Mitigation for {incident_data['predicted_label']}
                {incident_data['incident_packet']}
                """

                mitigation_embedding = embedder.embed(mitigation_query)

                mitigation_matches = retrieve_mitigations(
                    mitigation_index,
                    mitigation_embedding,
                    incident_data["predicted_label"]
                )

                top_mitigations = rerank_matches(
                    mitigation_query,
                    mitigation_matches,
                    reranker
                )

                system_prompt, user_prompt = build_final_prompt(
                    incident_data,
                    format_historical_cases(top_cases, incident_data["predicted_label"]),
                    format_mitigation_items(top_mitigations)
                )

                llm_output = run_llm(GROQ_API_KEY, system_prompt, user_prompt)

                response = f"""### 🧠 SOC Analysis Report

{llm_output}
"""

    else:
        response = """
Try:
• generate incident  
• analyze
"""

    # ------------------------
    # ASSISTANT RESPONSE
    # ------------------------
    chat["messages"].append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)