import streamlit as st
from src.chains import ask_question

st.set_page_config(
    page_title="Financial RAG Chatbot",
    layout="wide",
)

st.title("Financial RAG Chatbot")
st.caption(
    "Hi,Ask questions about NVIDIA annual reports"
    
)

SAMPLE_QUESTIONS = [
    "What was NVIDIA's Data Center revenue in fiscal year 2020?",
    "What was NVIDIA's total revenue in fiscal year 2024?",
    "What are NVIDIA's main business segments?",
    "What products does NVIDIA develop?",
]

with st.sidebar:
    st.header("Sample Questions")
    st.write("Choose a question to try the chatbot.")
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.pending_question = sample

    st.divider()
    st.subheader("About")
    st.write(
        "This application retrieves relevant passages from "
        "NVIDIA financial reports and uses an LLM to answer "
        "questions from the retrieved context."
    )
    st.caption("LLM: Qwen 2.5:3B · Local inference with Ollama")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SEQUENTIAL FULL-WIDTH CHAT HISTORY (NO EMOJIS, NO SIDE COLUMNS, NO SOURCES) ---
for message in st.session_state.messages:
    with st.container(border=True):
        if message["role"] == "user":
            st.markdown("**User:**")
            st.markdown(message["content"])
        else:
            st.markdown("**Assistant:**")
            st.markdown(message["content"])

# --- CHAT INPUT HANDLING ---
typed_question = st.chat_input("Ask a question about NVIDIA's financial reports...")
sample_question = st.session_state.pop("pending_question", None)
question = typed_question or sample_question

if question:
    # Save and display user question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.container(border=True):
        st.markdown("**User:**")
        st.markdown(question)

    # Generate and display assistant answer sequentially
    with st.container(border=True):
        st.markdown("**Assistant:**")
        with st.spinner("Searching reports and generating an answer..."):
            try:
                # Execute the chain (ignoring the returned documents metadata)
                answer, _ = ask_question(question)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                })
                
            except Exception as error:
                st.error("I couldn't generate an answer. Check that Ollama is running and the FAISS index is available.")
                st.caption(f"Error details: {error}")
