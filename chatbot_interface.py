import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical interface styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d3748 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        color: white;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Medical disclaimer */
    .medical-disclaimer {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #fbbf24;
        font-size: 0.9rem;
    }
    
    /* Chat container styling */
    .chat-container {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(30, 41, 59, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar header */
    .sidebar-header {
        background: rgba(99, 102, 241, 0.8);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Stats cards */
    .stats-card {
        background: rgba(51, 65, 85, 0.8);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stats-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #6366f1;
        font-size: 0.9rem;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.85rem;
    }
    
    .stat-item:last-child {
        border-bottom: none;
    }
    
    .stat-value {
        background: rgba(99, 102, 241, 0.8);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Example questions */
    .example-questions {
        background: rgba(51, 65, 85, 0.8);
        border-radius: 8px;
        padding: 1rem;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .example-header {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #fbbf24;
        font-size: 0.9rem;
    }
    
    .example-question {
        background: rgba(30, 41, 59, 0.8);
        padding: 0.7rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid transparent;
    }
    
    .example-question:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }
    
    /* Chat messages */
    .user-message {
        background: rgba(99, 102, 241, 0.8);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }
    
    .bot-message {
        background: rgba(51, 65, 85, 0.8);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input styling */
    .stTextArea textarea {
        background: rgba(51, 65, 85, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: rgba(51, 65, 85, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: rgba(51, 65, 85, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Medical compliance checker
def check_medical_compliance(response):
    """Check if response includes medical disclaimers and professional language"""
    disclaimer_keywords = ["disclaimer", "educational purposes", "professional medical advice", "consult"]
    medical_terms = ["medical", "healthcare", "clinical", "diagnosis", "treatment"]

    has_disclaimer = any(keyword in response.lower() for keyword in disclaimer_keywords)
    has_medical_terms = any(term in response.lower() for term in medical_terms)

    return has_disclaimer, has_medical_terms

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vectorstore: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        huggingfacehub_api_token=hf_token
    )
    return llm

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical AI Assistant</h1>
        <p>Your conversational health companion</p>
    </div>
    """, unsafe_allow_html=True)

    # Medical disclaimer
    st.markdown("""
    <div class="medical-disclaimer">
        ⚠️ <strong>Important:</strong> This AI provides general health information only. Always consult healthcare professionals for medical advice, diagnosis, or treatment. In emergencies, contact emergency services immediately.
    </div>
    """, unsafe_allow_html=True)

    # Create layout with columns
    col1, col2 = st.columns([3, 1])

    with col1:
        # Chat section header
        st.markdown("""
        <div class="sidebar-header">
            💬 Medical Chat
        </div>
        """, unsafe_allow_html=True)

        # Initialize session state for messages and stats
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'questions_asked' not in st.session_state:
            st.session_state.questions_asked = 0
        if 'avg_response_time' not in st.session_state:
            st.session_state.avg_response_time = 0.0

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>🏥 MediBot:</strong><br>{message['content']}
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_button = st.columns([4, 1])

            with col_input:
                prompt = st.text_area(
                    "Ask your health question:",
                    placeholder="Type your medical question here...",
                    height=100,
                    label_visibility="collapsed"
                )

            with col_button:
                st.write("")  # Add some spacing
                st.write("")  # Add some spacing
                submitted = st.form_submit_button("Send 🚀", use_container_width=True)

        # Clear chat button
        if st.button("Clear Chat 🗑️", type="secondary"):
            st.session_state.messages = []
            st.session_state.questions_asked = 0
            st.rerun()

    with col2:
        # Chat Statistics
        st.markdown("""
        <div class="stats-card">
            <div class="stats-header">📊 Chat Stats</div>
            <div class="stat-item">
                <span>Questions Asked:</span>
                <span class="stat-value">{}</span>
            </div>
            <div class="stat-item">
                <span>Avg Response:</span>
                <span class="stat-value">{:.1f}s</span>
            </div>
        </div>
        """.format(st.session_state.questions_asked, st.session_state.avg_response_time), unsafe_allow_html=True)

        # Example Questions
        st.markdown("""
        <div class="example-questions">
            <div class="example-header">💡 Example Questions</div>
        </div>
        """, unsafe_allow_html=True)

        example_questions = [
            "What are normal blood pressure ranges?",
            "How can I improve my sleep quality?",
            "What causes frequent headaches?",
            "Tips for managing stress and anxiety",
            "Signs of dehydration to watch for",
            "When should I see a doctor for a cough?"
        ]

        for question in example_questions:
            if st.button(question, key=f"example_{question[:20]}", use_container_width=True):
                # Add the example question to chat
                st.session_state.messages.append({'role': 'user', 'content': question})
                st.session_state.questions_asked += 1
                st.rerun()

    # Process chat input
    if submitted and prompt:
        # Add user message to chat
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.session_state.questions_asked += 1

        CUSTOM_PROMPT_TEMPLATE = """
You are a medical AI assistant providing evidence-based healthcare information.

Use the pieces of information provided in the context to answer the user's medical question.
If you don't know the answer, say that you don't know - do not make up medical information.
Always provide professional, clinical language appropriate for healthcare communication.

Context: {context}
Question: {question}

**IMPORTANT MEDICAL DISCLAIMER**: This information is for educational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns.

Answer:
"""

        # Configuration
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        # Check if HF_TOKEN is available
        if not HF_TOKEN:
            st.error("🔑 HF_TOKEN environment variable is not set. Please set your Hugging Face token.")
            return

        try:
            with st.spinner("🤔 Thinking..."):
                import time
                start_time = time.time()

                # Load vector store
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("❌ Failed to load the vector store")
                    return

                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                # Get response
                response = qa_chain.invoke({'query': prompt})

                end_time = time.time()
                response_time = end_time - start_time

                # Update average response time
                if st.session_state.questions_asked > 1:
                    st.session_state.avg_response_time = (
                            (st.session_state.avg_response_time * (st.session_state.questions_asked - 1) + response_time)
                            / st.session_state.questions_asked
                    )
                else:
                    st.session_state.avg_response_time = response_time

                result = response["result"]
                source_documents = response["source_documents"]

                # Format the response with cleaner source information
                formatted_response = result + "\n\n**📚 Source Documents:**\n"
                for i, doc in enumerate(source_documents, 1):
                    metadata = doc.metadata
                    book_name = metadata.get('source', 'Unknown Book')
                    page_number = metadata.get('page', 'Unknown Page')
                    formatted_response += f"{i}. Book: {book_name}, Page: {page_number}\n"

                # Add bot response to chat
                st.session_state.messages.append({'role': 'assistant', 'content': formatted_response})

                # Rerun to display the new messages
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Please try rephrasing your question or check your internet connection.")

if __name__ == "__main__":
    main()