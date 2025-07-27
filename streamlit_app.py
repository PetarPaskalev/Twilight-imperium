"""
Twilight Imperium Fourth Edition Assistant - Streamlit Web Interface

A beautiful web interface for the Twilight Imperium chatbot using Streamlit.
Features chat bubbles, conversation history, and responsive design.
"""

import streamlit as st
import time
from typing import List
import os
from dotenv import load_dotenv

# Import our chatbot
from twilight_chatbot_final import TwilightImperiumFinalBot

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Twilight Imperium Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    .user-message {
        background-color: #3b82f6;
        color: white;
        padding: 0.75rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #e2e8f0;
        color: #1e293b;
        padding: 0.75rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-right: 2rem;
    }
    
    .sidebar-info {
        background-color: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    
    .stats-box {
        background-color: #ecfdf5;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #10b981;
        text-align: center;
    }
    
    .error-box {
        background-color: #fef2f2;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ef4444;
        color: #dc2626;
    }
</style>
""", unsafe_allow_html=True)

def initialize_chatbot():
    """Initialize the chatbot with error handling"""
    try:
        if 'chatbot' not in st.session_state:
            with st.spinner("🚀 Initializing Twilight Imperium Assistant..."):
                st.session_state.chatbot = TwilightImperiumFinalBot()
            st.success("✅ Assistant ready!")
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize chatbot: {e}")
        st.markdown("""
        <div class="error-box">
        <strong>Troubleshooting:</strong><br>
        1. Ensure your OpenAI API key is set<br>
        2. Run the PDF processing steps first<br>
        3. Check that all dependencies are installed
        </div>
        """, unsafe_allow_html=True)
        return False

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False

def display_welcome_message():
    """Display welcome message and instructions"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Twilight Imperium Fourth Edition Assistant</h1>
        <p>Your AI guide to mastering the galaxy!</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.chat_started:
        st.markdown("""
        ### 🌟 Welcome to your personal TI4 rules expert!
        
        I can help you with:
        - **Strategy Cards & Action Cards** (Leadership, Flank Speed, etc.)
        - **Faction Abilities** (All 17 factions + FAQs)
        - **Combat Mechanics** and movement rules
        - **Victory Conditions** and objectives
        - **Card Interactions** and complex scenarios
        
        **💡 Pro Tips:**
        - Ask specific questions for best results
        - Mention faction names when asking about abilities
        - For complex interactions, describe both elements
        
        **Ready to explore the galaxy? Ask me anything below!** 👇
        """)

def display_sidebar():
    """Display sidebar with stats and information"""
    with st.sidebar:
        st.markdown("## 📊 Assistant Stats")
        
        # Stats
        if 'chatbot' in st.session_state:
            st.markdown("""
            <div class="stats-box">
                <strong>🗂️ Knowledge Base</strong><br>
                409 Rule Chunks<br>
                17 Faction Guides<br>
                Official Rulebooks
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Example questions
        st.markdown("## 💡 Example Questions")
        examples = [
            "What does the Leadership strategy card do?",
            "How do the Arborec's production abilities work?",
            "Which faction has a teleporting flagship?",
            "How does space combat work?",
            "What is Flank Speed action card?",
            "How do wormholes work with Ghosts of Creuss?"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.example_question = example
        
        st.markdown("---")
        
        # Controls
        st.markdown("## 🛠️ Controls")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.chat_started = False
            st.rerun()
        
        if st.button("🔄 Restart Assistant", use_container_width=True):
            if 'chatbot' in st.session_state:
                del st.session_state.chatbot
            st.rerun()
        
        # API Key status
        st.markdown("---")
        st.markdown("## 🔑 API Status")
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and len(api_key) > 10:
            st.success("✅ OpenAI API Key Found")
        else:
            st.error("❌ OpenAI API Key Missing")
            st.markdown("""
            <div class="sidebar-info">
            <small>Set your API key:<br>
            1. Create .env file<br>
            2. Add: OPENAI_API_KEY=your_key</small>
            </div>
            """, unsafe_allow_html=True)

def display_chat_interface():
    """Display the main chat interface"""
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle example question selection
    if 'example_question' in st.session_state:
        prompt = st.session_state.example_question
        del st.session_state.example_question
    else:
        # Chat input
        prompt = st.chat_input("Ask me anything about Twilight Imperium Fourth Edition...")
    
    # Process user input
    if prompt:
        st.session_state.chat_started = True
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching the rules..."):
                try:
                    response = st.session_state.chatbot.chat(
                        prompt, 
                        st.session_state.conversation_history
                    )
                    
                    # Stream the response for better UX
                    placeholder = st.empty()
                    full_response = ""
                    
                    # Simulate streaming (Streamlit effect)
                    words = response.split()
                    for i, word in enumerate(words):
                        full_response += word + " "
                        placeholder.markdown(full_response + "▌")
                        if i % 3 == 0:  # Update every 3 words
                            time.sleep(0.02)
                    
                    placeholder.markdown(full_response)
                    
                    # Add to session state
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Update conversation history for the chatbot
                    from langchain_core.messages import HumanMessage, AIMessage
                    st.session_state.conversation_history.append(HumanMessage(content=prompt))
                    st.session_state.conversation_history.append(AIMessage(content=response))
                    
                    # Keep history manageable
                    if len(st.session_state.conversation_history) > 20:
                        st.session_state.conversation_history = st.session_state.conversation_history[-20:]
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display welcome message
    display_welcome_message()
    
    # Initialize chatbot
    if not initialize_chatbot():
        st.stop()
    
    # Create layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main chat interface
        st.markdown("### 💬 Chat with your TI4 Assistant")
        display_chat_interface()
    
    with col2:
        # Sidebar content in column for better mobile experience
        display_sidebar()

if __name__ == "__main__":
    main() 