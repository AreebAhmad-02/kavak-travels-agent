import streamlit as st
import json
from main import TravelAssistant,KnowledgeBase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Kavak Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "assistant" not in st.session_state:
    st.session_state.assistant = TravelAssistant()


def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ Kavak Travel Assistant</h1>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🎯 Quick Actions")

        # Quick action buttons
        if st.button("🔍 Search Flights"):
            st.session_state.messages.append(
                {"role": "user", "content": "I want to search for flights"})

        if st.button("📋 Visa Information"):
            st.session_state.messages.append(
                {"role": "user", "content": "Tell me about visa requirements"})

        if st.button("💰 Refund Policy"):
            st.session_state.messages.append(
                {"role": "user", "content": "What are the refund policies?"})

        if st.button("💡 Travel Tips"):
            st.session_state.messages.append(
                {"role": "user", "content": "Give me some travel tips"})

        st.markdown("---")

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        # Features section
        st.header("✨ Features")
        st.markdown("""
        - **Flight Search**: Find flights with specific criteria
        - **Visa Information**: Get visa requirements for destinations
        - **Refund Policies**: Understand cancellation and refund rules
        - **Travel Tips**: Get helpful travel advice
        """)

        # Sample queries
        st.header("💭 Sample Queries")
        st.markdown("""
        - "Find me a round-trip to Tokyo in August with Star Alliance airlines"
        - "What are the visa requirements for UAE citizens visiting Japan?"
        - "Can I get a refund if I cancel my flight?"
        - "Give me travel tips for visiting Paris"
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")

        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)

        # Chat input
        user_input = st.text_input(
            "Ask me anything about travel...",
            placeholder="e.g., Find me flights to Tokyo in August",
            key="user_input"
        )

        # Send button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Send", key="send_button"):
                if user_input.strip():
                    # Add user message
                    st.session_state.messages.append(
                        {"role": "user", "content": user_input})

                    # Get assistant response
                    with st.spinner("Thinking..."):
                        response = st.session_state.assistant.chat(user_input)

                    # Add assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})

                    # Clear input
                    st.session_state.user_input = ""
                    st.rerun()

    with col2:
        st.header("📊 Flight Database")

        # Show flight database stats
        try:
            flight_db = FlightDatabase()
            total_flights = len(flight_db.flights)
            st.metric("Total Flights", total_flights)

            # Show sample routes
            st.subheader("Sample Routes")
            routes = set()
            for flight in flight_db.flights[:10]:
                routes.add(f"{flight['from']} → {flight['to']}")

            for route in list(routes)[:5]:
                st.write(f"✈️ {route}")

        except Exception as e:
            st.error(f"Error loading flight database: {e}")

        st.markdown("---")

        st.header("📚 Knowledge Base")
        try:
            kb = KnowledgeBase()
            st.success("✅ Knowledge base loaded successfully")
            st.write("Contains information about:")
            st.write("- Visa requirements")
            st.write("- Refund policies")
            st.write("- Travel tips")
            st.write("- Security guidelines")
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")


if __name__ == "__main__":
    main()
