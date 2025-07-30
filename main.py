#!/usr/bin/env python3
"""
Kavak Travel Assistant - Conversational AI for International Travel Planning
"""

from datetime import datetime
import json
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chains import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.json.base import create_json_agent,JsonToolkit
from langchain_community.llms import OpenAI
from langchain_community.tools.json.tool import JsonSpec
from langchain.schema import HumanMessage

from langchain.agents import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit

# Load environment variables
load_dotenv()


class KnowledgeBase:
    """RAG-based knowledge base for travel information."""

    def __init__(self, data_path: str = "data/visa_rules.md", vector_db_path: str = "./vector_db"):
        self.data_path = data_path
        self.vector_db_path = vector_db_path
        self.vectorstore = self._create_vectorstore()
        self.qa_chain = self._create_qa_chain()

    def _create_vectorstore(self) -> FAISS:
        try:
            # Check if vector store already exists
            if os.path.exists(self.vector_db_path):
                print(
                    f"📚 Loading existing vector store from {self.vector_db_path}")
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.load_local(self.vector_db_path, embeddings)
                print("✅ Vector store loaded successfully")
                return vectorstore

            # Create new vector store
            print(f"🔄 Creating new vector store from {self.data_path}")
            loader = TextLoader(self.data_path)
            documents = loader.load()
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)

            # Save vector store for future use
            print(f"💾 Saving vector store to {self.vector_db_path}")
            vectorstore.save_local(self.vector_db_path)
            print("✅ Vector store saved successfully")

            return vectorstore
        except Exception as e:
            print(f"Warning: Could not create vector store: {e}")
            # Return empty vector store
            embeddings = OpenAIEmbeddings()
            return FAISS.from_texts(["No knowledge base available"], embeddings)

    def rebuild_vectorstore(self):
        """Force rebuild the vector store from source documents."""
        print(f"🔄 Rebuilding vector store from {self.data_path}")

        # Remove existing vector store
        if os.path.exists(self.vector_db_path):
            import shutil
            shutil.rmtree(self.vector_db_path)
            print(f"🗑️ Removed existing vector store")

        # Create new vector store
        self.vectorstore = self._create_vectorstore()
        self.qa_chain = self._create_qa_chain()
        print("✅ Vector store rebuilt successfully")

    def _create_qa_chain(self) -> RetrievalQA:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )

    def query_knowledge(self, question: str) -> str:
        try:
            result = self.qa_chain.invoke({"query": question})
            return result["result"]
        except Exception as e:
            return f"Sorry, I couldn't retrieve information about that. Error: {e}"

# Initialize components
# flight_db = FlightDatabase()
knowledge_base = KnowledgeBase()
flight_filterer = FlightFilterer()

# Define tools


@tool
def search_flights(query_text: str) -> str:
    """Search for flights based on natural language query."""

    with open("data/flights.json", "r") as f:
        parsed_json = json.load(f)

    # parsed_json = json.loads(json_data)
    # Define the JSON spec
    json_spec = JsonSpec(dict_=parsed_json, max_value_length=400)
    # ✅ Create the toolkit from the spec
    toolkit = JsonToolkit(spec=json_spec)
    # Initialize the LLM
    llm = OpenAI(temperature=0,model="gpt-4o-mini")

    # Create the JSON agent
    agent_executor = create_json_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    # Query the JSON data
    # question = "Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers, provide complete details"
    question = query_text
    response = agent_executor.run(question)
    return response


@tool
def get_travel_information(query: str) -> str:
    """Get travel information including visa requirements, refund policies, and travel tips."""
    return knowledge_base.query_knowledge(query)


def create_travel_agent():
    """Create the travel assistant agent."""
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

    # Create memory buffer
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Kavak's intelligent travel assistant. Help users with flight searches, 
        visa information, refund policies, and travel tips. Be helpful and professional."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    tools = [search_flights, get_travel_information]
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        memory=memory
    )

    return agent_executor


class TravelAssistant:
    """Main interface for the travel assistant."""

    def __init__(self):
        self.agent = create_travel_agent()

    def chat(self, user_input: str) -> str:
        """Process user input and return response."""
        try:
            response = self.agent.invoke({"input": user_input})
            return response["output"]
        except Exception as e:
            return f"Sorry, I  encountered an error in main: {e}"

    def clear_memory(self):
        """Clear conversation memory."""
        self.agent.memory.clear()


def main():
    """Main function to run the travel assistant."""
    print("🌍 Welcome to Kavak Travel Assistant!")
    print("I can help you with flights, visas, refunds, and travel tips.")
    print("Type 'quit' to exit, 'clear' to clear memory, 'rebuild' to rebuild vector store.\n")

    assistant = TravelAssistant()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thank you for using Kavak Travel Assistant! ✈️")
                break

            if user_input.lower() == 'clear':
                assistant.clear_memory()
                print("Memory cleared! Starting fresh conversation.\n")
                continue

            if user_input.lower() == 'rebuild':
                knowledge_base.rebuild_vectorstore()
                print("Vector store rebuilt! You can now ask questions.\n")
                continue

            if not user_input:
                continue

            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nThank you for using Kavak Travel Assistant! ✈️")
            break
        except Exception as e:
            print(f"\nSorry, I encountered an error: {e}")


if __name__ == "__main__":
    main()
