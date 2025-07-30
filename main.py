#!/usr/bin/env python3
"""
Kavak Travel Assistant - Conversational AI for International Travel Planning
"""

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
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class TravelQuery(BaseModel):
    """Structured representation of a travel query."""
    origin: Optional[str] = Field(None, description="Departure city")
    destination: Optional[str] = Field(None, description="Destination city")
    departure_date: Optional[str] = Field(None, description="Departure date")
    return_date: Optional[str] = Field(None, description="Return date")
    preferred_airlines: Optional[List[str]] = Field(
        None, description="Preferred airlines")
    preferred_alliances: Optional[List[str]] = Field(
        None, description="Preferred alliances")
    max_price: Optional[float] = Field(None, description="Maximum price")
    avoid_overnight_layovers: Optional[bool] = Field(
        None, description="Avoid overnight layovers")
    refundable_only: Optional[bool] = Field(
        None, description="Refundable tickets only")


class FlightDatabase:
    """Mock flight database with search capabilities."""

    def __init__(self, data_path: str = "data/flights.json"):
        self.flights = self._load_flights(data_path)

    def _load_flights(self, data_path: str) -> List[Dict]:
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Flight data file {data_path} not found.")
            return []

    def search_flights(self, query: TravelQuery) -> List[Dict]:
        """Search flights based on query criteria."""
        print(f"\n=== FLIGHT SEARCH DEBUG ===")
        print(f"Query received: {query}")
        print(f"Total flights in database: {len(self.flights)}")

        if not query.destination:
            print("❌ No destination specified")
            return []

        print(f"🔍 Looking for flights to: {query.destination}")
        print(f"Alliance filter: {query.preferred_alliances}")
        print(f"Avoid overnight layovers: {query.avoid_overnight_layovers}")
        print(f"Refundable only: {query.refundable_only}")
        print(f"Max price: {query.max_price}")

        results = []
        for i, flight in enumerate(self.flights):
            print(
                f"\n--- Flight {i+1}: {flight['airline']} {flight['from']} → {flight['to']} ---")
            print(f"Flight data: {flight}")

            # Check destination
            if flight['to'].lower() != query.destination.lower():
                print(
                    f"❌ Destination mismatch: '{flight['to']}' != '{query.destination}'")
                continue
            print(f"✅ Destination match")

            # Check origin
            if query.origin and flight['from'].lower() != query.origin.lower():
                print(
                    f"❌ Origin mismatch: '{flight['from']}' != '{query.origin}'")
                continue
            if query.origin:
                print(f"✅ Origin match")

            # Check departure date
            if query.departure_date and flight['departure_date'] != query.departure_date:
                print(
                    f"❌ Date mismatch: '{flight['departure_date']}' != '{query.departure_date}'")
                continue
            if query.departure_date:
                print(f"✅ Date match")

            # Check alliances
            if query.preferred_alliances and flight['alliance'] not in query.preferred_alliances:
                print(
                    f"❌ Alliance mismatch: '{flight['alliance']}' not in {query.preferred_alliances}")
                continue
            if query.preferred_alliances:
                print(f"✅ Alliance match: {flight['alliance']}")

            # Check price
            if query.max_price and flight['price_usd'] > query.max_price:
                print(
                    f"❌ Price too high: ${flight['price_usd']} > ${query.max_price}")
                continue
            if query.max_price:
                print(f"✅ Price within budget: ${flight['price_usd']}")

            # Check refundable
            if query.refundable_only and not flight['refundable']:
                print(f"❌ Not refundable: {flight['refundable']}")
                continue
            if query.refundable_only:
                print(f"✅ Refundable: {flight['refundable']}")

            print(f"🎯 FLIGHT MATCHES ALL CRITERIA!")
            results.append(flight)

        print(f"\n=== SEARCH RESULTS ===")
        print(f"Total matching flights: {len(results)}")
        for i, flight in enumerate(results):
            print(
                f"Result {i+1}: {flight['airline']} - ${flight['price_usd']} - {flight['alliance']}")

        results.sort(key=lambda x: x['price_usd'])
        return results


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


class QueryExtractor:
    """Extract structured travel queries from natural language."""

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract travel information from user messages and return a clean JSON object without extra whitespace or formatting:
            {
                "origin": "departure city or null",
                "destination": "destination city or null", 
                "departure_date": "YYYY-MM-DD or null",
                "return_date": "YYYY-MM-DD or null",
                "preferred_airlines": ["airline names"] or null,
                "preferred_alliances": ["alliance names"] or null,
                "max_price": price_in_usd or null,
                "avoid_overnight_layovers": true/false or null,
                "refundable_only": true/false or null
            }
            
            Return ONLY the JSON object, no additional text or formatting."""),
            ("human", "{user_message}")
        ])
        self.chain = LLMChain(llm=self.llm, prompt=self.extraction_prompt)

    def extract_query(self, user_message: str) -> TravelQuery:
        try:
            result = self.chain.invoke({"user_message": user_message})
            json_str = result['text'].strip()

            # Remove markdown code blocks
            if json_str.startswith('```json'):
                json_str = json_str[7:-3]
            elif json_str.startswith('```'):
                json_str = json_str[3:-3]

            # Clean up whitespace and newlines
            json_str = json_str.replace('\n', '').replace('    ', '')

            # Try to parse JSON
            try:
                query_data = json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract key-value pairs manually
                query_data = self._extract_json_manually(json_str)

            return TravelQuery(**query_data)
        except Exception as e:
            print(f"Error extracting query: {e}")
            return TravelQuery()

    def _extract_json_manually(self, text: str) -> Dict:
        """Manually extract JSON-like data from malformed text."""
        import re

        # Default values
        data = {
            "origin": None,
            "destination": None,
            "departure_date": None,
            "return_date": None,
            "preferred_airlines": None,
            "preferred_alliances": None,
            "max_price": None,
            "avoid_overnight_layovers": None,
            "refundable_only": None
        }

        # Extract destination
        dest_match = re.search(r'"destination":\s*"([^"]+)"', text)
        if dest_match:
            data["destination"] = dest_match.group(1)

        # Extract origin
        origin_match = re.search(r'"origin":\s*"([^"]+)"', text)
        if origin_match:
            data["origin"] = origin_match.group(1)

        # Extract alliances
        alliance_match = re.search(
            r'"preferred_alliances":\s*\[([^\]]+)\]', text)
        if alliance_match:
            alliances = alliance_match.group(1).replace('"', '').split(',')
            data["preferred_alliances"] = [a.strip()
                                           for a in alliances if a.strip()]

        # Extract max price
        price_match = re.search(r'"max_price":\s*(\d+)', text)
        if price_match:
            data["max_price"] = float(price_match.group(1))

        # Extract boolean values
        if '"avoid_overnight_layovers":\s*true' in text.lower():
            data["avoid_overnight_layovers"] = True
        elif '"avoid_overnight_layovers":\s*false' in text.lower():
            data["avoid_overnight_layovers"] = False

        if '"refundable_only":\s*true' in text.lower():
            data["refundable_only"] = True
        elif '"refundable_only":\s*false' in text.lower():
            data["refundable_only"] = False

        return data


# Initialize components
flight_db = FlightDatabase()
knowledge_base = KnowledgeBase()
query_extractor = QueryExtractor()

# Define tools


@tool
def search_flights(query_text: str) -> str:
    """Search for flights based on natural language query."""
    query = query_extractor.extract_query(query_text)
    flights = flight_db.search_flights(query)

    if not flights:
        return "No flights found matching your criteria."

    result_lines = [f"Found {len(flights)} flights:\n"]
    for i, flight in enumerate(flights[:5], 1):
        result_lines.append(f"{i}. {flight['airline']} ({flight['alliance']})")
        result_lines.append(f"   Flight: {flight['flight_number']}")
        result_lines.append(f"   Route: {flight['from']} → {flight['to']}")
        result_lines.append(f"   Price: ${flight['price_usd']}")
        result_lines.append(f"   Duration: {flight['duration_hours']} hours")
        result_lines.append(
            f"   Refundable: {'Yes' if flight['refundable'] else 'No'}")
        result_lines.append("")

    return "\n".join(result_lines)


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
            return f"Sorry, I encountered an error: {e}"

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
