# Kavak Travel Assistant 🌍✈️

A comprehensive conversational AI travel assistant that helps users plan international travel by handling flight queries, retrieving policy information, and answering visa-related questions through natural language interaction.

## 🎯 Project Overview

This technical case study demonstrates advanced conversational AI capabilities using LangChain and LangGraph for intelligent travel planning. The assistant can:

- **Interpret Natural Queries**: Understand complex travel requests like "Find me a round-trip to Tokyo in August with Star Alliance airlines only"
- **Extract & Normalize Search Criteria**: Identify origin/destination, dates, airline preferences, and constraints
- **Search & Retrieve Results**: Use mock flight database with intelligent filtering
- **Answer Policy Questions**: Provide visa and refund information using RAG (Retrieval-Augmented Generation)
- **Maintain Context**: Handle multi-turn conversations with memory


### Core Components
- 🗣 **Natural Language Understanding**: Interprets complex travel-related queries
- 📅 **Structured Data Extraction**: Detects destination, dates, airlines, cost, and preferences
- 🔍 **Flight Search Engine**: Filters a mock JSON dataset using agent + JSON toolkit
- 📚 **Knowledge Retrieval**: Uses RAG on markdown policies to answer visa, refund, and travel rule questions
- 🧠 **Memory Support**: Maintains context with `ConversationBufferMemory`
- 🛠 **LangChain Agents**: Modular orchestration using OpenAI Tools + tool routing
- 🌐 **Streamlit Web UI**: Interactive frontend for chat-based planning


## 🏗️ Architecture Overview
## Diagram
```
+------------------------------------------------------------+
|         Kavak Conversational Travel Assistant              |
+------------------------------------------------------------+
|                                                            |
|  +---------------------+     +--------------------------+  |
|  |   LangChain Agent   |<--->|  Conversation Memory      |  |
|  | (Tool-Calling Core) |     | (BufferWindowMemory)      |  |
|  +---------------------+     +--------------------------+  |
|             |                                  |            |
|             v                                  v            |
|     +----------------------------------------------+        |
|     |              AgentExecutor                   |        |
|     | (Handles tool routing, memory, tool output)  |        |
|     +-------------------+--------------------------+        |
|                         |                                   |
|       +-----------------+-------------------+               |
|       |                                     |               |
|       v                                     v               |
| +--------------------+        +--------------------------+  |
| |  FlightSearchTool  |        |  KnowledgeBaseTool (RAG) |  |
| | (Structured JSON)  |        |  (Markdown + FAISS)       |  |
| +---------+----------+        +-------------+------------+  |
|           |                                 |               |
|     +-----v-----+                   +--------v--------+     |
|     | flights.json|                |  visa_rules.md   |     |
|     +-----------+|                | (Unstructured KB) |     |
|                 ||                +--------+----------+     |
|  +--------------+|                         |                |
|  |  ReAct Agent  |                         v                |
|  | (LangChain)   |              +------------------------+  |
|  +--------------+              |   FAISS Vector Store    |  |
|                                +-----------+------------+  |
|                                            |               |
|                                            v               |
|                                +------------------------+  |
|                                |  OpenAI Chat Model     |  |
|                                |   (ChatOpenAI / GPT)   |  |
|                                +------------------------+  |
+------------------------------------------------------------+
```
### Agent Tools

| Tool | Function |
|------|----------|
| `search_flights` | Uses a JSON agent using react technique using langchian agent executor to query `flights.json` for matching flights |
| `get_travel_information` | RAG-based QA using `visa_rules.md` and FAISS vector store |

### Key Components

- **LangChain Agents**: Uses `create_openai_tools_agent` with tool routing
- **Vector DB (FAISS)**: Embeds `visa_rules.md` using `OpenAIEmbeddings`
- **Prompt Template**: Mixes system message, scratchpad, and memory
- **Streamlit UI**: Simple chat interface with sidebar actions and state handling


### Technology Stack

- **Language**: Python 3.x
- **LLM Framework**: LangChain 
- **LLM Provider**: OpenAI GPT-4.1-mini
- **Vector Embedding**: OpenAI embeddings
- **Vector Database**: FAISS
- **Web Framework**: Streamlit
- **Data Formats**: JSON, Markdown

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AreebAhmad-02/kavak-travels-agent
   cd kavak-travels-agent
   ```
2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```


4. **Set up environment variables** (optional)

```bash
# For Windows PowerShell:
$env:OPENAI_API_KEY = "your-api-key-here"

# For Unix/Linux:
export OPENAI_API_KEY="your-api-key-here"
```

4. **Run the application**

   **Option A: Command Line Interface**

   ```bash
   python main.py
   ```

   **Option B: Web Interface**

   ```bash
   streamlit run streamlit_app.py
   ```

## 📁 Project Structure

```
kavak-travel-assistant/
├── main.py                 # Core implementation with LangChain/LangGraph
├── streamlit_app.py        # Web interface using Streamlit
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   ├── flights.json       # Mock flight database
│   └── visa_rules.md      # Knowledge base for policies
└── .env                   # Environment variables (create this)
```

## 🎮 Usage Examples

### Flight Search Queries

```
User: "Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers."

Assistant: Found 3 flights matching your criteria:

1. Turkish Airlines (Star Alliance)
   Flight: TK123
   Route: Dubai → Tokyo
   Price: $950
   Duration: 18 hours
   Refundable: Yes

2. Lufthansa (Star Alliance)
   Flight: LH456
   Route: Dubai → Tokyo
   Price: $1100
   Duration: 20 hours
   Refundable: No

3. Singapore Airlines (Star Alliance)
   Flight: SQ234
   Route: Dubai → Tokyo
   Price: $1300
   Duration: 14 hours
   Refundable: Yes
   
```

### Visa Information Queries

```
User: "What are the visa requirements for UAE citizens visiting Japan?"

Assistant: UAE citizens can enter Japan visa-free for short stays, usually up to 90 days, for purposes such as tourism, business, or visiting friends and relatives. However, visa policies may change, so I recommend verifying the latest entry requirements on the official website of the Embassy of Japan or the Japanese Ministry of Foreign Affairs before your trip. If you need further assistance or travel tips for Japan, feel free to ask!
```

### Policy Questions

```
User: "Can I get a refund if I cancel my flight?"

Assistant: Refundable tickets can be canceled up to 48 hours before departure, subject to a 10% processing fee. Full refund minus processing fee for cancellations made 48+ hours before departure. Partial refund available for cancellations made 24-48 hours before departure. Non-refundable tickets cannot be canceled for a refund but may be eligible for credit voucher with airline fees.
```

## 🔧 Technical Implementation

### Prompt Engineering Strategy

The system uses carefully crafted prompts for:

1. **Query Extraction**: Converts natural language to structured JSON
2. **Agent Behavior**: Defines helpful, professional travel assistant persona
3. **Tool Selection**: Routes queries to appropriate tools (flight search, knowledge base, etc.)
4. **REACT** : Uses Reasoning and Action is used for Quering with the json and Vector DB for multi-hop Queries

### LangChain Features

- **Agent Orchestration**: Intelligent tool routing based on query type
- **Context Management**: Maintains conversation history across turns
- **Tool Integration**: Seamless integration of flight search and knowledge retrieval
- **Error Handling**: Graceful fallbacks for failed operations

### RAG Implementation

- **Document Loading**: Markdown files loaded with TextLoader
- **Text Splitter**: Langchain Markdown text splitter is used fot splitting text.
- **Vector Embeddings**: OpenAI embeddings for semantic search
- **Retrieval**: FAISS vector store with k=3 nearest neighbors
- **Generation**: LLM-based answer synthesis from retrieved context

### Data Management

- **Knowledge Base**: Comprehensive markdown covering visas, policies, tips
- **Query Normalization**: Standardized date formats, alliance names, etc.

## 🎨 Features & Capabilities

### Sample Queries Handled

- "Find flights to Paris in September under $1000"
- "What documents do I need for a US visa?"
- "Tell me about Emirates' refund policy"
- "I want to avoid overnight layovers on my trip to Tokyo"
- "What are the COVID requirements for traveling to France?"

## 🧪 Testing & Validation

### Test Cases

1. **Flight Search**

   - ✅ Basic destination search
   - ✅ Multi-criteria filtering
   - ✅ Alliance-specific searches
   - ✅ Price range filtering

2. **Knowledge Retrieval**

   - ✅ Visa requirement queries
   - ✅ Policy information requests
   - ✅ Travel tips and advice

3. **Conversation Flow**
   - ✅ Multi-turn conversations
   - ✅ Context preservation
   - ✅ Error recovery

### Performance Metrics

- **Query Processing**: < 10 seconds for most requests
- **Accuracy**: High precision for structured queries
- **Scalability**: Modular design supports easy expansion

## 🔮 Future Enhancements

### Potential Extensions

- **Real Flight APIs**: Integration with actual flight booking systems
- **Multi-language Support**: Arabic and other language support
- **Booking Integration**: Direct flight booking capabilities
- **Personalization**: User preference learning and storage
- **Advanced Analytics**: Travel pattern analysis and recommendations

### Technical Improvements

- **Caching**: Redis-based response caching
- **Monitoring**: Application performance monitoring
- **Testing**: Comprehensive unit and integration tests
- **CI/CD**: Automated deployment pipeline



---

**Built with ❤️ for Kavak's Conversational AI Assessment**
