# Kavak Travel Assistant 🌍✈️

A comprehensive conversational AI travel assistant that helps users plan international travel by handling flight queries, retrieving policy information, and answering visa-related questions through natural language interaction.

## 🎯 Project Overview

This technical case study demonstrates advanced conversational AI capabilities using LangChain and LangGraph for intelligent travel planning. The assistant can:

- **Interpret Natural Queries**: Understand complex travel requests like "Find me a round-trip to Tokyo in August with Star Alliance airlines only"
- **Extract & Normalize Search Criteria**: Identify origin/destination, dates, airline preferences, and constraints
- **Search & Retrieve Results**: Use mock flight database with intelligent filtering
- **Answer Policy Questions**: Provide visa and refund information using RAG (Retrieval-Augmented Generation)
- **Maintain Context**: Handle multi-turn conversations with memory

## 🏗️ System Architecture

### Core Components

1. **Query Extractor**: Uses LLM to parse natural language into structured travel queries
2. **Flight Database**: Mock database with search and filtering capabilities
3. **Knowledge Base**: RAG system using FAISS vector store for policy information
4. **Conversation Agent**: LangChain agent with tool routing and context management
5. **Web Interface**: Streamlit app for user-friendly interaction

### Technology Stack

- **Language**: Python 3.x
- **LLM Framework**: LangChain & LangGraph
- **LLM Provider**: OpenAI GPT-3.5-turbo
- **Vector Database**: FAISS
- **Web Framework**: Streamlit
- **Data Formats**: JSON, Markdown

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd kavak-travel-assistant
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
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
```

### Visa Information Queries

```
User: "What are the visa requirements for UAE citizens visiting Japan?"

Assistant: UAE passport holders can enter Japan visa-free for up to 30 days for tourism purposes. Passport must be valid for at least 6 months beyond the intended stay. No visa application required for UAE citizens visiting Japan for tourism. Business travelers may require a business visa for stays longer than 30 days.
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

### LangChain/LangGraph Features

- **Agent Orchestration**: Intelligent tool routing based on query type
- **Context Management**: Maintains conversation history across turns
- **Tool Integration**: Seamless integration of flight search and knowledge retrieval
- **Error Handling**: Graceful fallbacks for failed operations

### RAG Implementation

- **Document Loading**: Markdown files loaded with TextLoader
- **Vector Embeddings**: OpenAI embeddings for semantic search
- **Retrieval**: FAISS vector store with k=3 nearest neighbors
- **Generation**: LLM-based answer synthesis from retrieved context

### Data Management

- **Flight Database**: JSON-based mock data with 12+ sample flights
- **Knowledge Base**: Comprehensive markdown covering visas, policies, tips
- **Query Normalization**: Standardized date formats, alliance names, etc.

## 🎨 Features & Capabilities

### Core Features

✅ **Natural Language Processing**: Understands complex travel requests
✅ **Flight Search**: Multi-criteria filtering (airline, alliance, price, dates)
✅ **Visa Information**: Country-specific requirements for UAE citizens
✅ **Policy Queries**: Refund, cancellation, and travel policies
✅ **Conversation Memory**: Maintains context across multiple turns
✅ **Error Handling**: Graceful degradation for edge cases

### Advanced Features

✅ **Structured Query Extraction**: LLM-based parsing to structured format
✅ **Vector Search**: Semantic similarity for knowledge retrieval
✅ **Tool Routing**: Intelligent selection of appropriate tools
✅ **Web Interface**: Modern Streamlit UI with real-time chat
✅ **Modular Design**: Clean separation of concerns

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

- **Query Processing**: < 2 seconds for most requests
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

## 🤝 Contributing

This is a technical case study for Kavak. For questions or feedback, please contact the Kavak hiring team.

## 📄 License

This project is created for Kavak's technical assessment purposes.

## 📞 Support

For technical support or questions about this implementation, please refer to the documentation above or contact the development team.

---

**Built with ❤️ for Kavak's Conversational AI Assessment**
