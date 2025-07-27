"""
Twilight Imperium Chatbot with LangGraph (Fixed Version)
Updated with correct imports for current LangGraph version

This version fixes import issues and uses the proper LangGraph API.
"""

import os
import operator
from typing import Annotated, Dict, List, TypedDict, Sequence
from dotenv import load_dotenv

# LangGraph and LangChain imports (updated)
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Import our custom search tool
from twilight_rules_tool import create_twilight_rules_tool

# Load environment variables
load_dotenv()


class ChatState(TypedDict):
    """State for the chatbot conversation"""
    messages: Annotated[Sequence[BaseMessage], operator.add]


class TwilightImperiumLangGraphBot:
    """
    Modern Twilight Imperium chatbot using LangGraph framework (Fixed Version)
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        """
        Initialize the LangGraph chatbot
        
        Args:
            model_name: OpenAI model to use
            temperature: Response creativity (0.0 = focused, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components
        self._setup_llm()
        self._setup_tools()
        self._setup_graph()
        
        print(f"✅ LangGraph Twilight Imperium Chatbot initialized with {model_name}")
    
    def _setup_llm(self):
        """Initialize the OpenAI LLM with tool calling"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=api_key
        )
        
        print(f"✅ Initialized {self.model_name} with temperature {self.temperature}")
    
    def _setup_tools(self):
        """Setup tools for the agent"""
        # Get our custom search tool
        search_tool = create_twilight_rules_tool()
        
        # Convert to LangGraph format
        @tool
        def search_twilight_rules(query: str) -> str:
            """Search the official Twilight Imperium Fourth Edition rules. 
            Use this when users ask about game rules, mechanics, combat, movement, 
            strategy cards, victory conditions, faction abilities, or any gameplay questions."""
            return search_tool.func(query)
        
        self.tools = [search_twilight_rules]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        print("✅ Tools setup complete with LangGraph integration")
    
    def _setup_graph(self):
        """Setup the LangGraph conversation flow"""
        
        def should_continue(state: ChatState) -> str:
            """Decide whether to continue with tools or end"""
            messages = state['messages']
            last_message = messages[-1]
            
            # If the last message has tool calls, execute them
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "call_tool"
            else:
                return "end"
        
        def call_model(state: ChatState) -> Dict:
            """Call the LLM with system prompt and conversation history"""
            messages = state['messages']
            
            # Add system message if this is the start of a new conversation
            if not messages or not any(isinstance(msg, AIMessage) for msg in messages[:2]):
                
                system_message = HumanMessage(content="""You are a knowledgeable assistant for Twilight Imperium Fourth Edition. 

IMPORTANT GUIDELINES:
• Always search the official rules when asked about game mechanics, faction abilities, or gameplay questions
• Analyze ALL search results to find the most relevant information  
• Consider context carefully (e.g., "Leadership" strategy card vs "Leaders" faction units)
• Be precise and accurate based on official rules
• Cite sources when referencing specific rules
• Ask for clarification if questions are ambiguous
• Be encouraging and help players learn this complex game

I excel at explaining:
- Strategy Cards (Leadership, Diplomacy, Politics, etc.)
- Combat mechanics (Space & Ground Combat)
- Movement and Tactical Actions
- Victory Conditions and Objectives
- Faction abilities, technologies, and unique units
- Trading and negotiation rules

Let me help you master the galaxy!""")
                
                messages = [system_message] + messages
            
            # Call the LLM
            response = self.llm_with_tools.invoke(messages)
            
            return {"messages": [response]}
        
        def call_tool(state: ChatState) -> Dict:
            """Execute any tool calls"""
            messages = state['messages']
            last_message = messages[-1]
            
            # Execute tool calls
            tool_messages = []
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    # Get the tool function
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    
                    # Execute the tool
                    if tool_name == "search_twilight_rules":
                        result = self.tools[0].func(**tool_input)
                    else:
                        result = f"Unknown tool: {tool_name}"
                    
                    # Create tool message
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call['id']
                    )
                    tool_messages.append(tool_message)
            
            return {"messages": tool_messages}
        
        # Create the graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("call_tool", call_tool)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "call_tool": "call_tool",
                "end": END
            }
        )
        
        # Add edge from tool back to agent
        workflow.add_edge("call_tool", "agent")
        
        # Compile the graph
        self.app = workflow.compile()
        
        print("✅ LangGraph conversation flow initialized")
    
    def chat(self, user_input: str, conversation_history: List[BaseMessage] = None) -> str:
        """
        Process a user message and return the chatbot's response
        
        Args:
            user_input: User's question or message
            conversation_history: Previous conversation messages
            
        Returns:
            Chatbot's response
        """
        try:
            # Prepare initial state
            if conversation_history is None:
                conversation_history = []
            
            # Add user message
            user_message = HumanMessage(content=user_input)
            messages = conversation_history + [user_message]
            
            # Create initial state
            initial_state = {
                "messages": messages
            }
            
            # Run the graph
            result = self.app.invoke(initial_state)
            
            # Extract the final response
            final_messages = result.get("messages", [])
            if final_messages:
                # Get the last AI message
                for message in reversed(final_messages):
                    if isinstance(message, AIMessage) and message.content:
                        return message.content
            
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            error_msg = f"I encountered an error: {e}. Please try rephrasing your question."
            print(f"❌ Error in chat: {e}")
            return error_msg
    
    def start_conversation(self):
        """Start an interactive chat session with memory"""
        print("\n" + "="*80)
        print("🚀 TWILIGHT IMPERIUM FOURTH EDITION ASSISTANT (LangGraph Fixed)")
        print("="*80)
        print("Ask me anything about Twilight Imperium rules and gameplay!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for example questions.")
        print("Type 'clear' to clear conversation history.")
        print("-"*80)
        
        # Conversation history
        conversation_history = []
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤖 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for playing! May the galaxy be yours!")
                    break
                
                # Handle clear command
                if user_input.lower() in ['clear', 'reset']:
                    conversation_history = []
                    print("🗑️  Conversation history cleared!")
                    continue
                
                # Handle help command
                if user_input.lower() in ['help', 'h']:
                    self._show_help()
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Get response from the agent
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.chat(user_input, conversation_history)
                print(response)
                
                # Update conversation history
                conversation_history.append(HumanMessage(content=user_input))
                conversation_history.append(AIMessage(content=response))
                
                # Keep conversation history manageable (last 10 exchanges)
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                
            except KeyboardInterrupt:
                print("\n\n👋 Conversation ended. Thanks for using the Twilight Imperium Assistant!")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Please try again or restart the assistant.")
    
    def _show_help(self):
        """Show example questions"""
        examples = [
            "What does the Leadership strategy card do?",
            "How do I move ships in combat?",
            "What are the victory conditions?",
            "How does ground combat work?",
            "What happens when I activate a system?",
            "What are the Arborec's special abilities?",
            "How does the Federation of Sol play differently?",
            "What's unique about the Ghosts of Creuss?",
            "Tell me about Sardakk N'orr's faction technology",
            "How do I use the Construction strategy card?"
        ]
        
        print("\n📝 Example Questions:")
        print("-" * 40)
        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")
        print("-" * 40)
        print("💡 Pro tip: Be specific about what you want to know!")
        print("🔄 Use 'clear' to reset conversation history")


def test_fixed_chatbot():
    """Test the fixed LangGraph chatbot"""
    print("🧪 Testing Fixed LangGraph Twilight Imperium Chatbot")
    print("="*60)
    
    try:
        # Initialize the chatbot
        chatbot = TwilightImperiumLangGraphBot()
        
        # Test questions including faction-specific ones
        test_questions = [
            "What does leadership do in the game?",
            "What are the Arborec's special abilities?",
            "How does space combat work?"
        ]
        
        for i, question in enumerate(test_questions[:2], 1):  # Test first 2
            print(f"\n🔸 Test {i}: '{question}'")
            print("-" * 50)
            
            response = chatbot.chat(question)
            print(f"🤖 Response: {response}")
        
        print("\n✅ Fixed LangGraph chatbot test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing chatbot: {e}")
        return False


if __name__ == "__main__":
    """
    Run this script to start the interactive LangGraph chatbot
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        test_fixed_chatbot()
    else:
        # Run interactive chat
        try:
            chatbot = TwilightImperiumLangGraphBot()
            chatbot.start_conversation()
        except Exception as e:
            print(f"❌ Failed to start chatbot: {e}")
            print("Please ensure:")
            print("1. Your OpenAI API key is set")
            print("2. You've completed Steps 1-6 (PDFs processed and faction data integrated)")
            print("3. All required packages are installed") 