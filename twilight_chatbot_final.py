"""
Twilight Imperium Chatbot - Final Version
Fixed recursion issues and improved complex question handling

This version addresses the recursion limit problem and provides better
handling for multi-part questions and complex interactions.
"""

import os
import operator
from typing import Annotated, Dict, List, TypedDict, Sequence
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Import our custom search tool
from twilight_rules_tool import create_twilight_rules_tool

# Load environment variables
load_dotenv()


class ChatState(TypedDict):
    """Enhanced state for the chatbot conversation"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tool_call_count: int  # Track number of tool calls to prevent loops
    has_searched: bool    # Track if we've already searched


class TwilightImperiumFinalBot:
    """
    Final Twilight Imperium chatbot with recursion protection and improved question handling
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        """
        Initialize the final chatbot with recursion protection
        
        Args:
            model_name: OpenAI model to use
            temperature: Response creativity (0.0 = focused, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tool_calls = 3  # Limit tool calls per conversation turn
        
        # Initialize components
        self._setup_llm()
        self._setup_tools()
        self._setup_graph()
        
        print(f"✅ Final Twilight Imperium Chatbot initialized with {model_name}")
        print(f"🛡️  Recursion protection: Max {self.max_tool_calls} tool calls per turn")
    
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
        
        # Enhanced search tool with broader queries
        @tool
        def search_twilight_rules(query: str) -> str:
            """Search the official Twilight Imperium Fourth Edition rules. 
            Use this when users ask about game rules, mechanics, combat, movement, 
            strategy cards, victory conditions, faction abilities, or any gameplay questions.
            Try to make your search query comprehensive to get all relevant information at once."""
            return search_tool.func(query)
        
        self.tools = [search_twilight_rules]
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        print("✅ Tools setup complete with enhanced search capabilities")
    
    def _setup_graph(self):
        """Setup the LangGraph conversation flow with recursion protection"""
        
        def should_continue(state: ChatState) -> str:
            """Enhanced decision logic with recursion protection"""
            messages = state['messages']
            tool_call_count = state.get('tool_call_count', 0)
            has_searched = state.get('has_searched', False)
            
            if not messages:
                return "end"
            
            last_message = messages[-1]
            
            # RECURSION PROTECTION: Limit tool calls
            if tool_call_count >= self.max_tool_calls:
                print(f"🛡️  Tool call limit reached ({self.max_tool_calls}), providing answer with available information")
                # Force the agent to answer with what it has
                return "end"
            
            # If we have tool calls and haven't exceeded the limit, execute them
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "call_tool"
            else:
                return "end"
        
        def call_model(state: ChatState) -> Dict:
            """Call the LLM with enhanced prompting for complex questions"""
            messages = state['messages']
            tool_call_count = state.get('tool_call_count', 0)
            has_searched = state.get('has_searched', False)
            
            # Enhanced system message based on conversation state
            if not messages or not any(isinstance(msg, AIMessage) for msg in messages[:2]):
                
                system_content = """You are a knowledgeable assistant for Twilight Imperium Fourth Edition.

IMPORTANT GUIDELINES:
• For complex or multi-part questions, search ONCE with a comprehensive query that covers all aspects
• Analyze ALL search results carefully to find interconnected information
• Consider context (e.g., "Leadership" strategy card vs "Leaders" faction units, "Flank Speed" action card interactions)
• Be precise and accurate based on official rules
• If you don't find complete information in your search, work with what you have rather than searching again
• Help identify factions, cards, and mechanics even with partial descriptions

SEARCH STRATEGY:
• For questions about card interactions, search for both card names and the interaction type
• For faction-specific questions, include the faction name or key identifying features
• Make your search queries comprehensive to get all relevant information at once

I excel at explaining:
- Strategy Cards and Action Cards (Leadership, Flank Speed, etc.)
- Combat mechanics and movement interactions
- Faction abilities, flagships, and unique mechanics
- Card and ability interactions
- Victory conditions and objectives

Let me help you master the galaxy!"""
                
                # Add recursion protection info if we've made tool calls
                if tool_call_count > 0:
                    system_content += f"\n\nNOTE: You have made {tool_call_count}/{self.max_tool_calls} tool calls. Try to answer with the information you have."
                
                if tool_call_count >= self.max_tool_calls:
                    system_content += "\n\nIMPORTANT: You have reached the tool call limit. You MUST provide an answer now using any information from previous searches, even if incomplete. Do not make any more tool calls."
                
                system_message = HumanMessage(content=system_content)
                messages = [system_message] + messages
            
            # Call the LLM
            response = self.llm_with_tools.invoke(messages)
            
            return {
                "messages": [response],
                "tool_call_count": tool_call_count,
                "has_searched": has_searched
            }
        
        def call_tool(state: ChatState) -> Dict:
            """Execute tool calls with tracking"""
            messages = state['messages']
            tool_call_count = state.get('tool_call_count', 0)
            last_message = messages[-1]
            
            # Execute tool calls
            tool_messages = []
            new_tool_calls = 0
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    # Check if we're exceeding the limit
                    if tool_call_count + new_tool_calls >= self.max_tool_calls:
                        print(f"🛡️  Skipping additional tool calls to prevent recursion")
                        break
                    
                    # Get the tool function
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    
                    print(f"🔍 Tool call {tool_call_count + new_tool_calls + 1}: {tool_input.get('query', 'Unknown query')}")
                    
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
                    new_tool_calls += 1
            
            return {
                "messages": tool_messages,
                "tool_call_count": tool_call_count + new_tool_calls,
                "has_searched": True
            }
        
        # Create the graph with configuration
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
        
        # Compile the graph with recursion limit
        self.app = workflow.compile(
            checkpointer=None,
            debug=False
        )
        
        print("✅ LangGraph conversation flow initialized with recursion protection")
    
    def chat(self, user_input: str, conversation_history: List[BaseMessage] = None) -> str:
        """
        Process a user message with recursion protection
        
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
            
            # Create initial state with tracking
            initial_state = {
                "messages": messages,
                "tool_call_count": 0,
                "has_searched": False
            }
            
            # Run the graph with recursion limit configuration
            config = {"recursion_limit": 50}  # Increase from default 25
            result = self.app.invoke(initial_state, config=config)
            
            # Extract the final response
            final_messages = result.get("messages", [])
            tool_call_count = result.get("tool_call_count", 0)
            
            if final_messages:
                # Get the last AI message
                for message in reversed(final_messages):
                    if isinstance(message, AIMessage) and message.content:
                        response = message.content
                        
                        # Add info about tool usage if helpful
                        if tool_call_count >= self.max_tool_calls:
                            response += f"\n\n💡 *Note: I searched the rules {tool_call_count} times to answer your question.*"
                        
                        return response
            
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question or asking about specific aspects one at a time."
            
        except Exception as e:
            if "recursion" in str(e).lower():
                return "I encountered a complex question that requires breaking down into smaller parts. Could you ask about specific aspects separately? For example:\n• What is Flank Speed?\n• Which faction has a teleporting flagship?\n• How do these abilities interact?"
            else:
                error_msg = f"I encountered an error: {e}. Please try rephrasing your question."
                print(f"❌ Error in chat: {e}")
                return error_msg
    
    def start_conversation(self):
        """Start an interactive chat session with enhanced error handling"""
        print("\n" + "="*80)
        print("🚀 TWILIGHT IMPERIUM FOURTH EDITION ASSISTANT (Final Version)")
        print("="*80)
        print("Ask me anything about Twilight Imperium rules and gameplay!")
        print("For complex questions, I'll search efficiently and provide comprehensive answers.")
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
                print("Please try again or break complex questions into smaller parts.")
    
    def _show_help(self):
        """Show example questions with tips for complex queries"""
        examples = [
            "What does the Leadership strategy card do?",
            "How do I move ships in combat?",
            "What are the Arborec's special abilities?",
            "How does Flank Speed action card work?",
            "Which faction has a flagship that can teleport?",
            "How does the Ghosts of Creuss wormhole travel work?",
            "What happens when I activate a system?",
            "How do action cards interact with faction abilities?",
            "Tell me about space combat mechanics",
            "What are the victory conditions?"
        ]
        
        print("\n📝 Example Questions:")
        print("-" * 40)
        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")
        print("-" * 40)
        print("💡 Tips for complex questions:")
        print("   • Ask about specific cards or factions by name when possible")
        print("   • For interactions, mention both elements (e.g., 'Flank Speed with Ghosts flagship')")
        print("   • If you get recursion errors, break the question into smaller parts")
        print("🔄 Use 'clear' to reset conversation history")


def test_final_chatbot():
    """Test the final chatbot with complex questions"""
    print("🧪 Testing Final Twilight Imperium Chatbot")
    print("="*60)
    
    try:
        # Initialize the chatbot
        chatbot = TwilightImperiumFinalBot()
        
        # Test questions including the problematic one
        test_questions = [
            "What does leadership do in the game?",
            "How does Flank Speed action card work with faction abilities?",
            "Which faction has a flagship that can teleport and how does it work?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🔸 Test {i}: '{question}'")
            print("-" * 50)
            
            response = chatbot.chat(question)
            print(f"🤖 Response: {response[:300]}...")
            if len(response) > 300:
                print("[Response truncated for test display]")
        
        print("\n✅ Final chatbot test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing chatbot: {e}")
        return False


if __name__ == "__main__":
    """
    Run this script to start the final chatbot
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        test_final_chatbot()
    else:
        # Run interactive chat
        try:
            chatbot = TwilightImperiumFinalBot()
            chatbot.start_conversation()
        except Exception as e:
            print(f"❌ Failed to start chatbot: {e}")
            print("Please ensure:")
            print("1. Your OpenAI API key is set")
            print("2. You've completed all previous steps (PDFs processed and faction data integrated)")
            print("3. All required packages are installed") 