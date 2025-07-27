"""
Twilight Imperium Chatbot Agent
Step 5: Complete LangChain agent with GPT-4o

This module creates the final chatbot that combines rule searching with
intelligent reasoning to answer Twilight Imperium questions accurately.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate

# Import our custom search tool
from twilight_rules_tool import create_twilight_rules_tool

# Load environment variables
load_dotenv()


class TwilightImperiumChatbot:
    """
    Complete Twilight Imperium Fourth Edition chatbot with rule search capabilities
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        """
        Initialize the chatbot
        
        Args:
            model_name: OpenAI model to use (gpt-4o, gpt-4, etc.)
            temperature: Response creativity (0.0 = focused, 1.0 = creative)
        """
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize components
        self._setup_llm()
        self._setup_tools()
        self._setup_memory()
        self._setup_agent()
        
        print(f"✅ Twilight Imperium Chatbot initialized with {model_name}")
    
    def _setup_llm(self):
        """Initialize the OpenAI LLM"""
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
        """Setup the rule search tool"""
        try:
            self.search_tool = create_twilight_rules_tool()
            self.tools = [self.search_tool]
            print("✅ Rules search tool loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load search tool: {e}")
    
    def _setup_memory(self):
        """Setup conversation memory"""
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Remember last 5 exchanges
        )
        print("✅ Conversation memory initialized")
    
    def _setup_agent(self):
        """Setup the LangChain agent with custom prompt"""
        
        # Create the system prompt
        system_prompt = """You are a helpful and knowledgeable assistant for Twilight Imperium Fourth Edition, 
a complex board game. Your role is to help players understand rules, mechanics, and strategies.

IMPORTANT GUIDELINES:
1. **Always search the rules first** when asked about game mechanics, rules, or gameplay questions
2. **Analyze ALL search results carefully** to find the most relevant information
3. **Consider context** - distinguish between similar terms (e.g., "Leadership" strategy card vs "Leaders" units)
4. **Be precise and accurate** - base your answers on the official rules
5. **Cite your sources** when referencing specific rules
6. **Ask for clarification** if a question is ambiguous
7. **Be encouraging** - help players learn this complex game

RESPONSE FORMAT:
- Give clear, direct answers
- Include relevant rule details
- Mention source (Learn to Play vs Rulebook)
- Suggest follow-up topics when helpful

COMMON TOPICS:
- Strategy Cards (Leadership, Diplomacy, Politics, etc.)
- Combat (Space Combat, Ground Combat)
- Movement and Tactical Actions
- Victory Conditions and Objectives
- Faction Abilities
- Technology and Upgrades
- Trading and Negotiation

Remember: If search results seem conflicting or unclear, analyze the context to determine 
which information actually answers the user's question."""

        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,  # Shows thinking process
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": system_prompt
            }
        )
        
        print("✅ LangChain agent initialized with Twilight Imperium expertise")
    
    def chat(self, user_input: str) -> str:
        """
        Process a user message and return the chatbot's response
        
        Args:
            user_input: User's question or message
            
        Returns:
            Chatbot's response
        """
        try:
            response = self.agent.run(user_input)
            return response
        except Exception as e:
            error_msg = f"I encountered an error: {e}. Please try rephrasing your question."
            print(f"❌ Error in chat: {e}")
            return error_msg
    
    def start_conversation(self):
        """Start an interactive chat session"""
        print("\n" + "="*80)
        print("🚀 TWILIGHT IMPERIUM FOURTH EDITION ASSISTANT")
        print("="*80)
        print("Ask me anything about Twilight Imperium rules and gameplay!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for example questions.")
        print("-"*80)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤖 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n👋 Thanks for playing! May the galaxy be yours!")
                    break
                
                # Handle help command
                if user_input.lower() in ['help', 'h']:
                    self._show_help()
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Get response from the agent
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
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
            "How do I research technology?",
            "What are action cards?",
            "How does trading work?",
            "What are faction abilities?",
            "How do I resolve space combat?"
        ]
        
        print("\n📝 Example Questions:")
        print("-" * 40)
        for i, example in enumerate(examples, 1):
            print(f"{i:2d}. {example}")
        print("-" * 40)
        print("💡 Pro tip: Be specific about what you want to know!")


def test_chatbot():
    """Test the chatbot with the context problem from earlier"""
    print("🧪 Testing Twilight Imperium Chatbot")
    print("="*60)
    
    try:
        # Initialize the chatbot
        chatbot = TwilightImperiumChatbot()
        
        # Test the context problem from earlier
        test_question = "What does leadership do in the game?"
        
        print(f"\n🔸 Test Question: '{test_question}'")
        print("-" * 50)
        
        response = chatbot.chat(test_question)
        
        print(f"\n🤖 Response:\n{response}")
        
        print("\n✅ Chatbot test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing chatbot: {e}")
        return False


if __name__ == "__main__":
    """
    Run this script to start the interactive chatbot
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        test_chatbot()
    else:
        # Run interactive chat
        try:
            chatbot = TwilightImperiumChatbot()
            chatbot.start_conversation()
        except Exception as e:
            print(f"❌ Failed to start chatbot: {e}")
            print("Please ensure:")
            print("1. Your OpenAI API key is set")
            print("2. You've completed Steps 1-4 (PDFs processed and vector store created)")
            print("3. All required packages are installed") 