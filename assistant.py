from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") 

assistant = Assistant(tools=[DuckDuckGo()], show_tool_calls=True)
assistant.print_response("Whats happening in France?", markdown=True)
