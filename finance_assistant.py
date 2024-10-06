

import os
from dotenv import load_dotenv
import logging

from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
import chainlit as cl  # Import Chainlit for app integration

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set OpenAI API key from .env file

# Manually define identity function


# Initialize Assistant using GPT-4 and YFinance tools
assistant = Assistant(
    llm=OpenAIChat(model="gpt-4"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    show_tool_calls=True,
    markdown=True,
)

# Query the assistant for stock price of NVDA (NVIDIA)
assistant.print_response("What is the stock price of NVDA")

# Query the assistant for a comparison between NVIDIA (NVDA) and AMD using all available tools
assistant.print_response("Write a comparison between NVDA and AMD, use all tools available.")
