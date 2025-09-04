from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools

from dotenv import load_dotenv
import os
load_dotenv()

# 创建基础智能体
base_url=os.environ.get("OPENAI_API_BASE")
print(base_url)
agent_storage: str = "tmp/agents.db"

web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o", base_url=os.environ.get("OPENAI_API_BASE")),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    # Store the agent sessions in a sqlite database
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    # Adds the current date and time to the instructions
    add_datetime_to_instructions=True,
    # Adds the history of the conversation to the messages
    add_history_to_messages=True,
    # Number of history responses to add to the messages
    num_history_responses=5,
    # Adds markdown formatting to the messages
    markdown=True,
)

news_agent = Agent(
    name="News Agent",
    model=OpenAIChat(id="gpt-4o", base_url=os.environ.get("OPENAI_API_BASE")),
    tools=[DuckDuckGoTools(search=True, news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteStorage(table_name="news_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

simple_test_app = Playground(agents=[web_agent, news_agent])
app = simple_test_app.get_app()

if __name__ == "__main__":
    simple_test_app.serve("simple_test:app", reload=True)
