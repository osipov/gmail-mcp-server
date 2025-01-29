from typing import Optional
import asyncio
import time

from mcp import ListToolsResult
import streamlit as st
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.human_input.types import (
    HumanInputRequest,
    HumanInputResponse,
)

async def streamlit_input_callback(request: HumanInputRequest) -> HumanInputResponse:
    if st.session_state.get("tool_approval", False):
        response = "Approved."
    else:
        response = "Denied."
    
    return HumanInputResponse(request_id=request.request_id, response=response)


def request_permission():
    if st.button("Approve"):
        st.session_state['permission_granted'] = True
    if st.button("Deny"):
        st.session_state['permission_granted'] = False

# Initialize the MCPApp at module level
app = MCPApp(name="gmail_agent", human_input_callback=streamlit_input_callback)

def format_list_tools_result(list_tools_result: ListToolsResult):
    res = ""
    for tool in list_tools_result.tools:
        res += f"- **{tool.name}**: {tool.description}\n\n"
    return res


async def get_gmail_agent():
    """Get existing agent from session state or create a new one"""
    if "agent" not in st.session_state:
        gmail_agent = Agent(
            name="gmail",
            instruction="""You are an agent with access to a specific Gmail account.
            Your job is to execute email tasks based on a user's request.
            Based on the user's request make the appropriate tool calls.
            Always check human input before sending or removing any emails.
            """,
            server_names=["gmail"],
            connection_persistence=False,
            # functions=[request_permission],
            human_input_callback=streamlit_input_callback,
        )
        await gmail_agent.initialize()
        st.session_state["agent"] = gmail_agent
        
    if "llm" not in st.session_state:
        st.session_state["llm"] = await st.session_state["agent"].attach_llm(AnthropicAugmentedLLM)
    
    return st.session_state["agent"], st.session_state["llm"]

async def main():
    # Initialize app
    await app.initialize()

    with st.sidebar:
        _ = st.toggle("Approve Tool Usage",
                  key="tool_approval",
                  value=True)
    
    # Get or create agent and LLM
    agent, llm = await get_gmail_agent()
    
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
        
    # Get tools only once and store in session state
    if "tools_str" not in st.session_state:
        tools = await agent.list_tools()
        st.session_state["tools_str"] = format_list_tools_result(tools)

    with st.expander("View Tools"):
        st.markdown(st.session_state["tools_str"])

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = ""
            with st.spinner("Thinking..."):
                response = await llm.generate(
                    message=prompt,
                    use_history=True,
                    parallel_tool_calls=False,
                )
                
                placeholder = st.empty()
                for msg in response:
                    if msg.stop_reason == "tool_use":
                        for c in msg.content:
                            if c.type == "tool_use":
                                placeholder.write(f"Calling tool: {c.name}")
                                
                    else:
                        text_response = ''
                        for c in msg.content:
                            placeholder.write(c.text)
                            text_response += c.text
                        st.session_state["messages"].append({"role": "assistant", "content": text_response})
                            
        st.rerun()

        # st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    asyncio.run(main())