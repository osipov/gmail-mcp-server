from mcp import ListToolsResult
import streamlit as st
import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.human_input.types import (
    HumanInputRequest,
    HumanInputResponse,
)

def streamlit_input_callback(request: HumanInputRequest) -> HumanInputResponse:
    """Request input from a human user via Streamlit."""
    # Display the description if available
    if request.description:
        st.info(f"**[HUMAN INPUT NEEDED]** {request.description}")

    # Display the prompt
    response = st.text_input(request.prompt, key=request.request_id)

    # A submit button to finalize the input
    if st.button("Submit", key=f"submit_{request.request_id}"):
        if response.strip():
            return HumanInputResponse(request_id=request.request_id, response=response.strip())
        else:
            st.error("Response cannot be empty. Please enter a valid response.")

    # Return None to indicate that no response has been finalized yet
    st.warning("Awaiting response...")
    return None

# Initialize the MCPApp at module level
app = MCPApp(name="gmail_agent", human_input_callback=streamlit_input_callback)

def format_list_tools_result(list_tools_result: ListToolsResult):
    res = ""
    for tool in list_tools_result.tools:
        res += f"- **{tool.name}**: {tool.description}\n\n"
    return res

def get_user_feedback():
    selected = st.feedback("thumbs")
    return bool(selected)


async def get_gmail_agent():
    """Get existing agent from session state or create a new one"""
    if "agent" not in st.session_state:
        gmail_agent = Agent(
            name="gmail",
            instruction="""You are an agent with access to a specific Gmail account.
            Your job is to execute email tasks based on a user's request.
            Based on the user's request make the appropriate tool calls.
            Always request permission from the user before sending any emails.
            To ask the user for permission, use the tool that requests customer/human input.""",
            server_names=["gmail"],
            connection_persistence=False,
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
                response = await llm.generate_str(
                    message=prompt,
                    use_history=True
                )
            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    asyncio.run(main())