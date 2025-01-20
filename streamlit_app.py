from mcp import ListToolsResult
import streamlit as st
import asyncio
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

app = MCPApp(name="gmail_agent")

def format_list_tools_result(list_tools_result: ListToolsResult):
    res = ""
    for tool in list_tools_result.tools:
        res += f"- **{tool.name}**: {tool.description}\n\n"
    return res


async def main():
    await app.initialize()

    gmail_agent = Agent(
        name="gmail",
        instruction="""You are an agent with access to a specific Gmail account.
        Your job is to execute email tasks based on a user's request.
        Based on the user's request make the appropriate tool calls.""",
        server_names=["gmail"],
    )
    await gmail_agent.initialize()
    llm = await gmail_agent.attach_llm(AnthropicAugmentedLLM)

    tools = await gmail_agent.list_tools()
    tools_str = format_list_tools_result(tools)

    with st.expander("View Tools"):
        st.markdown(tools_str)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            response = ""
            with st.spinner("Thinking..."):
                response = await llm.generate_str(message=st.session_state["messages"], use_history=True)
            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})
    st.write(st.session_state.messages)

if __name__ == "__main__":
    

    asyncio.run(main())