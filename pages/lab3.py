import streamlit as st
from openai import OpenAI


st.title("Lab 3: Chatbot Basics")

openai_api_key = st.secrets.get("API_KEY")  
if not openai_api_key:
    st.error("Missing API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()
client = OpenAI(api_key=openai_api_key)

st.sidebar.header("Model Settings")

model = st.sidebar.selectbox("Use advanced model", ("Standard", "Mini"))
if model == "Standard":
    model = "gpt-5"
else:
    model = "gpt-5-mini"

if 'client' not in st.session_state:
    st.session_state.client = OpenAI(api_key=openai_api_key)

if 'messages' not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": "You are a helpful assistant who explains things in a way that a 10-year-old can understand. Use simple words, short sentences, and lighthearted examples. After answering a question, always ask 'Do you want more info?' If the user says yes, provide more details and ask again. If the user says no, go back to asking 'How can I help?'"
        },
        {
            "role": "assistant",
            "content": "Hey, how can I help you today?"
        }
    ]

def maintain_buffer(messages, max_user_messages=2):
    system_message = None
    non_system_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg
        else:
            non_system_messages.append(msg)
    
    if len(non_system_messages) <= 1:
        return messages
    
    # Find user messages and indices
    user_indices = [i for i, msg in enumerate(non_system_messages) if msg["role"] == "user"]
    
    if len(user_indices) <= max_user_messages:
        return messages
    
    # Keep only last user message
    indices_to_keep = set(user_indices[-max_user_messages:])
    
    # Keep the assistant response that follows
    for idx in list(indices_to_keep):
        if idx + 1 < len(non_system_messages) and non_system_messages[idx + 1]["role"] == "assistant":
            indices_to_keep.add(idx + 1)
    
    # Rebuild messages list
    filtered = [non_system_messages[i] for i in range(len(non_system_messages)) if i in indices_to_keep]
    
    # Prepend system message
    if system_message:
        return [system_message] + filtered
    return filtered

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    chat_role = st.chat_message(msg["role"])
    chat_role.write(msg["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model,
        messages=st.session_state.messages,
        stream=True,
    )
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Maintain buffer
    st.session_state.messages = maintain_buffer(st.session_state.messages)
    
    # Show latest response
    st.rerun()

