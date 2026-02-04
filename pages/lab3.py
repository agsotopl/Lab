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
        {"role": "assistant", "content": "How can I help?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
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
    

