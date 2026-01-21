import streamlit as st
from openai import OpenAI

# --- Load API key from Streamlit secrets (NO user input) ---
openai_api_key = st.secrets.get("API_KEY")  # change to "OPEN_API_KEY" if that's your key name

if not openai_api_key:
    st.error("Missing API_KEY in Streamlit secrets. Add it to .streamlit/secrets.toml.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# --- Page header ---
st.title("Lab 2")
st.write("Upload a document and choose a summary style from the sidebar.")

# --- Sidebar summary options ---
st.sidebar.header("Summary Style")

summary_choice = st.sidebar.radio(
    "Choose one:",
    (
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points",
    ),
    index=0,
)

# Build an instruction based on sidebar choice
if summary_choice == "100 words":
    format_instruction = (
        "Summarize the document in exactly 100 words. "
        "No title. No bullet points. Plain text only."
    )
elif summary_choice == "2 connecting paragraphs":
    format_instruction = (
        "Summarize the document in exactly two connected paragraphs. "
        "No bullet points. No headings."
    )
else:  # "5 bullet points"
    format_instruction = (
        "Summarize the document in exactly 5 bullet points. "
        "Each bullet must be one complete sentence."
    )

# --- File upload + action ---
uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

generate = st.button("Generate Summary", disabled=not uploaded_file)

if uploaded_file and generate:
    document = uploaded_file.read().decode("utf-8", errors="replace")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Follow the requested summary format exactly.",
        },
        {
            "role": "user",
            "content": f"DOCUMENT:\n{document}\n\nINSTRUCTIONS:\n{format_instruction}",
        },
    ]

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    st.write_stream(stream)
