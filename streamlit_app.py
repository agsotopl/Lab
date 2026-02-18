import streamlit as st

st.set_page_config(
    page_title="Labs",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define each lab page
lab1 = st.Page("pages/lab1.py", title="Lab 1")
lab2 = st.Page("pages/lab2.py", title="Lab 2")
lab3 = st.Page("pages/lab3.py", title="Lab 3")
lab4 = st.Page("pages/lab4.py", title="Lab 4")
lab5 = st.Page("pages/lab5.py", title="Lab 5", default=True)

# Build navigation
pg = st.navigation([lab1, lab2, lab3, lab4, lab5])

# Run whichever page the user selects
pg.run()
