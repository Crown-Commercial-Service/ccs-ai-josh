import streamlit as st
import datetime

# --- Configuration ---
st.set_page_config(
    page_title="Azure Streamlit Test",
    layout="centered"
)

# --- App Content ---
st.title("✅ Azure Streamlit Deployment Test")

st.markdown("""
This simple application confirms that the Azure App Service:
1. Successfully installed the Streamlit dependency (`requirements.txt`).
2. Executed the custom startup command (`streamlit run demo_app.py ...`).
3. Correctly handled the web socket connections necessary for interactivity.
""")

# --- Interactivity Test 1: Slider ---
st.header("1. Interactivity Test")
value = st.slider(
    "Select a value:",
    min_value=0,
    max_value=100,
    value=50,
    help="Moving the slider confirms the application state is updating correctly."
)

st.info(f"The current selected value is: **{value}**")

# --- Interactivity Test 2: Button ---
st.header("2. State & Button Test")

if st.button("Click Me to Confirm"):
    st.success(f"Button clicked at {datetime.datetime.now().strftime('%H:%M:%S')}")
    st.balloons()
else:
    st.write("Awaiting confirmation...")

# --- Environment Check (For advanced debugging) ---
with st.expander("Deployment Info"):
    st.write("If you see this page, your Terraform setup for Azure App Service is fundamentally working.")
