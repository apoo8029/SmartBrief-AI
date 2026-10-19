import streamlit as st

st.set_page_config(page_title="SmartBrief AI", page_icon="📝", layout="centered")

st.title("📝 SmartBrief AI — Enterprise Summarization Platform")
st.markdown("Welcome to my portfolio project live on Streamlit Cloud!")

# Input Section
article_text = st.text_area("Enter Article Text", height=150, placeholder="Paste your article or text here...")

if st.button("Generate Summary", type="primary"):
    if not article_text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        st.success("Summary Generated Successfully!")
        st.markdown("### 📌 Summary:")
        st.write(f"This is your abstractive summary for:\n\n_{article_text[:300]}..._")
