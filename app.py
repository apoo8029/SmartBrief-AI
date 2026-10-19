
import streamlit as st
from src.summarizer import SmartBriefSummarizer

st.set_page_config(
    page_title="SmartBrief AI",
    page_icon="📝",
    layout="centered",
)

@st.cache_resource(show_spinner="Loading BART AI model...")
def load_summarizer():
    return SmartBriefSummarizer()

st.title("📝 SmartBrief AI")
st.caption("Transform long articles into concise summaries using BART.")

st.markdown(
    "Paste an English article below and generate an "
    "AI-written summary."
)

article_text = st.text_area(
    "Article text",
    height=260,
    max_chars=30000,
    placeholder="Paste your article here...",
)

col1, col2 = st.columns(2)

with col1:
    max_length = st.slider(
        "Maximum summary tokens",
        min_value=40,
        max_value=150,
        value=100,
    )

with col2:
    min_length = st.slider(
        "Minimum summary tokens",
        min_value=10,
        max_value=50,
        value=30,
    )

if st.button("Generate Summary", type="primary"):
    if not article_text.strip():
        st.warning("Please enter some text first.")

    elif len(article_text.split()) < 30:
        st.warning(
            "Please provide a longer article "
            "(at least 30 words)."
        )

    elif min_length >= max_length:
        st.warning(
            "Minimum summary length must be less "
            "than maximum length."
        )

    else:
        try:
            with st.spinner(
                "Generating summary with BART..."
            ):
                summarizer = load_summarizer()

                result = summarizer.summarize_long_document(
                    article_text,
                    max_length=max_length,
                    min_length=min_length,
                )

            summary = result["summary"]

            if not summary or not summary.strip():
                st.error("The model returned an empty summary.")
            elif summary.startswith("[Error:"):
                st.error(
                    "Summarization failed. Check the app logs."
                )
            else:
                st.success("AI summary generated successfully!")

                st.subheader("📌 Summary")
                st.write(summary)

                st.subheader("📊 Document statistics")

                a, b, c = st.columns(3)

                a.metric(
                    "Original tokens",
                    result["original_tokens"],
                )
                b.metric(
                    "Chunks processed",
                    result["chunks_processed"],
                )
                b_metric = len(summary.split())
                c.metric("Summary words", b_metric)

                st.caption(
                    "Summary word count is approximate; "
                    "model token count is different."
                )

                st.download_button(
                    "Download summary",
                    data=summary,
                    file_name="smartbrief_summary.txt",
                    mime="text/plain",
                )

        except Exception:
            st.error(
                "Could not generate the summary. "
                "Check the app logs for the technical error."
            )
            st.exception(Exception("Model initialization or inference failed"))
