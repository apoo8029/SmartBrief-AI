
import streamlit as st
from src.summarizer import SmartBriefSummarizer

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SmartBrief AI | Text Summarizer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #808080;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD MODEL ONCE AND CACHE IT
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_summarizer():
    return SmartBriefSummarizer()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Customize your summary.")

    max_length = st.slider(
        "Maximum summary tokens",
        min_value=40,
        max_value=150,
        value=100,
        step=10,
    )

    min_length = st.slider(
        "Minimum summary tokens",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
    )

    st.divider()

    st.markdown("### 🤖 AI Model")
    st.write("BART Large CNN")
    st.caption("Pretrained abstractive summarization model")

    st.markdown("### 💡 Tips")
    st.caption(
        "Use clear English text for better results. "
        "Long documents may take more time."
    )

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">📝 SmartBrief AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Turn lengthy articles into concise AI summaries.'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Paste an English article below. "
    "SmartBrief AI uses a pretrained BART model "
    "to generate an abstractive summary."
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
article_text = st.text_area(
    "📄 Enter your article or document",
    height=280,
    max_chars=30000,
    placeholder=(
        "Paste your article here...\n\n"
        "Example: Climate change is affecting weather "
        "patterns around the world..."
    ),
)

word_count = len(article_text.split())
character_count = len(article_text)

col1, col2 = st.columns(2)

with col1:
    st.metric("Input words", word_count)

with col2:
    st.metric("Characters", character_count)

# --------------------------------------------------
# VALIDATE SETTINGS
# --------------------------------------------------
if min_length >= max_length:
    st.warning(
        "Minimum summary tokens must be less than "
        "maximum summary tokens. Adjust the sidebar settings."
    )

# --------------------------------------------------
# GENERATE SUMMARY
# --------------------------------------------------
generate_clicked = st.button(
    "✨ Generate Summary",
    type="primary",
    use_container_width=True,
)

if generate_clicked:

    if not article_text.strip():
        st.warning("Please enter some text before generating a summary.")

    elif word_count < 30:
        st.warning(
            "Please enter a longer article containing at least 30 words."
        )

    elif min_length >= max_length:
        st.error(
            "Invalid summary length settings. "
            "Minimum length must be smaller than maximum length."
        )

    else:
        try:
            with st.spinner(
                "Loading AI model and generating your summary..."
            ):
                summarizer = load_summarizer()

                result = summarizer.summarize_long_document(
                    text=article_text,
                    max_length=max_length,
                    min_length=min_length,
                )

            summary = result.get("summary", "").strip()

            if not summary:
                st.error(
                    "The model returned an empty summary. "
                    "Try a different article."
                )

            elif summary.startswith("[Error:"):
                st.error(
                    "Summarization failed. Please check the application logs."
                )

            else:
                st.success("Summary generated successfully!")

                st.divider()

                # SUMMARY OUTPUT
                st.subheader("📌 Generated Summary")
                st.write(summary)

                # SUMMARY STATISTICS
                st.subheader("📊 Summary Statistics")

                original_tokens = result.get("original_tokens", 0)
                chunks_processed = result.get("chunks_processed", 0)
                summary_words = len(summary.split())

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Original tokens",
                    original_tokens,
                )

                metric2.metric(
                    "Chunks processed",
                    chunks_processed,
                )

                metric3.metric(
                    "Summary words",
                    summary_words,
                )

                st.caption(
                    "Token count and word count are different measurements."
                )

                # DOWNLOAD SUMMARY
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name="smartbrief_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                # OPTIONAL INPUT/OUTPUT VIEW
                with st.expander("View original article"):
                    st.write(article_text)

        except Exception:
            st.error(
                "Unable to generate the summary. "
                "The model may have failed to load, or the app "
                "may have reached its available memory limit."
            )

            st.info(
                "Check the Streamlit application logs for the "
                "technical error. If this is the first run, "
                "the model may need time to download."
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "SmartBrief AI | NLP & Abstractive Text Summarization "
    "| Built with Python, PyTorch, Transformers and Streamlit"
)
