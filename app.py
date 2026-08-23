import streamlit as st

from main import RAGPipeline


st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF Question Answering System")

st.write(
    "Upload your PDF documents and ask questions."
)


# =========================================================
# CREATE RAG PIPELINE
# =========================================================

if "rag" not in st.session_state:

    st.session_state.rag = RAGPipeline()

    st.session_state.processed = False


# =========================================================
# UPLOAD PDFs
# =========================================================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# =========================================================
# PROCESS PDFs
# =========================================================

if uploaded_files:

    st.info(
        f"{len(uploaded_files)} PDF(s) selected."
    )

    with st.expander("View uploaded files"):

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )

    if st.button(
        "Process PDFs",
        type="primary"
    ):

        with st.spinner(
            "Processing PDFs..."
        ):

            try:

                total_chunks = 0

                for uploaded_file in uploaded_files:

                    chunks = (
                        st.session_state.rag
                        .process_document(
                            uploaded_file
                        )
                    )

                    total_chunks += chunks

                st.session_state.processed = True

                st.success(
                    f"{len(uploaded_files)} PDF(s) processed successfully!"
                )

                st.info(
                    f"Total chunks created: {total_chunks}"
                )

            except Exception as e:

                st.error(
                    f"Error while processing PDFs: {e}"
                )


# =========================================================
# ASK QUESTION
# =========================================================

if st.session_state.get(
    "processed",
    False
):

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button(
        "Ask Question",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching documents..."
            ):

                try:

                    answer = (
                        st.session_state.rag
                        .ask(question)
                    )

                    st.subheader("🤖 Answer")

                    st.write(answer)

                except Exception as e:

                    st.error(
                        f"Error while generating answer: {e}"
                    )