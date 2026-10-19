import os
import pypdf
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError("Uploaded PDF file not found.")
    extracted_text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        if not extracted_text.strip():
            raise ValueError("No extractable text found in this PDF.")
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise RuntimeError(f"Failed to extract text from PDF: {e}")
