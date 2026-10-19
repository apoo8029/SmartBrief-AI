import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SmartBriefSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn") -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing summarizer with model: {self.model_name} on device: {self.device.upper()}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("Model and tokenizer successfully loaded into memory.")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization error: {e}")

    def summarize_chunk(self, text_chunk: str, max_length: int = 80, min_length: int = 25) -> str:
        if not text_chunk or not text_chunk.strip():
            return ""
        try:
            inputs = self.tokenizer(
                [text_chunk],
                max_length=1024,
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            with torch.inference_mode():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error during chunk summarization: {str(e)}")
            return f"[Error: {str(e)}]"

    def summarize_long_document(self, text: str, max_length: int = 100, min_length: int = 30) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")
            
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)
        
        if total_tokens <= 1000:
            summary = self.summarize_chunk(text, max_length=max_length, min_length=min_length)
            return {"summary": summary, "chunks_processed": 1, "original_tokens": total_tokens}
            
        chunk_size = 800
        overlap = 100
        chunk_summaries = []
        
        for i in range(0, total_tokens, chunk_size - overlap):
            chunk_tokens = tokens[i: i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunk_summary = self.summarize_chunk(chunk_text, max_length=60, min_length=20)
            chunk_summaries.append(chunk_summary)
            
        combined_summaries = " ".join(chunk_summaries)
        final_summary = self.summarize_chunk(combined_summaries, max_length=max_length, min_length=min_length)
        
        return {"summary": final_summary, "chunks_processed": len(chunk_summaries), "original_tokens": total_tokens}
