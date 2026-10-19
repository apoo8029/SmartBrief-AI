import pytest
from unittest.mock import MagicMock, patch
from src.summarizer import SmartBriefSummarizer
from src.url_extractor import is_safe_url

def test_is_safe_url_valid():
    assert is_safe_url("https://example.com") == True

def test_is_safe_url_private_ip():
    assert is_safe_url("http://127.0.0.1") == False
    assert is_safe_url("http://localhost") == False

def test_summarizer_empty_input():
    with patch("src.summarizer.AutoTokenizer.from_pretrained"), \
         patch("src.summarizer.AutoModelForSeq2SeqLM.from_pretrained"):
        
        summarizer = SmartBriefSummarizer()
        with pytest.raises(ValueError):
            summarizer.summarize_long_document("")
        with pytest.raises(ValueError):
            summarizer.summarize_long_document("   ")

def test_chunking_logic():
    with patch("src.summarizer.AutoTokenizer.from_pretrained"), \
         patch("src.summarizer.AutoModelForSeq2SeqLM.from_pretrained"):
        
        summarizer = SmartBriefSummarizer()
        summarizer.tokenizer = MagicMock()
        summarizer.tokenizer.encode.return_value = list(range(1200)) 
        summarizer.tokenizer.decode.side_effect = lambda x, **kwargs: "Mock chunk text"
        summarizer.summarize_chunk = MagicMock(return_value="Mock summary chunk")
        
        result = summarizer.summarize_long_document("A very long document text...", max_length=80, min_length=20)
        
        assert "summary" in result
        assert result["chunks_processed"] > 1
        assert result["original_tokens"] == 1200
