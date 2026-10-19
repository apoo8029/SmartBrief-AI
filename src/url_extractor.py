import requests
from bs4 import BeautifulSoup
import ipaddress
import socket
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        ip_addr = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip_addr)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved:
            return False
        return True
    except Exception:
        return False

def fetch_article_text(url: str) -> str:
    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")
    if not is_safe_url(url):
        raise ValueError("Unsafe or private URL detected.")
    headers = {"User-Agent": "SmartBriefAI-Bot/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        paragraphs = soup.find_all("p")
        text = "\n".join([p.get_text() for p in paragraphs])
        if len(text.strip()) < 100:
            text = soup.get_text()
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to fetch URL content: {e}")
        raise RuntimeError(f"Could not retrieve article from URL: {e}")
