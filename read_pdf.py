import sys
import subprocess
import os

try:
    import fitz
except ImportError:
    print("Installing PyMuPDF...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF"])
    import fitz

pdf_path = r"C:\Users\My Computer\Downloads\Dự án TAD-AI-3\Bài_sâu_răng_tiếng việt.pdf"
doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text() + "\n"

with open(r"C:\Users\My Computer\Downloads\Dự án TAD-AI-3\pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("Extracted text to pdf_text.txt successfully.")
