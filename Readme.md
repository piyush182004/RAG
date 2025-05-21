# Multi-Modal RAG Q&A

A professional, clean Streamlit app for multi-modal Retrieval-Augmented Generation (RAG) question answering over PDF documents. The app extracts both text and images from uploaded PDFs, builds a FAISS vector store for semantic search, and uses OpenRouter's Llama-3.2-11B-Vision model to answer user questions with relevant text and images.

---

## Features
- **PDF Upload:** Upload any PDF document for analysis.
- **Text & Image Extraction:** Extracts both text and images from the PDF.
- **Semantic Search:** Uses FAISS and MiniLM embeddings for fast, relevant retrieval.
- **Multi-Modal RAG:** Answers questions using both text and images from the document.
- **Vision LLM:** Utilizes OpenRouter's Llama-3.2-11B-Vision model for multi-modal understanding.
- **Modern UI:** Clean, white background with subtle styling for a professional look.

---

## Setup Instructions

### 1. Clone the Repository
```
git clone <your-repo-url>
cd rag-pro
```

### 2. Install Dependencies
Make sure you have Python 3.9+ and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) installed (for PDF image extraction).

Install Python dependencies:
```
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the project root with your OpenRouter API key:
```
OPENROUTER_API_KEY=sk-or-<your-openrouter-key>
```

### 4. Set Poppler Path (Windows)
Edit the `POPPLER_PATH` variable in `app.py` to match your Poppler install location. Example:
```
POPPLER_PATH = r"C:\Users\yourname\Downloads\poppler-xx\Library\bin"
```

### 5. Run the App
```
streamlit run app.py
```

---

## Usage
1. **Upload a PDF** using the file uploader.
2. **Ask a question** about the document in the input box (e.g., "What is the summary of page 2?").
3. The app will display an answer, referencing both text and images from the PDF.

---

## Tech Stack
- **Frontend/UI:** Streamlit (with custom CSS for a clean, modern look)
- **PDF Processing:** PyPDF2, pdf2image, Pillow
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace MiniLM
- **RAG Chain:** LangChain
- **Vision LLM:** OpenRouter (Llama-3.2-11B-Vision)

---

## Notes
- For best results, use high-quality, text-based PDFs.
- Poppler is required for image extraction from PDFs.
- The app is designed for local use and demo purposes. For production, secure your API keys and consider additional error handling.

---

## License
MIT License

---

## Credits
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenRouter](https://openrouter.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace](https://huggingface.co/)
- [Poppler](https://poppler.freedesktop.org/)
