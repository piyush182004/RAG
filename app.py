import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
from pdf2image import convert_from_bytes
from PIL import Image
import io
import base64
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAI

# Set custom Streamlit page config for a modern look
st.set_page_config(
    page_title="Multi-Modal RAG Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for a clean, white background and black text with subtle detailing
st.markdown(
    """
    <style>
    body, .stApp {
        background: #fff !important;
        color: #111 !important;
    }
    .main .block-container {
        background: #fafbfc !important;
        border-radius: 12px;
        box-shadow: 0 2px 12px 0 #e0e0e0;
        padding: 2rem 2rem 2rem 2rem;
        margin-top: 1.5rem;
    }
    .stTitle h1 {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 2.2rem;
        letter-spacing: 0.04em;
        color: #222;
    }
    .stMarkdown, .stAlert, .stTextInput, .stFileUploader, .stButton, .stSpinner {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1.08rem;
        color: #222 !important;
    }
    .stTextInput > div > input {
        background: #fff !important;
        color: #111 !important;
        border-radius: 7px;
        border: 1px solid #bbb;
        font-size: 1.08rem;
    }
    .stFileUploader > div {
        background: #fff !important;
        border: 1.2px solid #bbb;
        border-radius: 7px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #f5f5f5 0%, #e0e0e0 100%) !important;
        color: #222 !important;
        border-radius: 7px;
        font-weight: 500;
        font-size: 1.08rem;
        box-shadow: 0 1px 4px #e0e0e0;
        transition: 0.2s;
        border: 1px solid #bbb;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #e0e0e0 0%, #f5f5f5 100%) !important;
        color: #111 !important;
        box-shadow: 0 2px 8px #e0e0e0;
    }
    img {
        border-radius: 8px;
        box-shadow: 0 1px 8px #e0e0e0;
        margin-bottom: 1.2rem;
    }
    .answer-block {
        background: #f3f6fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1.2rem;
        box-shadow: 0 1px 8px #e0e0e0;
        color: #111;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Set Poppler path to the provided directory
POPPLER_PATH = r"C:\Users\hp\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
if not os.path.exists(POPPLER_PATH):
    st.error(f"Poppler directory not found at {POPPLER_PATH}. Ensure the directory exists and contains pdftoppm.exe.")
    st.stop()
if not os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm.exe")):
    st.error(f"pdftoppm.exe not found in {POPPLER_PATH}. Ensure Poppler is correctly installed.")
    st.stop()
os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("🤖 Multi-Modal RAG Q&A")
st.markdown("<span style='font-size:1.1rem; color:#222;'>Upload a PDF and ask questions. Answers will include relevant text and images.</span>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)
st.info("Tip: Ensure dependencies are installed via setup.sh or pip install -r requirements.txt before running.")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'images' not in st.session_state:
    st.session_state.images = []  # Explicitly initialize images

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

def process_pdf(file):
    """Extract text and images from a PDF file using PyPDF2 and pdf2image."""
    try:
        # Read PDF file for text
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            text += extracted_text if extracted_text else ""
        
        if not text.strip():
            st.warning("No text extracted from PDF. Proceeding with image extraction only.")
        
        # Reset file pointer to beginning for pdf2image
        file.seek(0)
        # Convert PDF to images using Poppler
        try:
            images = convert_from_bytes(file.read(), poppler_path=POPPLER_PATH)
            st.info(f"Extracted {len(images)} images from PDF.")
        except Exception as e:
            st.error(f"Failed to process PDF images: {str(e)}. Ensure Poppler binaries (pdftoppm.exe) and required DLLs are in {POPPLER_PATH}.")
            return [], []
        
        # Create documents for text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(text) if text.strip() else []
        documents = [Document(page_content=chunk, metadata={"source": "pdf", "page": i+1}) 
                     for i, chunk in enumerate(text_chunks)]
        
        # Store images with metadata
        image_data = []
        for i, img in enumerate(images):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data.append({"base64": img_base64, "page": i+1})
        
        return documents, image_data
    
    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}. Check if the PDF is valid and not corrupted.")
        return [], []

def create_vectorstore(documents):
    """Create a FAISS vector store from documents."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def plt_img_base64(img_base64):
    """Display base64-encoded image in Streamlit."""
    try:
        st.image(base64.b64decode(img_base64), caption="Relevant Image")
    except Exception as e:
        st.warning(f"Failed to display image: {str(e)}")

def format_docs_with_images(docs):
    """Format retrieved documents and include relevant images."""
    context = "\n".join(doc.page_content for doc in docs)
    relevant_images = []
    if 'images' in st.session_state and st.session_state.images:
        for doc in docs:
            page_num = doc.metadata.get("page", None)
            if page_num:
                for img in st.session_state.images:
                    if img["page"] == page_num:
                        relevant_images.append(img["base64"])
    st.info(f"Retrieved {len(relevant_images)} relevant images for context.")
    return {"context": context, "images": relevant_images}

def process_with_openrouter_vision(prompt_text, image_base64_list=None):
    """Process text and optional images with OpenRouter's vision model."""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        messages = [{"role": "user", "content": []}]
        
        messages[0]["content"].append({
            "type": "text", 
            "text": prompt_text
        })
        
        if image_base64_list and len(image_base64_list) > 0:
            for img_base64 in image_base64_list:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
        
        response = client.chat.completions.create(
            model="meta-llama/llama-3.2-11b-vision-instruct:free",
            messages=messages,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error with OpenRouter: {str(e)}")
        return None

# Process uploaded PDF
if uploaded_file:
    with st.spinner("Processing PDF..."):
        st.session_state.documents, st.session_state.images = process_pdf(uploaded_file)
        if st.session_state.documents or st.session_state.images:
            if st.session_state.documents:
                st.session_state.vectorstore = create_vectorstore(st.session_state.documents)
            st.success(f"PDF processed successfully! Extracted {len(st.session_state.documents)} text chunks and {len(st.session_state.images)} images.")
        else:
            st.error("No content (text or images) extracted from PDF. Please upload a valid PDF.")

# Query input
st.markdown("<hr style='border:1px solid #bbb; margin:1.5rem 0;'>", unsafe_allow_html=True)
query = st.text_input("Ask a question about the document:", placeholder="e.g. What is the summary of page 2?", help="Type your question and press Enter.")

# Set up multi-modal RAG chain
if st.session_state.vectorstore and query:
    # Create prompt template
    prompt_template = """
    You are an assistant that answers questions based on a document's text and images.
    Use the provided context to answer the question accurately.
    If an image is relevant, describe its content in the answer and reference it as [Image: Page X].
    Question: {question}
    Context: {context}
    """
    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_template)

    # Create RAG chain
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def run_chain(input_data):
        retrieved_docs = retriever.invoke(input_data["question"])
        context_data = format_docs_with_images(retrieved_docs)
        text_context = context_data["context"]
        images = context_data.get("images", [])
        
        formatted_prompt = prompt.format(
            question=input_data["question"],
            context=text_context
        )
        
        return process_with_openrouter_vision(formatted_prompt, images)
    
    # Run query
    with st.spinner("Generating answer..."):
        try:
            response = run_chain({"question": query})
            if response:
                # Display response
                st.markdown("<div class='answer-block'><b>Answer:</b><br>" + response + "</div>", unsafe_allow_html=True)
                
                # Display relevant images
                retrieved_docs = retriever.invoke(query)
                displayed_images = set()  # Track displayed images to avoid duplicates
                if 'images' in st.session_state and st.session_state.images:
                    st.markdown("### Relevant Images")
                    for doc in retrieved_docs:
                        page_num = doc.metadata.get("page", None)
                        if page_num:
                            for img in st.session_state.images:
                                if img["page"] == page_num and img["base64"] not in displayed_images:
                                    plt_img_base64(img["base64"])
                                    displayed_images.add(img["base64"])
                    if not displayed_images:
                        st.info("No relevant images found for the retrieved documents.")
                else:
                    st.info("No images available to display.")
            else:
                st.error("No response from OpenRouter. Please try again.")
        except Exception as e:
            st.error(f"Failed to process query: {str(e)}")
elif query and not st.session_state.vectorstore:
    st.error("Please upload and process a PDF before asking a question.")