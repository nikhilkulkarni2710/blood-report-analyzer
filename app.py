import os
import base64
import re
from io import BytesIO
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pytesseract
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Page Configuration & UI Styling ---
st.set_page_config(page_title="Blood Report AI Analyzer", page_icon="🩸", layout="wide")
st.title("🩸 Multimodal Blood Report AI Analyzer")
st.caption("Upload historical reports, parse unstructured data, and analyze trends using NVIDIA NIM & RAG.")

# --- 2. Sidebar Configuration ---
with st.sidebar:
    api_key = os.environ.get("NVIDIA_API_KEY", "")
if not api_key:
    st.error("Missing NVIDIA_API_KEY environment variable. Please add it to your Space Secrets.")
    st.stop()

with st.sidebar:
    st.header("📂 Upload Reports")
    uploaded_files = st.file_uploader(
        "Upload your 30 blood report PDFs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    process_btn = st.button("🚀 Process & Build Vector DB", disabled=not uploaded_files)

# Initialize models globally
@st.cache_resource
def load_models():
    # Using VILA for layout/vision reasoning, and NV-EmbedQA for matching medical terminology
    llm = ChatNVIDIA(model="nvidia/vila-15-8b", temperature=0.1)
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    return llm, embeddings

llm, embeddings = load_models()

# --- 3. Document Processing Pipeline ---
def redact_personal_information(page_image):
    ocr_data = pytesseract.image_to_data(
        page_image,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )
    lines = {}
    for index, text in enumerate(ocr_data["text"]):
        text = text.strip()
        if not text:
            continue
        line_key = (
            ocr_data["block_num"][index],
            ocr_data["par_num"][index],
            ocr_data["line_num"][index],
        )
        lines.setdefault(line_key, []).append(index)

    sensitive_labels = re.compile(
        r"\b(name|patient|address|phone|mobile|email|dob|birth|mrn|uhid|id|gender)\b",
        re.IGNORECASE,
    )
    address_line_keys = []
    redact_boxes = []
    ordered_lines = list(lines)
    for line_position, line_key in enumerate(ordered_lines):
        indexes = lines[line_key]
        line_text = " ".join(ocr_data["text"][index] for index in indexes)
        if sensitive_labels.search(line_text):
            redact_boxes.append(indexes)
            if re.search(r"\b(address|patient address)\b", line_text, re.IGNORECASE):
                address_line_keys.extend(ordered_lines[line_position + 1:line_position + 3])

    for line_key in address_line_keys:
        redact_boxes.append(lines[line_key])

    if not redact_boxes:
        return page_image

    redacted_image = page_image.copy()
    draw = ImageDraw.Draw(redacted_image)
    for indexes in redact_boxes:
        left = min(ocr_data["left"][index] for index in indexes)
        top = min(ocr_data["top"][index] for index in indexes)
        right = max(
            ocr_data["left"][index] + ocr_data["width"][index]
            for index in indexes
        )
        bottom = max(
            ocr_data["top"][index] + ocr_data["height"][index]
            for index in indexes
        )
        draw.rectangle((left - 8, top - 8, right + 8, bottom + 8), fill="black")
    return redacted_image


def process_uploaded_pdfs(files):
    documents_to_embed = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a temporary directory to store files during processing
    if not os.path.exists("temp_pdf_dir"):
        os.makedirs("temp_pdf_dir")

    for idx, uploaded_file in enumerate(files):
        status_text.text(f"Processing ({idx+1}/{len(files)}): {uploaded_file.name}")
        
        # Save uploaded file locally temporarily
        temp_pdf_path = os.path.join("temp_pdf_dir", f"report_{idx + 1}.pdf")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Convert pages to images to read tables / bypass marketing banners
        try:
            pages = convert_from_path(temp_pdf_path, dpi=200)
            
            for page_num, page_image in enumerate(pages):
                page_image = redact_personal_information(page_image)
                image_buffer = BytesIO()
                page_image.save(image_buffer, format="PNG")
                image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
                
                # Instruction to filter out promotions and grab structured data
                prompt = (
                    "Extract all medical biomarkers, test names, results, units, and reference ranges "
                    "from this image into a clean markdown table. Strictly ignore any advertisements, "
                    "promotional headers, pricing, or non-medical clinic banners. Never extract, repeat, "
                    "or infer names, addresses, phone numbers, email addresses, dates of birth, patient "
                    "IDs, or any other personally identifying information."
                )
                
                clean_text_output = llm.invoke([
                    HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
                    ])
                ])
                
                doc = Document(
                    page_content=clean_text_output.content,
                    metadata={"source": f"Report {idx + 1}", "page": page_num + 1}
                )
                documents_to_embed.append(doc)
                
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
        progress_bar.progress((idx + 1) / len(files))
        
    status_text.text("Splitting text into chunks for the Vector DB...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    final_chunks = text_splitter.split_documents(documents_to_embed)
    
    status_text.text("Generating embeddings and building Vector DB...")
    # Creates a persistent local folder to save the vector DB index
    vector_db = Chroma.from_documents(final_chunks, embeddings, persist_directory="./blood_reports_db")
    
    status_text.empty()
    progress_bar.empty()
    return vector_db

# Trigger processing when button is clicked
if process_btn:
    with st.spinner("Analyzing layouts and constructing data mapping... This takes a few moments per PDF."):
        st.session_state.vector_db = process_uploaded_pdfs(uploaded_files)
        st.success(f"Successfully processed {len(uploaded_files)} files! Vector Database built locally.")

# --- 4. Interactive Chat Interface ---
if "vector_db" in st.session_state:
    st.write("---")
    st.subheader("💬 Ask Your Reports Anything")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User chat query
    if user_query := st.chat_input("e.g., Extract my fasting blood sugar over time, or check if any values are consistently high."):
        # Display user query
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Set up a highly strict medical RAG Prompt
        custom_prompt_template = """You are an expert AI clinical health data analyzer. 
Use the retrieved context (historical blood reports) below to accurately answer the question.
If the data required to answer is missing or cannot be found in the provided context, state that the data is not available. Do not hallucinate or guess numbers or reference ranges.

Context:
{context}

Question: {question}
Answer:"""

        PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        # Execute Retrieval Augmented Generation
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing health history..."):
                response = rag_chain.run(user_query)
                st.markdown(response)
                st.caption("⚠️ *Disclaimer: Generated metrics are for informational purposes based on files provided. Always consult a physician.*")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("💡 Upload your blood report PDFs and click 'Process & Build Vector DB' to unlock the analysis chat interface.")
