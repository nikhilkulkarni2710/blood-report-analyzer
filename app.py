import os
import base64
import re
from io import BytesIO
import streamlit as st
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import pytesseract
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. Page Configuration & UI Styling ---
st.set_page_config(page_title="Blood Report AI Analyzer", page_icon="🩸", layout="wide")
st.title("🩸 Multimodal Blood Report AI Analyzer")
st.caption("Upload historical reports, parse unstructured data, and analyze trends using NVIDIA NIM & RAG.")

# --- 2. Sidebar Configuration ---
try:
    streamlit_api_key = st.secrets.get("NVIDIA_API_KEY", "")
    streamlit_use_vision = st.secrets.get("USE_VISION_MODEL", "false")
except st.errors.StreamlitSecretNotFoundError:
    streamlit_api_key = ""
    streamlit_use_vision = "false"

with st.sidebar:
    api_key = os.environ.get("NVIDIA_API_KEY", streamlit_api_key)
    use_vision_model = str(
        os.environ.get("USE_VISION_MODEL", streamlit_use_vision)
    ).lower() == "true"
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
class LocalChromaEmbeddings(Embeddings):
    def __init__(self):
        self.embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [embedding.tolist() for embedding in self.embedding_function(texts)]

    def embed_query(self, text):
        return self.embedding_function([text])[0].tolist()


@st.cache_resource
def load_models():
    llm = ChatNVIDIA(model="nvidia/neva-22b", temperature=0.1) if use_vision_model else None
    
    # Use Chroma's local embedding model to avoid hosted embedding API failures.
    embeddings = LocalChromaEmbeddings()
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
                if use_vision_model:
                    page_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                    image_buffer = BytesIO()
                    page_image.save(image_buffer, format="JPEG", quality=85, optimize=True)
                    image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
                    prompt = (
                        "Extract all medical biomarkers, test names, results, units, and reference ranges "
                        "from this image into a clean markdown table. Strictly ignore any advertisements, "
                        "promotional headers, pricing, or non-medical clinic banners. Never extract, repeat, "
                        "or infer names, addresses, phone numbers, email addresses, dates of birth, patient "
                        "IDs, or any other personally identifying information."
                    )
                    try:
                        clean_text_output = llm.invoke([
                            HumanMessage(content=[
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            ])
                        ])
                        extracted_text = clean_text_output.content
                    except Exception:
                        st.warning(
                            f"Vision model unavailable for {uploaded_file.name}; using local OCR instead."
                        )
                        extracted_text = pytesseract.image_to_string(page_image, config="--psm 6")
                else:
                    extracted_text = pytesseract.image_to_string(page_image, config="--psm 6")
                if not extracted_text.strip():
                    st.warning(f"No text could be extracted from page {page_num + 1}.")
                    continue
                
                doc = Document(
                    page_content=extracted_text,
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


def relevant_report_text(documents, query):
    query_lower = query.lower()
    biomarker_aliases = {
        "vitamin d": ("vitamin d", "25-oh", "25 oh", "25(oh)d", "cholecalciferol"),
        "vitamin b12": ("vitamin b12", "b12", "cobalamin"),
        "hemoglobin": ("hemoglobin", "haemoglobin", "hb"),
        "glucose": ("glucose", "blood sugar", "hba1c", "glycated hemoglobin"),
        "cholesterol": ("cholesterol", "ldl", "hdl", "triglyceride"),
    }
    aliases = next(
        (terms for name, terms in biomarker_aliases.items() if name in query_lower),
        (),
    )

    if "vitamin d" in query_lower:
        vitamin_d_pattern = re.compile(
            r"vitamin\s*d\s*\(?(?:25\s*-?\s*oh)?\)?[^\d]{0,120}"
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ng\s*/?\s*ml|nmol\s*/?\s*l)",
            re.IGNORECASE,
        )
    else:
        vitamin_d_pattern = None

    results = []
    for document in documents:
        lines = document.page_content.splitlines()
        if aliases:
            matched_lines = [
                line.strip()
                for line in lines
                if any(alias in line.lower() for alias in aliases)
            ]
            content = "\n".join(matched_lines)
        else:
            content = document.page_content
        if vitamin_d_pattern:
            match = vitamin_d_pattern.search(document.page_content)
            if match:
                value = float(match.group("value"))
                unit = re.sub(r"\s+", "", match.group("unit")).lower()
                interpretation = "Deficient" if unit == "ng/ml" and value < 20 else ""
                content = (
                    f"Vitamin D (25-OH): {match.group('value')} {unit}"
                    + (f" ({interpretation})" if interpretation else "")
                )
            else:
                content = ""
        if content.strip():
            results.append(
                f"**{document.metadata.get('source', 'Report')}, "
                f"page {document.metadata.get('page', '?')}**\n\n{content}"
            )
    return "\n\n".join(results)


def biomarker_chart_data(documents, query):
    query_lower = query.lower()
    biomarker_aliases = {
        "vitamin d": ("vitamin d", "25-oh", "25 oh", "25(oh)d", "cholecalciferol"),
        "vitamin b12": ("vitamin b12", "b12", "cobalamin"),
        "hemoglobin": ("hemoglobin", "haemoglobin", "hb"),
        "glucose": ("glucose", "blood sugar", "hba1c", "glycated hemoglobin"),
        "cholesterol": ("cholesterol", "ldl", "hdl", "triglyceride"),
    }
    biomarker = next(
        (name for name in biomarker_aliases if name in query_lower),
        None,
    )
    if not biomarker:
        return None, None

    aliases = biomarker_aliases[biomarker]
    value_pattern = re.compile(
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?:ng\s*/?\s*ml|mg\s*/?\s*dl|g\s*/?\s*dl|%)?",
        re.IGNORECASE,
    )
    vitamin_d_value_pattern = re.compile(
        r"vitamin\s*d\s*\(?(?:25\s*-?\s*oh)?\)?[^\d]{0,120}"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?:ng\s*/?\s*ml|nmol\s*/?\s*l)",
        re.IGNORECASE,
    )
    points = []
    for document in documents:
        for line in document.page_content.splitlines():
            line_lower = line.lower()
            if not any(alias in line_lower for alias in aliases):
                continue
            match = (
                vitamin_d_value_pattern.search(line)
                if biomarker == "vitamin d"
                else value_pattern.search(line)
            )
            if not match:
                continue
            points.append({
                "Report": document.metadata.get("source", "Report"),
                "Page": document.metadata.get("page", "?"),
                "Value": float(match.group("value")),
            })
            break

    if not points:
        return None, None
    chart = pd.DataFrame(points)
    chart["Reading"] = chart["Report"] + " / page " + chart["Page"].astype(str)
    return chart.set_index("Reading")["Value"], biomarker.title()

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

        custom_prompt_template = """You are an expert AI Clinical Data Analyst. 
Your task is to analyze the historical medical blood reports provided in the context below. 

When the user asks about a specific biomarker (like Vitamin D, Iron, or Sugar), do NOT just repeat the raw text. Instead, generate a comprehensive trend analysis by executing these steps:
1. Extract ALL instances of that biomarker across every provided report date.
2. Structure them into a clean markdown timeline table (Date | Result Value | Reference Range | Status).
3. Identify the trend: State clearly if the levels are steady, rising, falling, or consistently out of range.
4. Give a brief, clinical summary explaining what this trend implies based on standard reference ranges.

If the data is completely missing from the reports, explicitly state: "Biomarker not found in historical records."

Context:
{context}

Question: {question}
Answer:"""


        PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})
        with st.chat_message("assistant"):
            with st.spinner("Analyzing health history..."):
                if llm is not None:
                    rag_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        chain_type_kwargs={"prompt": PROMPT},
                    )
                    response = rag_chain.run(user_query)
                else:
                    matching_documents = retriever.invoke(user_query)
                    response = relevant_report_text(matching_documents, user_query)
                    if not response:
                        response = "No matching report data was found."
                    chart_data, chart_name = biomarker_chart_data(matching_documents, user_query)
                    if chart_data is not None:
                        st.subheader(f"{chart_name} trend")
                        st.line_chart(chart_data)
                st.markdown(response)
                st.caption("⚠️ *Results are extracted from the uploaded files. Consult a physician for interpretation.*")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("💡 Upload your blood report PDFs and click 'Process & Build Vector DB' to unlock the analysis chat interface.")
