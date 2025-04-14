import os
import json
import pymupdf  # PDF processing
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama  # Assuming you use Ollama for LLM
from langchain.schema import Document  # Import Document class

# Ensure storage directory exists
os.makedirs("static", exist_ok=True)  

# Function to extract text from a PDF
def load_pdf(file):
    file_path = f"uploaded_{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load PDF: {str(e)}")
        return []  # Return empty list if loading fails

# Function to get the first page image of the uploaded PDF
def get_pdf_first_page_image(file):
    file_path = f"uploaded_{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    try:
        doc = pymupdf.open(file_path)
        pix = doc[0].get_pixmap()
        image_path = "static/first_page.png"
        pix.save(image_path)
        return image_path
    except Exception as e:
        st.error(f"Failed to extract first page: {str(e)}")
        return None

# Function to split PDF text into chunks for better retrieval
def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

# Function to create a retriever from document embeddings
def create_retriever(docs_json):
    try:
        docs_dicts = json.loads(docs_json)

        # Convert dictionary objects into Document objects
        docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in docs_dicts]

        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vector = FAISS.from_texts(
            [doc.page_content for doc in docs],  # Now, doc.page_content will work
            embedder,
            metadatas=[doc.metadata for doc in docs]
        )

        return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error in creating retriever: {str(e)}")
        return None

# Function to build a prompt for the LLM
def build_prompt():
    prompt = """
    1. Use the following context to answer the question.
    2. If you don't know the answer, say "I don't know."
    3. Keep the answer concise (3-4 sentences).

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    return PromptTemplate.from_template(prompt)

# Function to create the LLM-based QA chain
def build_qa_chain(retriever, llm):
    prompt = build_prompt()
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
    )

    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
    )

# Function to process uploaded PDF (loads, chunks, and creates retriever)
def process_pdf(file):
    print("🔹 Loading PDF...")  # Debugging step
    docs = load_pdf(file)

    if not docs:
        st.error("No text extracted from PDF. Please check the file.")
        return None

    print(f"🔹 Total pages loaded: {len(docs)}")  # Check how many pages are loaded

    chunked_docs = chunk_documents(docs)
    print(f"🔹 Total chunks created: {len(chunked_docs)}")  # Check how many chunks are created

    # Convert to JSON for retriever
    docs_json = json.dumps([
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in chunked_docs
    ])

    print("🔹 Creating retriever...")
    retriever = create_retriever(docs_json)

    if retriever is None:
        st.error("Error in retriever creation.")
        return None

    print("✅ PDF Processing Done!")  # Final debug check
    return retriever

# Function to load the LLM
def get_llm():
    return Ollama(model="llama3")  # Change model if needed

# Main function to run the Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("🚀 Fast RAG-based QA with Bhagavad Gita")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file:
            try:
                image_path = get_pdf_first_page_image(uploaded_file)
                if image_path:
                    st.image(image_path, caption="First Page Preview", use_column_width=True)
            except Exception as e:
                st.error("Failed to load preview: " + str(e))

    if uploaded_file:
        with st.spinner("🔄 Processing PDF..."):
            retriever = process_pdf(uploaded_file)

        if retriever is None:
            return  # Stop execution if retriever creation fails

        llm = get_llm()
        qa_chain = build_qa_chain(retriever, llm)

        user_input = st.text_input("Enter your question:")

        if user_input:
            with st.spinner("🤖 Generating response..."):
                result = qa_chain.invoke({"query": user_input})

                # Extracting answer properly
                response = result.get("result", "No answer found.")

                st.write("### 📜 Answer:")
                st.write(response)
    else:
        st.info("📥 Please upload a PDF file to proceed.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
