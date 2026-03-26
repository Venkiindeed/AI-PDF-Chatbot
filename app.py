import streamlit as st
import tempfile


st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


from tavily import TavilyClient


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()


with st.sidebar:
    st.title("⚙️ Settings")

    uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")
    web_toggle = st.toggle("🌐 Enable Web Search")

    st.markdown("---")
    st.markdown("🚀 Built by Venkatesh R | AI Developer")


st.title("🤖 AI PDF Chatbot")
st.caption("Ask questions from your PDF or use web search 🌐")


if uploaded_file is not None:

   
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    
    loader = PyPDFLoader(file_path, extract_images=True)
    documents = loader.load()

  
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)


    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success("✅ PDF Loaded Successfully")

    
    query = st.chat_input("Ask your question...")

   
    llm = OllamaLLM(model="mistral")

    
    tavily = TavilyClient(api_key="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")

    if query:

        
        with st.chat_message("user"):
            st.write(query)

        
        if any(word in query.lower() for word in ["best", "better", "top", "compare"]):
            with st.chat_message("assistant"):
                st.write("This is a subjective question. Please ask a factual question.")
            st.stop()

        
        docs = retriever.invoke(query)

        
        stopwords = ["what", "is", "the", "of", "in", "a", "an", "and", "to"]
        query_words = [word for word in query.lower().split() if word not in stopwords]

        filtered_docs = []
        for doc in docs:
            if any(word in doc.page_content.lower() for word in query_words):
                filtered_docs.append(doc)

        docs = filtered_docs

        
        relevant_text = " ".join([doc.page_content for doc in docs]).lower()
        match_count = sum(word in relevant_text for word in query_words)

        if len(query_words) <= 2:
            pdf_has_answer = match_count >= 1
        else:
            pdf_has_answer = match_count >= 2

        
        pdf_context = ""
        sources = []

        if pdf_has_answer and len(docs) > 0:
            for doc in docs:
                pdf_context += doc.page_content + "\n"

                
                page = doc.metadata.get("page", "Unknown")
                sources.append(f"Page {page + 1}")  

        
        if not pdf_has_answer and not web_toggle:
            with st.chat_message("assistant"):
                st.write("Not found in document")
            st.stop()

        
        web_context = ""

        if web_toggle:
            web_results = tavily.search(query=query)

            for result in web_results["results"][:5]:
                content = result.get("content", "")
                if len(content.strip()) > 50:
                    web_context += content[:800] + "\n"

        
        if web_toggle and web_context.strip() == "":
            with st.chat_message("assistant"):
                st.write("No relevant web data found")
            st.stop()

        
        final_context = (pdf_context + "\n" + web_context)[:2000]

        
        prompt = f"""
You are a strict AI assistant.

Rules:
- If the question asks for a list, you MUST provide EXACT number of items requested
- Do NOT stop early
- No explanation, only list if list is asked
- Keep answers clean and concise

Context:
{final_context}

Question:
{query}
"""

        
        with st.chat_message("assistant"):

            with st.spinner("Thinking... 🤖"):
                response = llm.invoke(prompt)

            
            clean_response = response.replace("Question:", "").replace("Answer:", "")
            clean_response = "\n".join(clean_response.split("\n")[:6])

            
            st.markdown(f"""
            <div style="
                background-color:#1e1e1e;
                padding:15px;
                border-radius:10px;
                border-left:5px solid #4CAF50;
            ">
            {clean_response}
            </div>
            """, unsafe_allow_html=True)

           
            if pdf_context:
                unique_sources = list(set(sources))

                st.markdown("#### 📄 Source from PDF:")
                for s in unique_sources:
                    st.write(f"• {s}")



st.markdown("---")
st.markdown("💡 Powered by LangChain + Ollama + FAISS + Tavily")


