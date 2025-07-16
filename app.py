import streamlit as st
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from langchain_community.callbacks import StreamlitCallbackHandler # For streaming answers
from langchain_core.messages import HumanMessage, AIMessage # For LangChain chat history format

# Load environment variables (for local development)
load_dotenv()

# --- 1. Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Gozem RH Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Configuration and Secrets Management ---
# Prioritize Streamlit secrets for deployment, fall back to .env for local
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
MONGO_URI = os.getenv('MONGO_URI') or st.secrets.get('MONGO_URI')
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME') or st.secrets.get('ATLAS_VECTOR_SEARCH_INDEX_NAME')

# Display errors and stop if critical secrets are missing
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()
if not MONGO_URI:
    st.error("MongoDB URI not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()
if not ATLAS_VECTOR_SEARCH_INDEX_NAME:
    st.error("Atlas Vector Search Index Name not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()

# --- 3. HR Prompt in French ---
HR_PROMPT_TEMPLATE = """
Vous êtes un assistant RH pour Gozem Africa.
Fournissez des réponses précises et professionnelles basées sur le contexte fourni.
Si vous n'êtes pas sûr(e), invitez l'utilisateur à contacter le service RH.
vous pouvez repondre à des salutations et des remerciements tout simplement.

Contexte : {context}

Question : {question}
Répondez de manière professionnelle et utile :
"""

HR_PROMPT = PromptTemplate(
    template=HR_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# --- 4. Initialize MongoDB Atlas Vector Store and Embeddings (Cached) ---
# @st.cache_resource ensures this function runs only once across reruns
@st.cache_resource
def get_qa_chain(openai_api_key, mongo_uri, index_name):
    try:
        client = MongoClient(mongo_uri)
        db_name = "hr_rag_db"
        collection_name = "hr_documents"
        collection = client[db_name][collection_name]

        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_model,
            index_name=index_name,
            text_key="text",
            embedding_key="embedding"
        )

        # Initialize LLM with streaming=True for token-by-token output
        llm = ChatOpenAI(model_name="GPT-4o mini", temperature=0.2, openai_api_key=openai_api_key, streaming=True)

        # ConversationBufferMemory for LangChain's internal chat history management
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=memory, # Pass memory for internal LangChain history
            combine_docs_chain_kwargs={
                "prompt": HR_PROMPT.partial(company_name="Gozem Africa")
            },
            return_source_documents=True # Still retrieve and return source documents
        )

        return qa_chain
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        st.stop()

# Get the RAG chain instance
qa_chain = get_qa_chain(OPENAI_API_KEY, MONGO_URI, ATLAS_VECTOR_SEARCH_INDEX_NAME)


# --- 5. Streamlit App Interface (Chat-based UX) ---
st.title("supeRH - Gozem Africa 🤖")

st.markdown("""
Bienvenue dans votre assistant RH intelligent ! Posez vos questions sur les politiques,
procédures et informations relatives aux ressources humaines de Gozem Africa.
""")

# Initialize chat history in Streamlit's session state
# Each message is a dictionary: {"role": "user" or "assistant", "content": "message", "sources": [Document objects]}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it's an assistant message and has sources, display them in a collapsed expander
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("🔎 Voir les documents sources"): # This expander is for sources ONLY, defaults to collapsed
                st.markdown("Ces informations sont basées sur les documents suivants :")
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source Document {i+1}:**")
                    st.write(f"- Fichier: `{doc.metadata.get('source', 'Inconnue')}`")
                    st.write(f"- Page/Ligne: {doc.metadata.get('page', 'N/A')}")
                    st.write(f"- Type: `{doc.metadata.get('document_type', 'N/A')}`")
                    # Display a snippet of the document content
                    st.markdown(f"```text\n{doc.page_content[:400]}...\n```")
                    st.markdown("---")

# Use st.chat_input for new user messages – it automatically clears after submission
if user_question := st.chat_input("Posez votre question RH ici :"):
    # Add user message to the session's chat history for display
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Display the user's message immediately in the chat UI
    with st.chat_message("user"):
        st.markdown(user_question)

    # Prepare chat history for LangChain (it expects HumanMessage/AIMessage objects)
    langchain_chat_history = []
    for msg in st.session_state.messages[:-1]: # Exclude the very last user message
        if msg["role"] == "user":
            langchain_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_chat_history.append(AIMessage(content=msg["content"]))

    # Prepare for assistant's streaming response
    with st.chat_message("assistant"):
        # This message_placeholder will stream the answer directly. NO expander here.
        message_placeholder = st.empty()
        full_response_content = ""
        sources = []

        with st.spinner("Recherche et génération de réponse..."):
            try:
                # Initialize StreamlitCallbackHandler to stream output to the placeholder
                st_callback = StreamlitCallbackHandler(message_placeholder)

                result = qa_chain.invoke(
                    {"question": user_question, "chat_history": langchain_chat_history},
                    config={"callbacks": [st_callback]} # Pass the callback here
                )
                full_response_content = result["answer"]
                sources = result["source_documents"]

            except Exception as e:
                full_response_content = f"Désolé, une erreur est survenue lors de la génération de la réponse : {e}"
                st.error(full_response_content)

        # --- FIX: Ensure final content is displayed after spinner ---
        # Explicitly update the placeholder with the final, full content
        # after the spinner is gone and result is complete.
        message_placeholder.markdown(full_response_content)
        # --- END FIX ---

        # After streaming is complete, add the full response and sources to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response_content, "sources": sources})

        # Display source documents in an expander below the streamed answer
        # This expander is now only for the sources and is collapsed by default.
        if sources:
            with st.expander("🔎 Voir les documents sources"): # Defaults to collapsed (expanded=False)
                st.markdown("Ces informations sont basées sur les documents suivants :")
                for i, doc in enumerate(sources):
                    st.markdown(f"**Source Document {i+1}:**")
                    st.write(f"- Fichier: `{doc.metadata.get('source', 'Inconnue')}`")
                    st.write(f"- Page/Ligne: {doc.metadata.get('page', 'N/A')}")
                    st.write(f"- Type: `{doc.metadata.get('document_type', 'N/A')}`")
                    st.markdown(f"```text\n{doc.page_content[:400]}...\n```")
                    st.markdown("---")


# --- Sidebar for additional info ---
with st.sidebar:
    # Use columns to center the image
    col1_sb, col2_sb, col3_sb = st.columns([1, 2, 1])
    with col2_sb: # Place the image in the middle column
        # Make sure you have 'gozem_logo.png' in the same directory as app.py
        if os.path.exists("gozem_logo.png"):
            st.image("gozem_logo.png", width=80)
        else:
            st.markdown("<h2 style='text-align: center;'>Gozem Africa</h2>", unsafe_allow_html=True)
            st.warning("Logo image 'gozem_logo.png' not found. Displaying text fallback.")

    st.header("À propos de l'Assistant")
    st.markdown("""
    Cet assistant RH est un outil basé sur l'IA, conçu spécifiquement pour
    les employés de Gozem Africa. Il utilise les documents RH officiels
    pour fournir des réponses rapides et précises.

    *Powered by Gozem AI.*
    """)
    st.markdown("---")
    st.markdown("Pour toute question complexe ou spécifique à votre situation, veuillez contacter directement le service des Ressources Humaines.")