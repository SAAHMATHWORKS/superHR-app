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

# Load environment variables (for local development)
load_dotenv()

# --- 1. Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Gozem superRH",
    page_icon="🤖",  # An emoji as an icon
    layout="wide",   # Use "wide" for more content space
    initial_sidebar_state="expanded" # Sidebar will be expanded by default
)

# --- 2. Configuration and Secrets Management ---
# For local development, it loads from .env.
# For Streamlit Cloud, it will automatically pull from Streamlit Secrets.
# Make sure your .streamlit/secrets.toml file contains these:
# OPENAI_API_KEY="sk-..."
# MONGO_URI="mongodb+srv://..."
# ATLAS_VECTOR_SEARCH_INDEX_NAME="vector_index"

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
MONGO_URI = os.getenv('MONGO_URI') or st.secrets.get('MONGO_URI')
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv('ATLAS_VECTOR_SEARCH_INDEX_NAME') or st.secrets.get('ATLAS_VECTOR_SEARCH_INDEX_NAME')

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set it in .env or Streamlit secrets.")
    st.stop()
if not MONGO_URI:
    st.error("MongoDB URI not found. Please set it in .env or Streamlit secrets.")
    st.stop()
if not ATLAS_VECTOR_SEARCH_INDEX_NAME:
    st.error("Atlas Vector Search Index Name not found. Please set it in .env or Streamlit secrets.")
    st.stop()

# --- 3. HR Prompt in French ---
HR_PROMPT_TEMPLATE = """
Vous êtes un assistant RH pour Gozem Africa.
Fournissez des réponses précises et professionnelles basées sur le contexte fourni.
Si vous n'êtes pas sûr(e), invitez l'utilisateur à contacter le service RH.

Contexte : {context}

Question : {question}
Répondez de manière professionnelle et utile :
"""

HR_PROMPT = PromptTemplate(
    template=HR_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# --- 4. Initialize MongoDB Atlas Vector Store and Embeddings (Cached) ---

# Use st.cache_resource to cache expensive objects like the vector store and LLM
@st.cache_resource
def get_qa_chain(openai_api_key, mongo_uri, index_name):
    try:
        # Initialize MongoDB Client
        client = MongoClient(mongo_uri)
        db_name = "hr_rag_db" # Replace with your actual database name
        collection_name = "hr_documents" # Replace with your actual collection name
        collection = client[db_name][collection_name]

        # Initialize OpenAI Embeddings for the vector store
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

        # Initialize MongoDBAtlasVectorSearch as the vector store
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embedding_model,
            index_name=index_name,        # The name of your Atlas Vector Search index
            text_key="text",              # Field in MongoDB that contains the text content
            embedding_key="embedding"     # Field in MongoDB that contains the embedding vector
        )

        # Initialize LLM
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

        # Initialize memory for conversational chain
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        # Initialize the ConversationalRetrievalChain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5, # Retrieve top 5 most similar documents
                    # Example of pre_filter:
                    # "pre_filter": {"metadata.is_confidential": False}
                    # This requires 'metadata.is_confidential' to be a 'filter' type in your Atlas index
                }
            ),
            memory=memory,
            combine_docs_chain_kwargs={
                "prompt": HR_PROMPT.partial(company_name="Gozem Africa")
            },
            return_source_documents=True
        )

        return qa_chain
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        st.stop()

# Get the QA chain (cached for performance)
qa_chain = get_qa_chain(OPENAI_API_KEY, MONGO_URI, ATLAS_VECTOR_SEARCH_INDEX_NAME)


# --- 5. Streamlit App Interface ---
st.title("Assistant RH - Gozem Africa 🤖")

# Optional: Add a welcoming message or brief description
st.markdown("""
Bienvenue dans votre assistant RH intelligent ! Posez vos questions sur les politiques,
procédures et informations relatives aux ressources humaines de Gozem Africa.
""")

# --- Input Section ---
with st.container(border=True):
    st.markdown("#### Posez votre question :")
    user_question = st.text_input("Tapez votre question ici :", label_visibility="collapsed", placeholder="Ex: Quelle est la politique de Gozem en matière de congés annuels ?")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Main Content (Answer and Sources) ---
# Use columns for the answer and source documents for better visual flow
col1, col2 = st.columns([2, 1]) # 2:1 ratio for answer and sources

with col1:
    if user_question:
        with st.spinner("Recherche et génération de réponse..."):
            # LangChain expects chat_history in a specific format for ConversationalRetrievalChain
            # (list of tuples (human_message_str, ai_message_str))
            chat_history_for_chain = [(q, a) for q, a in st.session_state.chat_history]

            result = qa_chain({"question": user_question, "chat_history": chat_history_for_chain})

            # Update session state with the new conversation
            st.session_state.chat_history.append((user_question, result["answer"]))

        st.markdown("### Réponse :")
        # Use st.info for a visually distinct answer box
        st.info(result["answer"])

with col2:
    if user_question:
        with st.expander("🔎 Voir les documents sources", expanded=True):
            if result.get("source_documents"):
                st.markdown("Ces informations sont basées sur les documents suivants :")
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Source Document {i+1}:**")
                    st.write(f"- Fichier: `{doc.metadata.get('source', 'Inconnue')}`")
                    st.write(f"- Page/Ligne: {doc.metadata.get('page', 'N/A')}")
                    st.write(f"- Type: `{doc.metadata.get('document_type', 'N/A')}`")
                    # You can conditionally show confidential status if relevant for your use case
                    # if doc.metadata.get('is_confidential'):
                    #    st.write(f"- Confidentiel: {doc.metadata.get('is_confidential')}")
                    st.markdown(f"```text\n{doc.page_content[:400]}...\n```") # Display snippet in a code block
                    st.markdown("---")
            else:
                st.write("Aucun document source pertinent trouvé.")


# --- Conversation History ---
st.markdown("---") # Visual separator
with st.expander("💬 Historique de la conversation", expanded=False): # Start collapsed
    if st.session_state.chat_history:
        # Display in reverse order for newest at top
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Question {len(st.session_state.chat_history)-i}:**")
            st.write(q)
            st.markdown(f"**Réponse {len(st.session_state.chat_history)-i}:**")
            st.success(a) # Use st.success for a distinct style
            st.markdown("---")
    else:
        st.write("Commencez la conversation pour voir l'historique.")


# --- Sidebar for additional info (optional) ---
# --- Sidebar for additional info (optional) ---
with st.sidebar:
    # Use columns to center the image
    col1, col2, col3 = st.columns([1, 2, 1]) # Adjust ratio as needed: [left_space, image_width, right_space]
                                            # Using a 1:2:1 ratio generally works well for centering a medium-sized image

    with col2: # Place the image in the middle column
        # Make sure you have 'gozem_logo.png' in the same directory as app.py
        if os.path.exists("gozem_logo.png"):
            st.image("gozem_logo.png", width=80) # Your original image call
        else:
            st.markdown("<h2 style='text-align: center;'>Gozem Africa</h2>", unsafe_allow_html=True)
            st.warning("Logo image 'gozem_logo.png' not found. Displaying text fallback.")

    st.header("À propos de l'Assistant")
    st.markdown("""
    Cet assistant RH est un outil basé sur l'IA, conçu spécifiquement pour
    les employés de Gozem Africa. Il utilise les documents RH officiels
    pour fournir des réponses rapides et précises.

    *Gozem AI Team*
    """)
    st.markdown("---")
    st.markdown("Pour toute question complexe ou spécifique à votre situation, veuillez contacter directement le service des Ressources Humaines.")