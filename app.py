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
    page_title="Gozem RH Assistant",
    page_icon="🤖",  # An emoji as an icon
    layout="wide",   # Use "wide" for more content space
    initial_sidebar_state="expanded" # Sidebar will be expanded by default
)

# --- 2. Configuration and Secrets Management ---
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

        llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

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

qa_chain = get_qa_chain(OPENAI_API_KEY, MONGO_URI, ATLAS_VECTOR_SEARCH_INDEX_NAME)


# --- 5. Streamlit App Interface ---
st.title("SuperRH - Gozem Africa 🤖")

st.markdown("""
Bienvenue dans votre assistant RH intelligent ! Posez vos questions sur les politiques,
procédures et informations relatives aux ressources humaines de Gozem Africa.
""")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- NEW: Callback function to handle input and clear it ---
def handle_user_input():
    # Get the current question from the text input widget via its key
    user_question = st.session_state.user_question_input

    if user_question: # Process only if the input is not empty
        # Process the question
        chat_history_for_chain = [(q, a) for q, a in st.session_state.chat_history]
        with st.spinner("Recherche et génération de réponse..."):
            result = qa_chain({"question": user_question, "chat_history": chat_history_for_chain})

            # Update session state with the new conversation
            st.session_state.chat_history.append((user_question, result["answer"]))
            st.session_state.last_answer = result["answer"] # Store last answer for display below
            st.session_state.last_sources = result["source_documents"] # Store sources
            
        # IMPORTANT: Clear the input field in session state AFTER processing
        st.session_state.user_question_input = ""
    # This function doesn't return anything as it directly modifies session_state

# --- Input Section ---
with st.container(border=True):
    st.markdown("#### Posez votre question :")
    # --- MODIFICATION HERE: Add on_change callback ---
    st.text_input(
        "Tapez votre question ici :",
        key="user_question_input", # Unique key for the text input widget
        label_visibility="collapsed",
        placeholder="Ex: Quelle est la politique de Gozem en matière de congés annuels ?",
        on_change=handle_user_input # Call the function when input changes and user presses Enter
    )
    # The actual user_question variable that will trigger the display logic
    # will now come from the session state after the callback runs.
    # We will display the results based on st.session_state.last_answer

# --- Main Content (Answer and Sources) ---
col1, col2 = st.columns([2, 1])

# Display the last answer and sources if available in session state
if "last_answer" in st.session_state and st.session_state.last_answer:
    with col1:
        st.markdown("### Réponse :")
        st.info(st.session_state.last_answer)

    with col2:
        with st.expander("🔎 Voir les documents sources", expanded=True):
            if st.session_state.last_sources:
                st.markdown("Ces informations sont basées sur les documents suivants :")
                for i, doc in enumerate(st.session_state.last_sources):
                    st.markdown(f"**Source Document {i+1}:**")
                    st.write(f"- Fichier: `{doc.metadata.get('source', 'Inconnue')}`")
                    st.write(f"- Page/Ligne: {doc.metadata.get('page', 'N/A')}")
                    st.write(f"- Type: `{doc.metadata.get('document_type', 'N/A')}`")
                    st.markdown(f"```text\n{doc.page_content[:400]}...\n```")
                    st.markdown("---")
            else:
                st.write("Aucun document source pertinent trouvé.")


# --- Conversation History ---
st.markdown("---")
with st.expander("💬 Historique de la conversation", expanded=False):
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**Question {len(st.session_state.chat_history)-i}:**")
            st.write(q)
            st.markdown(f"**Réponse {len(st.session_state.chat_history)-i}:**")
            st.success(a)
            st.markdown("---")
    else:
        st.write("Commencez la conversation pour voir l'historique.")


# --- Sidebar for additional info (optional) ---
with st.sidebar:
    col1_sb, col2_sb, col3_sb = st.columns([1, 2, 1])
    with col2_sb:
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