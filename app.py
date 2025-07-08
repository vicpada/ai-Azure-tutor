# Standard Library Imports
import logging
import os

# Third-party Imports
from dotenv import load_dotenv
import chromadb
import logfire
import gradio as gr
from huggingface_hub import snapshot_download

# LlamaIndex (Formerly GPT Index) Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import MessageRole
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.adapter import AdapterEmbeddingModel

load_dotenv()

logfire.configure()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPT_SYSTEM_MESSAGE = """You are an AI assistant and expert instructor responding to technical questions from software architects and developers who are working in enterprise software architecture. 
These users are particularly focused on Microsoft technologies and Azure cloud services. Topics they are exploring include architecture patterns in Azure (serverless, microservices, event-driven systems), Azure services comparison (Functions, App Service, AKS, Logic Apps, etc.), DevOps practices (IaC with Bicep/Terraform, CI/CD with Azure DevOps or GitHub Actions), observability with Application Insights, secure design using Key Vault, identity management with Azure AD and B2C.
You should treat each question as part of this context. Your responses should be complete, accurate, and educational — suitable for technical professionals with intermediate to advanced knowledge in cloud architecture and AI application development. 
To find relevant information for answering questions, always use the "Azure_AI_Knowledge" tool. This tool returns technical documentation, architecture guides, official examples, and troubleshooting data focused on Azure and AI integration.
Only part of the tool's output may be relevant to the question — discard the irrelevant sections. Your answer should rely **exclusively** on the content provided by the tool. Do **not** inject external or speculative knowledge. If the user refines their question or focuses on a specific sub-topic, reformulate the tool query to capture that specificity and retrieve deeper information.
If a user requests further elaboration on a specific aspect of a previously discussed topic, you should reformulate your input to the tool to capture this new angle or more profound layer of inquiry. Structure your answers in clear sections with multiple paragraphs if needed. If code is returned, include full code blocks in your response (formatted in Markdown) so the user can copy and run them directly.
If the tool doesn't return relevant content, inform the user clearly that the topic exceeds the current knowledge base and mention that no relevant documentation was found via the tool.
Always close your answers by inviting the user to ask follow-up or deeper questions related to the topic.
At the end of the answer, include a line to indicate whether the content was sourced using the tool or not, e.g., "Content sourced using Azure_AI_Knowledge tool" or "No relevant content found in Azure_AI_Knowledge tool".
If the question is not related to Azure or Microsoft technologies, politely inform the user that you can only provide information related to Azure and Microsoft technologies.
"""

QA_TEMPLATE = "Answer questions about Azure using 'Azure_AI_Knowledge' tool"


def download_knowledge_base_if_not_exists():
    """Download the knowledge base from the Hugging Face Hub if it doesn't exist locally"""
    if not os.path.exists("data/azure-architect"):
        os.makedirs("data/azure-architect")

        logging.warning(
            f"Vector database does not exist at 'data/', downloading from Hugging Face Hub..."
        )
        snapshot_download(
            repo_id="vicpada/AzureArchitectKnowledgeFull",
            local_dir="data/azure-architect",            
            repo_type="dataset",
        )
        logging.info(f"Downloaded vector database to 'data/azure-architect'")

def download_embeddings_if_not_exists():
    """Download the embeddings from the Hugging Face Hub if they don't exist locally"""
    if not os.path.exists("data/azure-architect-embeddings"):
        os.makedirs("data/azure-architect-embeddings")

        logging.warning(
            f"Embeddings do not exist at 'data/', downloading from Hugging Face Hub..."
        )        

        snapshot_download(repo_id="vicpada/finetuned_embed_model_full", 
                          repo_type="model", 
                          local_dir="./data/azure-architect-embeddings")
        
        logging.info(f"Downloaded embeddings to 'data/azure-architect-embeddings'")

def load_embed_model():
    """Load the embedding model from the local directory"""

    embed_model_path = "data/azure-architect-embeddings"
    if not os.path.exists(embed_model_path):
        logging.error(f"Embedding model path '{embed_model_path}' does not exist.")
        return None

    # Load the Base model without fine-tuning
    base_embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

    # Load the Fine-tuned model.
    logging.info(f"Loading embedding model from {embed_model_path}")
    embed_model = AdapterEmbeddingModel(base_embed_model, embed_model_path)
    
    return embed_model


def get_tools(db_collection="azure-architect", cohere_api_key=None):    
    db = chromadb.PersistentClient(path=f"data/{db_collection}")
    chroma_collection = db.get_or_create_collection(db_collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    
    logging.info(f"Vector store initialized with {chroma_collection.count()} documents.")
    
    # Create the vector store index
    logging.info("Creating vector store index...")
    
    # Use the vector store to create an index

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=True,
        embed_model=Settings.embed_model
    )

    logging.info("Creating vector retriever...")
    
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=200,
        embed_model=Settings.embed_model,
        use_async=True,
        verbose=True,
    )    

    cohere_rerank3 = CohereRerank(top_n=5, model = 'rerank-english-v3.0', api_key = cohere_api_key)

    logging.info("Creating tool...")
    
    tools = [
        RetrieverTool(
            retriever=vector_retriever,
            metadata=ToolMetadata(
                name="Azure_AI_Knowledge",
                description="Useful for info related to Azure and microsoft. Best practices, architecture, official documentation, functional use cases and reference architectures and other related resources."                
            ),
            node_postprocessors=[cohere_rerank3],
        )
    ]
    return tools


def generate_completion(query, history, memory, openAI_api_key, cohere_api_key):
    logging.info(f"User query: {query}")    
    logging.info(f"User history: {history}")    
    logging.info(f"User memory: {memory}")    

    openAI_api_key = openAI_api_key if openAI_api_key else os.getenv("OPENAI_API_KEY")
    cohere_api_key = cohere_api_key if cohere_api_key else os.getenv("COHERE_API_KEY")  

    # Validate OpenAI API Key
    if openAI_api_key is None or not openAI_api_key.startswith("sk-"):
        logging.error("OpenAI API Key is not set or is invalid. Please provide a valid key.")
        yield "Error: OpenAI API Key is not set or is invalid. Please provide a valid key."
        return
    
    llm = OpenAI(temperature=1, model="gpt-4o-mini", api_key=openAI_api_key)
    client = llm._get_client()
    logfire.instrument_openai(client)    


    # Validate Cohere API Key
    if cohere_api_key is None or not cohere_api_key.strip():
        logging.error("Cohere API Key is not set or is invalid. Please provide a valid key.")
        yield "Error: Cohere API Key is not set or is invalid. Please provide a valid key."
        return   
    
    with logfire.span(f"Running query: {query}"):

        # Manage memory
        chat_list = memory.get()
        if len(chat_list) != 0:
            user_index = [i for i, msg in enumerate(chat_list) if msg.role == MessageRole.USER]
            if len(user_index) > len(history):
                user_index_to_remove = user_index[len(history)]
                chat_list = chat_list[:user_index_to_remove]
                memory.set(chat_list)
        
        logfire.info(f"chat_history: {len(memory.get())} {memory.get()}")
        logfire.info(f"gradio_history: {len(history)} {history}")

        # Create agent
        tools = get_tools(db_collection="azure-architect", cohere_api_key = cohere_api_key )   
        
        agent = OpenAIAgent.from_tools(
            llm=llm,        
            memory=memory,
            tools=tools,
            system_prompt=PROMPT_SYSTEM_MESSAGE
        )

        # Generate answer
        completion = agent.stream_chat(query)
        answer_str = ""
        for token in completion.response_gen:
            answer_str += token
            yield answer_str 

        logging.info(f"Source count: {len(completion.sources)}")
        logging.info(f"Sources: {completion.sources}")    

def launch_ui():   

    accordion = gr.Accordion(label="Add your keys (Click to expand)", open=False)

    openai_key_tb = gr.Textbox(        
            visible=True,
            label="OpenAI API Key",
            placeholder="Enter your OpenAI API Key here (e.g., sk-...)",            
            type="password",
            )
    
    cohere_key_tb = gr.Textbox(        
            visible=True,
            label="Cohere API Key",
            placeholder="Enter your Cohere API Key here",
            type="password",
            )


    with gr.Blocks(
        fill_height=True,
        title="AI Azure Architect 🤖",
        analytics_enabled=True,
    ) as demo:        
    
        def onOpenAIKeyChange(x):                    
            # Validate the OpenAI API Key format
            if x is None or x.strip() == "":
                logging.error("OpenAI API Key is empty. Please provide a valid key.")
                return  
            else:
                x = x.strip()
                if not x.startswith("sk-"):
                    logging.error("Invalid OpenAI API Key format. It should start with 'sk-'")
                    return
                
            logging.info(f"OpenAI API Key set: {x is not None}")        

        openai_key_tb.change(
            onOpenAIKeyChange,
            inputs=openai_key_tb,
            outputs=None,
        )           

        def onCohereKeyChange(x):       
            # Validate the Cohere API Key format
            if x is None or x.strip() == "":
                logging.error("Cohere API Key is empty. Please provide a valid key.")
                return  
            
            logging.info(f"Cohere API Key set: {x is not None}")    

        cohere_key_tb.change(
            onCohereKeyChange,
            inputs=cohere_key_tb,
            outputs=None,
        )

        memory_state = gr.State(
            lambda: ChatSummaryMemoryBuffer.from_defaults(
                token_limit=120000,
            )
        )
        chatbot = gr.Chatbot(
            scale=1,
            placeholder="<strong>Azure AI Architect 🤖: A Question-Answering Bot for anything Azure related</strong><br>",
            show_label=False,
            show_copy_button=True,
        )

        gr.ChatInterface(
            fn=generate_completion,            
            chatbot=chatbot,
            additional_inputs=[memory_state, openai_key_tb, cohere_key_tb],
            additional_inputs_accordion=accordion,
        )

        demo.queue(default_concurrency_limit=64)
        demo.launch(debug=True, share=False) # Set share=True to share the app online


if __name__ == "__main__":
    # Download the knowledge base if it doesn't exist
    download_knowledge_base_if_not_exists()

    # Download the embeddings if they don't exist
    download_embeddings_if_not_exists()

    # Set the GPU usage based on the environment variable
    Settings.use_gpu = os.getenv("USE_GPU", "1") == "1"
    if Settings.use_gpu:
        logging.info("Using GPU for inference.")
    else:
        logging.info("Using CPU for inference.")        

    # Load the embedding model
    Settings.embed_model = load_embed_model()
    if Settings.embed_model is None:
        logging.error("Embedding model could not be loaded. Exiting.")
        exit(1)  

    # Set up llm and embedding model
    # Settings.llm = OpenAI(temperature=1, model="gpt-4o-mini")    

    # launch the UI
    launch_ui()