import gradio as gr
import sys
import os
import time
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import UniversityRAGPipeline
from src.vector_store import UniversityVectorStore

# Global variables
rag_pipeline = None
conversation_history = []


def initialize_system():
    """Initialize the RAG pipeline system"""
    global rag_pipeline
    
    try:
        # Check vector store
        vector_store = UniversityVectorStore()
        stats = vector_store.get_collection_stats()
        
        if stats.get('total_documents', 0) == 0:
            return " Error: No documents found in vector store. Please run setup first.", "<div class='stats-box'>System not initialized.</div>"
        
        # Initialize pipeline
        rag_pipeline = UniversityRAGPipeline(vector_store=vector_store)
        
        return f" System initialized successfully! Ready to answer questions about University of Sargodha.\n\n Knowledge base: {stats['total_documents']} documents loaded.", get_quick_stats()
        
    except Exception as e:
        return f" Failed to initialize system: {e}", "<div class='stats-box'>System not initialized.</div>"


def chat_response(message, history):
    """Generate response for user message"""
    global rag_pipeline, conversation_history
    
    if not rag_pipeline:
        return " System not initialized. Please wait for initialization to complete.", history
    
    if not message.strip():
        return "Please ask a question about University of Sargodha rules and regulations.", history
    
    try:
        # Get response from RAG pipeline
        response = rag_pipeline.query(message)
        
        # Format response with metadata
        avg_confidence = 0
        if response.confidence_scores:
            avg_confidence = sum(response.confidence_scores) / len(response.confidence_scores) * 100

        formatted_response = f"""**Answer:**
{response.answer}

---
** Response Details:**
-  Response time: {response.response_time:.2f}s
-  Sources used: {response.sources_used}
-  Average confidence: {avg_confidence:.1f}% (from {len(response.confidence_scores)} sources)
-  Tokens used: {response.total_tokens_used}"""
        
        # Add to conversation history
        conversation_history.append({
            "user": message,
            "assistant": response.answer,
            "metadata": {
                "response_time": response.response_time,
                "sources_used": response.sources_used,
                "confidence_scores": response.confidence_scores,
                "tokens_used": response.total_tokens_used
            },
            "timestamp": datetime.now().isoformat()
        })
        
        return formatted_response, history + [[message, formatted_response]]
        
    except Exception as e:
        error_msg = f" Error processing your question: {e}\n\nPlease try rephrasing your question or contact the university directly."
        return error_msg, history + [[message, error_msg]]

def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return [], ""

def get_sample_questions():
    """Get list of sample questions"""
    return [
        "What are the admission requirements for undergraduate programs?",
        "How much are the fees for Master's degree programs?",
        "What are the examination rules and procedures?", 
        "What documents are required for admission?",
        "What are the library rules and regulations?",
        "What disciplinary actions can be taken against students?",
        "How can I apply for a postgraduate program?",
        "What is the semester system at University of Sargodha?",
        "What are the hostel accommodation rules?",
        "How are examination results calculated?"
    ]

def load_sample_question(question):
    """Load a sample question into the chatbox"""
    return question

def get_system_statistics():
    """Get system statistics"""
    global rag_pipeline
    
    if not rag_pipeline:
        return "System not initialized"
    
    try:
        stats = rag_pipeline.get_pipeline_stats()
        
        categories = 'N/A'
        if 'vector_store' in stats and 'distributions' in stats['vector_store'] and 'categories' in stats['vector_store']['distributions']:
            categories = ', '.join(stats['vector_store']['distributions']['categories'].keys())

        formatted_stats = f"""** University of Sargodha Chatbot Statistics**

** Query Statistics:**
- Total queries processed: {stats['queries']['total_queries']}
- Successful responses: {stats['queries']['successful_responses']}
- Success rate: {stats['queries']['success_rate']}%

** Performance:**
- Average response time: {stats['performance']['average_response_time']}s
- Total tokens used: {stats['performance']['total_tokens_used']:,}
- Average tokens per query: {stats['performance']['average_tokens_per_query']}

** Configuration:**
- Embedding service: {stats['configuration']['embedding_service']}
- LLM available: {stats['configuration']['llm_available']}
- Model: {stats['configuration']['llm_model']}
- Max context chunks: {stats['configuration']['max_context_chunks']}

** Knowledge Base:**
- Total documents: {stats.get('vector_store', {}).get('total_documents', 'N/A')}
- Document categories available: {categories}
"""
        
        return formatted_stats
    except Exception as e:
        return f"Error getting statistics: {e}"

def download_conversation():
    """Generate conversation download data"""
    global conversation_history
    
    if not conversation_history:
        return None
    
    download_data = {
        "export_date": datetime.now().isoformat(),
        "university": "University of Sargodha",
        "total_interactions": len(conversation_history),
        "conversation": conversation_history
    }
    
    filename = f"su_chatbot_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Create a temporary file to save the conversation
    import tempfile
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    with open(file_path, 'w') as f:
        json.dump(download_data, f, indent=2)
        
    return file_path

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2196f3);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    """
    
    with gr.Blocks(title="University of Sargodha - RAG Chatbot", css=custom_css) as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1> University of Sargodha</h1>
            <h2>Rules & Regulations Chatbot</h2>
            <p>Get instant answers about university policies, procedures, and regulations!</p>
        </div>
        """)
        
        # Initialize system on startup
        init_status = gr.Textbox(
            label=" System Status",
            value="Initializing system...",
            interactive=False
        )
        
        # Main chat interface
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label=" Chat with University Assistant",
                    height=500,
                    show_label=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask me about University of Sargodha rules and regulations...",
                        label="Your Question",
                        lines=2,
                        max_lines=4
                    )
                    submit_btn = gr.Button(" Send", variant="primary")
                
                with gr.Row():
                    clear_btn = gr.Button(" Clear Chat")
                    stats_btn = gr.Button(" Show Statistics") 
                    download_btn = gr.DownloadButton(" Download Chat", visible=False)
            
            with gr.Column(scale=1):
                gr.HTML("<h3> Sample Questions</h3>")
                
                sample_questions = get_sample_questions()
                for i, question in enumerate(sample_questions):
                    btn = gr.Button(f" {question[:40]}...", size="sm")
                    btn.click(
                        fn=lambda q=question: q,
                        outputs=msg
                    )
                
                gr.HTML("<h3> Quick Stats</h3>")
                quick_stats = gr.HTML()
        
        # Statistics modal
        with gr.Row():
            stats_display = gr.Textbox(
                label=" System Statistics",
                lines=15,
                max_lines=20,
                visible=False
            )
        
        # Event handlers
        def handle_submit(message, history):
            response, updated_history = chat_response(message, history)
            return "", updated_history, get_quick_stats()
        
        def handle_clear():
            clear_conversation()
            return [], "", get_quick_stats()
        
        def handle_stats():
            stats = get_system_statistics()
            return gr.update(value=stats, visible=True)
        
        def handle_download():
            if conversation_history:
                filepath = download_conversation()
                return filepath
            return None
        
        # Wire up events
        submit_btn.click(
            fn=handle_submit,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, quick_stats]
        )
        
        msg.submit(
            fn=handle_submit,
            inputs=[msg, chatbot], 
            outputs=[msg, chatbot, quick_stats]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, msg, quick_stats]
        )
        
        stats_btn.click(
            fn=handle_stats,
            outputs=stats_display
        )
        
        download_btn.click(
            fn=handle_download,
            outputs=download_btn
        )
        
        # Initialize system when interface loads
        def update_ui_on_load():
            status, stats = initialize_system()
            return status, stats, gr.update(visible=True) if rag_pipeline else gr.update(visible=False)

        interface.load(
            fn=update_ui_on_load,
            outputs=[init_status, quick_stats, download_btn]
        )
    
    return interface

# Launch interface
if __name__ == "__main__":
    interface = create_interface()
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )