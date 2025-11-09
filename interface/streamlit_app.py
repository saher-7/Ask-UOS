import streamlit as st
import sys
import os
import time
from datetime import datetime
import json
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import UniversityRAGPipeline
from src.vector_store import UniversityVectorStore

# Page configuration
st.set_page_config(
    page_title="University of Sargodha - Rules & Regulations Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        margin-bottom: 30px;
        background: linear-gradient(90deg, #1f4e79, #2196f3);
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4caf50;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.3rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    .stButton > button {
        width: 100%;
    }
    .sample-question {
        margin: 0.2rem 0;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.3rem;
        border: 1px solid #e0e0e0;
        cursor: pointer;
    }
    .sample-question:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline with caching"""
    try:
        with st.spinner(" Initializing chatbot system..."):
            # Check vector store
            vector_store = UniversityVectorStore()
            stats = vector_store.get_collection_stats()
            
            if stats.get('total_documents', 0) == 0:
                st.error(" No documents found in vector store. Please run setup first.")
                st.info("Run these commands in order:")
                st.code("""
python collect_documents.py
python test_processing.py
python test_chunking.py
python test_vector_store.py
                """)
                return None, None
            
            # Initialize pipeline
            rag_pipeline = UniversityRAGPipeline(vector_store=vector_store)
            
            return rag_pipeline, stats
    except Exception as e:
        st.error(f" Failed to initialize chatbot: {e}")
        st.info("Please check the error above and ensure all setup steps are completed.")
        return None, None

def display_chat_message(role, content, metadata=None):
    """Display a chat message with proper formatting"""
    # Avoid f-strings for multiline HTML, use triple quotes directly
    if role == "user":
        html_content = (
            '<div class="chat-message user-message">'
            '<strong> You:</strong><br>'
            f'{content}'
            '</div>'
        )
    else:
        # Replace newlines with <br> for bot messages
        safe_content = content.replace('\n', '<br>')
        html_content = (
            '<div class="chat-message bot-message">'
            '<strong> University Assistant:</strong><br>'
            f'{safe_content}'
            '</div>'
        )
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Display source information if available
    if metadata and role != "user":
        sources_used = metadata.get('sources_used', 0)
        response_time = metadata.get('response_time', 0)
        confidence_scores = metadata.get('confidence_scores', [])
        tokens_used = metadata.get('tokens_used', 0)
        
        if sources_used > 0:
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            source_html = f"""
            <div class="source-info">
                <strong> Response Details:</strong><br>
                 Sources consulted: {sources_used}<br>
                 Response time: {response_time:.2f}s<br>
                 Average confidence: {avg_confidence:.1%}<br>
                 Tokens used: {tokens_used}
            </div>
            """
            st.markdown(source_html, unsafe_allow_html=True)

def get_sample_questions():
    """Get categorized sample questions"""
    return {
        " Academic & Admission": [
            "What are the admission requirements for undergraduate programs?",
            "How do I apply for a Master's degree program?",
            "What documents are required for PhD admission?",
            "What is the minimum CGPA requirement for admission?"
        ],
        " Fees & Financial": [
            "What are the fee structures for different degree programs?",
            "How can I pay university fees?",
            "Are there any scholarship opportunities available?",
            "What are the fee submission deadlines?"
        ],
        " Examinations & Grading": [
            "What are the examination rules and procedures?",
            "How is the grading system structured?",
            "What happens if I miss an examination?",
            "How can I apply for re-evaluation of papers?"
        ],
        " Campus Life": [
            "What are the hostel accommodation rules?",
            "What are the library borrowing policies?",
            "What disciplinary actions can be taken against students?",
            "What are the campus facility usage guidelines?"
        ]
    }

def main():
    # Header (unchanged, already using triple quotes without f-string)
    st.markdown("""
    <div class="main-header">
        <h1> University of Sargodha</h1>
        <h2>Rules & Regulations AI Assistant</h2>
        <p>Get instant, accurate answers about university policies, procedures, and regulations!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG pipeline
    rag_pipeline, vector_stats = initialize_rag_pipeline()
    
    if not rag_pipeline:
        st.stop()
    
    # Sidebar with information and controls
    with st.sidebar:
        st.header(" System Information")
        
        # Knowledge Base Stats
        if vector_stats:
            # Use string concatenation to build the stats string
            categories = vector_stats.get('distributions', {}).get('categories', {})
            stats_content = f"**Documents Loaded:** {vector_stats.get('total_documents', 0):,}\n\n"
            stats_content += "**Categories Available:**\n"
            stats_content += "\n".join([f" {cat.title()}: {count}" for cat, count in categories.items()])
            stats_content += "\n\n**Sources:** Official University Documents"
            st.success(stats_content)
        
        # Usage Instructions (unchanged, already using triple quotes)
        st.markdown("###  How to Use")
        st.info("""
        **Simple Steps:**
        1. Type your question in the chat box below
        2. Click a sample question for quick start
        3. Get instant answers from official documents
        
        **Best Results:**
         Be specific in your questions
         Ask about official university policies
         Use keywords like "admission", "fees", "examination"
        """)
        
        # Quick Stats (if available)
        if 'pipeline_stats' in st.session_state:
            stats = st.session_state.pipeline_stats
            st.markdown("###  Session Stats")
            stats_info = f"""
            **This Session:**
             Questions asked: {stats.get('queries_count', 0)}
             Average response time: {stats.get('avg_response_time', 0):.1f}s
            """
            st.info(stats_info)
        
        # Sample Questions by Category
        st.markdown("###  Sample Questions")
        sample_questions = get_sample_questions()
        
        for category, questions in sample_questions.items():
            with st.expander(category):
                for question in questions:
                    if st.button(f" {question[:40]}...", key=f"sample_{hash(question)}", help=question):
                        st.session_state.selected_question = question
        
        # System Controls
        st.markdown("---")
        st.markdown("###  Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Clear Chat"):
                st.session_state.messages = [st.session_state.messages[0]] if 'messages' in st.session_state else []
                st.session_state.pipeline_stats = {"queries_count": 0, "avg_response_time": 0}
                st.rerun()
        
        with col2:
            if st.button(" Show Stats"):
                st.session_state.show_stats = True
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Welcome message (unchanged)
        welcome_msg = """**Welcome to the University of Sargodha AI Assistant!** 

I'm here to help you find accurate information about:

** Academic Matters:**
 Admission requirements and procedures
 Degree programs and course details
 Academic policies and regulations

** Financial Information:**  
 Fee structures and payment methods
 Scholarship and financial aid
 Payment deadlines and procedures

** Rules & Procedures:**
 Examination rules and grading
 Disciplinary policies
 Campus facility usage

** Student Services:**
 Hostel and accommodation
 Library services
 General student guidelines

** How can I help you today?**

*Tip: Use the sample questions in the sidebar for quick starts!*"""
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": welcome_msg,
            "metadata": None
        })
    
    if "pipeline_stats" not in st.session_state:
        st.session_state.pipeline_stats = {"queries_count": 0, "avg_response_time": 0}
    
    # Handle sample question selection
    if "selected_question" in st.session_state:
        st.session_state.user_input = st.session_state.selected_question
        del st.session_state.selected_question
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            display_chat_message(
                message["role"], 
                message["content"], 
                message.get("metadata")
            )
    
    # Chat input
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                " Ask your question about University of Sargodha...",
                key="user_input",
                placeholder="e.g., What are the admission requirements for MS programs?"
            )
        
        with col2:
            send_button = st.button(" Send", type="primary", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "metadata": None
        })
        
        # Display user message immediately
        with chat_container:
            display_chat_message("user", user_input)
        
        # Generate response
        with st.spinner(" Thinking..."):
            try:
                # Get response from RAG pipeline
                start_time = time.time()
                response = rag_pipeline.query(user_input)
                processing_time = time.time() - start_time
                
                # Prepare metadata
                metadata = {
                    "sources_used": response.sources_used,
                    "response_time": response.response_time,
                    "confidence_scores": response.confidence_scores,
                    "tokens_used": response.total_tokens_used
                }
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "metadata": metadata
                })
                
                # Update pipeline stats
                current_stats = st.session_state.pipeline_stats
                current_stats["queries_count"] += 1
                
                # Update average response time
                prev_avg = current_stats.get("avg_response_time", 0)
                prev_count = current_stats["queries_count"] - 1
                new_avg = (prev_avg * prev_count + processing_time) / current_stats["queries_count"]
                current_stats["avg_response_time"] = new_avg
                
                # Clear input and rerun
                st.session_state.user_input = ""
                st.rerun()
                
            except Exception as e:
                st.error(f" Error generating response: {e}")
                
                # Add error message
                error_msg = """I apologize, but I encountered an error processing your question. 

**Possible solutions:**
 Try rephrasing your question
 Use simpler language
 Check if the system is properly initialized

**For immediate help:** Contact the university directly at su.edu.pk"""

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": None
                })
                st.rerun()
    
    # Show detailed stats if requested
    if st.session_state.get('show_stats', False):
        st.markdown("---")
        st.markdown("##  Detailed System Statistics")
        
        if rag_pipeline:
            try:
                with st.spinner("Loading statistics..."):
                    detailed_stats = rag_pipeline.get_pipeline_stats()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Query Performance")
                    query_stats = detailed_stats['queries']
                    st.metric("Total Queries", query_stats['total_queries'])
                    st.metric("Success Rate", f"{query_stats['success_rate']}%")
                
                with col2:
                    st.markdown("### Response Performance")
                    perf_stats = detailed_stats['performance']
                    st.metric("Avg Response Time", f"{perf_stats['average_response_time']:.2f}s")
                    st.metric("Total Tokens Used", f"{perf_stats['total_tokens_used']:,}")
                
                with col3:
                    st.markdown("### System Configuration")
                    config_stats = detailed_stats['configuration']
                    st.metric("Embedding Service", config_stats['embedding_service'])
                    st.metric("LLM Available", "" if config_stats['llm_available'] else "")
                
                # Show raw stats in expandable section
                with st.expander(" Detailed Statistics"):
                    st.json(detailed_stats)
                
                # Reset stats display flag
                st.session_state.show_stats = False
                
            except Exception as e:
                st.error(f"Error loading statistics: {e}")
    
    # Download conversation functionality
    if len(st.session_state.messages) > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(" Download Conversation History", use_container_width=True):
                # Prepare conversation data
                conversation_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "university": "University of Sargodha",
                    "session_stats": st.session_state.pipeline_stats,
                    "total_messages": len(st.session_state.messages) - 1,  # Exclude welcome message
                    "conversation": [
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                            "metadata": msg.get("metadata"),
                            "timestamp": datetime.now().isoformat()
                        }
                        for msg in st.session_state.messages[1:]  # Skip welcome message
                    ]
                }
                
                # Create download
                json_str = json.dumps(conversation_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label=" Download as JSON",
                    data=json_str,
                    file_name=f"su_chatbot_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    # Footer (unchanged, already using triple quotes)
    st.markdown("""
    <div class="footer">
        <p><strong> University of Sargodha Official AI Assistant</strong></p>
        <p>For official inquiries and the most current information, visit 
        <a href="https://su.edu.pk" target="_blank">su.edu.pk</a> or contact the relevant department directly.</p>
        <p><em>This AI assistant provides information based on official university documents. 
        While every effort is made to ensure accuracy, please verify important information with the university.</em></p>
        <p><small>Powered by RAG (Retrieval-Augmented Generation) Technology  Built for University of Sargodha</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()