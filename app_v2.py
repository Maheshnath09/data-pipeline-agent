"""
Gradio App V2 - Multi-Agent Data Pipeline with RAG
Dark theme with purple/pink gradient matching HF Spaces style.
"""
import os
import gradio as gr


def create_multi_agent_app():
    """Create the multi-agent Gradio application."""
    
    # Import here to avoid startup delay
    from multi_agent_pipeline import run_multi_agent_pipeline
    
    def run_pipeline(file_obj, target_column, progress=gr.Progress(track_tqdm=True)):
        """Run the multi-agent pipeline with progress tracking."""
        if file_obj is None:
            return "<h2>Error</h2><p>Please upload a CSV or Excel file.</p>", None
        
        if not target_column or not target_column.strip():
            return "<h2>Error</h2><p>Please enter a target column name.</p>", None
        
        # Get file path
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        # Show initial progress
        progress(0.1, desc="Loading data...")
        
        # Run the multi-agent pipeline with progress callback
        html_report, model_path = run_multi_agent_pipeline(
            file_path=file_path,
            target_column=target_column.strip(),
            progress=progress
        )
        
        progress(1.0, desc="Complete!")
        
        return html_report, model_path
    
    # Custom CSS - Dark theme with purple/pink gradient
    custom_css = """
    /* Hide PWA elements */
    .install-pwa-btn, .update-banner {
        display: none !important;
    }
    
    /* Container styling */
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    
    /* Agent badges - purple/pink gradient */
    .agent-badges {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin: 15px 0 25px 0;
    }
    
    .agent-badge {
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    """
    
    # Build the interface with system default theme
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue"
        ),
        title="Data Pipeline Agent",
        analytics_enabled=False,
        css=custom_css
    ) as app:
        
        # Header
        gr.Markdown("# 🧠 Data Pipeline Agent (Multi-Agent + RAG)")
        gr.Markdown("Upload dataset → Auto-clean → Visualize → Train → Generate Report")
        
        gr.HTML("""
        <div class="agent-badges">
            <span class="agent-badge">🎯 Orchestrator</span>
            <span class="agent-badge">📊 Data Analyst</span>
            <span class="agent-badge">🤖 ML Engineer</span>
            <span class="agent-badge">📈 Visualization</span>
            <span class="agent-badge">💡 Insight</span>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="📁 Upload Dataset (CSV/Excel)",
                    file_types=[".csv", ".xlsx", ".xls"],
                    type="filepath"
                )
                
                target_input = gr.Textbox(
                    label="🎯 Target Column Name",
                    placeholder="e.g., price, target, label",
                    lines=1
                )
                
                run_btn = gr.Button(
                    "Run Pipeline",
                    variant="primary"
                )
            
            with gr.Column(scale=2):
                report_output = gr.HTML(
                    label="📊 Data Pipeline Report"
                )
                
                model_output = gr.File(
                    label="💾 Download Trained Model (.pkl)"
                )
        
        # Connect the button
        run_btn.click(
            fn=run_pipeline,
            inputs=[file_input, target_input],
            outputs=[report_output, model_output]
        )
        
        # Set max file size
        app.max_file_size = 50 * 1024 * 1024  # 50MB
        app.queue()
    
    return app


if __name__ == "__main__":
    app = create_multi_agent_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
