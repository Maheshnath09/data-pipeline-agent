import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the pipeline functions and Gradio app creator
from main import create_gradio_app

# Create the Gradio app
app = create_gradio_app()

# For Hugging Face Spaces deployment
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
