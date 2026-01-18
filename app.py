import gradio as gr
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the pipeline functions and Gradio app creator
from main import create_gradio_app

# Create and launch the Gradio app
app = create_gradio_app()

if __name__ == "__main__":
    app.launch()
