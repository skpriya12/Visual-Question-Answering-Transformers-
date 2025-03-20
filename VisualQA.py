from transformers.utils import logging

logging.set_verbosity_error()

import requests
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")


def answer_question(image, question):
    """Generates an answer to a question based on the provided image."""
    if image is None or not question.strip():
        return "Please upload an image and enter a question."

    # Process image and question
    inputs = processor(image, question, return_tensors="pt")

    # Generate answer
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


# Gradio UI
iface = gr.Interface(
    fn=answer_question,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Enter your question")],
    outputs=gr.Textbox(label="Answer"),
    title="Visual Question Answering with BLIP",
    description="Upload an image and ask a question. The model will generate an answer.",
)

# Launch the UI
iface.launch()
iface.close()