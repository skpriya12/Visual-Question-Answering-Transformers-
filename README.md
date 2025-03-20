# Visual Q&A using Transformers

This repository provides an AI-powered Visual Question Answering (VQA) system using the BLIP (Bootstrapped Language-Image Pretraining) model from Salesforce. The model takes an image and a natural language question as input and generates an answer based on the image's content.

# Features

🏞️ Image-based Question Answering – Ask any question about an image, and the model will generate an intelligent response.

🤖 Powered by BLIP – Utilizes the Salesforce/blip-vqa-base model for high-quality answers.

⚡ Fast and User-Friendly – Implements an interactive Gradio UI for easy use.

🔌 Plug & Play – Works with local images and can be extended to handle URLs.

# Installation

Ensure you have Python 3.8+ installed, then install the required dependencies:

pip install transformers pillow gradio torch requests

# How It Works
1. Loads the BLIP processor and BLIP-VQA model.
2. Accepts an image and a text question.
3. Processes inputs into a format suitable for the model.
4. Uses generate() to predict an answer.
5. Displays the result in a simple, user-friendly UI.
