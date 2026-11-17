import gradio as gr
from transformers import pipeline

# Load a ready-made text classification model
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

def classify(text):
    # Handle empty input
    if not text or not text.strip():
        return "Please type some text first."
    result = classifier(text)[0]
    return f"{result['label']} (confidence: {result['score']:.0%})"

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Type your text here", lines=4),
    outputs=gr.Textbox(label="Prediction"),
    title="Text Classifier",
)

demo.launch()