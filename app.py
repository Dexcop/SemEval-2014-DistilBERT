from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import torch

CATEGORIES = ["food", "service", "price", "ambience", "anecdotes/miscellaneous"]

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    # Multi-label classification
    probs = torch.sigmoid(outputs.logits).flatten()
    preds = (probs >= 0.5).int().tolist()

    # Only keep categories where prediction == 1
    active = [cat for cat, p in zip(CATEGORIES, preds) if p == 1]

    # If nothing predicted, return "None"
    return ", ".join(active) if active else "No aspect detected"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Review"),
    outputs=gr.Textbox(label="Detected Aspects"),
    title="Aspect Category Detector"
)

demo.launch()
