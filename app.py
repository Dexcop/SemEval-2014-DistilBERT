from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import torch

CATEGORIES = ["food", "service", "price", "ambience", "anecdotes/miscellaneous"]

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)

    # Multi-label classification → use sigmoid
    probs = torch.sigmoid(outputs.logits).flatten()

    # Threshold (you can tune this)
    preds = (probs >= 0.5).int().tolist()

    # Format result
    result = {cat: pred for cat, pred in zip(CATEGORIES, preds)}
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Review"),
    outputs=gr.JSON(label="Predicted Aspect Categories"),
    title="Restaurant Aspect Category Classifier"
)

demo.launch()
