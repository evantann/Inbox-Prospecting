import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Define the model repo
model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input
inputs = tokenizer("Dettan@ucdavis.edu  25/25 25/25 0/25 0/25 total = 50 C+", return_tensors="pt")

# Model apply
with torch.no_grad():  # No need to track gradients for inference
    outputs = model(**inputs)

# Get the logits (raw prediction scores)
logits = outputs.logits

# Apply softmax to get probabilities
probabilities = F.softmax(logits, dim=-1)

# Get predicted class
predicted_class = torch.argmax(probabilities, dim=-1).item()

# Map predicted class to label (spam or not spam)
labels = ['not spam', 'spam']
predicted_label = labels[predicted_class]

print(f"Predicted label: {predicted_label}")

# On Thu, May 30, 2024 at 2:12 PM Evan Tan <ettan@ucdavis.edu> wrote:
# From: Evan Tan <ettan@ucdavis.edu>