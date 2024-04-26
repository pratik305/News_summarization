from flask import Flask, render_template, request
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load your model and tokenizer
model_path = "./pegasus-dailymail-model"
tokenizer_path = "./tokenizer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Define your text summarization function
def summarize_text(text):
    # Preprocess your text (tokenization, padding, etc.)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    # Forward pass through your model
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated summary
    summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summarized_text

@app.route('/')
def index():
    text = request.args.get('text', '')  # Get the input text from the query parameter
    return render_template('index.html', text=text, summarized_text="")

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    summarized_text = summarize_text(text)  # Use the summarize_text function to generate the summarized text
    return render_template('index.html', text=text, summarized_text=summarized_text)

if __name__ == '__main__':
    app.run(debug=True)
