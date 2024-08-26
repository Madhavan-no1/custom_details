import re
import spacy

# Load spaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Clean the text by removing unwanted characters
    text = re.sub(r'\n', ' ', text)  # Replace newline characters with spaces
    text = re.sub(r'[^A-Za-z0-9,.!? ]+', '', text)  # Remove special characters
    return text

# Example usage
transcript = """
Customer: I'm looking for an SUV, preferably with an automatic transmission AND Petrol car.
Salesperson: Sure! We have a few options that might suit your needs. Any specific color in mind?
Customer: I would love a black one, and it should be a recent model, like 2020 or 2021.
Salesperson: Great! We have exactly what you're looking for.
"""

cleaned_transcript = preprocess_text(transcript)
print(cleaned_transcript)


def extract_information(text):
    doc = nlp(text)
    customer_requirements = {
        "Car Type": None,
        "Fuel Type": None,
        "Color": None,
        "Distance Travelled": None,
        "Make Year": None,
        "Transmission Type": None,
    }

    for token in doc:
        # Extract car type
        if token.text.lower() in ["suv", "hatchback", "sedan"]:
            customer_requirements["Car Type"] = token.text

        # Extract transmission type
        if token.text.lower() in ["manual", "automatic"]:
            if token.nbor().text.lower() == "transmission":
                customer_requirements["Transmission Type"] = token.text + " transmission"
            else:
                customer_requirements["Transmission Type"] = token.text

        # Extract make year
        if token.like_num:
            try:
                number = int(token.text)
                if 1990 < number < 2024:
                    customer_requirements["Make Year"] = number
            except ValueError:
                continue  # Skip non-numeric tokens like "one"

        # Extract color
        if token.text.lower() in ["black", "white", "red", "blue", "silver", "gray", "green"]:
            customer_requirements["Color"] = token.text

        # Extract fuel type
        if token.text.lower() in ["diesel", "petrol", "electric", "hybrid"]:
            customer_requirements["Fuel Type"] = token.text

        # Extract distance travelled
        if token.like_num and token.nbor().text.lower() in ["miles", "km", "kilometers"]:
            customer_requirements["Distance Travelled"] = token.text + " " + token.nbor().text

    return customer_requirements

# Example usage
extracted_info = extract_information(cleaned_transcript)
print(extracted_info)

#pipelinec  

import warnings
from transformers import pipeline

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")
# Load a zero-shot classification pipeline with the clean_up_tokenization_spaces parameter set to True
classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", clean_up_tokenization_spaces=True
)

def detect_objections(text):
    candidate_labels = ["refurbishment quality", "car issues", "price issues", "customer experience issues"]
    result = classifier(text, candidate_labels)
    return result

# Example usage
objections = detect_objections(cleaned_transcript)
print(objections)


#flask

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    transcript = file.read().decode('utf-8')
    
    # Process the transcript
    cleaned_transcript = preprocess_text(transcript)
    extracted_info = extract_information(cleaned_transcript)
    objections = detect_objections(cleaned_transcript)
    
    # Combine all the results
    response = {
        "Customer Requirements": extracted_info,
        "Objections": objections
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
