import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load your dataset
input_csv_path = r"C:\Users\om\Desktop\unlabeled.csv" # add path of unlabeled dataset
output_csv_path = r"C:\Users\om\Desktop\labeled3.csv" # add path where we want to store labeled dataset
df = pd.read_csv(input_csv_path)

# Load pre-trained BERT model and tokenizer from Hugging Face
model_name = "nateraw/bert-base-uncased-emotion"  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Define emotion labels and their corresponding target labels
emotion_categories = ['anger', 'joy', 'sadness', 'neutral']
emotion_labels = {
    'anger': 1,
    'joy': 2,
    'sadness': 3,
    'neutral': 4
}

# Function to classify text
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs, dim=1).item()
    
    if label < len(emotion_categories):
        emotion = emotion_categories[label]
        if emotion in emotion_labels:
            return emotion_labels[emotion]  # Return the corresponding label
    return None  # Return None if the label is not in the desired categories

# Apply classification and filter rows
df['label'] = df['text'].apply(classify_text)
df = df.dropna(subset=['label'])  # Remove rows where 'label' is None

# Save the new dataframe to a CSV file
df.to_csv(output_csv_path, index=False)

print(f"Labeled data saved to {output_csv_path}")
