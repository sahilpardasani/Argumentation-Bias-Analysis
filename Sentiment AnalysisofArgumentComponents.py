import os
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to parse .ann files
def parse_ann_file(ann_filepath):
    annotations = []
    with open(ann_filepath, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            annotation_id = parts[0]
            annotation_type_info = parts[1].split()
            annotation_text = parts[2]
            label = annotation_type_info[0]
            annotations.append({"id": annotation_id, "label": label, "text": annotation_text})
    return annotations

# Function to perform sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to process dataset and analyze sentiment
def process_sentiment_analysis(dataset_path):
    sentiment_data = []
    for file in os.listdir(dataset_path):
        if file.endswith(".txt"):
            text_filepath = os.path.join(dataset_path, file)
            ann_filepath = text_filepath.replace(".txt", ".ann")
            if not os.path.exists(ann_filepath):
                continue
            
            annotations = parse_ann_file(ann_filepath)
            for ann in annotations:
                sentiment = get_sentiment(ann["text"])
                sentiment_data.append({"Essay": file, "Label": ann["label"], "Sentiment": sentiment})
    
    return pd.DataFrame(sentiment_data)

# Function to visualize sentiment distribution
def visualize_sentiment_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Label", y="Sentiment")
    plt.title("Sentiment Distribution Across Argument Components")
    plt.axhline(0, color='red', linestyle='dashed')
    plt.show()

# Define dataset path (change accordingly)
dataset_path = "/Users/sahil.pardasani/Desktop/ResearchPaperForConference/ArgumentAnnotatedEssays-1.0/brat-project"  # Update this path to your dataset directory

# Process dataset
df = process_sentiment_analysis(dataset_path)

# Generate visualization
visualize_sentiment_distribution(df)
