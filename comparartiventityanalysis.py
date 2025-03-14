import os
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

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

# Function to extract named entities
def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to categorize topics
def categorize_topic(text):
    social_keywords = {"society", "culture", "education", "justice", "discrimination", "rights", "equality"}
    economic_keywords = {"economy", "finance", "market", "business", "money", "tax", "trade", "employment"}
    technology_keywords = {"AI", "technology", "innovation", "science", "engineering", "internet"}
    politics_keywords = {"government", "policy", "election", "law", "politics", "democracy"}

    text_lower = text.lower()
    if any(word in text_lower for word in social_keywords):
        return "Social Issues"
    elif any(word in text_lower for word in economic_keywords):
        return "Economic Issues"
    elif any(word in text_lower for word in technology_keywords):
        return "Technology"
    elif any(word in text_lower for word in politics_keywords):
        return "Politics"
    else:
        return "Other"

# Function to process dataset
def process_dataset(dataset_path):
    all_annotations = []
    for file in os.listdir(dataset_path):
        if file.endswith(".txt"):
            text_filepath = os.path.join(dataset_path, file)
            ann_filepath = text_filepath.replace(".txt", ".ann")
            if not os.path.exists(ann_filepath):
                continue
            
            with open(text_filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            topic = categorize_topic(text)
            annotations = parse_ann_file(ann_filepath)
            for ann in annotations:
                entities = extract_named_entities(ann["text"])
                for entity, ent_type in entities:
                    all_annotations.append({"Essay": file, "Topic": topic, "Label": ann["label"], "Entity": entity, "Type": ent_type})
    
    return pd.DataFrame(all_annotations)

# Function to visualize entity distribution across topics
def visualize_entity_distribution_by_topic(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x="Type", hue="Topic", order=df["Type"].value_counts().index)
    plt.title("Entity Distribution by Topic")
    plt.xticks(rotation=45)
    plt.show()

# Define dataset path (change accordingly)
dataset_path = "/Users/sahil.pardasani/Desktop/ResearchPaperForConference/ArgumentAnnotatedEssays-1.0/brat-project"  # Update this path to your dataset directory

# Process dataset
df = process_dataset(dataset_path)

# Generate visualization
visualize_entity_distribution_by_topic(df)
