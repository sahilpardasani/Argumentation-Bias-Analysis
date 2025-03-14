import os
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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
            start_idx = int(annotation_type_info[1])
            end_idx = int(annotation_type_info[2])
            annotations.append({"id": annotation_id, "label": label, "start": start_idx, "end": end_idx, "text": annotation_text})
    return annotations

def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

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
            annotations = parse_ann_file(ann_filepath)
            for ann in annotations:
                entities = extract_named_entities(ann["text"])
                for entity, ent_type in entities:
                    all_annotations.append({"Essay": file, "Label": ann["label"], "Entity": entity, "Type": ent_type})
    return pd.DataFrame(all_annotations)

def visualize_entity_distribution(df):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="Type", order=df["Type"].value_counts().index)
    plt.title("Entity Distribution in Argument Components")
    plt.xticks(rotation=45)
    plt.show()

dataset_path = "/Users/sahil.pardasani/Desktop/ResearchPaperForConference/ArgumentAnnotatedEssays-1.0/brat-project"  # Change this path if needed
df = process_dataset(dataset_path)
visualize_entity_distribution(df)
