import os
import pandas as pd
import spacy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Load spaCy small model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities using spaCy's NER.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def parse_ann_file(ann_filepath):
    """
    Parse .ann files to extract annotations.
    """
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
            annotations.append({
                "id": annotation_id,
                "label": label,
                "start": start_idx,
                "end": end_idx,
                "text": annotation_text
            })
    return annotations

def process_dataset(dataset_path):
    """
    Process the dataset to extract entities and labels.
    """
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
                entities = extract_entities(ann["text"])
                for entity, ent_type in entities:
                    all_annotations.append({
                        "Essay": file,
                        "Label": ann["label"],
                        "Entity": entity,
                        "Type": ent_type
                    })
    
    return pd.DataFrame(all_annotations)

def analyze_bias(df):
    """
    Analyze bias in entity usage (e.g., gender, race).
    """
    # Focus on PERSON and NORP entities
    bias_entities = ["PERSON", "NORP"]
    filtered_df = df[df["Type"].isin(bias_entities)]

    # Count occurrences of each entity type per label
    entity_label_counts = filtered_df.groupby(["Type", "Label"]).size().unstack(fill_value=0)

    # Normalize counts by label
    entity_label_freq_norm = entity_label_counts.div(entity_label_counts.sum(axis=0), axis=1)

    return entity_label_counts, entity_label_freq_norm

def visualize_bias(entity_label_counts, entity_label_freq_norm):
    """
    Visualize bias in entity usage.
    """
    # Plot raw counts using a bar plot
    plt.figure(figsize=(12, 6))
    entity_label_counts.plot(kind="bar", stacked=True, colormap="viridis", edgecolor="black")
    plt.title("Entity-Label Counts (PERSON and NORP)", fontsize=16)
    plt.xlabel("Entity Type", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Argument Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # Plot normalized frequencies using a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(entity_label_freq_norm, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Normalized Entity-Label Frequencies (PERSON and NORP)", fontsize=16)
    plt.xlabel("Argument Label", fontsize=14)
    plt.ylabel("Entity Type", fontsize=14)
    plt.tight_layout()
    plt.show()

# Define dataset path
dataset_path = "/Users/sahil.pardasani/Desktop/ResearchPaperForConference/ArgumentAnnotatedEssays-1.0/brat-project"  # Update this path

# Process dataset
df = process_dataset(dataset_path)

# Analyze bias
entity_label_counts, entity_label_freq_norm = analyze_bias(df)

# Visualize results
visualize_bias(entity_label_counts, entity_label_freq_norm)