import os
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load spaCy small model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities using spaCy's NER.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

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
            annotations.append({
                "id": annotation_id,
                "label": label,
                "start": start_idx,
                "end": end_idx,
                "text": annotation_text
            })
    return annotations

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
                entities = extract_entities(ann["text"])
                for entity, ent_type in entities:
                    all_annotations.append({
                        "Essay": file,
                        "Label": ann["label"],
                        "Entity": entity,
                        "Type": ent_type
                    })
    
    return pd.DataFrame(all_annotations)

def calculate_pmi(df):
    """
    Calculate PMI scores for entity-label associations.
    """
    entity_label_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    label_counts = defaultdict(int)
    total = len(df)

    for _, row in df.iterrows():
        entity_label_counts[(row["Type"], row["Label"])] += 1
        entity_counts[row["Type"]] += 1
        label_counts[row["Label"]] += 1

    pmi_scores = {}
    for (entity, label), count in entity_label_counts.items():
        p_entity_label = count / total
        p_entity = entity_counts[entity] / total
        p_label = label_counts[label] / total
        pmi = np.log2(p_entity_label / (p_entity * p_label))
        pmi_scores[(entity, label)] = pmi

    return pmi_scores

def analyze_entity_influence(df, entity_types=None):
    """
    Analyze the influence of specific entity types on argument labels.
    """
    if entity_types is None:
        entity_types = ["ORG", "PERSON", "GPE", "DATE"]

    filtered_df = df[df["Type"].isin(entity_types)]
    entity_label_freq = filtered_df.groupby(["Type", "Label"]).size().unstack(fill_value=0)
    entity_label_freq_norm = entity_label_freq.div(entity_label_freq.sum(axis=0), axis=1)

    return entity_label_freq, entity_label_freq_norm

def visualize_pmi_scores(pmi_scores):
    """
    Visualize PMI scores using a bar plot.
    """
    entities, labels, pmi_values = zip(*[(k[0], k[1], v) for k, v in pmi_scores.items()])
    pmi_df = pd.DataFrame({"Entity": entities, "Label": labels, "PMI": pmi_values})

    plt.figure(figsize=(12, 6))
    sns.barplot(data=pmi_df, x="Entity", y="PMI", hue="Label", palette="viridis")
    plt.title("PMI Scores for Entity-Label Associations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_entity_influence(entity_label_freq_norm):
    """
    Visualize normalized entity-label frequencies using a heatmap.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(entity_label_freq_norm, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Normalized Entity-Label Frequency")
    plt.xlabel("Argument Label")
    plt.ylabel("Entity Type")
    plt.tight_layout()
    plt.show()

# Define dataset path
dataset_path = "/Users/sahil.pardasani/Desktop/ResearchPaperForConference/ArgumentAnnotatedEssays-1.0/brat-project"  # Update this path

# Process dataset
df = process_dataset(dataset_path)

# Calculate PMI scores
pmi_scores = calculate_pmi(df)

# Analyze entity influence
entity_label_freq, entity_label_freq_norm = analyze_entity_influence(df)

# Visualize results
visualize_pmi_scores(pmi_scores)
visualize_entity_influence(entity_label_freq_norm)