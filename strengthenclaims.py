import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from textblob import TextBlob

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
                all_annotations.append({
                    "Essay": file,
                    "Label": ann["label"],
                    "Text": ann["text"]
                })
    
    return pd.DataFrame(all_annotations)

def calculate_sentiment(text):
    """
    Calculate sentiment polarity using TextBlob.
    """
    return TextBlob(text).sentiment.polarity

def chi_square_test(df):
    """
    Perform a Chi-Square test to check if entity distributions across labels are significant.
    """
    # Create a contingency table of entity types vs. labels
    contingency_table = pd.crosstab(df["Type"], df["Label"])

    # Perform Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-Square Statistic: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of Freedom: {dof}")
    print("Expected Frequencies Table:")
    print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))

    if p < 0.05:
        print("The distribution of entities across labels is statistically significant (p < 0.05).")
    else:
        print("The distribution of entities across labels is not statistically significant (p >= 0.05).")

def perform_anova(df):
    """
    Perform ANOVA to test if sentiment differences across labels are significant.
    """
    # Group sentiment scores by label
    sentiment_groups = [df[df["Label"] == label]["Sentiment"] for label in df["Label"].unique()]

    # Perform ANOVA
    f_stat, p_value = f_oneway(*sentiment_groups)

    print(f"F-statistic: {f_stat}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("There are significant differences in sentiment across labels (p < 0.05).")
    else:
        print("There are no significant differences in sentiment across labels (p >= 0.05).")

# Define dataset path
dataset_path = "/Users/sahil.pardasani/Desktop/ResearchPaperForConference/ArgumentAnnotatedEssays-1.0/brat-project"  # Update this path

# Process dataset
df = process_dataset(dataset_path)

# Add sentiment scores to the dataset
df["Sentiment"] = df["Text"].apply(lambda x: calculate_sentiment(x))

# Perform Chi-Square test (if entity types are available)
# Uncomment the following lines if you have entity types in your dataset
# chi_square_test(df)

# Perform ANOVA
perform_anova(df)