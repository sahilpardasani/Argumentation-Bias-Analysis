This project aims to investigate bias in argumentation mining using the Argument Annotated Essays Dataset. It analyses the distribution of argumentative components, sentiment variability, entity usage across topics, and ethical bias in argument structures. The project implements various statistical tests (ANOVA, Chi-Square) to determine the significance of sentiment and entity distribution differences.

Features

Argument Component Analysis: Identifies Claims, Premises, and Major Claims from annotated essays.
Named Entity Recognition (NER): Extracts and categorizes demographic, geopolitical, and economic entities.
Sentiment Analysis: Uses TextBlob to measure sentiment across argument types.
Statistical Bias Evaluation: Employs Chi-Square and ANOVA tests to detect significant biases in entity and sentiment distributions.
Topic Classification: Categorizes essays into Social Issues, Economic Issues, Politics, Technology, and Other.
Data Visualization: Uses Matplotlib and Seaborn to generate insights into bias patterns.

Installation

Ensure you have Python 3.8+ installed. Then, install the required dependencies:
pip install pandas numpy spacy matplotlib seaborn textblob scipy
python -m spacy download en_core_web_sm

Dataset
The Argument Annotated Essays dataset consists of 90+ essays, each with a corresponding .ann file containing argument annotations. Download the dataset from the web

Results
Claims are more positive in sentiment compared to Premises and Major Claims.
Premises contain the highest number of factual entities (e.g., DATE, MONEY, GPE, ORG).
Social Issues essays tend to have more PERSON and NORP entities, while Economic essays feature more MONEY and ORG entities.

