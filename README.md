
🛠️ Support Ticket Classifier

This project uses NLP and machine learning techniques to classify customer support tickets based on:

- 🔷 **Urgency Level** (`Low`, `Medium`, `High`)
- 🟢 **Issue Type** (like `Payment`, `Technical`, etc.)

It uses both **classical ML** (Random Forest + SBERT) for issue type classification, and **transformers** (`DistilBERT`) for urgency classification with data rebalancing and weighted loss.


📁 Dataset

The dataset used is:
ai_dev_assignment_tickets_complex_1000.xls

## 🧪 Features

- ✅ Data cleaning and lemmatization using SpaCy
- ✅ Feature engineering: sentiment, length, complaint keyword score
- ✅ Sentence embeddings using SBERT
- ✅ Classifier 1: Random Forest for `issue_type`
- ✅ Classifier 2: DistilBERT for `urgency_level` (weighted + balanced)
- ✅ Gradio interface for batch ticket prediction
- ✅ Matplotlib/Seaborn visualizations
- ✅ Confusion matrices for both classifiers

## DEPENDENCEIES
pip install pandas numpy scikit-learn nltk spacy matplotlib seaborn gradio openpyxl
pip install sentence-transformers
pip install --upgrade transformers