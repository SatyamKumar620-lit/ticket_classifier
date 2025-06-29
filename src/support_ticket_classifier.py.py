
import os
import re
import nltk
import torch
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.utils import resample
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from textblob import TextBlob
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer
import gradio as gr

# ----------------------- NLTK & SpaCy Setup -----------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
spacy.cli.download("en_core_web_sm")

stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
os.environ["WANDB_DISABLED"] = "true"

# ----------------------- Load and Preprocess -----------------------
df = pd.read_excel("ai_dev_assignment_tickets_complex_1000.xls")
df.dropna(subset=['ticket_text', 'issue_type', 'urgency_level', 'product'], inplace=True)
df = df.drop_duplicates(subset=['ticket_text']).sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]
    return ' '.join(tokens)

df['clean_text'] = df['ticket_text'].apply(clean_text)
df['ticket_length'] = df['ticket_text'].apply(lambda x: len(str(x)))
df['sentiment'] = df['ticket_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

complaint_keywords = ['broken', 'error', 'late', 'missing', 'faulty', 'damaged', 'cracked', 'slow', 'delayed', 'urgent', 'immediately', 'asap']
df['complaint_score'] = df['ticket_text'].apply(lambda x: sum(1 for token in str(x).lower().split() if token in complaint_keywords))

# ----------------------- SBERT Embeddings -----------------------
X_sbert = sbert_model.encode(df['clean_text'].tolist())
X_aug = np.hstack((X_sbert, df[['ticket_length', 'sentiment', 'complaint_score']].values))

# ----------------------- Issue Type Classification (Random Forest) -----------------------
y_issue = df['issue_type']
X_train_issue, X_test_issue, y_train_issue, y_test_issue = train_test_split(X_aug, y_issue, test_size=0.2, random_state=42)

clf_issue = RandomForestClassifier(random_state=42)
clf_issue.fit(X_train_issue, y_train_issue)
y_pred_iss = clf_issue.predict(X_test_issue)
print("\n Issue Type Classification Report:")
print(classification_report(y_test_issue, y_pred_iss))
le_issue = LabelEncoder()
le_issue.fit(y_issue)
y_val_iss = le_issue.transform(y_test_issue)
y_pred_iss = le_issue.transform(y_pred_iss)

# ----------------------- Urgency Level Classification (DistilBERT) -----------------------
le = LabelEncoder()
df['urgency_label'] = le.fit_transform(df['urgency_level'])

train_texts, val_texts, y_train, y_val = train_test_split(
    df['ticket_text'].tolist(), df['urgency_label'].tolist(),
    test_size=0.2, stratify=df['urgency_label'], random_state=42
)

# Upsample Low class
df_train = pd.DataFrame({"text": train_texts, "label": y_train})
low_class = le.transform(['Low'])[0]
df_majority = df_train[df_train['label'] != low_class]
df_minority = df_train[df_train['label'] == low_class]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1, random_state=42)

train_texts_balanced = df_balanced['text'].tolist()
y_train_balanced = df_balanced['label'].tolist()

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_balanced), y=y_train_balanced)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts_balanced, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

train_dataset = TicketDataset(train_encodings, y_train_balanced)
val_dataset = TicketDataset(val_encodings, y_val)

class WeightedDistilBertForClassification(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(le.classes_))
model = WeightedDistilBertForClassification(base_model.config, class_weights_tensor)
model.load_state_dict(base_model.state_dict(), strict=False)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir="./logs",
    logging_steps=10
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average='macro')}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

print("\n🔍 Final Evaluation on Validation Set:")
print(classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0))

# ----------------------- Visualizations -----------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='urgency_level', data=df)
plt.title("Ticket Distribution by Urgency Level")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, display_labels=le.classes_, cmap="Blues", values_format="d", ax=ax[0])
ax[0].set_title("Urgency Level")
ConfusionMatrixDisplay.from_predictions(y_val_iss, y_pred_iss, display_labels=le_issue.classes_, cmap="Purples", values_format="d", ax=ax[1])
ax[1].set_title("Issue Type")
plt.tight_layout()
plt.show()

# ----------------------- Gradio Batch Prediction -----------------------
def predict_batch(texts):
    lines = texts.strip().split("\n")
    inputs = tokenizer(lines, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).tolist()
    return "\n".join(f"{i+1}. {le.inverse_transform([p])[0]}" for i, p in enumerate(preds))

gr.Interface(
    fn=predict_batch,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste multiple ticket texts separated by new lines",
        label="Ticket Texts"
    ),
    outputs=gr.Textbox(label="Urgency Predictions"),
    title="🛠️ Batch Urgency Classifier",
    description="Paste multiple support ticket texts (one per line) and get predicted urgency levels."
).launch()
