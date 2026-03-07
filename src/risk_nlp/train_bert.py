import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── CONFIG ────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
NUM_LABELS  = 5
MAX_LENGTH  = 128
EPOCHS      = 4
BATCH_SIZE  = 8
MODEL_PATH  = "models/bert_risk_classifier/"
os.makedirs(MODEL_PATH, exist_ok=True)

LABEL_NAMES = [
    "Normal",
    "Port_Strike",
    "Natural_Disaster",
    "Raw_Material_Shortage",
    "Geopolitical"
]

# ─── DATASET CLASS ─────────────────────────────────────────
class RiskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─── METRICS FUNCTION ──────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    acc            = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# ─── LOAD DATA ─────────────────────────────────────────────
print("📦 Loading training data...")
df = pd.read_csv("data/processed/risk_training_data.csv")
print(f"   Total samples: {len(df)}")
print(f"   Label distribution:\n{df['label_name'].value_counts()}\n")

# Train / test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"   Train: {len(train_texts)} | Test: {len(test_texts)}")

# ─── LOAD TOKENIZER & MODEL ────────────────────────────────
print(f"\n🤗 Loading {MODEL_NAME}...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model     = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)
print("   ✅ Model loaded")

# ─── CREATE DATASETS ───────────────────────────────────────
train_dataset = RiskDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
test_dataset  = RiskDataset(test_texts,  test_labels,  tokenizer, MAX_LENGTH)

# ─── TRAINING ARGUMENTS ────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = MODEL_PATH,
    num_train_epochs        = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    eval_strategy           = "epoch",        # ← renamed from evaluation_strategy
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "accuracy",
    logging_steps           = 10,
    warmup_steps            = 10,
    weight_decay            = 0.01,
    report_to               = "none"
)

# ─── TRAINER ───────────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = test_dataset,
    compute_metrics = compute_metrics
)

# ─── TRAIN ─────────────────────────────────────────────────
print("\n🚀 Fine-tuning DistilBERT...")
trainer.train()

# ─── EVALUATE ──────────────────────────────────────────────
print("\n📊 Evaluating model...")
predictions_output = trainer.predict(test_dataset)
y_pred = np.argmax(predictions_output.predictions, axis=-1)
y_true = test_labels

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

# ─── CONFUSION MATRIX ──────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d',
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    cmap='Blues'
)
plt.title('Confusion Matrix — Risk Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{MODEL_PATH}confusion_matrix.png")
plt.show()
print(f"✅ Confusion matrix saved")

# ─── SAVE MODEL & TOKENIZER ────────────────────────────────
model.save_pretrained(f"{MODEL_PATH}final/")
tokenizer.save_pretrained(f"{MODEL_PATH}final/")
print(f"\n✅ Model saved to {MODEL_PATH}final/")