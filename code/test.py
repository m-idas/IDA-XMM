import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import load_preprocess_data
def get_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.readlines()
    return [label.strip() for label in labels]
model_path = "./model/IDA-XMM-finetune"

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('./model/IDA-XMM-tokenizer')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load evaluation data
inputs = load_preprocess_data("./data/","data_test", tokenizer)
inputs = {k: v.to(device) for k, v in inputs.items()}
labels=get_labels("./data/label_test")

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'],labels)
dataloader = DataLoader(dataset, batch_size=8)

# Prediction and evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = batch[2].detach().cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")
