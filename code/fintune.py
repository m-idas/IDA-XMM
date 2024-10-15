from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import load_preprocess_data
K=10
EPOCHS=100
BATCH=8
def get_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.readlines()
    return [label.strip() for label in labels]

model_name = './model/IDA-XMM-pretrained'
tokenizer = BertTokenizer.from_pretrained('./model/IDA-XMM-tokenizer')
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=K)  


inputs = load_preprocess_data("./data/","data_finetune", tokenizer)
inputs = {k: v.to(device) for k, v in inputs.items()}
labels=get_labels("./data/label_finetune")

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'],labels)
dataloader = DataLoader(dataset, batch_size=BATCH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
epochs=EPOCHS
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        total_loss += loss.item()
        
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

model.save_pretrained('./model/MIDAS-finetune')

