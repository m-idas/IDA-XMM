from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import load_preprocess_data
EPOCHS=100
BATCH=8
# Load the model and tokenizer
model_name = 'bert'
tokenizer = BertTokenizer.from_pretrained('./model/IDA-XMM-tokenizer')
model = BertForMaskedLM.from_pretrained(model_name)
model.train()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Prepare data
inputs = load_preprocess_data("./data/","data_pretrain", tokenizer)
inputs = {k: v.to(device) for k, v in inputs.items()}
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
dataloader = DataLoader(dataset, batch_size=BATCH)

# Model pre-training
num_epochs = EPOCHS
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
#save model
model.save_pretrained('./model/IDA-XMM-pretrained')

