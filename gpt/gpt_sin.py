import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer


# 1. 데이터 생성
pi = np.pi
x = np.linspace(-pi, pi, 1000)
y = np.sin(x)

# Define a custom dataset for the sin function values
class SinDataset(Dataset):
    def __init__(self, x, y, tokenizer, block_size=128):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.block_size = block_size
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        xi = self.x[idx]
        yi = self.y[idx]
        tokenized_input = self.tokenizer(f"{xi:.5f}", return_tensors="pt")
        tokenized_output = self.tokenizer(f"{yi:.5f}", return_tensors="pt")
        return tokenized_input.input_ids.squeeze(), tokenized_output.input_ids.squeeze()

# 2. 데이터 준비
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

train_dataset = SinDataset(x, y, tokenizer)
data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. 모델 및 토크나이저 정의
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# 4. 학습
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
)

def pad_sequences(sequences, pad_token):
    max_length = max(len(seq) for seq in sequences)
    return [seq.tolist() + [pad_token] * (max_length - len(seq)) for seq in sequences]

def collate_fn(batch):
    input_ids = [item[0].tolist() for item in batch]
    labels = [item[1].tolist() for item in batch]
    pad_token_id = tokenizer.pad_token_id

    # Calculate the max length for both input and labels
    max_length = max(max(len(seq) for seq in input_ids), max(len(seq) for seq in labels))

    # Pad sequences based on the max length
    input_ids = [seq + [pad_token_id] * (max_length - len(seq)) for seq in input_ids]
    labels = [seq + [pad_token_id] * (max_length - len(seq)) for seq in labels]

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels)
    }

# Update the Trainer with the new data_collator
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
)

trainer.train()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 학습 결과
model.eval()
predicted_ys = []

for xi in x:
    tokenized_input = tokenizer(f"{xi:.5f}", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(tokenized_input.input_ids)
        predictions = outputs.logits
    predicted_token = torch.argmax(predictions, dim=2)
    predicted_text = tokenizer.decode(predicted_token[0], skip_special_tokens=True)
    
    # 텍스트를 숫자로 변환하는 함수
    try:
        predicted_ys.append(float(predicted_text))
    except ValueError:  # 숫자 변환 오류 처리
        predicted_ys.append(0)  # 잘못된 예측값은 0으로 처리 (적절한 처리 방법을 선택해야 함)

# 6. 결과 시각화
plt.figure(figsize=(10,6))
plt.plot(x, y, label='True Sin function')
plt.plot(x, predicted_ys, label='Predicted by GPT-2', linestyle='dashed')
plt.legend()
# save image
plt.savefig('gpt_sin.png')