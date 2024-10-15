import os
from torch.utils.data import DataLoader, TensorDataset
max_length=512
def load_and_preprocess_data(directory, file_name, tokenizer, max_length=512):
    # 构建完整的文件路径
    file_path = os.path.join(directory, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # 读取二进制文件内容
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    
    # 假设二进制文件直接包含UTF-8编码的文本
    raw_data = raw_data.decode('utf-8')
    
    # 将文本按行分割成列表（每行一个文本实例）
    data = raw_data.split('\n')
    
    # 使用tokenizer处理文本
    inputs = tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    
    # 创建TensorDataset
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    return dataset