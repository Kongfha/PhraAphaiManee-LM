from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset


class WannakadeeDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            cur = ""
            for line in f:
                line = line.strip()
                if line:
                    if(line[0]=="๏"):
                        self.lines.append(cur)
                        cur = ""
                    cur += line + " \n "
                else:
                    lines.append(cur + line + " \n ")
        self.lines = self.lines[1:]        

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        encoded = self.tokenizer.encode_plus(
            line,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

def get_loader(tokenizer, train_path, eval_path):
    train_dataset = WannakadeeDataset(train_path, tokenizer, max_length=400)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataset = WannakadeeDataset(eval_path, tokenizer, max_length=400)
    valid_loader = DataLoader(valid_dataset, batch_size=8)

    return train_loader, valid_loader
