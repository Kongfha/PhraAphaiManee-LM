from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
import torch
from tqdm import tqdm

from make_data import get_loader

import argparse

parser = argparse.ArgumentParser(description='use the model')

parser.add_argument('--train_path', metavar="TRAIN-PATH", type=str, default="./Dataset/phra_aphai-train.txt",help="training dataset path")
parser.add_argument('--val_path', metavar="VAL-PATH", type =str, default="./Dataset/phra_aphai-val.txt",help="validation dataset path")
parser.add_argument('--pretrained_path', metavar="PRE-PATH", type=str,default="tupleblog/generate-thai-lyrics",help="pretrained model path")
parser.add_argument('--epochs',metavar='NUM-EPOCH', type=int, required=True, help="number of epochs")
parser.add_argument('--lr',metavar='LR', type = float, default=2e-5, help="learning rate")
parser.add_argument('--save_path',metavar='SAVE-PATH', type = str, default="./", help="saving path for model and tokenizer")

def make_model(pretrained_path, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(pretrained_path)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr, no_deprecation_warning=True)
    return tokenizer, model, optimizer

def train_loop(pretrained_path, train_path, eval_path, num_epochs, lr):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Getting model")
    tokenizer, model, optimizer = make_model(pretrained_path, lr)

    print("Getting data")
    train_loader, valid_loader = get_loader(tokenizer,train_path, eval_path)

    print("Training")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        for batch in tqdm(valid_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    return model, tokenizer

def save_model(model,path):
    model.save_pretrained(path)

def save_tokenizer(tokenizer,path):
    tokenizer.save_pretrained(path)


if __name__ == "__main__":
    args = parser.parse_args()
    pretrained_path = args.pretrained_path
    train_path = args.train_path
    eval_path = args.val_path
    num_epochs = args.epochs 
    lr = args.lr
    save_path = args.save_path

    print("Running")
    model, tokenizer = train_loop(pretrained_path, train_path, eval_path, num_epochs, lr)

    if save_path[-1] != "/":
        save_path += "/"

    model_path = save_path+"model"
    print(f"Saving Model to {model_path}")
    save_model(model,model_path)

    tokenizer_path = save_path+"tokenizer"
    print(f"Saving Tokenizer to {tokenizer_path}")
    save_tokenizer(tokenizer,tokenizer_path)

    print("Finished")

