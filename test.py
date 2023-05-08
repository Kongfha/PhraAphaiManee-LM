from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import argparse

parser = argparse.ArgumentParser(description='use the model')

parser.add_argument('--input', metavar='TEXT', type=str, required=True, help='starter text')
parser.add_argument('--model_path', metavar='MODEL-PATH', type=str, default = "./model", help='path to model')
parser.add_argument('--tokenizer_path', metavar='TOK-PATH', type=str, default = "./tokenizer", help='path to tokenizer')
parser.add_argument('--max_length', metavar='MAXLENGTH', type=int, default = 140, help = "max length of output")
parser.add_argument('--top_k', metavar='TOPK', type=int, default = 25, help = "top_k")
parser.add_argument('--temperature', metavar='TEMP', type=float, default = 1.0, help = "temperature")



def test(model, tokenizer, text, max_length,top_k,temperature):
    nlp = pipeline("text-generation",model=model,tokenizer=tokenizer)    
    generated_text = nlp(text,max_length=max_length,top_k=top_k,temperature=temperature)
    print(f"Input: {text}")
    print(f"Output:\n {generated_text[0]['generated_text']}")

if __name__ == "__main__":
    args = parser.parse_args()
    mode_path = args.model_path
    tokenizer_path = args.tokenizer_path
    text = args.input
    max_length = args.max_length
    top_k = args.top_k
    temperature = args.temperature
    print("Getting model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(mode_path)

    print("Generating output")
    test(model,tokenizer,text,max_length,top_k,temperature)

