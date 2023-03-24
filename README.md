# PhraAphaiManee-LM 

PhraAphaiManee-LM or GPT-2 for Thai poem (PhraAphaiManee-Style).
I use [GPT-2 for Thai lyrics](https://huggingface.co/tupleblog/generate-thai-lyrics), 
which is based on [GPT-2 base Thai](https://huggingface.co/flax-community/gpt2-base-thai) as a pre-trained model for 
[PhraAphaiManee (พระอภัยมณี)](https://vajirayana.org/%e0%b8%9e%e0%b8%a3%e0%b8%b0%e0%b8%ad%e0%b8%a0%e0%b8%b1%e0%b8%a2%e0%b8%a1%e0%b8%93%e0%b8%b5) dataset.

You can access the model at [this link](https://huggingface.co/Kongfha/PhraAphaiManee-LM), To try out the deployed model, please visit [this link](https://kongfha-phraaphaimanee-generation.hf.space).

## Calling the model from Hugging Face
``` py
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Kongfha/PhraAphaiManee-LM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generate = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer)

input_sentence = "๏ สัมผัสเส้นขอบฟ้าชลาลัย"
generated_text = generate(input_sentence,
                          max_length=140,
                          top_k=25,
                          temperature=1)
# generation parameters can be varied 

print(f"Input: {text}")
print(f"Output:\n {generated_text[0]['generated_text']}")
```
