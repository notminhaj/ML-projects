from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt', padding=True, return_attention_mask=True)
attention_mask = tokenizer(input_text, return_tensors='pt', padding=True).attention_mask

output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1,
                        temperature=0.000001, top_k=50, top_p=0.7, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)