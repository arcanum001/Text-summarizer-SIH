from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download and save T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer.save_pretrained("/app/t5-small")
model.save_pretrained("/app/t5-small")