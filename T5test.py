from transformers import T5ForConditionalGeneration, T5Tokenizer

# Cargar el modelo y el tokenizador
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Definir la función de generación de texto
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generar texto
prompt = '¿Cuál es el propósito de la vida?'
generated_text = generate_text(prompt)
print(generated_text)
