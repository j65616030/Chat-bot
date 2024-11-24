from transformers import T5Tokenizer, T5ForConditionalGeneration

# Cargar el modelo T5 y el tokenizador
tokenizer = T5Tokenizer.from_pretrained('t5-base')
modelo = T5ForConditionalGeneration.from_pretrained('t5-base')

# Definir la función de generación de respuesta
def generar_respuesta(pregunta, max_length=50):
    # Tokenizar la pregunta
    inputs = tokenizer(pregunta, return_tensors='pt')
    
    # Generar la respuesta
    outputs = modelo.generate(inputs['input_ids'], max_length=max_length)
    
    # Decodificar la respuesta
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return respuesta

# Probar la función de generación de respuesta
pregunta = "¿Cuál es el significado de la vida?"
respuesta = generar_respuesta(pregunta)
print(respuesta)
