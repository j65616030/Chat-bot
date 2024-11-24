import torch
from transformers import BertTokenizer, BertModel

# Cargar el modelo BERT y el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
modelo = BertModel.from_pretrained('bert-base-uncased')

# Definir la función principal
def respuesta(pregunta):
    # Tokenizar la pregunta
    inputs = tokenizer(pregunta, return_tensors='pt')
    
    # Ejecutar el modelo BERT
    outputs = modelo(**inputs)
    
    # Obtener la respuesta
    respuesta = outputs.last_hidden_state[:, 0, :]
    
    return respuesta

# Ejecutar la función principal
pregunta = "¿Cuál es el significado de la vida?"
respuesta = respuesta(pregunta)
print(respuesta)
