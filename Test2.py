from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# Cargar el modelo BERT y el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
modelo = BertModel.from_pretrained('bert-base-uncased')

# Definir la función de generación de respuesta
def generar_respuesta(pregunta, max_length=50):
    # Tokenizar la pregunta
    inputs = tokenizer(pregunta, return_tensors='pt')
    
    # Ejecutar el modelo BERT
    outputs = modelo(**inputs)
    
    # Inicializar la respuesta
    respuesta = []
    
    # Generar la respuesta
    for i in range(max_length):
        # Obtener el token más probable
        token = torch.argmax(outputs.last_hidden_state[:, -1, :])
        
        # Decodificar el token
        token_decodificado = tokenizer.decode(token.item(), skip_special_tokens=True)
        
        # Agregar el token decodificado a la respuesta
        respuesta.append(token_decodificado)
        
        # Actualizar los inputs
        nuevo_input = tokenizer(token_decodificado, return_tensors='pt')
        inputs = nuevo_input
        
        # Ejecutar el modelo BERT
        outputs = modelo(**inputs)
    
    # Unir la respuesta en un solo string
    respuesta = ' '.join(respuesta)
    
    return respuesta

# Probar la función de generación de respuesta
pregunta = "¿Cuál es el significado de la vida?"
respuesta = generar_respuesta(pregunta)
print(respuesta)
