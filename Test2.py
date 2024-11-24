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
        
        # Agregar el token a la respuesta
        respuesta.append(token.item())
        
        # Actualizar los inputs
        nuevo_input = torch.tensor([[token.item()]])
        nuevo_input = nuevo_input.expand(1, inputs['input_ids'].size(1))
        inputs['input_ids'] = torch.cat((inputs['input_ids'], nuevo_input), dim=1)
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones(1, 1)), dim=1)
        
        # Ejecutar el modelo BERT
        outputs = modelo(**inputs)
    
    # Decodificar la respuesta
    respuesta = tokenizer.decode(respuesta, skip_special_tokens=True)
    
    return respuesta

# Probar la función de generación de respuesta
pregunta = "¿Cuál es el significado de la vida?"
respuesta = generar_respuesta(pregunta)
print(respuesta)
