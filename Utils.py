import subprocess
import requests

def respuesta(pregunta):
    # Lógica para responder a la pregunta
    with open('respuestas.txt', 'r') as f:
        for linea in f:
            p, r = linea.strip().split(':')
            if p == pregunta:
                if r.startswith('comando:'):
                    comando = r[8:]
                    salida = subprocess.check_output(comando, shell=True)
                    return salida.decode('utf-8')
                elif r.startswith('api:'):
                    api = r[4:]
                    respuesta_api = requests.get(api)
                    return respuesta_api.text
                else:
                    return r
    return "Lo siento, no entiendo tu pregunta."
