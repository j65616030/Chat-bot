from flask import Flask, request, render_template
from utils import respuesta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot():
    pregunta = request.form['pregunta']
    respuesta = respuesta(pregunta)
    return respuesta

if __name__ == '__main__':
    app.run(debug=True)
