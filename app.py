from flask import Flask, request, render_template_string
import sqlite3
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import os

app = Flask(__name__)

# Configurações do modelo
MAX_WORDS = 1000
MAX_LENGTH = 20
EMBEDDING_DIM = 50
class IAModel:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        self.model = None
        self.response_map = {}
        self.num_classes = 0
        
    def create_model(self, vocab_size, num_classes):
        vocab_size = max(1, vocab_size)
        model = Sequential([
            Embedding(input_dim=MAX_WORDS, 
                     output_dim=EMBEDDING_DIM, 
                     input_length=MAX_LENGTH,
                     mask_zero=True),
            LSTM(100),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')  # Número fixo de classes
        ])
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def prepare_data(self, perguntas, respostas):
        # Preparar perguntas
        self.tokenizer.fit_on_texts(perguntas)
        X = self.tokenizer.texts_to_sequences(perguntas)
        X = pad_sequences(X, maxlen=MAX_LENGTH, padding='post', truncating='post')
        X = np.minimum(X, MAX_WORDS - 1)
        
        # Criar mapeamento único para respostas
        unique_responses = sorted(list(set(respostas)))  # Ordenar para consistência
        self.response_map = {resp: idx for idx, resp in enumerate(unique_responses)}
        self.num_classes = len(unique_responses)
        
        # Criar one-hot encoding para respostas
        y = np.zeros((len(respostas), self.num_classes))
        for i, resp in enumerate(respostas):
            y[i, self.response_map[resp]] = 1
        
        return X, y
    
    def train(self, X, y, epochs=50):
        try:
            if len(X) < 2:
                return "Necessário pelo menos 2 exemplos para treinar"
            
            if self.num_classes < 1:
                return "Nenhuma resposta única encontrada"
            
            # Recriar modelo para garantir dimensões corretas
            self.model = self.create_model(MAX_WORDS, self.num_classes)
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Treinar modelo
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=min(32, len(X)),
                callbacks=[early_stopping],
                verbose=1
            )
            return "Modelo treinado com sucesso!"
        except Exception as e:
            return f"Erro durante o treinamento: {str(e)}"
    
    def predict(self, texto):
        try:
            if not self.model or not self.response_map:
                return "Modelo não está pronto para previsões."

            sequence = self.tokenizer.texts_to_sequences([texto])
            sequence = np.minimum(sequence, MAX_WORDS - 1)
            padded = pad_sequences(sequence, maxlen=MAX_LENGTH, 
                                 padding='post', truncating='post')
            
            pred = self.model.predict(padded, verbose=0)
            predicted_index = np.argmax(pred[0])
            
            # Converter índice previsto de volta para resposta
            for resposta, idx in self.response_map.items():
                if idx == predicted_index:
                    return resposta
            
            return "Não sei responder a essa pergunta."
        except Exception as e:
            return f"Erro na predição: {str(e)}"

    def save(self):
        try:
            if self.model:
                self.model.save('modelo_ia.h5')
                with open('model_data.pkl', 'wb') as f:
                    pickle.dump({
                        'tokenizer': self.tokenizer,
                        'response_map': self.response_map,
                        'num_classes': self.num_classes
                    }, f)
                return True
            return False
        except Exception as e:
            print(f"Erro ao salvar modelo: {str(e)}")
            return False

    def load(self):
        try:
            if os.path.exists('modelo_ia.h5') and os.path.exists('model_data.pkl'):
                self.model = load_model('modelo_ia.h5')
                with open('model_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.tokenizer = data['tokenizer']
                    self.response_map = data['response_map']
                    self.num_classes = data['num_classes']
                return True
            return False
        except Exception as e:
            print(f"Erro ao carregar modelo: {str(e)}")
            return False
# Instância global do modelo
ia_model = IAModel()

# Funções do banco de dados
def init_db():
    conn = sqlite3.connect('memoria.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pergunta TEXT NOT NULL,
            resposta TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def adicionar_memoria(pergunta, resposta):
    conn = sqlite3.connect('memoria.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO memoria (pergunta, resposta) VALUES (?, ?)',
                  (pergunta, resposta))
    conn.commit()
    conn.close()

def get_all_data():
    conn = sqlite3.connect('memoria.db')
    cursor = conn.cursor()
    cursor.execute('SELECT pergunta, resposta FROM memoria')
    data = cursor.fetchall()
    conn.close()
    return data

def treinar_modelo():
    try:
        data = get_all_data()
        if len(data) < 2:
            return "Necessário pelo menos 2 exemplos para treinar o modelo"
        
        perguntas, respostas = zip(*data)
        X, y = ia_model.prepare_data(perguntas, respostas)
        ia_model.train(X, y)
        ia_model.save()
        return "Modelo treinado com sucesso!"
    except Exception as e:
        return f"Erro durante o treinamento: {str(e)}"

# Estilo CSS
style = """
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f2f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            box-sizing: border-box;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        .response {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #4CAF50;
        }
        .nav {
            margin-bottom: 20px;
        }
        .nav a {
            color: #4CAF50;
            text-decoration: none;
            margin-right: 20px;
        }
    </style>
"""

# Templates HTML
index_html = """
<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8">
    <title>Sistema de IA</title>
    """ + style + """
  </head>
  <body>
    <div class="container">
        <h1>Sistema de IA</h1>
        <div class="nav">
            <a href="{{ url_for('inserir') }}">Inserir Dados</a>
            <a href="{{ url_for('consultar') }}">Consultar</a>
            <a href="{{ url_for('treinar') }}">Treinar Modelo</a>
        </div>
        <p>Bem-vindo ao Sistema de IA com Aprendizado Contínuo</p>
    </div>
  </body>
</html>
"""

inserir_html = """
<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8">
    <title>Inserir Dados</title>
    """ + style + """
  </head>
  <body>
    <div class="container">
        <div class="nav">
            <a href="{{ url_for('index') }}">Início</a>
            <a href="{{ url_for('consultar') }}">Consultar</a>
            <a href="{{ url_for('treinar') }}">Treinar Modelo</a>
        </div>
        <h1>Inserir Novos Dados</h1>
        <form method="post">
            <label>Pergunta:</label><br>
            <input type="text" name="pergunta" required><br>
            <label>Resposta:</label><br>
            <input type="text" name="resposta" required><br>
            <button type="submit">Adicionar</button>
        </form>
        {% if mensagem %}
        <div class="response">
            <p>{{ mensagem }}</p>
        </div>
        {% endif %}
    </div>
  </body>
</html>
"""

consultar_html = """
<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8">
    <title>Consultar</title>
    """ + style + """
  </head>
  <body>
    <div class="container">
        <div class="nav">
            <a href="{{ url_for('index') }}">Início</a>
            <a href="{{ url_for('inserir') }}">Inserir Dados</a>
            <a href="{{ url_for('treinar') }}">Treinar Modelo</a>
        </div>
        <h1>Consultar IA</h1>
        <form method="post">
            <label>Sua pergunta:</label><br>
            <input type="text" name="pergunta" required><br>
            <button type="submit">Consultar</button>
        </form>
        {% if resposta %}
        <div class="response">
            <h3>Resposta:</h3>
            <p>{{ resposta }}</p>
        </div>
        {% endif %}
    </div>
  </body>
</html>
"""

treinar_html = """
<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8">
    <title>Treinar Modelo</title>
    """ + style + """
  </head>
  <body>
    <div class="container">
        <div class="nav">
            <a href="{{ url_for('index') }}">Início</a>
            <a href="{{ url_for('inserir') }}">Inserir Dados</a>
            <a href="{{ url_for('consultar') }}">Consultar</a>
        </div>
        <h1>Treinar Modelo de IA</h1>
        <form method="post">
            <button type="submit" name="treinar">Iniciar Treinamento</button>
        </form>
        {% if mensagem %}
        <div class="response">
            <p>{{ mensagem }}</p>
        </div>
        {% endif %}
    </div>
  </body>
</html>
"""

# Rotas Flask
@app.route('/')
def index():
    return render_template_string(index_html)

@app.route('/inserir', methods=['GET', 'POST'])
def inserir():
    mensagem = ''
    if request.method == 'POST':
        pergunta = request.form['pergunta']
        resposta = request.form['resposta']
        adicionar_memoria(pergunta, resposta)
        mensagem = 'Dados adicionados com sucesso!'
    return render_template_string(inserir_html, mensagem=mensagem)

@app.route('/consultar', methods=['GET', 'POST'])
def consultar():
    resposta = ''
    if request.method == 'POST':
        pergunta = request.form['pergunta']
        if ia_model.model is not None:
            resposta = ia_model.predict(pergunta)
        else:
            resposta = "Modelo ainda não foi treinado!"
    return render_template_string(consultar_html, resposta=resposta)

@app.route('/treinar', methods=['GET', 'POST'])
def treinar():
    mensagem = ''
    if request.method == 'POST':
        mensagem = treinar_modelo()
    return render_template_string(treinar_html, mensagem=mensagem)

if __name__ == '__main__':
    init_db()
    ia_model.load()  # Tenta carregar um modelo existente
    app.run(debug=True)