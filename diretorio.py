import os
from joblib import dump

# Caminho para o diretório onde você deseja salvar o arquivo
directory = 'C:/Users/NOT-ACER/Downloads/Tese/Treinamento1/'
file_path = os.path.join(directory, 'text_classifier.joblib')

# Crie o diretório se não existir
os.makedirs(directory, exist_ok=True)

# Suponha que você já treinou seu modelo e ele está na variável `model`
# Salve o modelo
dump(model, file_path)
print(f"Modelo salvo em {file_path}")
