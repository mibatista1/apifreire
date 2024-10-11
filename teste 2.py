import urllib.request
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lista de URLs das APIs para cada ano de 2000 a 2004
urls = [
    'https://dadosabertos.capes.gov.br/api/3/action/datastore_search?resource_id=6cb47471-ffc1-44e0-b60b-324993dc0e8f&limit=5',  # 2000
    'https://dadosabertos.capes.gov.br/api/3/action/datastore_search?resource_id=b6ab4de3-6d51-4073-8452-5c29eaf48e94&limit=5',  # 2001
    'https://dadosabertos.capes.gov.br/api/3/action/datastore_search?resource_id=28bcef08-3743-4400-a215-57cb10c5b72e&limit=5',  # 2002
    'https://dadosabertos.capes.gov.br/api/3/action/datastore_search?resource_id=424182de-c430-452c-ae7f-1793545b07bf&limit=5',  # 2003
    'https://dadosabertos.capes.gov.br/api/3/action/datastore_search?resource_id=7926fa44-b9ac-41fa-a832-9b3a92feedea&limit=5'   # 2004
]

# Função para obter dados de uma URL
def fetch_data(url):
    try:
        response = urllib.request.urlopen(url)
        return json.loads(response.read())
    except Exception as e:
        print(f'Error fetching data from {url}: {e}')
        return {}

# Obter e combinar dados de todas as APIs
all_records = []

for url in urls:
    print(f'Fetching data from {url}')
    data = fetch_data(url)
    
    # Exibir dados brutos para diagnóstico
    print(json.dumps(data, indent=2))  
    
    if 'result' in data and 'records' in data['result']:
        records = data['result']['records']
        if records:
            all_records.extend(records)
        else:
            print(f'No records found in data from {url}')
    else:
        print(f'Unexpected structure in data from {url}')

# Criar DataFrame com todos os dados combinados
df = pd.DataFrame(all_records)

# Exibindo as primeiras linhas e colunas do DataFrame para verificar os dados
print("\nDados iniciais:")
print(df.head())
print("\nColunas disponíveis:")
print(df.columns)

# Se não houver dados, parar a execução
if df.empty:
    print("Nenhum dado disponível para análise.")
else:
    # Contagem de Publicações
    num_publications = len(df)
    print(f'\nNúmero de Publicações: {num_publications}')

    # Análise e Visualização com base nas colunas disponíveis
    # Distribuição por Ano
    if 'AnoBase' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='AnoBase', data=df)
        plt.title('Distribuição de Publicações por Ano')
        plt.xticks(rotation=45)
        plt.xlabel('Ano')
        plt.ylabel('Número de Publicações')
        plt.show()
    else:
        print('Coluna "AnoBase" não encontrada.')

    # Autores mais frequentes
    if 'Autor' in df.columns:
        author_counts = df['Autor'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=author_counts.index, y=author_counts.values)
        plt.title('Autores mais Frequentes')
        plt.xticks(rotation=45)
        plt.xlabel('Autor')
        plt.ylabel('Número de Publicações')
        plt.show()
    else:
        print('Coluna "Autor" não encontrada.')

    # Palavras-chave mais comuns
    if 'PalavrasChave' in df.columns:
        keywords_series = df['PalavrasChave'].str.split(',').explode()
        keyword_counts = keywords_series.value_counts()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=keyword_counts.index, y=keyword_counts.values)
        plt.title('Palavras-chave mais Comuns')
        plt.xticks(rotation=45)
        plt.xlabel('Palavra-chave')
        plt.ylabel('Número de Ocorrências')
        plt.show()
    else:
        print('Coluna "PalavrasChave" não encontrada.')

    # Análise Estatística
    print("\nAnálise Descritiva:")
    print(df.describe(include='all'))

   
