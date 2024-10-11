
import urllib.request
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk import bigrams, trigrams
import networkx as nx

# Baixar recursos do NLTK (se necessário)
nltk.download('punkt')
nltk.download('stopwords')

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

# Verificando se há dados disponíveis
if df.empty:
    print("Nenhum dado disponível para análise.")
else:
    # Contagem de Publicações
    num_publications = len(df)
    print(f'\nNúmero de Publicações: {num_publications}')

    # Estatísticas Textuais Básicas
    if 'Resumo' in df.columns:  # Supondo que a coluna de texto seja 'Resumo'
        text_data = df['Resumo'].dropna()
        text_data = ' '.join(text_data)
        
        # Frequência de Palavras
        stop_words = set(stopwords.words('portuguese'))
        tokens = word_tokenize(text_data.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        word_freq = pd.Series(filtered_tokens).value_counts()
        
        # Gráfico de Frequência de Palavras
        plt.figure(figsize=(10, 6))
        word_freq.head(20).plot(kind='bar')
        plt.title('Frequência das 20 Palavras Mais Comuns')
        plt.xlabel('Palavra')
        plt.ylabel('Frequência')
        plt.show()

        # Nuvem de Palavras
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nuvem de Palavras')
        plt.show()

        # Análise de Co-ocorrência de Palavras
        bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
        bigram_matrix = bigram_vectorizer.fit_transform(text_data.split())
        bigram_freq = pd.DataFrame(bigram_matrix.toarray(), columns=bigram_vectorizer.get_feature_names_out())
        bigram_freq = bigram_freq.sum().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        bigram_freq.head(20).plot(kind='bar')
        plt.title('Frequência das 20 Bigramas Mais Comuns')
        plt.xlabel('Bigramas')
        plt.ylabel('Frequência')
        plt.show()
        
        # Classificação Hierárquica Descendente (CHD)
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(text_data.split())
        linkage_matrix = linkage(X.toarray(), method='ward')
        
        plt.figure(figsize=(10, 6))
        dendrogram(linkage_matrix, labels=vectorizer.get_feature_names_out())
        plt.title('Classificação Hierárquica Descendente')
        plt.xlabel('Palavras')
        plt.ylabel('Distância')
        plt.show()

        # Análise de Similitude
        bigram_model = list(bigrams(filtered_tokens))
        bigram_freq = pd.Series(bigram_model).value_counts()
        
        G = nx.Graph()
        for bigram, freq in bigram_freq.items():
            G.add_edge(bigram[0], bigram[1], weight=freq)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
        plt.title('Rede de Co-ocorrências de Bigramas')
        plt.show()
    
    else:
        print('Coluna "Resumo" não encontrada.')

    # Pesquisa de Especificidades
    if 'Categoria' in df.columns:
        categories = df['Categoria'].dropna().unique()
        for category in categories:
            category_df = df[df['Categoria'] == category]
            print(f'\nAnálise da Categoria: {category}')
            print(category_df.describe(include='all'))
            
    else:
        print('Coluna "Categoria" não encontrada.')

    # Análise Estatística
    print("\nAnálise Descritiva:")
    print(df.describe(include='all'))
