import tkinter as tk
from tkinter import filedialog
import fitz  # PyMuPDF para leitura de PDFs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk import bigrams

nltk.download('punkt')
nltk.download('stopwords')

# Função para processar o arquivo
def process_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf")])
    if file_path:
        if file_path.endswith('.pdf'):
            text_data = extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                text_data = file.read()
        analyze_text(text_data)

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Função para análise qualitativa do texto com categorias de Bardin
def analyze_text(text_data):
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(text_data.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    # Define categorias e palavras-chave associadas
    categories = {
        'Tecnologia': ['tecnologia', 'inovação', 'digital', 'computador', 'software'],
        'Educação': ['educação', 'ensino', 'aprendizagem', 'escola', 'universidade'],
        'Pesquisa': ['pesquisa', 'estudo', 'análise', 'dados', 'metodologia']
    }

    # Conta a ocorrência de palavras-chave em cada categoria
    category_counts = {category: 0 for category in categories}
    
    for word in filtered_tokens:
        for category, keywords in categories.items():
            if word in keywords:
                category_counts[category] += 1
    
    # Criação de DataFrame para visualização
    category_df = pd.DataFrame(list(category_counts.items()), columns=['Categoria', 'Frequência'])

    # Gráfico de Frequência das Categorias
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Categoria', y='Frequência', data=category_df, palette='Blues_d')
    plt.title('Frequência das Categorias')
    plt.xlabel('Categoria')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()

    # Nuvem de Palavras
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuvem de Palavras')
    plt.tight_layout()
    plt.show()

    # Análise de Bigramas
    bigram_model = list(bigrams(filtered_tokens))
    bigram_freq = pd.Series(bigram_model).value_counts()

    plt.figure(figsize=(10, 6))
    bigram_freq.head(10).plot(kind='bar', color='steelblue')
    plt.title('Frequência dos 10 Bigramas Mais Comuns')
    plt.xlabel('Bigramas')
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()

# Configuração da Interface Gráfica
root = tk.Tk()
root.title("Lantec - Análise Qualitativa de Texto")

# Configuração do tamanho da janela e cores
root.geometry('800x600')
root.configure(bg='lightblue')

frame = tk.Frame(root, bg='lightblue')
frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

label = tk.Label(frame, text="Lantec - Análise Qualitativa de Texto", font=('Arial', 16, 'bold'), bg='lightblue')
label.pack(pady=10)

upload_button = tk.Button(frame, text="Anexar Arquivo de Texto ou PDF", command=process_file, bg='deepskyblue', fg='white', font=('Arial', 12, 'bold'))
upload_button.pack(pady=20)

root.mainloop()
