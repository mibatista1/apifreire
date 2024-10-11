import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import textstat
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
from nltk.util import ngrams
from joblib import load
import fitz  # PyMuPDF
import rispy
import nltk
from textblob import TextBlob
from rake_nltk import Rake
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import requests

# Baixar recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Texto")

        # Carregar modelo treinado
        self.model_path = 'C:/Users/NOT-ACER/Downloads/Tese/text_classifier.joblib'
        if os.path.exists(self.model_path):
            self.model = load(self.model_path)
        else:
            raise FileNotFoundError(f"Modelo não encontrado no caminho: {self.model_path}")

        # Adicionar imagem e título
        self.image_path = r'C:\Users\NOT-ACER\Downloads\Tese\images.jfif'
        img = Image.open(self.image_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        self.image_label = tk.Label(root, image=img_tk, bg='lightblue')
        self.image_label.image = img_tk
        self.image_label.pack(pady=20)

        self.label = tk.Label(root, text="Lantec - Análise Quantitativa e Qualitativa", font=('Arial', 16), bg='lightblue')
        self.label.pack(pady=10)

        # Frame para carregar arquivo
        self.file_frame = tk.Frame(root)
        self.file_frame.pack(pady=10)

        self.load_button = tk.Button(self.file_frame, text="Carregar Arquivo", command=self.load_file)
        self.load_button.pack()

        # Frame para botões de análises
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(pady=10, side=tk.TOP)

        self.text_data = None
        self.processing_label = None

        # Adicionar botões para análises em um menu horizontal
        self.analysis_buttons = {
            "Similitude": self.analyze_similarity,
            "Complexidade": self.analyze_text_complexity,
            "Wordcloud": self.create_wordcloud,
            "Relações": self.analyze_word_relationships,
            "Frases": self.analyze_sentences,
            "Palavras-chave": self.analyze_keywords,
            "Sentimento": self.analyze_sentiment,
            "Freq. Palavras": self.analyze_word_frequency,
            "Coocorrência": self.analyze_word_cooccurrence,
            "N-gramas": self.analyze_ngrams,
            "Classificação": self.analyze_text_classification,
            "Conteúdo": self.analyze_content_analysis
        }

        # Configurar o menu horizontal
        self.button_widgets = {}
        for idx, (analysis_name, analysis_function) in enumerate(self.analysis_buttons.items()):
            button = tk.Button(self.buttons_frame, text=analysis_name, command=lambda func=analysis_function: self.run_analysis(func))
            button.grid(row=0, column=idx, padx=5, pady=5)
            self.button_widgets[analysis_name] = button

        # Adicionar campos para buscar artigos por DOI e palavras-chave
        self.search_frame = tk.Frame(root)
        self.search_frame.pack(pady=10, side=tk.BOTTOM)

        self.doi_label = tk.Label(self.search_frame, text="Digite o DOI do artigo:")
        self.doi_label.grid(row=0, column=0, padx=5)

        self.doi_entry = tk.Entry(self.search_frame)
        self.doi_entry.grid(row=0, column=1, padx=5)

        self.search_doi_button = tk.Button(self.search_frame, text="Buscar por DOI", command=self.search_article_by_doi)
        self.search_doi_button.grid(row=0, column=2, padx=5)

        self.keywords_label = tk.Label(self.search_frame, text="Digite palavras-chave:")
        self.keywords_label.grid(row=1, column=0, padx=5)

        self.keywords_entry = tk.Entry(self.search_frame)
        self.keywords_entry.grid(row=1, column=1, padx=5)

        self.search_keywords_button = tk.Button(self.search_frame, text="Buscar por Palavras-chave", command=self.search_articles_by_keywords)
        self.search_keywords_button.grid(row=1, column=2, padx=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("PDF Files", "*.pdf"), ("RIS Files", "*.ris")])
        if file_path:
            try:
                if file_path.endswith('.pdf'):
                    self.text_data = self.extract_text_from_pdf(file_path)
                elif file_path.endswith('.ris'):
                    self.text_data = self.extract_text_from_ris(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        self.text_data = file.read()
                if self.text_data:
                    print(f"Arquivo carregado. Tamanho do texto: {len(self.text_data)} caracteres.")
                    messagebox.showinfo("Sucesso", "Arquivo carregado com sucesso!")
                else:
                    messagebox.showwarning("Aviso", "O arquivo está vazio ou não pôde ser lido.")
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível carregar o arquivo: {e}")

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text()
        except Exception as e:
            print(f"Erro ao ler PDF: {e}")
        return text

    def extract_text_from_ris(self, ris_path):
        text = ""
        try:
            with open(ris_path, 'r', encoding='utf-8') as file:
                ris_data = rispy.load(file)
            text = '\n'.join([str(entry) for entry in ris_data])
        except Exception as e:
            print(f"Erro ao ler RIS: {e}")
        return text

    def split_text(self, text, max_length=10000):
        """Divide o texto em blocos menores com comprimento máximo especificado."""
        return [text[i:i+max_length] for i in range(0, len(text), max_length)]

    def run_analysis(self, analysis_function):
        if self.text_data:
            if not self.processing_label:
                self.processing_label = tk.Label(self.root, text="Processando...", bg='lightblue')
                self.processing_label.pack(pady=10)
            # Agende a execução da análise para a thread principal
            self.root.after(0, self._run_analysis, analysis_function)
        else:
            messagebox.showwarning("Aviso", "Nenhum arquivo carregado!")

    def _run_analysis(self, analysis_function):
        try:
            # Executa a função de análise
            analysis_function(self.text_data)
        except Exception as e:
            self.log_error(f"Erro durante a análise: {e}")
            messagebox.showerror("Erro", f"Erro durante a análise: {e}")
        finally:
            if self.processing_label:
                self.processing_label.destroy()
                self.processing_label = None

    def preprocess_text(self, text):
        """Realiza tokenização, remoção de stopwords e lematização em português."""
        # Remover caracteres especiais e deixar o texto em minúsculas
        text = re.sub(r'\W+', ' ', text).lower()

        # Tokenizar o texto
        tokens = word_tokenize(text)

        # Definir stopwords em português
        stop_words = set(stopwords.words('portuguese'))

        # Adicionar stopwords personalizadas (se necessário)
        stop_words.update({'a', 'o', 'e', 'de', 'do', 'da', 'em', 'para', 'que', 'com', 
                           'é', 'os', 'as', 'um', 'uma', 'no', 'na', 'se', 'por', 'dos', 
                           'das', 'ao', 'às', 'foi', 'era', 'ser', 'há', 'não', 'ele', 
                           'ela', 'eles', 'elas', 'me', 'nos', 'nós', 'vocês', 'você'})

        # Filtrar tokens que não estão nas stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # Lematizar os tokens (aqui usamos a lematização do NLTK para inglês, pois o suporte para português é limitado)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens

    def plot_and_show(self, plot_function, title):
        """Gera e exibe um gráfico usando a função fornecida."""
        plt.figure(figsize=(10, 6))
        plot_function()
        plt.tight_layout()
        plt.show()

    def analyze_similarity(self, text_data):
        def plot_similarity():
            texts = self.split_text(text_data)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            similarity_matrix = (X * X.T).A
            plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.xlabel("Blocos de Texto")
            plt.ylabel("Blocos de Texto")
            plt.title("Matriz de Similaridade")

        self.plot_and_show(plot_similarity, "Matriz de Similaridade")

    def analyze_text_complexity(self, text_data):
        def plot_complexity():
            texts = self.split_text(text_data)
            complexities = [textstat.flesch_reading_ease(text) for text in texts]
            plt.hist(complexities, bins=10, color='blue', edgecolor='black')
            plt.xlabel("Índice de Facilidade de Leitura")
            plt.ylabel("Frequência")
            plt.title("Distribuição da Complexidade do Texto")

        self.plot_and_show(plot_complexity, "Complexidade do Texto")

    def create_wordcloud(self, text_data):
        def plot_wordcloud():
            text = ' '.join(self.preprocess_text(text_data))
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')

        self.plot_and_show(plot_wordcloud, "Nuvem de Palavras")

    def analyze_word_relationships(self, text_data):
        def plot_relationships():
            tokens = self.preprocess_text(text_data)
            bigrams = list(ngrams(tokens, 2))
            bigram_freq = Counter(bigrams)
            G = nx.Graph()
            G.add_edges_from(bigram_freq.keys())
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_weight='bold')
            plt.title("Relações Entre Palavras")

        self.plot_and_show(plot_relationships, "Relações Entre Palavras")

    def analyze_sentences(self, text_data):
        def plot_sentences():
            sentences = text_data.split('.')
            sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence]
            plt.hist(sentence_lengths, bins=20, color='green', edgecolor='black')
            plt.xlabel("Número de Palavras por Frase")
            plt.ylabel("Frequência")
            plt.title("Comprimento das Frases")

        self.plot_and_show(plot_sentences, "Comprimento das Frases")

    def analyze_keywords(self, text_data):
        def plot_keywords():
            rake = Rake()
            rake.extract_keywords_from_text(text_data)
            keywords = rake.get_ranked_phrases_with_scores()
            keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
            labels, values = zip(*keywords[:10])
            plt.barh(labels, values, color='purple')
            plt.xlabel("Pontuação")
            plt.title("Palavras-chave")

        self.plot_and_show(plot_keywords, "Palavras-chave")

    def analyze_sentiment(self, text_data):
        def plot_sentiment():
            blob = TextBlob(text_data)
            sentiment = blob.sentiment.polarity
            plt.bar(['Sentimento'], [sentiment], color='orange')
            plt.ylabel("Polaridade")
            plt.title("Análise de Sentimento")

        self.plot_and_show(plot_sentiment, "Sentimento do Texto")

    def analyze_word_frequency(self, text_data):
        def plot_word_frequency():
            tokens = self.preprocess_text(text_data)
            freq_dist = nltk.FreqDist(tokens)
            most_common = freq_dist.most_common(10)
            words, counts = zip(*most_common)
            plt.bar(words, counts, color='blue')
            plt.xlabel("Palavras")
            plt.ylabel("Frequência")
            plt.title("Frequência das Palavras")

        self.plot_and_show(plot_word_frequency, "Frequência das Palavras")

    def analyze_word_cooccurrence(self, text_data):
        def plot_cooccurrence():
            tokens = self.preprocess_text(text_data)
            bigrams = list(ngrams(tokens, 2))
            bigram_freq = Counter(bigrams)
            most_common_bigrams = bigram_freq.most_common(10)
            bigrams, counts = zip(*most_common_bigrams)
            bigrams = [' '.join(bigram) for bigram in bigrams]
            plt.barh(bigrams, counts, color='teal')
            plt.xlabel("Frequência")
            plt.title("Coocorrência de Palavras")

        self.plot_and_show(plot_cooccurrence, "Coocorrência de Palavras")

    def analyze_ngrams(self, text_data):
        def plot_ngrams():
            tokens = self.preprocess_text(text_data)
            n = 3  # N-gram de 3 palavras
            ngrams_list = list(ngrams(tokens, n))
            ngram_freq = Counter(ngrams_list)
            most_common_ngrams = ngram_freq.most_common(10)
            ngrams, counts = zip(*most_common_ngrams)
            ngrams = [' '.join(ngram) for ngram in ngrams]
            plt.barh(ngrams, counts, color='cyan')
            plt.xlabel("Frequência")
            plt.title("N-gramas")

        self.plot_and_show(plot_ngrams, "N-gramas")

    def analyze_text_classification(self, text_data):
        def plot_classification():
            prediction = self.model.predict([text_data])[0]
            plt.bar(['Classificação'], [prediction], color='red')
            plt.ylabel("Classe")
            plt.title("Classificação do Texto")

        self.plot_and_show(plot_classification, "Classificação do Texto")

    def analyze_content_analysis(self, text_data):
        messagebox.showinfo("Análise de Conteúdo", "Análise de conteúdo não implementada no momento.")

    def fetch_article_info(self, doi):
        """Busca informações do artigo usando o DOI da API do CrossRef."""
        url = f"https://api.crossref.org/works/{doi}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Verifica se a solicitação foi bem-sucedida
            data = response.json()
            article_info = data['message']
            title = article_info.get('title', ['Título não disponível'])[0]
            authors = ', '.join([author['family'] for author in article_info.get('author', [])])
            published_date = article_info.get('created', {}).get('date-time', 'Data não disponível')
            journal = article_info.get('container-title', ['Revista não disponível'])[0]

            info_message = (f"Título: {title}\n"
                            f"Autores: {authors}\n"
                            f"Data de Publicação: {published_date}\n"
                            f"Revista: {journal}")

            messagebox.showinfo("Informações do Artigo", info_message)
        except requests.RequestException as e:
            self.log_error(f"Erro ao buscar informações do artigo: {e}")
            messagebox.showerror("Erro", f"Erro ao buscar informações do artigo: {e}")

    def search_article_by_doi(self):
        doi = self.doi_entry.get().strip()
        if doi:
            self.fetch_article_info(doi)
        else:
            messagebox.showwarning("Aviso", "Por favor, insira um DOI válido.")

    def search_articles_by_keywords(self):
        keywords = self.keywords_entry.get().strip()
        if keywords:
            self.fetch_articles_by_keywords(keywords)
        else:
            messagebox.showwarning("Aviso", "Por favor, insira palavras-chave válidas.")

    def fetch_articles_by_keywords(self, keywords):
        """Busca artigos usando palavras-chave na API do CrossRef."""
        url = f"https://api.crossref.org/works?query={keywords}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Verifica se a solicitação foi bem-sucedida
            data = response.json()
            articles = data['message'].get('items', [])

            if not articles:
                messagebox.showinfo("Resultados", "Nenhum artigo encontrado para as palavras-chave fornecidas.")
                return

            articles_info = []
            for article in articles:
                title = article.get('title', ['Título não disponível'])[0]
                authors = ', '.join([author['family'] for author in article.get('author', [])])
                published_date = article.get('created', {}).get('date-time', 'Data não disponível')
                journal = article.get('container-title', ['Revista não disponível'])[0]
                articles_info.append(f"Título: {title}\nAutores: {authors}\nData de Publicação: {published_date}\nRevista: {journal}\n")

            info_message = "\n\n".join(articles_info)
            messagebox.showinfo("Resultados da Busca", info_message)
        except requests.RequestException as e:
            self.log_error(f"Erro ao buscar artigos: {e}")
            messagebox.showerror("Erro", f"Erro ao buscar artigos: {e}")

    def log_error(self, error_message):
        with open('error_log.txt', 'a') as log_file:
            log_file.write(f"{error_message}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()
