

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
import seaborn as sns
import numpy as np

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
            messagebox.showerror("Erro", f"Modelo não encontrado no caminho: {self.model_path}")
            self.model = None

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
        self.buttons_frame.pack(pady=10)

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
        self.search_frame.pack(pady=10)

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
        """Gera e exibe um gráfico utilizando o matplotlib."""
        plt.figure(figsize=(14, 7))
        plot_function()
        plt.title(title, fontsize=16)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)  # Ajuste manual
        plt.show()

    def analyze_similarity(self, text_data):
        def plot_similarity():
            texts = self.split_text(text_data)
            if len(texts) > 1:
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(texts)
                similarity_matrix = (X * X.T).A
                plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
                plt.colorbar()
                plt.title("Matriz de Similaridade TF-IDF")
            else:
                messagebox.showwarning("Aviso", "Texto muito curto para análise de similaridade.")

        self.root.after(0, lambda: self.plot_and_show(plot_similarity, "Matriz de Similaridade TF-IDF"))

    def analyze_text_complexity(self, text_data):
        def plot_complexity():
            texts = self.split_text(text_data)
            readability_scores = [textstat.flesch_reading_ease(text) for text in texts]
            plt.plot(readability_scores, marker='o')
            plt.title("Facilidade de Leitura do Texto")
            plt.xlabel("Blocos de Texto")
            plt.ylabel("Pontuação de Facilidade de Leitura")

        self.root.after(0, lambda: self.plot_and_show(plot_complexity, "Facilidade de Leitura"))

    def create_wordcloud(self, text_data):
        def plot_wordcloud():
            # Usar tokens processados (sem stopwords e lematizados)
            tokens = self.preprocess_text(text_data)
            text_clean = ' '.join(tokens)  # Converter lista de tokens de volta para string
            
            # Gerar a nuvem de palavras
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                  colormap='viridis', contour_color='black', 
                                  contour_width=1).generate(text_clean)
            
            # Exibir a nuvem de palavras
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Nuvem de Palavras")

        self.root.after(0, lambda: self.plot_and_show(plot_wordcloud, "Nuvem de Palavras"))

    def analyze_word_relationships(self, text_data):
        def plot_word_relationships():
            tokens = self.preprocess_text(text_data)
            word_pairs = list(ngrams(tokens, 2))
            word_pair_freq = Counter(word_pairs)
            if word_pair_freq:
                words = [f"{pair[0]} - {pair[1]}" for pair in word_pair_freq.keys()][:20]
                frequencies = [word_pair_freq[pair] for pair in word_pair_freq.keys()][:20]
                plt.bar(words, frequencies, color='skyblue', edgecolor='black')
                plt.xlabel('Relações de Palavras')
                plt.ylabel('Frequência')
                plt.xticks(rotation=45, ha='right')
            else:
                messagebox.showwarning("Aviso", "Nenhuma relação de palavras encontrada.")

        self.root.after(0, lambda: self.plot_and_show(plot_word_relationships, "Distribuição de Frequência das Relações entre Palavras"))

    def analyze_sentences(self, text_data):
        def show_sentence_analysis():
            sentences = text_data.split('. ')
            avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
            messagebox.showinfo("Análise de Frases", f"Comprimento Médio das Frases: {avg_sentence_length:.2f} palavras")

        self.root.after(0, show_sentence_analysis)

    def analyze_keywords(self, text_data):
        def plot_keywords():
            rake = Rake(language='portuguese')
            rake.extract_keywords_from_text(text_data)
            keywords = rake.get_ranked_phrases()[:20]
            plt.barh(range(len(keywords)), range(len(keywords)), color='lightcoral')
            plt.yticks(range(len(keywords)), keywords)
            plt.xlabel('Relevância')

        self.root.after(0, lambda: self.plot_and_show(plot_keywords, "Palavras-Chave"))

    def analyze_sentiment(self, text_data):
        def show_sentiment_analysis():
            analysis = TextBlob(text_data)
            sentiment = analysis.sentiment.polarity
            sentiment_description = "Positivo" if sentiment > 0 else "Negativo" if sentiment < 0 else "Neutro"
            messagebox.showinfo("Análise de Sentimento", f"Sentimento: {sentiment_description} (Polaridade: {sentiment:.2f})")

        self.root.after(0, show_sentiment_analysis)

    def analyze_word_frequency(self, text_data):
        def plot_word_frequency():
            tokens = self.preprocess_text(text_data)
            word_freq = Counter(tokens)
            if word_freq:
                words = [word for word, _ in word_freq.most_common(20)]
                frequencies = [freq for _, freq in word_freq.most_common(20)]
                plt.bar(words, frequencies, color='lightblue', edgecolor='black')
                plt.xlabel('Palavras')
                plt.ylabel('Frequência')
                plt.xticks(rotation=45, ha='right')
            else:
                messagebox.showwarning("Aviso", "Nenhuma palavra encontrada.")

        self.root.after(0, lambda: self.plot_and_show(plot_word_frequency, "Distribuição de Frequência das Palavras"))

    def analyze_word_cooccurrence(self, text_data):
        def plot_word_cooccurrence():
            tokens = self.preprocess_text(text_data)
            word_pairs = list(ngrams(tokens, 2))
            word_pair_freq = Counter(word_pairs)
            nx_graph = nx.Graph()
            for pair, freq in word_pair_freq.items():
                nx_graph.add_edge(pair[0], pair[1], weight=freq)
            pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)
            nx.draw(nx_graph, pos, with_labels=True, node_size=3000, node_color='skyblue',
                    font_size=12, font_weight='bold', edge_color='gray',
                    width=[d['weight']*0.1 for (u, v, d) in nx_graph.edges(data=True)])
            plt.title("Coocorrência de Palavras")

        self.root.after(0, lambda: self.plot_and_show(plot_word_cooccurrence, "Coocorrência de Palavras"))

    def analyze_ngrams(self, text_data):
        def plot_ngrams():
            tokens = self.preprocess_text(text_data)
            bigrams = list(ngrams(tokens, 2))
            bigram_freq = Counter(bigrams)
            if bigram_freq:
                bigram_keys = [' '.join(bigram) for bigram in bigram_freq.keys()][:20]
                bigram_values = [freq for _, freq in bigram_freq.most_common(20)]
                plt.barh(bigram_keys, bigram_values, color='salmon', edgecolor='black')
                plt.xlabel('Frequência')
                plt.ylabel('N-gramas')
                plt.xticks(rotation=45, ha='right')
            else:
                messagebox.showwarning("Aviso", "Nenhum N-grama encontrado.")

        self.root.after(0, lambda: self.plot_and_show(plot_ngrams, "Distribuição de Frequência dos N-gramas"))

    def analyze_text_classification(self, text_data):
        def show_classification():
            prediction = self.model.predict([text_data])
            messagebox.showinfo("Classificação de Texto", f"Classificação do Texto: {prediction[0]}")

        self.root.after(0, show_classification)

    def analyze_content_analysis(self, text_data):
        def show_content_analysis():
            # Análise de temas principais usando LDA
            texts = self.split_text(text_data)
            if len(texts) < 2:
                messagebox.showwarning("Aviso", "Texto muito curto para análise de temas.")
                return

            vectorizer = TfidfVectorizer()  # Usar padrão de stopwords
            X = vectorizer.fit_transform(texts)
            lda = LatentDirichletAllocation(n_components=5, random_state=0)
            lda.fit(X)

            # Mostrar tópicos principais
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                topics.append(f"Tópico {topic_idx + 1}: " + ", ".join(topic_words))

            messagebox.showinfo("Análise de Conteúdo", "\n".join(topics))

        self.root.after(0, show_content_analysis)

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
