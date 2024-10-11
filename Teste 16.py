
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
from rake_nltk import Rake
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import seaborn as sns
import numpy as np
from textblob import TextBlob

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
            "Wordcloud": self.create_wordcloud,
            "Relações": self.analyze_word_relationships,
            "Frases": self.analyze_sentences,
            "Palavras-chave": self.analyze_keywords,
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

    def preprocess_text(self, text):
        """Realiza tokenização, remoção de stopwords e lematização em português."""
        text = re.sub(r'\W+', ' ', text).lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('portuguese'))
        stop_words.update({'a', 'o', 'e', 'de', 'do', 'da', 'em', 'para', 'que', 'com', 
                           'é', 'os', 'as', 'um', 'uma', 'no', 'na', 'se', 'por', 'dos', 
                           'das', 'ao', 'às', 'foi', 'era', 'ser', 'há', 'não', 'ele', 
                           'ela', 'eles', 'elas', 'me', 'nos', 'nós', 'vocês', 'você'})
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def run_analysis(self, analysis_function):
        if self.text_data:
            if not self.processing_label:
                self.processing_label = tk.Label(self.root, text="Processando...", bg='lightblue')
                self.processing_label.pack(pady=10)
            self.root.after(0, self._run_analysis, analysis_function)
        else:
            messagebox.showwarning("Aviso", "Nenhum arquivo carregado!")

    def _run_analysis(self, analysis_function):
        try:
            analysis_function(self.text_data)
        except Exception as e:
            self.log_error(f"Erro durante a análise: {e}")
            messagebox.showerror("Erro", f"Erro durante a análise: {e}")
        finally:
            if self.processing_label:
                self.processing_label.destroy()
                self.processing_label = None

    def plot_and_show(self, plot_function, title):
        """Cria e exibe um gráfico com base na função de plotagem fornecida."""
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_function(ax)
        ax.set_title(title)
        plt.show()

    def create_wordcloud(self, text_data):
        """Gera e exibe uma nuvem de palavras a partir do texto fornecido."""
        text_data = ' '.join(self.preprocess_text(text_data))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)

        def plot_wordcloud(ax):
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')

        self.plot_and_show(plot_wordcloud, "Nuvem de Palavras")

    def analyze_word_relationships(self, text_data):
        """Gera um gráfico de pizza das palavras mais frequentes."""
        tokens = self.preprocess_text(text_data)
        freq_dist = Counter(tokens)
        common_words = freq_dist.most_common(10)
        words, counts = zip(*common_words)

        def plot_word_relationships(ax):
            ax.pie(counts, labels=words, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        self.plot_and_show(plot_word_relationships, "Palavras Mais Frequentes")

    def analyze_keywords(self, text_data):
        """Gera um gráfico de pizza das palavras-chave mais frequentes."""
        tokens = self.preprocess_text(text_data)
        r = Rake()
        r.extract_keywords_from_text(' '.join(tokens))
        keywords = r.get_ranked_phrases_with_scores()
        
        if keywords:
            scores, keywords = zip(*keywords)
        else:
            scores, keywords = [], []

        def plot_keywords(ax):
            ax.pie(scores, labels=keywords, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')

        self.plot_and_show(plot_keywords, "Palavras-chave")

    def analyze_sentences(self, text_data):
        """Gera um gráfico de pizza do comprimento médio das frases."""
        sentences = text_data.split('.')
        sentence_lengths = [len(sentence.split()) for sentence in sentences if len(sentence.split()) > 0]
        average_length = np.mean(sentence_lengths) if sentence_lengths else 0

        def plot_sentence_lengths(ax):
            ax.pie([average_length, 100 - average_length], labels=['Média das Frases', 'Outros'], autopct='%1.1f%%', startangle=140)
            ax.axis('equal')

        self.plot_and_show(plot_sentence_lengths, "Comprimento Médio das Frases")

    def analyze_word_frequency(self, text_data):
        """Gera um gráfico de pizza da frequência das palavras mais comuns."""
        tokens = self.preprocess_text(text_data)
        freq_dist = Counter(tokens)
        most_common_words = freq_dist.most_common(10)
        words, counts = zip(*most_common_words)

        def plot_word_frequency(ax):
            ax.pie(counts, labels=words, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')

        self.plot_and_show(plot_word_frequency, "Frequência das Palavras")

    def analyze_word_cooccurrence(self, text_data):
        """Gera um gráfico de pizza da coocorrência de bigramas."""
        tokens = self.preprocess_text(text_data)
        bigrams_list = list(ngrams(tokens, 2))
        bigram_freq = Counter(bigrams_list)
        bigram_freq = dict(sorted(bigram_freq.items(), key=lambda item: item[1], reverse=True))

        top_bigrams = dict(list(bigram_freq.items())[:10])
        labels, counts = zip(*top_bigrams.items())

        def plot_word_cooccurrence(ax):
            ax.pie(counts, labels=[' '.join(bigram) for bigram in labels], autopct='%1.1f%%', startangle=140)
            ax.axis('equal')

        self.plot_and_show(plot_word_cooccurrence, "Coocorrência de Palavras")

    def analyze_ngrams(self, text_data):
        """Gera um gráfico de pizza dos trigrams mais frequentes."""
        tokens = self.preprocess_text(text_data)
        trigrams_list = list(ngrams(tokens, 3))
        trigram_freq = Counter(trigrams_list)
        trigram_freq = dict(sorted(trigram_freq.items(), key=lambda item: item[1], reverse=True))

        top_trigrams = dict(list(trigram_freq.items())[:10])
        labels, counts = zip(*top_trigrams.items())

        def plot_ngrams(ax):
            ax.pie(counts, labels=[' '.join(trigram) for trigram in labels], autopct='%1.1f%%', startangle=140)
            ax.axis('equal')

        self.plot_and_show(plot_ngrams, "N-gramas")

    def analyze_text_classification(self, text_data):
        """Classifica o texto e exibe o resultado como gráfico de pizza."""
        if self.model:
            prediction = self.model.predict([text_data])
            sentiment = 'positivo' if prediction[0] == 1 else 'negativo'
        else:
            sentiment = 'Modelo não carregado'

        def plot_classification(ax):
            ax.pie([1], labels=[sentiment], colors=['blue' if sentiment == 'positivo' else 'red'], startangle=140)
            ax.axis('equal')

        self.plot_and_show(plot_classification, f"Classificação do Texto: {sentiment}")

    def analyze_content_analysis(self, text_data):
        """Exibe uma análise do conteúdo como gráfico de pizza."""
        readability = textstat.flesch_reading_ease(text_data)
        num_sentences = textstat.sentence_count(text_data)
        num_words = textstat.lexicon_count(text_data, True)
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

        analysis_data = {
            'Legibilidade': readability,
            'Número de Sentenças': num_sentences,
            'Número de Palavras': num_words,
            'Comprimento Médio das Frases': avg_sentence_length
        }

        def plot_content_analysis(ax):
            ax.pie(list(analysis_data.values()), labels=list(analysis_data.keys()), autopct='%1.1f%%', startangle=140)
            ax.axis('equal')

        self.plot_and_show(plot_content_analysis, "Análise do Conteúdo")

    def log_error(self, message):
        print(message)
        # Optionally log to a file or other medium

# Inicia a aplicação
if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()
