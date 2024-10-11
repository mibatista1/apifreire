import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.util import ngrams
from collections import Counter
from rake_nltk import Rake
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
from textblob import TextBlob
import fitz  # PyMuPDF
import rispy
import nltk
from joblib import load
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer

# Baixar recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Texto")

        # Definir caminho para o modelo e vetorizador
        self.model_path = 'C:/Users/NOT-ACER/Downloads/Tese/text_classifier.joblib'
        self.vectorizer_path = 'C:/Users/NOT-ACER/Downloads/Tese/vectorizer.joblib'     
        
        # Carregar modelo treinado e vetorizador
        self.model = self.load_model()
        self.vectorizer = self.load_vectorizer()

        # Adicionar imagem e título
        self.image_path = 'images.jfif'
        self.image_label = None
        self.load_image()

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

        # Adicionar botões para análises
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

    def load_model(self):
        if os.path.exists(self.model_path):
            return load(self.model_path)
        else:
            messagebox.showerror("Erro", f"Modelo não encontrado no caminho: {self.model_path}")
            return None

    def load_vectorizer(self):
        if os.path.exists(self.vectorizer_path):
            return load(self.vectorizer_path)
        else:
            messagebox.showerror("Erro", f"Vetorizador não encontrado no caminho: {self.vectorizer_path}")
            return TfidfVectorizer()  # Retorne um vetor padrão se não encontrado

    def load_image(self):
        try:
            img = Image.open(self.image_path)
            img = img.resize((400, 400), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label = tk.Label(self.root, image=img_tk, bg='lightblue')
            self.image_label.image = img_tk
            self.image_label.pack(pady=20)
        except Exception as e:
            print(f"Erro ao carregar imagem: {e}")

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
        """Realiza tokenização, remoção de stopwords e lematização."""
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
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def create_wordcloud(self, text_data):
        """Gera e exibe uma nuvem de palavras a partir do texto fornecido."""
        tokens = self.preprocess_text(text_data)
        text = ' '.join(tokens)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        def plot_wordcloud(ax):
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')

        self.plot_and_show(plot_wordcloud, "Nuvem de Palavras")

    def analyze_word_relationships(self, text_data):
        """Exibe as relações entre palavras usando um gráfico de linhas."""
        tokens = self.preprocess_text(text_data)
        bigrams = list(ngrams(tokens, 2))
        bigram_freq = Counter(bigrams)
        common_bigrams = bigram_freq.most_common(10)
        bigram_labels, counts = zip(*common_bigrams)
        bigram_labels = [' '.join(bigram) for bigram in bigram_labels]
        if len(bigram_labels) > 5:
            bigram_labels = bigram_labels[:5]
            counts = counts[:5]

        def plot_word_relationships_line(ax):
            ax.plot(bigram_labels, counts, marker='o')
            ax.set_xlabel('Relações de Palavras')
            ax.set_ylabel('Frequência')

        self.plot_and_show(plot_word_relationships_line, "Relações de Palavras")

    def analyze_keywords(self, text_data):
        """Exibe as principais palavras-chave extraídas do texto."""
        rake = Rake()
        rake.extract_keywords_from_text(text_data)
        keywords = rake.get_ranked_phrases_with_scores()

        top_keywords = keywords[:10]
        if top_keywords:
            keyword_output = "\n".join([f"{score:.2f}: {phrase}" for score, phrase in top_keywords])
            self.show_results_window("Palavras-chave", keyword_output)
        else:
            messagebox.showwarning("Aviso", "Nenhuma palavra-chave encontrada.")

    def analyze_sentences(self, text_data):
        """Exibe o resumo das principais frases do texto."""
        parser = PlaintextParser.from_string(text_data, Tokenizer("portuguese"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 3)  # Resumir em 3 frases

        summary_text = "\n".join([str(sentence) for sentence in summary])
        if summary_text:
            self.show_results_window("Resumo", summary_text)
        else:
            messagebox.showwarning("Aviso", "Não foi possível gerar um resumo.")

    def analyze_word_frequency(self, text_data):
        """Gera um gráfico de linhas das palavras mais frequentes."""
        tokens = self.preprocess_text(text_data)
        freq_dist = Counter(tokens)
        common_words = freq_dist.most_common(10)
        words, counts = zip(*common_words)
        if len(words) > 5:
            words = words[:5]
            counts = counts[:5]

        def plot_word_frequency_line(ax):
            ax.plot(words, counts, marker='o')
            ax.set_xlabel('Palavras')
            ax.set_ylabel('Frequência')
            ax.set_title('Frequência das Palavras')

        self.plot_and_show(plot_word_frequency_line, "Frequência das Palavras")

    def analyze_word_cooccurrence(self, text_data):
        """Gera um gráfico de linhas da coocorrência de palavras."""
        tokens = self.preprocess_text(text_data)
        bigrams = list(ngrams(tokens, 2))
        bigram_freq = Counter(bigrams)
        common_bigrams = bigram_freq.most_common(10)
        bigram_labels, counts = zip(*common_bigrams)
        bigram_labels = [' '.join(bigram) for bigram in bigram_labels]
        if len(bigram_labels) > 5:
            bigram_labels = bigram_labels[:5]
            counts = counts[:5]

        def plot_word_cooccurrence_line(ax):
            ax.plot(bigram_labels, counts, marker='o')
            ax.set_xlabel('Coocorrência de Palavras')
            ax.set_ylabel('Frequência')

        self.plot_and_show(plot_word_cooccurrence_line, "Coocorrência de Palavras")

    def analyze_ngrams(self, text_data):
        """Gera e exibe um gráfico de linhas dos n-gramas mais frequentes."""
        tokens = self.preprocess_text(text_data)
        n = 2  # Tamanho do n-grama
        ngram_freq = Counter(ngrams(tokens, n))
        common_ngrams = ngram_freq.most_common(10)
        ngram_labels, counts = zip(*common_ngrams)
        ngram_labels = [' '.join(ngram) for ngram in ngram_labels]
        if len(ngram_labels) > 5:
            ngram_labels = ngram_labels[:5]
            counts = counts[:5]

        def plot_ngrams_line(ax):
            ax.plot(ngram_labels, counts, marker='o')
            ax.set_xlabel('N-gramas')
            ax.set_ylabel('Frequência')

        self.plot_and_show(plot_ngrams_line, "N-gramas")

    def analyze_text_classification(self, text_data):
        """Realiza a classificação de texto usando o modelo carregado e exibe os resultados."""
        if self.model and self.vectorizer:
            tokens = self.vectorizer.transform([text_data])
            prediction = self.model.predict(tokens)
            class_labels = self.model.classes_
            predicted_class = class_labels[prediction[0]]
            messagebox.showinfo("Classificação", f"O texto foi classificado como: {predicted_class}")
        else:
            messagebox.showwarning("Aviso", "Modelo ou vetorizador não carregados corretamente.")

    def analyze_content_analysis(self, text_data):
        """Exibe uma análise de conteúdo resumida do texto."""
        # Gerar o resumo
        parser = PlaintextParser.from_string(text_data, Tokenizer("portuguese"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 3)
        summary_text = "\n".join([str(sentence) for sentence in summary])
        
        # Gerar a nuvem de palavras
        tokens = self.preprocess_text(text_data)
        text = ' '.join(tokens)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        def plot_wordcloud(ax):
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')

        # Exibir resumo e nuvem de palavras
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.imshow(wordcloud, interpolation='bilinear')
        ax1.axis('off')
        ax1.set_title('Nuvem de Palavras')

        ax2.text(0.1, 0.9, summary_text, fontsize=12, ha='left', va='top', wrap=True)
        ax2.set_title('Resumo do Texto')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def show_results_window(self, title, text):
        """Exibe uma nova janela com os resultados fornecidos."""
        results_window = tk.Toplevel(self.root)
        results_window.title(title)
        results_text = tk.Text(results_window, wrap='word', height=20, width=80)
        results_text.insert('1.0', text)
        results_text.config(state='disabled')
        results_text.pack(padx=10, pady=10)

    def log_error(self, message):
        """Log de erros no console."""
        print(message)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()


