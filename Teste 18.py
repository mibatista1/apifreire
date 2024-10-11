import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib  # Adicione esta linha para importar joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.util import ngrams
from collections import Counter
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
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer  # Usando LSA para resumo

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
            "Frequência": self.analyze_word_frequency,
            "N-gramas": self.analyze_ngrams,
            "Classificação": self.analyze_text_classification,
            "Análise de Sentimento": self.analyze_sentiment,
            "Resumo": self.summarize_text
        }

        # Configurar o menu horizontal
        self.button_widgets = {}
        for idx, (analysis_name, analysis_function) in enumerate(self.analysis_buttons.items()):
            button = tk.Button(self.buttons_frame, text=analysis_name, command=lambda func=analysis_function: self.run_analysis(func))
            button.grid(row=0, column=idx, padx=5, pady=5)
            self.button_widgets[analysis_name] = button

    def load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)  # Alterado para usar joblib.load
        else:
            messagebox.showerror("Erro", f"Modelo não encontrado no caminho: {self.model_path}")
            return None

    def load_vectorizer(self):
        if os.path.exists(self.vectorizer_path):
            return joblib.load(self.vectorizer_path)  # Alterado para usar joblib.load
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

    def analyze_word_frequency(self, text_data):
        """Calcula e exibe a frequência das palavras."""
        tokens = self.preprocess_text(text_data)
        word_freq = Counter(tokens)

        def plot_word_frequency(ax):
            ax.bar(*zip(*word_freq.most_common(10)))
            ax.set_ylabel("Frequência")
            ax.set_xlabel("Palavras")

        self.plot_and_show(plot_word_frequency, "Frequência de Palavras")

    def analyze_ngrams(self, text_data):
        """Exibe n-gramas do texto."""
        tokens = self.preprocess_text(text_data)
        n = 3  # Para trigramas
        ngrams_list = list(ngrams(tokens, n))
        ngram_freq = Counter(ngrams_list)

        def plot_ngrams(ax):
            ax.bar(*zip(*ngram_freq.most_common(10)))
            ax.set_ylabel("Frequência")
            ax.set_xlabel(f"N-gramas (n={n})")

        self.plot_and_show(plot_ngrams, f"Frequência de N-gramas (n={n})")

    def analyze_text_classification(self, text_data):
        """Classifica o texto usando o modelo treinado."""
        vectorized_text = self.vectorizer.transform([text_data])
        prediction = self.model.predict(vectorized_text)
        messagebox.showinfo("Classificação", f"A classificação do texto é: {prediction[0]}")

    def analyze_sentiment(self, text_data):
        """Analisa o sentimento do texto."""
        analysis = TextBlob(text_data)
        polarity = analysis.sentiment.polarity
        sentiment = "Positivo" if polarity > 0 else "Negativo" if polarity < 0 else "Neutro"
        messagebox.showinfo("Análise de Sentimento", f"O sentimento do texto é: {sentiment}")

    def summarize_text(self, text_data):
        """Gera um resumo do texto usando LSA."""
        try:
            parser = PlaintextParser.from_string(text_data, Tokenizer("portuguese"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 2)  # Resumir para 2 frases
            summary_text = ' '.join(str(sentence) for sentence in summary)
            messagebox.showinfo("Resumo", summary_text if summary_text else "Não foi possível gerar um resumo.")
        except Exception as e:
            messagebox.showerror("Erro", "Erro ao gerar resumo: " + str(e))

    def log_error(self, message):
        """Registra erros."""
        with open("error_log.txt", "a") as log_file:
            log_file.write(message + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()

