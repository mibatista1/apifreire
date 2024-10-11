import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
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
from gensim import corpora, models
from rake_nltk import Rake
import os

nltk.download('punkt')
nltk.download('stopwords')

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
            "Entidades": self.analyze_entities,
            "Relações": self.analyze_word_relationships,
            "Frases": self.analyze_sentences,
            "Palavras-chave": self.analyze_keywords,
            "Sentimento": self.analyze_sentiment,
            "Freq. Palavras": self.analyze_word_frequency,
            "Coocorrência": self.analyze_word_cooccurrence,
            "N-gramas": self.analyze_ngrams,
            "Tópicos": self.analyze_topic_modeling,
            "Classificação": self.analyze_text_classification
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

    def analyze_similarity(self, text_data):
        def plot_similarity():
            texts = self.split_text(text_data)
            if len(texts) > 1:
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(texts)
                similarity_matrix = (X * X.T).A
                plt.figure(figsize=(10, 5))
                plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title("Matriz de Similaridade TF-IDF")
                plt.show()
            else:
                messagebox.showwarning("Aviso", "Texto muito curto para análise de similaridade.")
        
        self.root.after(0, plot_similarity)

    def analyze_text_complexity(self, text_data):
        def plot_complexity():
            texts = self.split_text(text_data)
            readability_scores = [textstat.flesch_reading_ease(text) for text in texts]
            avg_score = sum(readability_scores) / len(readability_scores)
            messagebox.showinfo("Complexidade Textual", f"Pontuação Média de Facilidade de Leitura: {avg_score:.2f}")
        
        self.root.after(0, plot_complexity)

    def create_wordcloud(self, text_data):
        def plot_wordcloud():
            sample_text = " ".join(text_data.split()[:10000])
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(sample_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Nuvem de Palavras")
            plt.show()
        
        self.root.after(0, plot_wordcloud)

    def analyze_entities(self, text_data):
        def show_entities():
            # Implementação fictícia para análise de entidades
            entities = "Entidades encontradas: Nenhuma entidade real implementada."
            messagebox.showinfo("Entidades", entities)
        
        self.root.after(0, show_entities)

    def analyze_word_relationships(self, text_data):
        def plot_word_relationships():
            words = text_data.split()
            word_pairs = list(ngrams(words, 2))
            word_pair_freq = Counter(word_pairs)
            plt.figure(figsize=(12, 6))
            nx_graph = nx.Graph()
            for pair, freq in word_pair_freq.items():
                nx_graph.add_edge(pair[0], pair[1], weight=freq)
            pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)
            nx.draw(nx_graph, pos, with_labels=True, node_size=5000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
            plt.title("Relações entre Palavras")
            plt.show()
        
        self.root.after(0, plot_word_relationships)

    def analyze_sentences(self, text_data):
        def plot_sentences():
            blob = TextBlob(text_data)
            sentences = blob.sentences
            sentence_lengths = [len(sentence.words) for sentence in sentences]
            plt.hist(sentence_lengths, bins=range(max(sentence_lengths) + 1), edgecolor='black')
            plt.xlabel("Número de Palavras")
            plt.ylabel("Número de Frases")
            plt.title("Distribuição de Comprimento das Frases")
            plt.show()
        
        self.root.after(0, plot_sentences)

    def analyze_keywords(self, text_data):
        def show_keywords():
            r = Rake()  # Certifique-se de que a importação está correta
            r.extract_keywords_from_text(text_data)
            keywords = r.get_ranked_phrases_with_scores()
            keywords_text = '\n'.join([f"{score}: {phrase}" for score, phrase in keywords])
            messagebox.showinfo("Palavras-chave", f"Palavras-chave extraídas:\n{keywords_text}")
        
        self.root.after(0, show_keywords)

    def analyze_sentiment(self, text_data):
        def show_sentiment():
            blob = TextBlob(text_data)
            sentiment = blob.sentiment
            messagebox.showinfo("Sentimento", f"Polaridade: {sentiment.polarity:.2f}\nSubjetividade: {sentiment.subjectivity:.2f}")
        
        self.root.after(0, show_sentiment)

    def analyze_word_frequency(self, text_data):
        def plot_word_frequency():
            words = text_data.split()
            word_freq = Counter(words)
            most_common_words = word_freq.most_common(20)
            words, freqs = zip(*most_common_words)
            plt.figure(figsize=(12, 6))
            plt.bar(words, freqs)
            plt.xticks(rotation=45)
            plt.xlabel("Palavras")
            plt.ylabel("Frequência")
            plt.title("Frequência das Palavras")
            plt.show()
        
        self.root.after(0, plot_word_frequency)

    def analyze_word_cooccurrence(self, text_data):
        def plot_word_cooccurrence():
            words = text_data.split()
            word_pairs = list(ngrams(words, 2))
            word_pair_freq = Counter(word_pairs)
            plt.figure(figsize=(12, 6))
            plt.bar([' '.join(pair) for pair in word_pair_freq.keys()], word_pair_freq.values())
            plt.xticks(rotation=45)
            plt.xlabel("Palavras")
            plt.ylabel("Coocorrência")
            plt.title("Coocorrência de Palavras")
            plt.show()
        
        self.root.after(0, plot_word_cooccurrence)

    def analyze_ngrams(self, text_data):
        def plot_ngrams():
            words = text_data.split()
            bigrams = list(ngrams(words, 2))
            trigrams = list(ngrams(words, 3))
            bigram_freq = Counter(bigrams)
            trigram_freq = Counter(trigrams)
            plt.figure(figsize=(12, 6))
            plt.bar([' '.join(pair) for pair in bigram_freq.keys()], bigram_freq.values(), label='Bigrams')
            plt.bar([' '.join(pair) for pair in trigram_freq.keys()], trigram_freq.values(), label='Trigrams', alpha=0.7)
            plt.xticks(rotation=45)
            plt.xlabel("N-gramas")
            plt.ylabel("Frequência")
            plt.title("Distribuição de N-gramas")
            plt.legend()
            plt.show()
        
        self.root.after(0, plot_ngrams)

    def analyze_topic_modeling(self, text_data):
        def show_topic_modeling():
            texts = self.split_text(text_data)
            tokenized_texts = [text.split() for text in texts]
            dictionary = corpora.Dictionary(tokenized_texts)
            corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
            lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
            topics = lda_model.print_topics(num_words=5)
            topics_text = '\n'.join([f"Topic {i}: {topic}" for i, topic in enumerate(topics)])
            messagebox.showinfo("Modelagem de Tópicos", f"Tópicos encontrados:\n{topics_text}")
        
        self.root.after(0, show_topic_modeling)

    def analyze_text_classification(self, text_data):
        def show_classification():
            prediction = self.model.predict([text_data])
            messagebox.showinfo("Classificação", f"Categoria prevista: {prediction[0]}")
        
        self.root.after(0, show_classification)

    def log_error(self, message):
        with open('error_log.txt', 'a') as log_file:
            log_file.write(message + '\n')

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()


