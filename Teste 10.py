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
            "Entidades": self.analyze_entities,
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
        text = re.sub(r'\W+', ' ', text).lower()
        tokens = word_tokenize(text, language='portuguese')
        stop_words = set(stopwords.words('portuguese'))
        stop_words.update({'a', 'o', 'as', 'os', 'um', 'uns', 'uma', 'umas', 'de', 'da', 'do', 'dos', 'das', 'em', 'para', 'com', 'por', 'entre', 'sobre', 'entre', 'mas', 'ou', 'nem', 'se', 'que', 'este', 'essa', 'aquele', 'aquela', 'me', 'te', 'nos', 'vos', 'lhe', 'lhes'})
        tokens = [word for word in tokens if word not in stop_words]
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
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', contour_color='black', contour_width=1).generate(text_data)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Nuvem de Palavras")

        self.root.after(0, lambda: self.plot_and_show(plot_wordcloud, "Nuvem de Palavras"))

    def analyze_entities(self, text_data):
        def show_entities():
            text_blob = TextBlob(text_data)
            entities = text_blob.noun_phrases
            unique_entities = set(entities)
            messagebox.showinfo("Entidades Nomeadas", f"Entidades Nomeadas Identificadas:\n" + '\n'.join(unique_entities))

        self.root.after(0, show_entities)

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

            # Criar uma lista de stopwords em português
            stop_words = set(stopwords.words('portuguese'))

            # Adicionar stopwords específicas
            stop_words.update({'a', 'o', 'as', 'os', 'um', 'uns', 'uma', 'umas', 'de', 'da', 'do', 'dos', 'das', 'em', 'para', 'com', 'por', 'entre', 'sobre', 'entre', 'mas', 'ou', 'nem', 'se', 'que', 'este', 'essa', 'aquele', 'aquela', 'me', 'te', 'nos', 'vos', 'lhe', 'lhes'})

            vectorizer = TfidfVectorizer(stop_words=stop_words)  # Usar a lista de stopwords
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

    def log_error(self, message):
        """Grava mensagens de erro em um arquivo de log."""
        with open('error_log.txt', 'a') as file:
            file.write(message + '\n')

# Função para executar a aplicação
def main():
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
