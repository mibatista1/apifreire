import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import textstat
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
from rake_nltk import Rake
import fitz  # PyMuPDF
import nltk
import rispy
import threading

nltk.download('punkt')

class TextAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Texto")
        
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
            "Palavras-chave": self.analyze_keywords
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
                # Verifica e imprime o tamanho do texto carregado
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
            if analysis_function in [self.create_wordcloud, self.analyze_similarity, self.analyze_word_relationships]:
                # Essas funções criam gráficos e devem rodar na thread principal
                threading.Thread(target=self._run_analysis_in_main_thread, args=(analysis_function,)).start()
            else:
                threading.Thread(target=self._run_analysis, args=(analysis_function,)).start()
        else:
            messagebox.showwarning("Aviso", "Nenhum arquivo carregado!")

    def _run_analysis(self, analysis_function):
        try:
            # Executa a função de análise
            analysis_function(self.text_data)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante a análise: {e}")
        finally:
            if self.processing_label:
                self.processing_label.destroy()
                self.processing_label = None

    def _run_analysis_in_main_thread(self, analysis_function):
        self.root.after(0, lambda: self._run_analysis(analysis_function))

    def analyze_similarity(self, text_data):
        try:
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
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise de similaridade: {e}")

    def analyze_text_complexity(self, text_data):
        try:
            texts = self.split_text(text_data)
            readability_scores = [textstat.flesch_reading_ease(text) for text in texts]
            avg_score = sum(readability_scores) / len(readability_scores)
            messagebox.showinfo("Complexidade Textual", f"Pontuação Média de Facilidade de Leitura: {avg_score:.2f}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise de complexidade: {e}")

    def create_wordcloud(self, text_data):
        try:
            sample_text = " ".join(text_data.split()[:10000])
            wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(sample_text)
            self.root.after(0, self._show_wordcloud, wordcloud)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na criação da nuvem de palavras: {e}")

    def _show_wordcloud(self, wordcloud):
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Nuvem de Palavras")
        plt.show()

    def analyze_entities(self, text_data):
        try:
            # Implementação fictícia para análise de entidades
            entities = "Entidades encontradas: Nenhuma"
            messagebox.showinfo("Análise de Entidades", entities)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise de entidades: {e}")

    def analyze_word_relationships(self, text_data):
        try:
            words = text_data.split()
            word_freq = Counter(words)
            graph = nx.Graph()
            for word in word_freq:
                graph.add_node(word)
            for i, word1 in enumerate(words):
                for word2 in words[i + 1:]:
                    if word_freq[word1] > 1 and word_freq[word2] > 1:
                        graph.add_edge(word1, word2)
            self.root.after(0, self._show_word_relationships, graph)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise de relações entre palavras: {e}")

    def _show_word_relationships(self, graph):
        plt.figure(figsize=(10, 8))
        nx.draw(graph, with_labels=True)
        plt.title("Relações entre Palavras")
        plt.show()

    def analyze_sentences(self, text_data):
        try:
            sentences = nltk.sent_tokenize(text_data)
            num_sentences = len(sentences)
            messagebox.showinfo("Análise de Frases", f"Número de Frases: {num_sentences}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise de frases: {e}")

    def analyze_keywords(self, text_data):
        try:
            r = Rake()
            r.extract_keywords_from_text(text_data)
            keywords = r.get_ranked_phrases()
            messagebox.showinfo("Palavras-chave", "Palavras-chave:\n" + "\n".join(keywords))
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na análise de palavras-chave: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()
