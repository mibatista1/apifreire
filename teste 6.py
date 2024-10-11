

import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk  # Biblioteca para manipulação de imagens
import fitz  # PyMuPDF para leitura de PDFs
import rispy  # Biblioteca para manipular arquivos RIS
from langdetect import detect, DetectorFactory


DetectorFactory.seed = 0

# Função para processar o arquivo
def process_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("RIS files", "*.ris")])
    if file_path:
        if file_path.endswith('.pdf'):
            text_data = extract_text_from_pdf(file_path)
        elif file_path.endswith('.ris'):
            text_data = extract_text_from_ris(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                text_data = file.read()
        detect_language(text_data)

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Função para extrair texto de um arquivo RIS
def extract_text_from_ris(ris_path):
    text = ""
    with open(ris_path, 'r', encoding='utf-8') as file:
        entries = rispy.load(file)
        for entry in entries:
            for field in entry.values():
                text += ' '.join(field) + ' '
    return text

# Função para detectar o idioma do texto
def detect_language(text_data):
    try:
        lang = detect(text_data)
        print(f"Idioma detectado: {lang}")
    except Exception as e:
        print(f"Erro na detecção do idioma: {e}")

# Configuração da Interface Gráfica
root = tk.Tk()
root.title("Lantec.IA")

# Configuração do tamanho da janela e cores
root.geometry('800x600')
root.configure(bg='lightblue')

frame = tk.Frame(root, bg='lightblue')
frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

image_path = r'C:\Users\NOT-ACER\Downloads\Tese\images.jfif'
img = Image.open(image_path)
img = img.resize((400, 400), Image.LANCZOS)  # Ajusta o tamanho da imagem se necessário
img_tk = ImageTk.PhotoImage(img)

image_label = Label(frame, image=img_tk, bg='lightblue')
image_label.image = img_tk
image_label.pack(pady=20)

label = tk.Label(frame, text="Lantec - Análise Quantitativa e Qualitativa", font=('Arial', 16, 'bold'), bg='lightblue')
label.pack(pady=10)

upload_button = tk.Button(frame, text="Anexar Arquivo de Texto, PDF ou RIS", command=process_file, bg='deepskyblue', fg='white', font=('Arial', 12, 'bold'))
upload_button.pack(pady=20)

root.mainloop()
