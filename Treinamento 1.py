from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

# Dados de exemplo
texts = ["texto exemplo 1", "texto exemplo 2"]  # Substitua pelos seus textos
labels = [0, 1]  # Substitua pelos seus rótulos

# Criação do vetor TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Criação e treinamento do classificador
classifier = LogisticRegression()
classifier.fit(X, labels)

# Salvando o modelo e o vectorizer
dump(classifier, 'text_classifier.joblib')
dump(vectorizer, 'vectorizer.joblib')
