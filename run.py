from waitress import serve
from api import app  # Ajuste o nome do arquivo se necessário

serve(app, host='0.0.0.0', port=5000)
