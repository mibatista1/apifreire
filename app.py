from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/paulo_freire', methods=['GET'])
def paulo_freire():
    conteudo = """
    Paulo Freire foi um educador e filósofo brasileiro, conhecido mundialmente por suas contribuições à educação e à pedagogia crítica. Nascido em 19 de setembro de 1921, em Recife, Freire teve uma infância marcada pela pobreza, o que influenciou sua visão sobre a educação e a sociedade.

    Freire acreditava que a educação deveria ser um ato de liberdade e não de dominação. Sua abordagem pedagógica enfatizava a conscientização, onde os alunos se tornam críticos e ativos na sua aprendizagem. Ele introduziu o conceito de "educação libertadora", que buscava transformar a relação entre educador e educando, promovendo um diálogo genuíno e respeitoso.

    Sua obra mais famosa, "Pedagogia do Oprimido", publicada em 1968, discute como a educação pode ser um meio de emancipação e libertação. Freire argumenta que a opressão não é apenas um estado político, mas uma condição social e cultural, e que a educação deve ajudar os oprimidos a se conscientizarem de sua realidade e a lutar contra a opressão.

    Além de "Pedagogia do Oprimido", Freire escreveu várias outras obras importantes, incluindo "Educação como Prática da Liberdade", "Pedagogia da Esperança" e "Pedagogia da Autonomia". Em cada uma dessas obras, ele reafirma a importância do diálogo, da reflexão crítica e da ação como pilares da educação.

    Freire também teve uma carreira política significativa, tendo atuado como Secretário da Educação em São Paulo e participado de movimentos sociais. Sua abordagem pedagógica influenciou não apenas o Brasil, mas educadores em todo o mundo, especialmente em contextos de opressão e desigualdade.

    Freire faleceu em 2 de maio de 1997, mas seu legado continua vivo na luta por uma educação mais justa e libertadora.
    """

    resumo_obras = {
        "Pedagogia do Oprimido": "Um dos principais trabalhos de Freire, onde ele discute a conscientização e a educação como um meio de libertação.",
        "Educação como Prática da Liberdade": "Neste livro, Freire explora a relação entre educação e liberdade, defendendo a importância da prática reflexiva.",
        "Pedagogia da Esperança": "Uma reflexão sobre suas experiências e um retorno à sua obra anterior, enfatizando a importância da esperança na educação.",
        "Pedagogia da Autonomia": "Freire argumenta que a autonomia é fundamental para a formação do indivíduo crítico e reflexivo."
    }

    return jsonify({"conteudo": conteudo, "resumo_obras": resumo_obras})

if __name__ == '__main__':
    app.run(debug=True)
