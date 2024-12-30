from pymorphy2 import MorphAnalyzer

def preprocess_text(text):
    '''Лемматизация'''
    morph = MorphAnalyzer()
    lemmas = []
    for word in text.split(' '):
        word = word.strip('!@#$%^&*()_+-=?><.,\'\":;][\{\}]`~\n\t\s—»«').lower()
        ana = morph.parse(word)
        lemmas.append(ana[0].normal_form)
    return lemmas