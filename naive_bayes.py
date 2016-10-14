import pandas as pd
from collections import Counter, defaultdict
import nltk

def extract_vocabulary(D):
    print "Extracting vocabulary..."
    all_terms = "\n".join(list(D.titel)) # + list(D.vraag) + list(D.antwoord)
    print "Tokenizing all words..."
    
    return Counter(nltk.word_tokenize(all_terms))

def text_in_class(D, c):
    print "Getting text from documents in class {}...".format(c)
    D_filtered = D[D.ministerie == c]
    all_terms = "\n".join(list(D_filtered.titel) + list(D_filtered.vraag) + list(D_filtered.antwoord))
    return nltk.word_tokenize(all_terms) 

def count_tokens(text_c, t):
    return text_c.count(t)

def train_multinomial(C, D):
    condprob = defaultdict(lambda: defaultdict(float))
    prior = defaultdict(float)
    V = extract_vocabulary(D)
    for c in C:
        print "\n   " + c
        prior[c] = C[c]
        text_c = text_in_class(D, c)
        for t, _ in V.most_common(3):
            T_ct = count_tokens(text_c, t)
            condprob[t][c] = float((T_ct + 1)) / (len(text_c) + 20)

    print V, prior, condprob

if __name__ == '__main__':
    # Change to KVR1000.csv.gz if this becomes too slow for you
    D = pd.read_csv('KVR.csv', sep='\t', encoding='utf-8', index_col=0, 
        names=['jaar', 'partij','titel','vraag','antwoord','ministerie']) 

    # Get value counts and create a dictionary of the classes we want
    cc = D.ministerie.value_counts(normalize=True).head(20)
    C = dict(zip(cc.index.tolist()[1::2], cc.tolist()[1::2]))

    # Filter out rows with unwanted classes
    D_filtered = D[D.ministerie.isin(C.keys())]
    cc_filtered = D_filtered.ministerie.value_counts(normalize=True).head(10)
    C_filtered = dict(zip(cc_filtered.index.tolist(), 
        D_filtered.ministerie.value_counts(normalize=True).head(10).tolist()))

    train_multinomial(C_filtered, D_filtered)