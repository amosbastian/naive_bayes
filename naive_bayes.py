import pandas as pd
from collections import Counter, defaultdict
import nltk

def extract_vocabulary(D):
    print "Extracting vocabulary..."
    all_terms = "\n".join(list(D.titel)) # + list(D.vraag) + list(D.antwoord)
    print "Tokenizing all words..."
    
    return Counter(nltk.word_tokenize(all_terms))

def docs_in_class(D, c):
    print "Getting amount of documents in class {}...".format(c)
    D_filtered = D[D.ministerie == c]
    return len(D_filtered.titel)

def text_in_class(D, c):
    print "Getting text from documents in class {}...".format(c)
    D_filtered = D[D.ministerie == c]
    all_terms = "\n".join(list(D_filtered.titel) + list(D_filtered.vraag) + list(D_filtered.antwoord))
    return nltk.word_tokenize(all_terms) 

def count_tokens(text_c, t):
    print t
    return text_c.count(t)

def train_multinomial(C, D):
    condition_probability = defaultdict(Counter)
    V = extract_vocabulary(D)
    N = len(D.titel)
    for c in C:
        print c
        N_c = docs_in_class(D, c)
        prior_c = float(N_c) / N
        text_c = text_in_class(D, c)
        for t, _ in V.most_common(20):
            T_ct = count_tokens(text_c, t)
            condition_probability[t][c] = "hehe" # probability uitrekenen hier

    print condition_probability

if __name__ == '__main__':
    # Change to KVR1000.csv.gz if this becomes too slow for you
    df = pd.read_csv('KVR.csv', sep='\t', encoding='utf-8', index_col=0, 
        names=['jaar', 'partij','titel','vraag','antwoord','ministerie']) 

    # Get value counts and create a dictionary of the classes we want
    class_counts = df.ministerie.value_counts(normalize=True).head(20)
    classes = dict(zip(class_counts.index.tolist()[1::2], 
        class_counts.tolist()[1::2])) #

    # Filter out rows with unwanted classes
    df_filtered = df[df.ministerie.isin(classes.keys())]
    train_multinomial(classes, df_filtered)