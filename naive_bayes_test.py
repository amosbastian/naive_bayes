import pandas as pd
from collections import Counter, defaultdict
import nltk
import math

def extract_vocabulary(C, D):
    V = defaultdict(list)
    for c in C:
        D_temporary = D[D.ministerie == c]
        text = "\n".join(list(D_temporary.titel))
        V[c] = nltk.word_tokenize(text)
        V["all_classes"] += V[c]
    
    return V

def text_in_class(D, c):
    D_filtered = D[D.ministerie == c]
    all_terms = "\n".join(list(D_filtered.titel))
    return nltk.word_tokenize(all_terms) 

def count_tokens(text_c, t):
    return text_c.count(t)

def train_multinomial(C, D):
    condprob = defaultdict(lambda: defaultdict(float))
    prior = defaultdict(float)
    V = extract_vocabulary(C, D)
    print V
    for c in C:
        print "\n" + c
        prior[c] = C[c]
        text_c = V[c]
        for t in set(V["all_classes"]):
            T_ct = count_tokens(text_c, t)
            condprob[t][c] = float((T_ct + 1)) / (len(text_c) + len(set(V["all_classes"])))
            print "P({} | {}) = ({} + 1) / ({} + {}) = {}".format(t, c, T_ct, 
                len(text_c), len(set(V["all_classes"])), condprob[t][c])

    return V, prior, condprob

def extract_tokens(V, d):
    return [token for token in d.split() if token in V]

def apply_multinomial(C, V, prior, condprob, d):
    W = extract_tokens(V, d)
    score = defaultdict(float)
    for c in C:
        score[c] = math.log(prior[c])
        for t in W:
            score[c] += math.log(condprob[t][c])

    assigned_class = max(score, key=score.get)
    print "\nDocument d '{}' gives class = {}".format(d, assigned_class)

if __name__ == '__main__':
    D = pd.read_csv('test.csv', sep='-', encoding='utf-8', index_col=0, 
        names=['docID', 'titel','ministerie']) 

    cc = D.ministerie.value_counts(normalize=True).head(20)
    C = dict(zip(cc.index.tolist(), cc.tolist()))
    V, prior, condprob = train_multinomial(C, D)
    apply_multinomial(C, V, prior, condprob, "Chinese Chinese Chinese Tokyo Japan")