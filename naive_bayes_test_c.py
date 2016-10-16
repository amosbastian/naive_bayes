import pandas as pd
from collections import Counter, defaultdict
import nltk
import math
import random

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
    # print V
    for c in C:
        # print "\n" + c
        prior[c] = C[c]
        text_c = V[c]
        #print c, text_c
        #print set(V["all_classes"])
        #print
        for t in set(V["all_classes"]):
            T_ct = count_tokens(text_c, t)
            condprob[t][c] = float((T_ct + 1)) / (len(text_c) + len(set(V["all_classes"])))
            # print "P({} | {}) = ({} + 1) / ({} + {}) = {}".format(t, c, T_ct, 
            #     len(text_c), len(set(V["all_classes"])), condprob[t][c])

    return V, prior, condprob

def extract_tokens(V, d):
    return [token for token in d if token in V["all_classes"]]

def apply_multinomial(C, V, prior, condprob, d):
    W = extract_tokens(V, d)
    score = defaultdict(float)
    
    for c in C:
        score[c] = math.log(prior[c])
        for t in W:
            score[c] += math.log(condprob[t][c])

    assigned_class = max(score, key=score.get)
    #print score
    return assigned_class
    # print "\nDocument d '{}' gives class = {}".format(d, assigned_class)

def get_mi(t, c, D):
    # counting of term occurances is represented as a 2d array, The first index
    # is in {0, 1} and represents whether a term is present (index 1) or not 
    # (index 0). The second index is in {0, 1} and represents whether a
    # document is in class c.
    count = [[0, 0], [0, 0]]
    ndocs = len(D.titel)
    for doc in D[D.ministerie == c].titel:
        if t in doc.split(' '):
            count[1][1] += 1 # num of documents with class c containing t
        else:
            count[0][1] += 1 # num of documents with class c without t
    
    for doc in D[D.ministerie != c].titel:
        if t in doc.split(' '):
            count[1][0] += 1 # num of documents not with class c containing t
        else:
            count[0][0] += 1 # num of documents not with class c without t
    
    # with each of the counts, we calculate the probabilities
    P = [[0, 0], [0, 0]]
    P[1][1] = float(count[1][1]) / ndocs
    P[0][1] = float(count[0][1]) / ndocs
    P[1][0] = float(count[1][0]) / ndocs
    P[0][0] = float(count[0][0]) / ndocs
    
    P_c = [0, 0] # probability that a document is in c (P_c[1]) or not (P_c[0])
    P_t = [0, 0] # probability that a term is in a document (P_t[1]) or not (P_t[0])
    
    P_c[1] = float(len(D[D.ministerie == c])) / ndocs
    P_c[0] = float(len(D[D.ministerie != c])) / ndocs
    P_t[1] = float(count[1][1] + count[1][0]) / ndocs
    P_t[0] = float(count[0][1] + count[0][0]) / ndocs
    
    I_u_c = 0.0
    
    # for e_t in range(2):
    #    for e_c in range(2):
    #        print P[e_t][e_c], "log2( (", P[e_t][e_c], ") / (", P_t[e_t],"*",P_c[e_c], ") )"
    
    for e_t in range(2):
        for e_c in range(2):
            # the number in the log should not be 0, nor should any component 
            # of the division be zero
            if (((P_t[e_t] * P_c[e_c]) != P[e_t][e_c]) and 
            (P[e_t][e_c] != 0 and (P_t[e_t] * P_c[e_c]) != 0)):
                I_u_c += P[e_t][e_c] * math.log((P[e_t][e_c]) / (P_t[e_t] * P_c[e_c]), 2)
    
    # return A(t, c)
    return I_u_c
    
def dd_float():
    return defaultdict(float)

def top_k_terms(V, C, D, k):
    mi = defaultdict(dd_float)
    
    for c in C:
        for t in V[c]:
            mi[c][t] = get_mi(t, c, D)

        # Hier alleen top k terms in dictionary laten per class
        print "top k termen: {}".format(dict(Counter(mi[c]).most_common(k)))
    
    return mi
    
    
if __name__ == '__main__':
    D = pd.read_csv('test2.csv', sep='-', encoding='utf-8', index_col=0, 
        names=['docID', 'titel','ministerie']) 
    
    #Get fractions of documents belonging to the classes
    cc = D.ministerie.value_counts(normalize=True)
    
    #Make dict out of those percentages
    C = dict(zip(cc.index.tolist(), cc.tolist()))
    
    #Take a 3/5 fraction as training set
    train = D.sample(frac=0.6)
    
    #Use the rest as set to predict class
    test = D.drop(train.index)
    
    V, prior, condprob = train_multinomial(C, train)
    
    top_kek = top_k_terms(V, C, train, 3)
    topkV = defaultdict(list)
    
    for cls in top_kek:
        topkV[cls] = dict(Counter(top_kek[cls]).most_common(5))
    
    for i in range(len(test.titel)):
        text = "\n".join(list(test.titel)[i].split())
        predicted_class = apply_multinomial(C, topkV, prior, condprob, nltk.word_tokenize(text))
        print "{} | {}".format(list(test.ministerie)[i], predicted_class)
