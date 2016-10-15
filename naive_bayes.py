import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from collections import Counter, defaultdict
import nltk
import math
import json

def extract_vocabulary(C, D):
    V = defaultdict(list)
    print "Extracting vocabulary:"
    for c in C:
        print c
        D_temporary = D[D.ministerie == c]
        text = "\n".join(list(D_temporary.titel))
        V[c] = nltk.word_tokenize(text)
        V["all_classes"] += V[c]
    
    return V

def count_tokens(text_c, t):
    return text_c.count(t)

def train_multinomial(C, D):
    condprob = defaultdict(lambda: defaultdict(float))
    prior = defaultdict(float)
    V = extract_vocabulary(C, D)
    B = len(set(V["all_classes"]))
    print "\nTraining (multinomial):"
    for c in C:
        print c
        prior[c] = C[c]
        text_c = V[c]
        A = len(text_c)
        for t in set(V["all_classes"]):
            # BOTTLENECK
            T_ct = count_tokens(text_c, t)
            condprob[t][c] = float((T_ct + 1)) / (A + B)

    return V, prior, condprob

def extract_tokens(V, d):
    return [token for token in d if token in V["all_classes"]]

def apply_multinomial(C, V, prior, condprob, d):
    W = extract_tokens(V, d)
    print W
    score = defaultdict(float)
    for c in C:
        score[c] = math.log(prior[c])
        for t in W:
            score[c] += math.log(condprob[t][c])

    assigned_class = max(score, key=score.get)
    return assigned_class

if __name__ == '__main__':
    # Change to KVR1000.csv.gz if this becomes too slow for you
    D = pd.read_csv('KVR.csv', sep='\t', encoding='utf-8', index_col=0, 
        names=['jaar', 'partij','titel','vraag','antwoord','ministerie']) 

    # Get value counts and create a dictionary of the classes we want
    cc = D.ministerie.value_counts(normalize=True).head(20)
    C = dict(zip(cc.index.tolist()[1::2], cc.tolist()[1::2]))

    # Filter out rows with unwanted classes
    D_filtered = D[D.ministerie.isin(C.keys())]
    cc_filtered = D_filtered.ministerie.value_counts(normalize=True)
    C_filtered = dict(zip(cc_filtered.index.tolist(), 
        D_filtered.ministerie.value_counts(normalize=True).tolist()))

    # Split into training and testing set
    train = D_filtered.head(100).sample(frac=0.8)
    test = D_filtered.head(100).drop(train.index)

    V, prior, condprob = train_multinomial(C_filtered, train)
    print "\nprior: {}".format(prior)
    
    for i in range(len(test.titel)):
        text = "\n".join(list(test.titel)[i].split())
        predicted_class = apply_multinomial(C_filtered, V, prior, condprob, nltk.word_tokenize(text))
        print "\n{}\n{}".format(list(test.ministerie)[i], predicted_class)
    # print "V: {}".format(json.dumps(V))
    # print "\nprior: {}".format(prior)
    # print "\ncondprod: {}".format(condprob)
    # apply_multinomial(C_filtered, V, prior, condprob, "de van")