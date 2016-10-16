from collections import Counter, defaultdict
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk, math, json, pickle

def extract_vocabulary(C, D):
    V = defaultdict(list)
    print "Extracting vocabulary:"
    for c in C:
        print c
        # Get all text (titles) from documents with class c
        D_temporary = D[D.ministerie == c]
        text = "\n".join(list(D_temporary.titel)) # + list(D_temporary.titel) etc?
        # Save all tokens to our dictionary
        V[c] = nltk.word_tokenize(text)
        V["all_classes"] += V[c]
    
    return V

def dd_float():
    return defaultdict(float)

def train_multinomial(C, D):
    # Initialise some variables
    condprob = defaultdict(dd_float)
    prior    = defaultdict(float)
    V        = extract_vocabulary(C, D)
    B        = len(set(V["all_classes"]))

    print "\nTraining (multinomial):"
    for c in C:
        print c
        # Get prior chance and all tokens in documents of class c
        prior[c]    = C[c]
        text_c      = V[c]
        terms_class = len(text_c)
        for t in set(V["all_classes"]):
            # Get number of occurrences of t in training documents from class c
            # and calculate the conditional probability
            T_ct = text_c.count(t)
            condprob[t][c] = float((T_ct + 1)) / (terms_class + B)

    return V, prior, condprob

def apply_multinomial(C, V, prior, condprob, d):
    # Get all tokens that are both in d and the vocabulary
    W     = [token for token in d if token in V["all_classes"]]
    score = defaultdict(float)

    for c in C:
        # Score is initialised as the prior score
        score[c] = math.log(prior[c])
        for t in W:
            score[c] += math.log(condprob[t][c])

    # Predicted class is the class with the highest score
    predicted_class = max(score, key=score.get)
    return predicted_class

if __name__ == '__main__':
    # Change to KVR1000.csv.gz if this becomes too slow for you
    D = pd.read_csv('KVR.csv', sep='\t', encoding='utf-8', index_col=0, 
        names=['jaar', 'partij','titel','vraag','antwoord','ministerie']) 

    classes = [u' Justitie (JUS)', 
               u' Volksgezondheid, Welzijn en Sport (VWS)', 
               u' Buitenlandse Zaken (BUZA)', 
               u' Verkeer en Waterstaat (VW)',
               u' Sociale Zaken en Werkgelegenheid (SZW)', 
               u' Onderwijs, Cultuur en Wetenschappen (OCW)',
               u' Volkshuisvesting, Ruimtelijke Ordening en Milieubeheer (VROM)',
               u' Financi\xebn (FIN)',
               u' Economische Zaken (EZ)',
               u' Defensie (DEF)']

    # Filter out rows with unwanted classes
    D_filtered  = D[D.ministerie.isin(classes)]
    cc_filtered = D_filtered.ministerie.value_counts(normalize=True)
    C_filtered  = dict(zip(cc_filtered.index.tolist(), 
        D_filtered.ministerie.value_counts(normalize=True).tolist()))

    # Split into training and testing set
    train = D_filtered.sample(frac=0.8, random_state=50)
    test  = D_filtered.drop(train.index)

    V, prior, condprob = train_multinomial(C_filtered, train)

    with open("prior_titel.pickle", "wb") as handle:
        pickle.dump(prior, handle)

    # correct, wrong = 0, 0

    # top_10(C, V)

    # # For each document in the test set, get the text in its title and predict
    # # its class 
    # for i in range(len(test.titel)):
    #     text = "\n".join(list(test.titel)[i].split())
    #     predicted_class = apply_multinomial(C_filtered, V, prior, 
    #         condprob, nltk.word_tokenize(text))
    #     if predicted_class == list(test.ministerie)[i]:
    #         correct += 1
    #     else:
    #         wrong += 1

    # print "\nAccuracy: {}%".format(100 * float(correct) / (correct + wrong))