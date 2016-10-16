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
    #print W
    for c in C:
        # Score is initialised as the prior score
        score[c] = math.log(prior[c])
        for t in W:
            score[c] += math.log(condprob[t][c])

    # Predicted class is the class with the highest score
    predicted_class = max(score, key=score.get)
    return predicted_class

def get_mi(t, c, D):
    # counting of term occurances is represented as a 2d array, The first index
    # is in {0, 1} and represents whether a term is present (index 1) or not 
    # (index 0). The second index is in {0, 1} and represents whether a
    # document is in class c.
    count = [[0, 0], [0, 0]]
    ndocs = len(D.titel)
    for doc in D[D.ministerie == c].titel:
        words = doc.split(' ')
        if t in words:
            count[1][1] += 1 # num of documents with class c containing t
        else:
            count[0][1] += 1 # num of documents with class c without t
    
    for doc in D[D.ministerie != c].titel:
        words = doc.split(' ')
        if t in words:
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
    
    for e_t in range(2):
        for e_c in range(2):
            # the number in the log should not be 0, nor should any component 
            # of the division be zero
            if (((P_t[e_t] * P_c[e_c]) != P[e_t][e_c]) and 
            (P[e_t][e_c] != 0 and (P_t[e_t] * P_c[e_c]) != 0)):
                I_u_c += P[e_t][e_c] * math.log((P[e_t][e_c]) / (P_t[e_t] * P_c[e_c]), 2)
    
    # return A(t, c)
    return I_u_c

def top_k_terms(V, C, D, k):
    print "\nGetting top {} terms of each class".format(k)
    mi = defaultdict(dd_float)
    for c in C:
        print c
        for t in V[c]:
            mi[c][t] = get_mi(t, c, D)

        mi[c] = dict(Counter(mi[c]).most_common(k))
    
    return mi


def get_F1(c, test, C, topkV, prior, condprob):
    relevant_retrieved = 0
    relevant_items = len(test[test.ministerie == c])
    items_retrieved = len(test.titel)
    
    
    for i in range(len(test.titel)):
        # retrieve text of a test document
        text = "\n".join(list(test.titel)[i].split(' '))
        
        # predict the class using apply_multinomial
        pred_c = apply_multinomial(C, topkV, prior, condprob, nltk.word_tokenize(text))
        real_c = list(test.ministerie)[i]
        
        # count relevant items retrieved if the class has been guessed
        # correctly and corresponds to class c
        if real_c == pred_c == c:
            relevant_retrieved += 1
            
    print "relevant retrieved", relevant_retrieved
    print "relevant_items", relevant_items
    print "items_retrieved", items_retrieved
    
    # Calculate precision and Recall
    Prec = 0
    Rec = 0
    if items_retrieved != 0:
        Prec = float(relevant_retrieved) / float(items_retrieved) 
    if items_retrieved != 0:
        Rec = float(relevant_retrieved) / float(relevant_items)
        
    # use default a = 0.5, calculate and return F1.
    a = 0.5
    F1 = 1 / ((a / Prec) + ((1 - a) / Rec))
    
    return F1


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
    train = D_filtered.head(1000).sample(frac=0.8, random_state=50)
    test  = D_filtered.head(1000).drop(train.index)

    V, prior, condprob = train_multinomial(C_filtered, train)

    # print json.dumps(top_k_terms(V, C_filtered, train, 10))
    correct, wrong = 0, 0


    # Get the top k terms for each class
    top_kek = top_k_terms(V, C_filtered, train, 20)

    # Create a dict and put all the top k terms from every class into
    # a dict as a new vocabulary
    topkV = defaultdict(list)
    for cls in top_kek:
        topkV[cls].extend(top_kek[cls])
        topkV["all_classes"].extend(top_kek[cls])
    
    # remove duplicates
    topkV["all_classes"] = list(set(topkV["all_classes"]))
    
    # Calculate the F1 for a class
    print get_F1(u' Justitie (JUS)', test, C_filtered, topkV, prior, condprob)
    
    
    # For each document in the test set, get the text in its title and predict
    # its class 
    for i in range(len(test.titel)):
        text = "\n".join(list(test.titel)[i].split())
        predicted_class = apply_multinomial(C_filtered, V, prior, 
            condprob, nltk.word_tokenize(text))
        if predicted_class == list(test.ministerie)[i]:
            correct += 1
        else:
            wrong += 1

    print "\nAccuracy: {}%".format(100 * float(correct) / (correct + wrong))