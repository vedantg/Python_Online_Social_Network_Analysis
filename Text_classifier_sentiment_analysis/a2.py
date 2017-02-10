# coding: utf-8

"""
In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.
You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.
The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.
Complete the 14 methods below, indicated by TODO.
As usual, completing one method at a time, and debugging with doctests, should
help.
"""
# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
from nltk.featstruct import FeatStruct


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.
    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    >>> tokenize("Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun?", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    ###TODO
    token_list = []    

    if keep_internal_punct == False:
        numpy_array=np.array(re.sub('\W+', ' ', doc.lower()).split())                    
    else:
        data = doc.lower().strip().split()
        for d in data:
            d = d.strip(string.punctuation)
            token_list.append(d)
            
        numpy_array=np.array(token_list)        

    return numpy_array
    

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
       #print("TokenGot::",tokens)
    for item in tokens:
      #print("CurrentToken: ", token)
      feats['token='+item]=feats['token='+item]+1
    pass

def take_window_slice(tokens,k):
    slice_list = []
    slice_list_return = []
    flag = True        
    while flag:
        result = tuple(tokens[:k])
        result_len = len(result)
        if(result_len<=0):
            flag = False
            continue
        slice_list.append(result)
        try:
            tokens = tokens.pop(0)
        except:
            tokens = tokens[1:]

    for sliced in slice_list:
        if (len(sliced) >= k):
            slice_list_return.append(sliced)

    return slice_list_return


def set_csr_matrix(column_dict,row_feats_dict_return,to_prune_dict_return,min_freq):
        row=[]
        column=[]
        data=[]
        for key,val in row_feats_dict_return.items():
            for x in val:
                if(to_prune_dict_return[x[0]] >= min_freq):
                    row.append(key)
                    column.append(column_dict[x[0]])
                    data.append(x[1])
        return row,column,data


def vocab_csr_matrix(row_feats_dict_return,vocab):
    temp_dict=vocab    
    row1=0
    col1=0
    row=[]
    column=[]
    data=[]
    
    for val in row_feats_dict_return.values():
        for t1 in val:
            if(t1[0] in temp_dict):
                row.append(col1)
                column.append(temp_dict[t1[0]])
                data.append(t1[1])
        col1=col1+1
        row1=row1+1
    return(row,column,data)
    
    
def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO     
    combination_slice = []
    windows_slice = take_window_slice(tokens,k)
    
    for sliced in windows_slice:
        com = np.array(sliced)
        combination_slice = (combinations(com, 2))
        for pair in combination_slice:
            token_pair_after = '__'.join(pair)
            token_pair = 'token_pair=' + token_pair_after
            feats[token_pair] = feats[token_pair] + 1

neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    feats['pos_words']=0
    feats['neg_words']=0
    token_list = map(lambda x:x.lower(), tokens)
    for item in token_list:
        if item in neg_words:
            feats['neg_words']=feats['neg_words']+1            
        elif item in pos_words:
            feats['pos_words']=feats['pos_words']+1
            
    pass

def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.
    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO

    featurize_dict = defaultdict(int)    
    
    for feature in feature_fns:
        feature(tokens,featurize_dict)
    
    return sorted(featurize_dict.items())

    pass

def call_feature(tokens_list,feature_fns):
    set_row=0
    row_feats_dict=defaultdict(lambda: 0)
    to_prune_dict=defaultdict(int)
    for row_list in tokens_list:
        row_feats_dict[set_row]=featurize(row_list,feature_fns)
        #print("row_feats_dict[set_row]::",row_feats_dict[set_row])
        set_row = set_row + 1
        #print("Row_Current_count::",set_row)
    
    for val in row_feats_dict.values():
        #print("Valuesss::",val)
        for x in val:
            to_prune_dict[x[0]]=to_prune_dict[x[0]]+1
            #print("to_prune_dict[x[0]]::",to_prune_dict[x[0]])
    return row_feats_dict,to_prune_dict

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.
    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),
    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    row_feats_dict_return=defaultdict(int)
    counter=0
    csr_row=[]
    csr_column=[]
    csr_data=[]
    to_prune_dict_return=defaultdict(int)
    prune_dict=defaultdict(int)
    columns=defaultdict(int)
    
    counter=0
    
    def getKey_one(item1):
        return item1[0]
    
    row_feats_dict_return,to_prune_dict_return = call_feature(tokens_list,feature_fns)
    
    for key,val in to_prune_dict_return.items():
        #print("Current Temp Value::",val)
        if(val>=min_freq):
            prune_dict[key]=val
            #print("Final_Dict::",final_dict)
            
    
    final_dict_list=sorted(prune_dict.items(), key=getKey_one)
    
    #print("Pruned List::",final_dict_list)
    
    for item in final_dict_list:
        #print("current_item::",item[0])
        columns[item[0]]=counter
        counter+=1
    
    if(vocab==None):
        csr_row,csr_column,csr_data=set_csr_matrix(columns,row_feats_dict_return,to_prune_dict_return,min_freq)
    else:
        csr_row,csr_column,csr_data=vocab_csr_matrix(row_feats_dict_return,vocab)  
    
                               
    return (csr_matrix((csr_data, (csr_row,csr_column)), dtype='int64'),columns)    
    
    pass

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    #use this version final
    
    """kfold = KFold(len(labels), k)
    accuracies=[]
    for train_ind, test_ind in kfold:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)
    pass"""

    lab = len(labels)    
    num_folds = KFold(lab, k)
    acc=[]
    for train_data, test_data in num_folds:
        clf.fit(X[train_data], labels[train_data])
        predicted_data = clf.predict(X[test_data])
        acc.append(accuracy_score(labels[test_data], predicted_data))
    return np.mean(acc)
    pass

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.
    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.
    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).
    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])
    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
      This list should be SORTED in descending order of accuracy.
      This function will take a bit longer to run (~20s for me).
    """
    ###TODO   
    feature=0
    false_list=[tokenize(d) for d in docs]    
    true_list=[tokenize(d,True) for d in docs]   
    function_list = []
    for fns in range(1, len(feature_fns)+1):
        for sub_fns in combinations(feature_fns, fns):
            function_list.append(list(sub_fns))
    list_of_dict=[]
    clf = LogisticRegression()
    #print(function_list)
    
    while feature<len(function_list):
        feat=function_list[feature]
        for punctuation in punct_vals:            
            for f in min_freqs:                
                if (punctuation==False):
                    X,vocab=vectorize(false_list, feat, f)                    
                else:
                    X,vocab=vectorize(true_list, feat, f)
                acc = cross_validation_accuracy(clf, X, labels, 5)                    
                feat_dict={}
                feat_dict['features']=feat
                feat_dict['punct']=punctuation                
                feat_dict['min_freq']=f                                               
                feat_dict['accuracy']= acc
                list_of_dict.append(feat_dict)
        feature = feature + 1
    
    
    list_of_dict=sorted(list_of_dict, key=lambda a:a['accuracy'],reverse=True)
                #print(settings.items())
                #print(list_of_dict)
    #list_of_dict=sorted(list_of_dict, key=lambda x:(-x['accuracy']))
    #print(sorted(list_of_dict,key=lambda x: "%.5f" % (x['accuracy'])))
    #print(dict_list)
    return list_of_dict
    pass


def prepare_all_combination_mean(results):
    settings_set=set()
    mean_feature_Accuracy=[]    
    for r in results:
        settings_set.add('min_freq='+str(r['min_freq']))
        settings_set.add('punct='+str(r['punct']))
        
    
    for feature in settings_set:
        total_accuracy=0
        counter=0
        for item in results:
            if(feature==('min_freq='+str(item['min_freq']))):
                try:
                    total_accuracy+=item['accuracy']
                    counter+=1
                except:
                    total_accuracy=item['accuracy']
                    counter=1
            elif(feature==('punct='+str(item['punct']))):
                try:
                    total_accuracy+=item['accuracy']
                    counter+=1
                except:
                    total_accuracy=item['accuracy']
                    counter=1
                    
            
                    
        mean_feature_Accuracy.append(((total_accuracy/counter),feature))
    
    return settings_set,mean_feature_Accuracy
    
def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    acc=[]
    for data in results:
        #print()
        for key,val in data.items():
            if(key == 'accuracy'):
                acc.append(data[key])
    #print(accuracy)
    
    #counts = Counter(results.values())
    vals = sorted(acc)
    #print(vals)
    #plt.bar(range(len(vals)),vals,1)
    #plt.xticks(vals)
    x = plt.plot(vals)
    plt.setp(x, color='r', linewidth=2.0,dash_capstyle='round')
    plt.xlabel('Settings')
    plt.ylabel('Accuracy')
    plt.savefig("accuracies.png")
    plt.show()
    
    pass

def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    min_freq=2.
    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    feat_set_1=set()
    s="features="
    
    def getKey_one(item1):
        return item1[0]
    
    mean_final=[]
    
    feat_set,mean_final = prepare_all_combination_mean(results) 
    
    for r in range(0,(len(results))):
        s="features="
        feature_typ=results[r]["features"]        
        for f in feature_typ:
            s=s+" "+f.__name__
        feat_set_1.add(s)
    #print("feat_set::",feat_set_1)
    
    
    for feature in feat_set_1:
        t=0
        denom=0
        #print("Features::",feature)
        for item in results:
            s1="features="
            for x in item['features']:
                #print("X::",x)
                s1 = s1 +" "+x.__name__
            if(feature==s1):
                try:
                    t+=item['accuracy']
                    denom+=1
                except:
                    t=item['accuracy']
                    denom=1
        mean_final.append(((t/denom),feature))
    
    return sorted(mean_final, key=getKey_one,reverse=True)
    
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)
    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """

    #feat_func = [lexicon_features,token_pair_features]
    clf = LogisticRegression()
    feat_func = best_result['features']
    freq = best_result['min_freq']
    punctuation = best_result['punct']
    #print("PUNCT::",punctuation)
    list1 = [tokenize(sub,punctuation) for sub in docs]
    #print("Current_list::",list1)
    x , vocab = vectorize(list1,feat_func,freq)
    #print("X::",x)    
    #print("vocab::",vocab)
    clf.fit(x,labels)
    #print("CLF::",clf)
    return clf, vocab
    
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.
    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO
    
    token_list = []
    zip_list = []
    
    def getKey_one(item1):
        return item1[1]
        
    vocab_list = sorted(vocab.items(), key=getKey_one)
    
    for t1,t2 in vocab_list:
        token_list.append(t1)

    token_array = np.array(token_list) 

    if label==0:
        first = np.argsort(clf.coef_[0])[:n]        
    else:
        first = np.argsort(clf.coef_[0])[::-1][:n]
    
    ele_one=[]    
    ele_one=token_array[first]
    ele_two = abs(clf.coef_[0][first])

    for t1 in zip(ele_one, ele_two):
        zip_list.append(t1)

    return sorted(zip_list,key=getKey_one,reverse=True)
    pass


def set_probability(test_docs,predicted_label,probablities_label,test_labels):
    
    dictlist = []
    for i in range(0,len(test_docs)):        
        set_dict = {}
        if(predicted_label[i] != test_labels[i]):
            set_dict["filename"] = test_docs[i]
            set_dict["index"] = i
            set_dict["predicted"] = predicted_label[i]
            set_dict["probas"] = probablities_label[i]
            set_dict["truth"] = test_labels[i]
            dictlist.append(set_dict)
    
    return dictlist    

def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.
    Note: use read_data function defined above to read the
    test data.
    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO 
    
    feat_func = best_result['features']
    freq = best_result['min_freq']
    punctuation = best_result['punct']
    
    docs,labels=read_data(os.path.join('data', 'test'))
    
    if(punctuation==False):
        tokens = [tokenize(d) for d in docs]        
    else:
        tokens = [tokenize(d,True) for d in docs]        

    x,vocab=vectorize(tokens,feat_func,freq,vocab=vocab)
    
    return docs,labels,x
    
    pass



def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.
    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.
    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    predicted_label = clf.predict(X_test)
    #print("predicted_label::",predicted_label)
    probablities_label = clf.predict_proba(X_test)
    #print("probablities_label::",probablities_label)
    dictlist_return = []
    
    dictlist_return = set_probability(test_docs,predicted_label,probablities_label,test_labels)

    dictlist_return = sorted(dictlist_return, key=lambda x: x['probas'][x['truth']])[:n]
    
    for value in dictlist_return:
        if(value['truth']==1):
            print("truth=",value['truth'],"predicted=",value['predicted'],"proba=",value['probas'][0])
        elif(value['truth']==0):
            print("truth=",value['truth'],"predicted=",value['predicted'],"proba=",value['probas'][1])
        print(value['filename'])
        
    pass


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    #download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))
     
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    

if __name__ == '__main__':
    main()