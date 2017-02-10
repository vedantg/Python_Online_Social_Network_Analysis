# coding: utf-8

# Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    movies['tokens'] = ""
    gen = ""
    gen_return = ""
     
    for index,row in movies.iterrows():
        gen = row['genres']
        gen_return=tokenize_string(gen)
        movies.set_value(index,'tokens',gen_return)
  
    return (movies)
        
        

def cal_unique_features(movies): #num_features

        h = set()
        for index,row in movies.iterrows():
            gen = row['tokens']
            for item in gen:
                h.add(item)                
       
        return sorted(h)
        
def cal_unique_vocab(get_h):
    return_dict = {}
    counter = 0 
    for item in get_h:
        #print("current_item::",item[0]) # vocab complete
        return_dict[item]=counter
        counter+=1
    
    return return_dict
    
     
def cal_unique_docs(h,movies): #df(i)
        df_dict = {}
        #check_set = set()
        for item in h:
            #print("ITEM::",item)
            count = 0
            for index,row in movies.iterrows():
                check_set = set()
                gen = row['tokens']
                #print("GEN::",gen)
                for gen_item in gen:
                    #print("GEN_ITEM",gen_item)
                    check_set.add(gen_item)
                    #print("Check_set:",check_set)
                    if item in check_set:
                        #print("Count_Before::",count)
                        count += 1
                        #print("Count_After::",count)                        
                        break
      
                df_dict[item]=count
                
         
        return(df_dict)
        #print("HII::",df_dict)

def get_tf_value(index_dict, tok, ind):
    
    for t_list in index_dict[tok]:
                if t_list[0] == ind:
                    tf_val = t_list[1]
                    return(tf_val)
    

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO   
    movies['features'] = ""    
    get_h = set()  
    vocab_dict = {}
    df_dict_return = {}
    tup_list = []
    index_dict = {}
    index_dict_1 = {}
    movie_len = len(movies)    
    #print("MovieLength::",movie_len)
    #print("MOVIES:::",movies)
           
    get_h = cal_unique_features(movies)  # num_features

    vocab_dict = cal_unique_vocab(get_h) # vocab complete

    len_vocab = len(get_h)
    
    df_dict_return = cal_unique_docs(get_h,movies) # df(i)

    for token in get_h :
        #tup_list.clear()
        #print("token_GOTTTTT:::",token)
        for index,row in movies.iterrows():            
            #print("row_got::",row)
            gen_list = row['tokens']
            #print("gen_list::",gen_list)
            #mov_id = row['movieId'] 
            #print("mov_id::",mov_id)
            token_count_1 = Counter(gen_list).most_common()[:1]
            tok = token_count_1[0]
            index_dict_1[index] = tok[1]
            token_count = gen_list.count(token)
            #print("token_count::",token_count)
            tup = (index,token_count)
            #print("tuple::",tup)
            tup_list.append(tup)
            #print("LIST_PRINT:::::::::::::",tup_list)
        index_dict[token] = tup_list
        tup_list = []
        
    
    #print("INDEX_DICT:::",index_dict) # tf(i,d)
    #print("INDEX_DICT_1:::",index_dict_1) # max_k dict per docx
  
    
    for ind, row in movies.iterrows():
        data_list = []
        rows_list = []
        columns_list = []
        gen_list = row['tokens']
        #print("TOKENS GOTTT::",gen_list)   
        for gen in gen_list:
            tf = get_tf_value(index_dict,gen,ind)
            #print("TF GOTTT::",tf)   
            tf_weight = float( tf / index_dict_1[ind])
            #print("tf_weight::",tf_weight)
            df_weight = float( math.log10( movie_len / df_dict_return[gen] ) )
            #print("df_weight::",df_weight)
            final_tfidf = tf_weight * df_weight
            #print("final_tfidf::",final_tfidf)
            data_list.append(final_tfidf)
            columns_list.append(vocab_dict[gen])
            rows_list.append(0)            
        csr = csr_matrix((data_list, (rows_list,columns_list)), shape=(1,len_vocab))
        #print("TYPE of CSR GOTT::",type(csr))
        #print("CSR GOTT:::",csr)        
        movies.set_value(ind, 'features', csr)
    
    #print("UPDATE movies::",movies) 

    return(movies,vocab_dict)
                            

    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO    
    
    #print("SHAPE of AAA::",a.shape)
    a = a.toarray()
    #print("TYPE of AAA::",type(a))
    #print("AA:::",a)
    
    #print("SHAPE of BBB::",b.shape)
    b = b.toarray()
    #print("TYPE of BBB::",type(b))
    #print("BBB_TEST:::",b)
    
    b_new = b.reshape(22,1)
    
    dot_product = np.dot(a, b_new)
   
    norm_a = np.linalg.norm(a)
    
    #print("NORM_a::",norm_a)
    
    #print("TYPE of NORM_a::",type(norm_a))
    
    norm_b = np.linalg.norm(b)
    
    #print("NORM_b::",norm_b)
    
    #print("TYPE of NORM_b::",type(norm_b))
    
    norm_total = np.multiply(norm_a, norm_b)
    
    #print("norm_total::",norm_total)
    
    #print("TYPE of norm_total::",type(norm_total))
    
    cos_sim = np.divide(dot_product, norm_total)
    
    #print("cos_sim::",cos_sim)
    
    #print("TYPE of cos_sim::",type(cos_sim))
    
    return_ans = cos_sim.item()
    
    #print("return_ans::",return_ans)
    
    #print("TYPE of return_ans::",type(return_ans))
    
    return (return_ans)
       
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    
    user_result = []    
    
    for index,row in ratings_test.iterrows():
        userid_test = row['userId']
        #print("userid_test::",userid_test) 
        movieid_test = row['movieId'] 
        #print("movieid_test::",movieid_test) 
        x = list(movies[movies.movieId==movieid_test]['features'])[0]
        #print("CSR_GOTT+X::",x)
        #print("TYPE of CSR_GOTT_X::",type(x))
        subset_train = ratings_train[ratings_train.userId == userid_test]
        #print("SUB MOVIE SET::",subset_train)
        #print("TYPE of SUB MOVIE SET::",type(x))
        total_if_zero=0
        rating_if_zero=0
        sum_main_result=0
        sum_cosine=0        
        for index1,row1 in subset_train.iterrows():
            userid_train = row1['userId']
            #print("userid_train::",userid_train)              
            if(userid_test == userid_train ):
                #print("HII IN IFFF:::")
                movieid_train = row1['movieId']
                #print("movieid_train::",movieid_train)
                rating_train = row1['rating']
                #print("rating_train::",rating_train)
                total_if_zero = total_if_zero + 1   
                rating_if_zero = rating_if_zero + rating_train
                y = list(movies[movies.movieId==movieid_train]['features'])[0]
                #print("CSR_GOTT_Y::",y)
                #print("TYPE of CSR_GOTT_Y::",type(y))
                result_cos = cosine_sim(x,y)
                sum_main_result += result_cos * rating_train
                sum_cosine += result_cos     
        
        if(sum_main_result != 0):
            user_result.append(sum_main_result/sum_cosine)
            #print("user_result::",user_result)             
        else:
            user_result.append(rating_if_zero / total_if_zero)
            #print("user_result::",user_result)  
            
    return_result_arr = np.array(user_result) 
    
    return return_result_arr
   
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
