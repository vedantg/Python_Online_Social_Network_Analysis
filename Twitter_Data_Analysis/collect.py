"""
collect.py

"""
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import sys
import time
from TwitterAPI import TwitterAPI
import json
from pprint import pprint
from config import *

consumer_key = 'Sdts6KDxe0lRrJv58xAW4reRA'
consumer_secret = 'qRps0KdoqWU8Y6BUntSwHwweA2xTrZf5nMrAjVIcdRZiF5aQzM'
access_token = '771103127063916544-NxseKPMKR7bZFEn3xptQK61weCECEmV'
access_token_secret = 'itLVtiuZ2aR35eBc9ohhiFTC7kaocyfADpxak8GibSiOQ'


write_response_file = 'HillaryClinton_new.json'
user_data_file = 'user_data_new.txt'
tweets_data_file = 'tweets_data_new.txt'
file_name = 'Log.txt'

def get_twitter():
    """ Constructing an instance of TwitterAPI using tokens.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    
def write_response_file_func(write_response_file, listOfDicts):
    
    with open(write_response_file, 'a',encoding='utf-8') as fileWriter:
        fileWriter.write(json.dumps(listOfDicts))
        fileWriter.write("\n")

def get_max_id(tweets_list):
    ids_list = []
    for tweet in tweets_list:
         ids_list.append(tweet['id_str'])
    return sorted(ids_list)[0]


def robust_request(twitter, resource, params):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    response = twitter.request(resource, params)
    if response.status_code == 200:
        dataInJson = response.json()
        tweets_list = dataInJson['statuses']
        ids = int( get_max_id(tweets_list))
        print("Minimum ID::",ids)
        return ids,dataInJson,tweets_list
    else:
        tweets_list = dataInJson['statuses']
        write_response_file_func(write_response_file, tweets_list)
        print('Got error %s \nsleeping for 15 minutes.' % response.text)
        sys.stderr.flush()
        time.sleep(61 * 15)
 
def get_query_tweets(twitter, q):
        
    listOfDicts = []
    loop_condition = False
    total = 0
    max_ids = 0
    
    while not loop_condition:
        print("Fetching Tweets-->>")
        if not max_ids:
            #get initial tweet/search parameters to get tweets without max_id
            search = {'q': q + '-filter:retweets', 'lang':'en','count': 100}
        search = {'q': q + '-filter:retweets', 'count': 100, 'lang':'en', 'max_id': max_ids}
        resource = 'search/tweets'
        ids,dataInJson,tweets_list = robust_request(twitter,resource,search)               
        total += len(tweets_list)        
        if total >= 5000:
            loop_condition = True
        write_response_file_func(write_response_file, tweets_list)
    

    pass
    
def create_user_data(fname,user_data_file):
    
    user_ptr = open(user_data_file, 'a',encoding='utf-8')

    with open(fname, 'r',encoding='utf-8') as fp:
        for line in fp:
            tweets_list = json.loads(line)
            for tweets in tweets_list:
                tw = tweets['user']
                desc = ' '.join(t for t in str(tw['description']).split())
                user_ptr.write( tw['id_str']+ " || "+tw['screen_name']+ " || "+tw['name']+ " || "+tw['location']+ " || "+desc+ " || "+ "\n")
    user_ptr.close() 


def create_user_tweet_data(fname,tweets_data_file):
    
    tweet_ptr = open(tweets_data_file, 'a',encoding='utf-8')

    with open(fname, 'r',encoding='utf-8') as fp:
        for line in fp:
            tweets_list = json.loads(line)
            for tweets in tweets_list:
                      tw = tweets['user']
                      tweet_text = ' '.join(t for t in str(tweets['text']).split())
                      tweet_ptr.write(tw['id_str']+ " || "+tw['screen_name']+ " || "+ tweet_text+ "\n")
    tweet_ptr.close()  
    


           
def get_follow_qry(twitter, screen_name):
	""" fetches followers of user ids """
	done = False
	followers_list = []
	params = {'screen_name':screen_name, 'count': 1000}
	while not done:
		response = twitter.request('followers/ids', params)
		data = response.json()
		if response.status_code == 200:
			followers_list = data['ids']
			print("Followers (no of account he/she following) received: %s" %len(followers_list))
			done = True
		else:
			print('Got error %s \nsleeping for 15 minutes.' % response.text)
			sys.stderr.flush()
			time.sleep(61 * 15)
	return followers_list


def pick_followers(twitter):
    count = 0
    follower_dict = {}
    uniq_user_list_Counter = []
    uniq_user_list = []

    with open('tweets_data_new.txt', 'r',encoding='utf-8') as fp:
        for line in fp:
            user = line.split(' || ')[1]
            uniq_user_list.append(user)
            uniq_user_list_Counter = Counter(elem for elem in uniq_user_list).most_common(10)
                
    #print('The top users dictionary is:::',uniq_user_list_Counter)
    final_screens = []
    for user_tuple in uniq_user_list_Counter:
        final_screens.append(user_tuple[0])
    #print('The top users LISTTT is:::',final_screens)

    for screen_name in final_screens:
        follow_ids_list = get_follow_qry(twitter, screen_name)
        if follow_ids_list:
            count += 1		
            follower_dict[screen_name] = follow_ids_list
            #write_follower_file('followers1.json', follower_dict)            
            if count >= 6 :
                  break
        
    #print('data in dict--',follower_dict.keys())
    #print('data in dict-->',follower_dict.values())
    
    #return follower_dict
    write_response_file_func('followers1_new.json', follower_dict)
    
def write_graph_edges(file_name,order,edges):
    
    with open(file_name, 'a',encoding='utf-8') as fileWriter:
        fileWriter.write(order)
        fileWriter.write(edges)
        fileWriter.write("\n")

def main():
    twitter = get_twitter()
    print('Twitter Connection Established')
    #q = '#Obamacare'
    q = '@HillaryClinton'
    get_query_tweets(twitter,q)
    create_user_data(write_response_file,user_data_file)
    create_user_tweet_data(write_response_file,tweets_data_file)
    pick_followers(twitter)
    get_total_users(user_data_file,tweets_data_file)
    

if __name__ == '__main__':
    main()

