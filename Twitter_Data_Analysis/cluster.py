from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time, json
from TwitterAPI import TwitterAPI

consumer_key = 'Sdts6KDxe0lRrJv58xAW4reRA'
consumer_secret = 'qRps0KdoqWU8Y6BUntSwHwweA2xTrZf5nMrAjVIcdRZiF5aQzM'
access_token = '771103127063916544-NxseKPMKR7bZFEn3xptQK61weCECEmV'
access_token_secret = 'itLVtiuZ2aR35eBc9ohhiFTC7kaocyfADpxak8GibSiOQ'

file_name = 'Log.txt'

# This method is done for you.
def get_twitter():
    """Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
        An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)



def girvan_newman(G, depth=0):

    if G.order() == 1:
        return [G.nodes()]

    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    components = [c for c in nx.connected_component_subgraphs(G)]
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]
    return components

def get_max_followers(followers):
    follow_list = []
    follow_dict = dict(followers)
    for key_name, values_list in follow_dict.items():
        if values_list > 1:
            follow_list.append(key_name)
    return follow_list
    
def get_all_followers(fname):
    followers_ids = set()
    followers = Counter()
    follow_dict = {}
    follow_list_all = []
    follow_list = []
    
    with open(fname, 'r',encoding='utf-8') as fp:
        for line in fp:
            data = json.loads(line)
            #print("values::",data)
            for key,value in data.items():
                followers.update(value)
                for val in value:
                    followers_ids.add(val)
     
    follow_list = get_max_followers(followers)
    #print("Followers_counter::",followers)
    #print("ALL::",len(list(followers_ids)))    
    follow_list_all = (list(followers_ids))[:40] 
     
    return follow_list,follow_list_all
    

def follows_target_check(twitter,top_followers_list):
    """
    check whether the top users follows hillary or not if they follow create their 
    list for further community dectectiion
    """
    yes_follow_list = []
    not_follow_list = []
    following_dict = {}
    target = 'HillaryClinton'
    
    for user in top_followers_list:
        params = {'source_id':user, 'target_screen_name':target}
        response = twitter.request('friendships/show', params)
        data = response.json()
        #print("DATAAA::",data)
        if response.status_code == 200:
            #print("IN BIGG IFFFFF:::")
            following_dict = data['relationship']['source']
            #print("following_dict::",following_dict)
            check = following_dict['following']
            #print("check::",check)
            if check:
                #print("IN IFFFFF:::")
                yes_follow_list.append(user)
                
            else:
                #print("IN ELSEEEE:::")
                not_follow_list.append(user)
                
        else:
            print('Got error %s \nsleeping for 15 minutes.' % response.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
    
    print("YES_LIST:::",yes_follow_list) 
    print("NO_LIST:::",not_follow_list)           
    return not_follow_list
    

def get_edges(top_followers_list):

    user_dict = {}
    with open('tweets_data_new.txt',encoding='utf-8') as fp:
        for line in fp:
            data = line.split(' || ')
            user_dict[data[1]] = data[0]
        
    edges_dict = {}
    print("creating edges file...")
    top_list = set(top_followers_list)
    with open('followers1_new.json', 'r',encoding='utf-8') as fp:
        for line in fp:
            data = json.loads(line)
            for screen_name, follow_list in data.items():
                common_list = set(follow_list).intersection(top_list)
                if common_list:
                    edges_dict[str(user_dict[screen_name])] = list(common_list)[:20]
                else:
                    # no common. consider random 10 followers
                    edges_dict[str(user_dict[screen_name])] = follow_list[:20]
    # now create edges.txt
    with open('edges_new.txt', 'w',encoding='utf-8') as fp:
        for k, v in edges_dict.items():
            for ids in v:
                fp.write(str(k)+ "\t"+ str(ids)+ "\n")
    print("edges file created....")

def draw_network(graph, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).
    """
    plt.figure(figsize=(12,12))
    nx.draw_networkx(graph, with_labels=False, alpha=.5, width=.1, node_size=100)
    plt.axis("off")
    plt.savefig(filename, format="PNG")

def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
    A networkx undirected graph.
    """
    return nx.read_edgelist('edges_new.txt', delimiter='\t')

def write_result(file_name, parameter):
    
    with open(file_name, 'a',encoding='utf-8') as fileWriter:
        fileWriter.write(parameter)
        fileWriter.write("\n")
        
def main():
    list_avg = []
    twitter = get_twitter()
    print('Twitter Connection Established')
    top_followers_list,list1 = get_all_followers('followers1_new.json')
    print("Most common followers are: %s" %len(top_followers_list))
    get_edges(top_followers_list)
    graph = read_graph()
    print('graph has %d nodes and %d edges' %(graph.order(), graph.number_of_edges()))
    draw_network(graph, 'network_new.png')
    i = 0
    while i < 4:
        return_component = girvan_newman(graph)
        i = i + 1
        graph = return_component[0]
        communities = len(return_component)
        write_result(file_name, communities)
        print("total clusters formed: %s" %len(return_component))
        for i in range(len(return_component)):
            print("cluster[%s] has %s nodes:" %(i, return_component[i].order()))
            list_avg.append(return_component[i].order())
   
    lent = len(list_avg)
    a = sum(list_avg)
    avg = a / lent
    write_result(file_name, avg) 


if __name__ == '__main__':
    main()