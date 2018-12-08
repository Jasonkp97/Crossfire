import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.neighbors import DistanceMetric
from collections import defaultdict
from scipy import spatial


# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def closeness(id1, id2):
    score = 0
    intersection = set(network_dict[id1]) & set(network_dict[id2])
    for i in intersection:
        score += 1 / len(network_dict[i])
    return score

def is_Expat(id):
    count=0
    errorcount=0

    for friend in network_dict[id]:
        try:
            if (spatial.distance.euclidean([train_dict[id][3],train_dict[id][4]],
                                           [train_dict[friend][3],train_dict[friend][4]]) >= 10
            and
                train_dict[id][3]*train_dict[id][4] != 0
            and
                train_dict[friend][3]*train_dict[friend][4] != 0
                ):
                count+=1
        except:
            errorcount+=1

        if (spatial.distance.euclidean([train_lat[id],train_lon[id]],[train_lat[friend],train_lon[friend]])>=2):
            count+=1
    if count>=(len(network_dict[id]) / 2):
        return True
    return False



if __name__ == "__main__":


### Load in data and preprocess
    network_crude = np.loadtxt('graph.txt').astype(int)
    network_dict = defaultdict(list)
    train_dict=defaultdict(list)

    training_crude=np.asarray(open('posts_train.txt',"r").readlines())[1:]
    test_crude=np.asarray(open('posts_test.txt','r').readlines())[1:]
    training_data=np.zeros((training_crude.shape[0]-1,7))
    test_data=np.zeros((test_crude.shape[0]-1,5))

    for i in range(training_crude.shape[0]-1):
        training_data[i]=training_crude[i].split(",",-1)
    for j in range(test_crude.shape[0]-1):
        test_data[j]=test_crude[j].split(",",-1)
    
    train_y=training_data[:,4:6]
    train_id=training_data[:,0]
    train_hour1=training_data[:,1]
    train_hour2=training_data[:,2]
    train_hour3=training_data[:,3]
    train_lat=training_data[:,4]
    train_lon=training_data[:,5]
    train_posts=training_data[:,6]

    test_id=test_data[:,0]
    test_hour1=test_data[:,1]
    test_hour2=test_data[:,2]
    test_hour3=test_data[:,3]
    test_posts=test_data[:,4]


    training_data_all_else=training_data[:,1:7]


    for i in range(49812):
        train_dict[train_id[i]]=training_data_all_else[i]
    [network_dict[a].append(b) for a, b in network_crude]

    print(closeness(2172,233))

    # for i in range(len(network)):
    #      if (closeness(5931, i) != 0):
    #          print("closeness of "+str(5931)+" and "+str(i)+" is: "+ str(closeness(5931,i)))

    # print(len(train_posts))
    # print(len(train_lat)
    # expat_count=0
    # for i in range(len(network_dict)):
    #     expat_count+=is_Expat(i)
    # print("expat count: "+str(expat_count))

    # friend_counts=np.array([0]*len(network_dict))
    # for i in range(len(network_dict)):
    #     friend_counts=friend_counts.append(len(network_dict[i]))
    #     print(len(network_dict[i]))
    # print(friend_counts.sort())


### Number of anti-socials and number of them in the test set

    # feizhai=np.array([0]*221)
    # count=0
    # for i in range(network.shape[0]-1):
    #     if(network[i+1][0]-network[i][0]>1):
    #         for j in range(network[i][0]+1,network[i+1][0]):
    #             feizhai[count]=j
    #             count+=1
    # count1=0
    # for i in feizhai:
    #     if i in test_crude:
    #         print(i)
    #         count1+=1
    # print(count)
    # print(count1)

### Scatterplot of users' longitude and latitude

    #fig = plt.figure()
    # ax=fig.add_subplot(1,1,1)
    # ax.set_title("Lon_Lat_Graph")
    # ax.set_xlabel("Longitude")
    # ax.set_ylabel("Latitude")
    #
    # ax=plt.scatter(train_lon,train_lat)
    #
    # plt.show()



### 44666666666666668888

    
    
