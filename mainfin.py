import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import dill as pickle
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC 
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.neighbors import DistanceMetric
from collections import defaultdict
from scipy import spatial
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge


# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=DeprecationWarning)

def get_cluster_info(data):
    cluster=AgglomerativeClustering()
    cluster.fit(data)

    return cluster.labels_

    #return cluster.cluster_centers_,cluster.labels_



def closeness(id1, id2):
    score = 0
    intersection = set(network_dict[id1]) & set(network_dict[id2])
    for i in intersection:
        score += 1 / len(network_dict[i])
    return score+0.01

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

def posting_pattern_lifting(h1,h2,h3):
    h1plus2=h1+h2
    h1plus3=h1+h3
    h2plus3=h2+h3
    h1minus2=h1-h2
    h1minus3=h1-h3
    h2minus3=h2-h3
    h1midnight=12-h1
    h2midnight=12-h2
    h3midnight=12-h3

    print(type(np.array([h1])))
    X_train_lifted=np.concatenate((np.array([h1]).T,
                                   np.array([h2]).T,
                                   np.array([h3]).T,
                                   np.array([h1plus2]).T,
                                   np.array([h1plus3]).T,
                                   np.array([h2plus3]).T,
                                   np.array([h1minus2]).T,
                                   np.array([h1minus2]).T,
                                   np.array([h2minus3]).T,
                                   np.array([h1midnight]).T,
                                   np.array([h2midnight]).T,
                                   np.array([h3midnight]).T
                                    ), axis=1)
    return X_train_lifted

def continent_classification(h1_tr,h2_tr,h3_tr, y_train, h1_te,h2_te,h3_te):
    X_train=posting_pattern_lifting(h1_tr,h2_tr,h3_tr)
    X_test=posting_pattern_lifting(h1_te,h2_te,h3_te)
    randomForest = RandomForestClassifier(n_estimators=100,
                                          criterion="entropy",
                                          max_depth=10,
                                          max_features=6,n_jobs=-1)

    randomForest.fit(X_train,y_train)
    return randomForest.predict(X_test)


if __name__ == "__main__":

### Load in data and preprocess
    network_crude = np.loadtxt('graph.txt').astype(int)
    network_dict = defaultdict(list)
    train_dict=defaultdict(list)

    training_crude=np.asarray(open('posts_train.txt',"r").readlines())[1:]
    test_crude=np.asarray(open('posts_test.txt','r').readlines())[1:]
    training_data=np.zeros((training_crude.shape[0]-1,7))
    test_data=np.zeros((test_crude.shape[0],5))

    for i in range(training_crude.shape[0]-1):
        training_data[i]=training_crude[i].split(",",-1)
    for j in range(test_crude.shape[0]):
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
    # prediction_result=np.array([[0]*2]*test_data.shape[0])

    for i in range(49812):
        train_dict[train_id[i]]=training_data_all_else[i]
    [network_dict[a].append(b) for a, b in network_crude]

    print("143")

###Learners(MLP Neural Network)
    #clf=MLPRegressor(hidden_layer_sizes=(100,3),activation='logistic',solver='adam')
    #clf.fit(training_data[:,1:4],training_data[:,4:6])


###Learners(Adaboosting with MLP Neural Network)

    # clf_boost_lat = AdaBoostRegressor(base_estimator=MLPRegressor(hidden_layer_sizes=(100,3),activation='logistic',solver='adam'),n_estimators=5,learning_rate=0.3,loss='square')
    # clf_boost_lon = AdaBoostRegressor(base_estimator=MLPRegressor(hidden_layer_sizes=(100, 3), activation='logistic', solver='adam'), n_estimators=5,learning_rate=0.3, loss='square')

    #clf_boost_lat.fit(training_data[:,1:4],training_data[:,4])
    #clf_boost_lon.fit(training_data[:,1:4],training_data[:,5])

###Adaboost Predictions

    # for i in range(test_data.shape[0]):
    #     closeness_vector=np.array([0]*test_data.shape[0])
    #     for j in range(test_data.shape[0]):
    #         closeness_vector[j]=closeness(i,j)
    #
    #
    # prediction_lat=clf_boost_lat.predict(test_data[:,1:4])
    # prediction_lon=clf_boost_lon.predict(test_data[:,1:4])
    # real_test_id=test_id.astype(np.int32)
    # final_result=np.concatenate((real_test_id,prediction_lat,prediction_lon),axis=0).reshape(1000,3,order='F').tolist()
    #
    # np.savetxt("answer1.csv",final_result,fmt=['% 4d','%1.3f','%1.3f'],delimiter=",")

###Learners(Forward Feeding)
    # error_term=np.concatenate((prediction_lat,prediction_lon),axis=0).reshape(1000,2,order='F').tolist()-training_data[:,4:6]
    # knn_learn= KNeighborsRegressor(weights=error_term)
    # knn_learn.fit(training_data[:,6],error_term)
    # final_prediction=knn_learn.predict()



### Find number of friends for each user
    # max=len(network_dict[1])
    # print(type(network_dict[1]))
    # for a in range(1,len(network_dict)):
    #     if len(network_dict[a]) > max:
    #         max=len(network_dict[a])
    #
    # friend_count=np.array([0]*(max+1))
    # print("len",len(network_dict))
    # for b in range(1,len(network_dict)):
    #     friend_count[len(network_dict[b])]+=1
    # f=0
    # sum=0
    # for element in friend_count:
    #     f+=1
    #     sum+=element
    #     print(element)
    #     if f>20:
    #         break
    # print("less than 20 friends:",sum)
    # # for i in range(len(network)):
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

###Scatterplot of users' longitude and latitude
    # count=0
    # fig = plt.figure()
    # ax=fig.add_subplot(1,1,1)
    # ax.set_title("Lon_Lat_Graph")
    # ax.set_xlabel("Longitude")
    # ax.set_ylabel("Latitude")

    # data=training_data[training_data[e][6]>100 for e in range(training_data.shape[0]) ]
    # class_selector = training_data[:, 6] > 100
    # if class_selector.any():
    #     data = training_data[class_selector]
    #
    # us_selector_all = np.logical_and(
    #                         np.logical_and(training_data[:,4]>25,training_data[:,4]<49),
    #                         np.logical_and(training_data[:,5]>-130,training_data[:,5]<-70))
    #
    # eu_selector_all = np.logical_and(
    #                         np.logical_and(training_data[:,4]>36,training_data[:,4]<72),
    #                         np.logical_and(training_data[:,5]>-9,training_data[:,5]<66))
    #
    # us_selector_active = np.logical_and(
    #                         np.logical_and(data[:,4]>25,data[:,4]<49),
    #                         np.logical_and(data[:,5]>-130,data[:,5]<-70))
    #
    # eu_selector_active = np.logical_and(
    #                         np.logical_and(data[:,4]>36,data[:,4]<72),
    #                         np.logical_and(data[:,5]>-9,data[:,5]<66))
    #
    #
    #
    #
    #
    #
    # print("US Active User: "+str(sum(us_selector_active)))
    # print("EU Active User: "+str(sum(eu_selector_active)))
    # print("All Active User: "+str(sum(class_selector)))
    # print("US All User: "+str(sum(us_selector_all)))
    # print("EU All User: "+str(sum(eu_selector_all)))
    #
    #


### Clustering
    train_continents=np.array([0] * 8)
    print("before clustering")
    #train_continents=get_cluster_info(training_data[:, 4:6])
    train_continents= np.random.randint(4, size=len(train_hour1)) #dummy code. Please comment out prior to deployment
    print("finish clustering")

###Predict the cluster labels of test data
    test_continents=continent_classification(train_hour1, train_hour2, train_hour3,
                                              train_continents,
                                              test_hour1, test_hour2, test_hour3)

# ###One Hot Encoding of Categories
#     #1. Encoder
#     enc = OneHotEncoder(c)
#
#     # 2. FIT & Transform
#     enc.fit([test_continents])
#     test_continents_OHC = enc.transform([test_continents]).toarray()
#     print(test_continents)
#     print(test_continents_OHC)
#     print(test_continents_OHC.shape)
#     enc = OneHotEncoder(categories=4)
#     enc.fit([train_continents])
#     train_continents_OHC =enc.transform([train_continents]).toarray()
#     print(train_continents_OHC.shape)

    test_continents_OHC = to_categorical(test_continents)
    train_continents_OHC = to_categorical(train_continents)

    test_pred = np.zeros(shape=len(train_id))
    test_pred_index = 0
    for test_point in test_data:
        id=test_point[0]
        weight=np.zeros(shape=len(train_id))
        weight_index = 0
        for train_id_single in train_id:
            weight[weight_index]=closeness(id,train_id_single)
            weight_index += 1
        clf_boost_multi = MultiOutputRegressor(AdaBoostRegressor(
            base_estimator=KernelRidge(kernel=closeness),
            n_estimators=5, learning_rate=0.3, loss='square'),-1)
        #Please zip/concat X_train, and X_test
        Xtrain=np.concatenate((training_data[:,1:4],[training_data[:,6]],train_continents_OHC),axis=1)
        Xtest=np.concatenate((test_data[:,1:5],train_continents_OHC),axis=1)
        ytrain=np.concatenate((training_data[:,4:6]))

        clf_boost_multi.fit(Xtrain,ytrain)
        clf_boost_multi.predict()
    #print("cluster_center",cluster_center)
    # max=np.array([0]*labels.max())
    # for e in labels:
    #     max[e-1]+=1
    # for a in max:
    #     print(a)

    #ax = plt.scatter(training_data[:, 5], training_data[:, 4],c=labels)
    #plt.show()

### 44666666666666668888

    
    
#233