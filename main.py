import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import sklearn
import geonames
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
#SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.neighbors import DistanceMetric

# Uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)





if __name__ == "__main__":

    network = np.loadtxt('graph.txt').astype(int)
    training_crude=np.asarray(open('posts_train.txt',"r").readlines())[1:]
    training_data=np.zeros((training_crude.shape[0]-1,7))
    for i in range(training_crude.shape[0]-1):
        training_data[i]=training_crude[i].split(",",-1)

    train_y=training_data[:,4:6]
    train_id=training_data[:,0]
    train_hour1=training_data[:,1]
    train_hour2=training_data[:,2]
    train_hour3=training_data[:,3]
    train_lon=training_data[:,4]
    train_lat=training_data[:,5]
    train_posts=training_data[:,6]

    
    
    
    
    geonames_client = geonames.GeonamesClient('demo')
    geonames_result = geonames_client.find_timezone({'lat': 48.871236, 'lng': 2.77928})
    print geonames_result['timezoneId']
 
    
