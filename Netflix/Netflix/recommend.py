import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from SoftImpute import SoftImpute
import csv

#Loads and parses train file 
def loadTrainData():
    with open('train.csv', 'rt') as file:
        data = csv.reader(file)       
        dataset = list(data)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = int(dataset[x][y])
    return np.array([el[0:3] for el in dataset])

#Loads and parses test file 
def loadTestData():
    with open('test.csv', 'rt') as file:
        data = csv.reader(file)       
        dataset = list(data)
        for x in range(len(dataset)):
            for y in range(2):
                dataset[x][y] = int(dataset[x][y])
    return [el[0:2] for el in dataset]

def calcEdges(data):
    n = len(data)        
    usersDic = {}
    usersId = 0
    moviesDic = {}
    moviesId = 0
    for i in range(n):
        r = data[i]
        if r[0] not in moviesDic:
            moviesDic[r[0]] = moviesId
            moviesId += 1
        if r[1] not in usersDic:
            usersDic[r[1]] = usersId
            usersId += 1
    E = np.zeros((moviesId, usersId))
    #E = np.full((moviesId, usersId), np.nan)
    for i in range(n):
        user = usersDic[data[i][1]]
        movie = moviesDic[data[i][0]]
        E[movie, user] = data[i][2]
    estimator = Imputer(0, strategy='mean')
    #estimator = SoftImpute()    
    #estimator.fit(E)
    #E = estimator.predict(E)
    E = estimator.fit_transform(E)
    return E, usersDic, moviesDic

def eval(userAverages, userClusters, usersDic, movieAverages, movieClusters, moviesDic, test):    
    #Determine the error rate
    mse = 0.0
    hits = 0
    for i in range(len(test)):
        if test[i][0] in moviesDic:
            mId = moviesDic[test[i][0]]
            cluster = movieClusters[mId]
            movieRating = movieAverages[cluster]
        else:
            movieRating = 4
        if test[i][1] in usersDic:
            uId = usersDic[test[i][1]]
            cluster = userClusters[uId]
            userRating = userAverages[cluster]
        else:
            userRating = 4        
        realRating = test[i][2]
        #50% each yields the best performance
        rating = 0.5*userRating + 0.5*movieRating        
        if abs(rating - realRating) < 0.5:
            hits += 1
        diff = realRating - rating
        mse += diff * diff
    return mse / float(len(test)), hits

def findAverages(clusters, dic, ind, k, data):
    res = [0] * k
    totals = [0] * k
    for i in range(len(data)):
        mId = dic[data[i][ind]]
        cluster = clusters[mId]
        res[cluster] += data[i][2]
        totals[cluster] += 1
    for i in range(k):
        res[i] /= float(totals[i])
    return res

def calcLaplacian(lowEdges):
    W = cosine_similarity(lowEdges, lowEdges)
    D = np.sum(W, 1)
    D = np.eye(len(D)) * D
    L = D - W
    return L

def clusterize(edges, k):
    #L = calcLaplacian(edges)
    #vals, vecs = np.linalg.eig(L)
    #vecs = vecs.real;        
    kmeans = KMeans(n_clusters=k)
    return kmeans.fit_predict(edges)

def predict(userAverages, userClusters, usersDic, movieAverages, movieClusters, moviesDic, test):
    predictions = [0] * len(test)
    for i in range(len(test)):
        if test[i][0] in moviesDic:
            mId = moviesDic[data[i][0]]
            cluster = movieClusters[mId]
            movieRating = movieAverages[cluster]
        else:
            movieRating = 4
        if test[i][1] in usersDic:
            uId = usersDic[data[i][1]]
            cluster = userClusters[uId]
            userRating = userAverages[cluster]
        else:
            userRating = 4        
        #50% each yields the best performance
        predictions[i] = 0.5*userRating + 0.5*movieRating
    return predictions

data = loadTrainData()
training = True
if training:
    # Do KFold validation to optimize for # of clusters
    n = len(data)
    kf = KFold(n, n_folds=10)
    for train_index, test_index in kf:        
        trainData = data[train_index]
        testData = data[test_index]    
        movieEdges, usersDic, moviesDic = calcEdges(trainData)
        userEdges = movieEdges.T    
        #Increasing k improves marginally perfomance (around 1% for k=1000) but it takes too much time to compute
        k = 50
        #Increasing n does not improve performance
        n = 25 
        est = TruncatedSVD(n_components=n)    
        lowMovieEdges = est.fit_transform(movieEdges)
        lowUserEdges = est.fit_transform(userEdges)
        movieClusters = clusterize(lowMovieEdges, k)
        userClusters = clusterize(lowUserEdges, k)
        movieAverages = findAverages(movieClusters, moviesDic, 0, k, trainData)
        userAverages = findAverages(userClusters, usersDic, 1, k, trainData)    
        mse, hits = eval(userAverages, userClusters, usersDic, movieAverages, movieClusters, moviesDic, testData)
        print "n: ", n, " - mse: ", mse, " - hits: ", hits
else:
    movieEdges, usersDic, moviesDic = calcEdges(data)
    userEdges = movieEdges.T    
    #Increasing k improves marginally perfomance (around 1% for k=1000) but it takes too much time to compute
    k = 50
    #Increasing n does not improve performance
    n = 25    
    est = TruncatedSVD(n_components=n)
    lowMovieEdges = est.fit_transform(movieEdges)
    lowUserEdges = est.fit_transform(userEdges)
    movieClusters = clusterize(lowMovieEdges, k)
    userClusters = clusterize(lowUserEdges, k)
    movieAverages = findAverages(movieClusters, moviesDic, 0, k, data)
    userAverages = findAverages(userClusters, usersDic, 1, k, data)
    test = loadTestData()
    predictions = predict(userAverages, userClusters, usersDic, movieAverages, movieClusters, moviesDic, test)
    for i in range(len(predictions)):
        print predictions[i]