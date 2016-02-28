import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
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
    return [el[0:2] for el in dataset]

def calcEdges(data):
    n = len(data)
    nMovieIDs = max([el[0] for el in data]) + 1
    nUserIDs = max([el[1] for el in data]) + 1
    #LIL format is fast for inserting
    E = sp.lil_matrix((nMovieIDs, nUserIDs))    
    for i in range(n):
        r = data[i]
        E[r[0],r[1]] = r[2]
    #CSR format is fast for operations
    return E.tocsr()

def eval(averages, clusters, test):    
    #Determine the error rate
    mse = 0.0
    for x in range(len(test)):
        cluster = clusters[test[x][0]]
        ranking = int(round(averages[cluster]))        
        realRanking = test[x][2]
        diff = realRanking - ranking
        mse += diff * diff
    return mse / float(len(test))

def findAverages(clusters, k, data):
    res = [0] * k
    totals = [0] * k
    for i in range(len(data)):
        cluster = clusters[data[i][0]]
        res[cluster] += data[i][2]
        totals[cluster] += 1
    for i in range(k):
        res[i] /= float(totals[i])
    return res

data = loadTrainData()
# Do KFold validation to optimize for # of clusters
n = len(data)
kf = KFold(n, n_folds=10)
for train_index, test_index in kf:        
    trainData = data[train_index]
    testData = data[test_index]
    edges = calcEdges(trainData)
    #W = edges*edges.transpose()
    W = edges.transpose()*edges
    W.setdiag(0,0)
    D = W.sum(1).flatten().tolist()[0]
    D = sp.diags(D, 0).tocsr()
    L = D - W    
    for k in [5]:
        vals, vecs = scipy.sparse.linalg.eigs(L, k)
        vecs = vecs.real;        
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(edges)
        clusters = kmeans.predict(edges)        
        averages = findAverages(clusters, k, trainData)
        mse = eval(averages, clusters, testData)
        print "k: ", k, " - mse: ", mse

test = loadTestData()
