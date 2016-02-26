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

def eval(averages, kmeans, vecs, test):
    #Initialize confusion matrix for 5 ratings
    cArray = [0] * 5
    cMatrix = [cArray[:] for i in range(5)]
    #Determine the error rate
    hits = 0
    for x in range(len(test)):        
        vec = vecs[test[x][0]]
        cluster = kmeans.predict(vec)
        ranking = averages[cluster] - 1
        realRanking = test[x][2] - 1
        cMatrix[realRanking][ranking] += 1
        if ranking == realRanking:
            hits += 1
    return [hits, cMatrix]

def findAverages(clusters, k, data):
    res = [0] * k
    totals = [0] * k
    print len(clusters)
    print len(data)
    for i in range(len(data)):
        cluster = clusters[data[i][0]]
        res[cluster] += data[i][2]
        totals[cluster] += 1
    for i in range(k):
        res[i] = int(round(res[i]/totals[i]))
    return res

def clusterize(kmeans, vecs, k):
    clusters = [0] * len(vecs)
    for i in range(len(vecs)):
        cluster = kmeans.predict(vecs[i])
        clusters[i] = cluster[0]
    return clusters

data = loadTrainData()
# Do KFold validation to optimize for # of clusters
n = len(data)
kf = KFold(n, n_folds=10)
for train_index, test_index in kf:        
    trainData = data[train_index]
    testData = data[test_index]
    edges = calcEdges(trainData)
    W = edges*edges.transpose()
    W.setdiag(0,0)
    D = W.sum(1).flatten().tolist()[0]
    D = sp.diags(D, 0).tocsr()
    L = D - W
    for k in range(6, 12):
        vals, vecs = scipy.sparse.linalg.eigs(L, k)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(vecs)
        clusters = clusterize(kmeans, vecs, k)
        averages = findAverages(clusters, k, trainData)
        [hits, cMatrix] = eval(averages, kmeans, vecs, testData)
        print "k: ", k, " - hits: ", hits

test = loadTestData()
