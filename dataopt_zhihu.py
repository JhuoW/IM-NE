import numpy as np
import os
from tqdm import tqdm
import pickle
import networkx as nx
import random
from IC import avgSize, runIC
import matplotlib.pyplot as plt
from CCHeuristic import CC_heuristic
from copy import deepcopy
from generalGreedy import generalGreedy
from degreeDiscount import degreeDiscountIC
from newGreedyIC import newGreedyIC
from Harvester import Harvester
from degreeHeuristic import degreeHeuristic, degreeHeuristic2
from matplotlib.font_manager import FontProperties
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

plt.rcParams['font.sans-serif'] = ['SimHei']

filename = 'zhihu/embedding.txt'
picklefile = 'zhihu/result.pkl'
datafile = 'zhihu/graph.txt'

S = []


def readEmbedding(filename):
    f = open(filename, 'r+')
    node_num, dim = [int(x) for x in f.readline().strip().split()]
    vectors = dict()
    while 1:
        l = f.readline()
        if l == '':
            break
        line = l.strip().split(' ')
        vectors[line[0]] = [float(x) for x in line[1:]]
    f.close()
    return node_num, dim, vectors


def Multivalency(G):
    Ep = dict()
    probabilities = [.03]
    if isinstance(G, nx.Graph):
        for e in G.edges():
            p = random.choice(probabilities)
            Ep[(e[0], e[1])] = p
            Ep[(e[1], e[0])] = p
    elif isinstance(G, nx.DiGraph):
        for e in G.out_edges():
            p = random.choice(probabilities)
            Ep[(e[0], e[1])] = p
    else:
        raise NotImplementedError
    return Ep


def calculate(vectors, node_num):
    result = dict()
    for i in tqdm(vectors.keys()):
        res = 0
        for j in vectors.keys():
            dist = np.linalg.norm(np.array(vectors[i]) - np.array(vectors[j]))  # l2 norm
            res += dist
        result[int(i)] = res / (node_num - 1)
    # return result   {440: 3.548125420654588, 50: 3.2733370799993415, 103: 3.379448188603178, ...}
    with open(picklefile, 'wb') as f:
        pickle.dump(result, f)
        f.close()


def getSeedSet(k, result):
    # 返回result中value值最小的key
    T = deepcopy(result)
    for i in tqdm(range(k)):
        key = min(T, key=T.get)
        S.insert(0, key)
        T.pop(key)
    print(S)  # [437, 704, 619, 449, 789]
    return S


def getNEIMData(G, maxk, p, result):
    data = dict()
    for i in range(1, maxk + 1):
        S = getSeedSet(i, result)
        size = avgSize(G, S, p, 200)
        data[i] = size
    return data


def getCCData(G, maxk, p):
    data = dict()
    for i in range(1, maxk + 1):
        S = CC_heuristic(G, i, p)
        print(S)
        M = []
        for j in S:
            M.insert(-1, j[0])
        size = avgSize(G, M, p, 200)
        data[i] = size
    return data


def getGreedyData(G, maxk, p):
    data = dict()
    for i in range(1, maxk + 1):
        S = generalGreedy(G, i, p)
        size = avgSize(G, S, p, 200)
        data[i] = size
    return data


def readGreedy():
    f = open('resu.txt', 'r+')
    line = f.readline()
    line = f.readline()
    line = line.strip()
    return eval(line)


def getDDData(G, maxk, p):
    data = dict()
    for i in range(1, maxk + 1):
        S = degreeDiscountIC(G, i, p)
        size = avgSize(G, S, p, 200)
        data[i] = size
    return data


def getNewGreedyIC(G, maxk, p):
    data = dict()
    for i in range(1, maxk + 1):
        S = newGreedyIC(G, i, p)
        size = avgSize(G, S, p, 1)
        data[i] = size
    return data


def getHarvesterData(G, maxk, Ep):
    data = dict()
    for i in range(1, maxk + 1):
        S = Harvester(G, i, Ep, 100)
        size = avgSize(G, S, .03, 200)
        data[i] = size
    return data


def getDegreeHeu(G, maxk, p):
    data = dict()
    for i in range(1, maxk + 1):
        S = degreeHeuristic2(G, i, p)
        size = avgSize(G, S, p, 200)
        data[i] = size
    return data


if __name__ == '__main__':
    node_num, dim, vectors = readEmbedding(filename)
    if not os.path.exists(picklefile):
        calculate(vectors, node_num)
        with open(picklefile, 'rb') as f:
            result = pickle.load(f)
            f.close()
    else:
        with open(picklefile, 'rb') as f:
            result = pickle.load(f)
            f.close()
    print(result)
    G = nx.Graph()
    with open(datafile, 'r') as f:
        for line in f.readlines():
            u, v = map(int, line.split())
            G.add_edge(u, v)
            G[u][v]['weight'] = 1.0
    # NE-IM
    data = getNEIMData(G, 50, .03, result)
    # CCHeuristic
    data2 = getCCData(G, 50, .03)

    # generalGreedy
    # data3 = getGreedyData(G, 50, .01)
    # data3 = readGreedy()

    # Degree Discount
    data4 = getDDData(G, 50, .02)

    # Representive Nodes
    # data5 = getNewGreedyIC(G, 50, .03)

    # Harvester
    Ep = Multivalency(G)
    data6 = getHarvesterData(G, 50, Ep)

    # Degree
    data7 = getDegreeHeu(G, 50, .01)

    fig = plt.figure()

    d1plt, = plt.plot(data.keys(), data.values(), 'r--')
    d2plt, = plt.plot(data2.keys(), data2.values(), 'g--')
    # d3plt, = plt.plot(data3.keys(), data3.values(), 'b--')
    d4plt, = plt.plot(data4.keys(), data4.values(), color='black', linestyle='--')
    d5plt, = plt.plot(data6.keys(), data6.values(), color='purple', linestyle='--')
    d6plt, = plt.plot(data7.keys(), data7.values(), color='blue', linestyle='--')

    plt.xlabel("种子节点数k",fontsize=16)
    plt.ylabel("影响范围",fontsize=16)
    plt.legend(handles=[d1plt, d2plt, d4plt, d5plt, d6plt],
               labels=["NE-IM", "CC-Heuristic", "Degree Discount", "Harvester", "Degree"],
               loc=9)
    fig.savefig('zhihu/influence_vs_k7.png', dpi=fig.dpi)
    #

    # data6 = getDegreeHeu(G, 50, .03)
    # fig = plt.figure()
    # d1plt, = plt.plot(data6.keys(), data6.values(), 'r--')
    # plt.xlabel("Seed set size k")
    # plt.ylabel("Influence spread")
    # plt.legend(handles=[d1plt], labels=["DegreeHeu"], loc=9)
    # fig.savefig('influence_vs_k4.png', dpi=fig.dpi)
