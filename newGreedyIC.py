''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 2)
'''
from __future__ import division
from copy import deepcopy  # copy graph object
from random import random
import sys

sys.path.append("..")
from priorityQueue import PriorityQueue as PQ
import networkx as nx
from runIAC import avgIAC


def bfs(E, S):
    ''' Finds all vertices reachable from subset S in graph E using Breadth-First Search
    Input: E -- networkx graph object
    S -- list of initial vertices
    Output: Rs -- list of vertices reachable from S
    '''
    Rs = []
    for u in S:
        if u in E:
            if u not in Rs: Rs.append(u)
            for v in E[u].keys():
                if v not in Rs: Rs.append(v)
    return Rs


def findCCs(G, Ep):
    # remove blocked edges from graph G
    E = deepcopy(G)
    edge_rem = [e for e in E.edges() if random() < (1 - Ep) ** (E[e[0]][e[1]]['weight'])]
    # edge_rem = []
    # for e in E.edges():
    #     if random() < (1 - Ep[e]) ** (E[e[0]][e[1]]['weight']):
    #         edge_rem = edge_rem.extend(e)

    E.remove_edges_from(edge_rem)

    # initialize CC
    CCs = dict()  # each component is reflection of the number of a component to its members
    explored = dict(zip(E.nodes(), [False] * len(E)))
    c = 0
    # perform BFS to discover CC
    for node in E:
        if not explored[node]:
            c += 1
            explored[node] = True
            CCs[c] = [node]
            component = list(E[node].keys())
            for neighbor in component:
                if not explored[neighbor]:
                    explored[neighbor] = True
                    CCs[c].append(neighbor)
                    component.extend(E[neighbor].keys())
    return CCs


def newGreedyIC(G, k, Ep, R=20):
    # Ep=.01
    import time
    S = []
    for i in range(k):
        print(i)
        time2k = time.time()
        scores = {v: 0 for v in G}
        for j in range(R):
            print(j, )
            CCs = findCCs(G, Ep)
            for CC in CCs:
                for v in S:
                    if v in range(CC):
                        break
                else:  # in case CC doesn't have node from S
                    for u in range(CC):
                        scores[u] += float(CC) / R
        max_v, max_score = max(scores.items(), key=lambda dv: dv)
        S.append(max_v)
        print(time.time() - time2k)
    return S


if __name__ == "__main__":
    import time

    start = time.time()

    G = nx.read_gpickle("../graphs/hep.gpickle")
    print('Read graph G')
    print(time.time() - start)

    model = "MultiValency"

    if model == "MultiValency":
        ep_model = "range"
    elif model == "Random":
        ep_model = "random"
    elif model == "Categories":
        ep_model = "degree"

    # get propagation probabilities
    Ep = dict()
    with open("Ep_hep_%s1.txt" % ep_model, 'w+') as f:
        for line in f:
            data = line.split()
            Ep[(int(data[0]), int(data[1]))] = float(data[2])
    print(len(G.edges()))
    for i in G.edges():
        Ep[i] = float(1)
    I = 1000
    S = newGreedyIC(G, 10, Ep)
    print(S)
    print(avgIAC(G, S, Ep, I))
