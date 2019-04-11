''' Independent cascade model for influence propagation
'''
__author__ = 'ivanovsergey'


def runIC(G, S, p=.01):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    from random import random
    T = deepcopy(S)  # copy already selected nodes 复制已选择的节点

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]:  # for neighbors of a selected node  遍历已选择节点的邻居
            if v not in T:  # if it wasn't selected yet  如果该节点不在S内
                w = G[T[i]][v]['weight']  # count the number of edges between two nodes  weight为0或1
                # random()表示一个0,1之间的随机数
                if random() <= 1 - (1 - p) ** w:  # if at least one of edges propagate influence w为0时，比不可能传播，为1是传播概率为p(对于无向图)
                    print(T[i], 'influences', v)  # 节点T[i] 影响 v
                    T.append(v)  # 将v加入激活节点中
        i += 1

    return T


def runIC2(G, S, p=.01):
    ''' Runs independent cascade model (finds levels of propagation).
    Let A0 be S. A_i is defined as activated nodes at ith step by nodes in A_(i-1).
    We call A_0, A_1, ..., A_i, ..., A_l levels of propagation.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    import random
    T = deepcopy(S)
    Acur = deepcopy(S)
    Anext = []
    i = 0
    while Acur:
        values = dict()
        for u in Acur:
            for v in G[u]:
                if v not in T:
                    w = G[u][v]['weight']
                    if random.random() < 1 - (1 - p) ** w:
                        Anext.append((v, u))
        Acur = [edge[0] for edge in Anext]
        print(i, Anext)
        i += 1
        T.extend(Acur)
        Anext = []
    return T


def avgSize(G, S, p, iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(set(runIC(G, S, p)))) / iterations
    return avg
