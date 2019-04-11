''' Implements greedy heuristic for IC model [1]

[1] -- Wei Chen et al. Efficient Influence Maximization in Social Networks (Algorithm 1)
'''
__author__ = 'ivanovsergey'
# 普通贪心算法
import sys

from priorityQueue import PriorityQueue as PQ
from IC import runIC


def generalGreedy(G, k, p=.01):
    ''' Finds initial seed set S using general greedy heuristic
    Input: G -- networkx Graph object
    k -- number of initial nodes needed
    p -- propagation probability  传播概率
    Output: S -- initial set of k nodes to propagate
    '''
    import time
    start = time.time()
    R = 20  # number of times to run Random Cascade
    S = []  # set of selected nodes
    # add node to S if achieves maximum propagation for current chosen + this node
    for i in range(k):
        s = PQ()  # priority queue
        for v in G.nodes():  # 遍历G中所有节点
            if v not in S:
                s.add_task(v, 0)  # initialize spread value  0为优先度
                for j in range(R):  # run R times Random Cascade  运行R次随机级联
                    [priority, count, task] = s.entry_finder[v]   # 获取v的优先度
                    # runIC(G, S + [v], p) 表示把S+[v]看做种子集，p为传播概率 返回Influence Spread
                    # priority - float(len(runIC(G, S + [v], p))) / R 为优先度
                    # v由于在上面已经加入pq, 所以会先执行remove_task 将v移出entry_finder 即把上面的task置为<removed-task>
                    # 此时，该方法用于更新v的优先度 如果在IC模型中 S=[v]的影响越大，那么v的优先度越小
                    s.add_task(v, priority - float(len(runIC(G, S + [v], p))) / R)  # add normalized spread value
        task, priority = s.pop_item()  # 移除并返回最低优先度的节点
        S.append(task)  # 将优先度最低的节点加入S  优先度低是因为在IC模型中扩散的很快
        print(i, k, time.time() - start)
    return S
