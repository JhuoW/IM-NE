__author__ = 'http://docs.python.org/2/library/heapq.html#priority-queue-implementation-notes'

import itertools
from heapq import *

# 优先队列
'''
堆是一种特殊的树形数据结构，每个节点都有一个值，通常我们所说的堆的数据结构指的是二叉树。
堆的特点是根节点的值最大（或者最小），而且根节点的两个孩子也能与孩子节点组成子树，亦然称之为堆。 

堆的主要操作是插入和删除最小元素(元素值本身为优先级键值，小元素享有高优先级)
在插入或者删除操作之后，我们必须保持该实现应有的性质: 1. 完全二叉树 2. 每个节点值都小于或等于它的子节点
'''


class PriorityQueue(object):
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap 按堆排列项的列表
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count  序列数

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:  # 如果v 在entry_finder中 那么移除v
            self.remove_task(task)
        count = next(self.counter)  # 从0开始往后...0,1,2,...
        entry = [priority, count, task]  # [0,0,v] count:0,1,2...
        self.entry_finder[task] = entry
        heappush(self.pq, entry)  # self.pq是堆， heappush是把entry放入self.pq中

    def remove_task(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)  # 删除entry_finder 中的key=v的value，返回为给定的key对应的value
        entry[-1] = self.REMOVED  # [priority, count, <removed-task>] 这是要删除的值

    def pop_item(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)  # heappop 返回 pq中的最小项 即最低优先度的节点
            if task is not self.REMOVED:  # 如果该节点没有被remove
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def __str__(self):
        return str([entry for entry in self.pq if entry[2] != self.REMOVED])


pq = PriorityQueue()
pq.add_task(pq.REMOVED, -100)
pq.add_task(1, -75)
pq.add_task(2, -50)
pq.add_task(pq.REMOVED, -25)
if __name__ == '_main__':
    console = []
