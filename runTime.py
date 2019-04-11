import matplotlib.pyplot as plt

name_list = ['HEP-TH', 'Cora', 'Zhihu']
num_NEIM = [0.11, 0.13, 3.2]
num_CCHEU = [1.2, 2, 30]
num_DD = [1.5,1.8,69]

x = list(range(len(num_NEIM)))
total_width, n = 0.5, 2
width = total_width / n

plt.bar(x, num_NEIM, width=width, label='NE-IM', fc='r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_CCHEU, width=width, label='CC-Heuristic', tick_label=name_list, fc='b')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_DD, width=width, label='Degree Discount', tick_label=name_list, fc='black')

plt.legend()
plt.show()
