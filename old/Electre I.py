#%%
import random
import networkx as nx
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(precision = 2)

def readtxt(path , separator = ',' , decimal = '.'):
    F = open(path, 'r' , encoding='utf8')
    data = []
    data_dict = {}
    for line in F:
        if line != '\n':
            data.append(line.replace(decimal, '.').replace('\n', '').split(separator))
    F.close()
    criteria = data.pop(0)[1:]
    weights = [float(cell) for cell in data.pop(0)[1:]]
    for i, line in enumerate(data):
        data_dict[i] = [float(cell) for cell in line[1:]]
    return  criteria, weights, data_dict

def normalize(data, amin = 0, amax = 1):
    max_per_col = [max(col) for col in np.matrix([row for row in data.values()]).T.tolist()]
    min_per_col = [min(col) for col in np.matrix([row for row in data.values()]).T.tolist()]  
    normalized_data = {}
    for key in data:
        normalized_row = []
        for i, cell in enumerate(data[key]):
            normalized_row.append( amin + (amax - amin) * (cell - min_per_col[i]) / (max_per_col[i] - min_per_col[i]) )
        normalized_data[key] = normalized_row
    return normalized_data

def get_delta(data_dict):
    return max([max(col) - min(col) for col in np.matrix([row for row in data_dict.values()]).T.tolist()])

criteria, weights, data_dict = readtxt('exeli.csv')

delta = get_delta(data_dict)

concordance_matrix = np.ones((len(data_dict), len(data_dict)))
for pair in itertools.combinations(data_dict, 2):
    # print(pair)
    concordance_f_s = []
    concordance_s_f = []
    for i, w in enumerate(weights):
        if data_dict[pair[0]][i] > data_dict[pair[1]][i]:
            concordance_f_s.append(w)
        elif data_dict[pair[0]][i] < data_dict[pair[1]][i]:
            concordance_s_f.append(w)
        else:
            concordance_f_s.append(w)
            concordance_s_f.append(w)
    concordance_matrix[pair[0], pair[1]] = np.sum(concordance_f_s)
    concordance_matrix[pair[1], pair[0]] = np.sum(concordance_s_f)


discordance_matrix = np.zeros((len(data_dict), len(data_dict)))
for pair in itertools.combinations(data_dict, 2):
    discordance_f_s = []
    discordance_s_f = []
    for i, w in enumerate(weights):
        discordance_f_s.append((data_dict[pair[1]][i] - data_dict[pair[0]][i]) / delta)
        discordance_s_f.append((data_dict[pair[0]][i] - data_dict[pair[1]][i]) / delta)
    discordance_f_s.append(0)
    discordance_s_f.append(0)
    discordance_matrix[pair[0], pair[1]] = max(discordance_f_s)
    discordance_matrix[pair[1], pair[0]] = max(discordance_s_f)

# SET P and Q HERE
p = 1
q = 0.4

G = nx.DiGraph()
for n in data_dict:
    G.add_node(n)
dominance_matrix = np.zeros((len(data_dict), len(data_dict)))
for pair in itertools.permutations(data_dict, 2):
    if concordance_matrix[pair[0], pair[1]] >= p and discordance_matrix[pair[0], pair[1]] <= q:
        dominance_matrix[pair[0], pair[1]] = 1
        G.add_edge(pair[0], pair[1])


fig = plt.figure()
plt.title("P = " + str(p) + '      ' + 'Q = ' + str(q) )
nx.draw_shell(G, with_labels = True, node_size=1000, node_color='g')
fig.savefig('teste.png')
# plt.show()