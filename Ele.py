

import collections
def readtxt(path , separator = ',' , decimal = '.'): # csv->header_dict, data_dict
    F = open(path, 'r' , encoding='utf8')
    data = []
    data_dict = collections.OrderedDict()
    header_dict = collections.OrderedDict()
    for line in F:
        if line != '\n':
            data.append(line.replace(decimal, '.').replace('\n', '').split(separator))
    F.close()
    i = 1 # criterium counter
    for line in data:
        if line[0] == 'C':
            header_dict['critérios'] = [cell.strip() for cell in line[1:]]
        elif line[0].strip() == 'W':
            header_dict['pesos'] = [float(cell) for cell in line[1:]]
        elif line[0].strip() == 'V':
            header_dict['vetos'] = [float(cell) for cell in line[1:]]
        elif line[0].strip() == 'P':
            header_dict['preferências estritas'] = [float(cell) for cell in line[1:]]
        elif line[0].strip() == 'Q':
            header_dict['preferências fracas'] = [float(cell) for cell in line[1:]]
        else:
            data_dict[line[0].strip()] = [float(cell) for cell in line[1:]]
            i += 1
    return  header_dict, data_dict


import numpy as np
def normalize_data(data_dict, amin = 0, amax = 1): # normalize data dictionary
    max_per_col = [max(col) for col in np.matrix([row for row in data_dict.values()]).T.tolist()]
    min_per_col = [min(col) for col in np.matrix([row for row in data_dict.values()]).T.tolist()]  
    normalized_data = {}
    for key in data_dict:
        normalized_row = []
        for i, cell in enumerate(data_dict[key]):
            normalized_row.append( amin + (amax - amin) * (cell - min_per_col[i]) / (max_per_col[i] - min_per_col[i]) )
        normalized_data[key] = normalized_row
    return normalized_data


def __get_delta(data_dict):# returns delta(max diference between max and min grades for all criteria) from data_dict
    return max([max(col) - min(col) for col in np.matrix([row for row in data_dict.values()]).T.tolist()])


import itertools
def concordance_discordance(data_dict, weights):#returns the concordance and dicordance matrix for a given data_dict
    delta = __get_delta(data_dict)
    concordance_matrix = np.ones((len(data_dict), len(data_dict)))
    discordance_matrix = np.zeros((len(data_dict), len(data_dict)))
    for pair in itertools.combinations(data_dict, 2):
        fst = list(data_dict.keys()).index(pair[0])
        scd = list(data_dict.keys()).index(pair[1])
        # print(pair)
        concordance_f_s = []
        concordance_s_f = []
        discordance_f_s = []
        discordance_s_f = []
        for i, w in enumerate(weights):
            discordance_f_s.append((data_dict[pair[1]][i] - data_dict[pair[0]][i]) / delta)
            discordance_s_f.append((data_dict[pair[0]][i] - data_dict[pair[1]][i]) / delta)
            if data_dict[pair[0]][i] > data_dict[pair[1]][i]:
                concordance_f_s.append(w)
            elif data_dict[pair[0]][i] < data_dict[pair[1]][i]:
                concordance_s_f.append(w)
            else:
                concordance_f_s.append(w)
                concordance_s_f.append(w)
        discordance_f_s.append(0)
        discordance_s_f.append(0)
        discordance_matrix[fst, scd] = max(discordance_f_s)
        discordance_matrix[scd, fst] = max(discordance_s_f)
        concordance_matrix[fst, scd] = np.sum(concordance_f_s)
        concordance_matrix[scd, fst] = np.sum(concordance_s_f)
    return concordance_matrix, discordance_matrix

def strong_weak_dominance(concordance_matrix, discordance_matrix, cminus, c0, cplus, d1, d2):
    dominance_matrix = np.empty(concordance_matrix.shape, dtype=str)
    dominance_matrix.fill('-')
    # weak_dominance_matrix = np.zeros(concordance_matrix.shape)
    for (line, row), conc in np.ndenumerate(concordance_matrix):
        disc = discordance_matrix[line, row]        
        if line == row:
            dominance_matrix[line, row] = '-'            
        elif (conc >= cplus and disc <= d2) or (conc >= c0 and disc <= d1):
            dominance_matrix[line, row] = 'F'
            # print(line, row, conc, disc, 'F')
        elif (conc >= cminus and disc <= d1):
            dominance_matrix[line, row] = 'f'
            # print(line, row, conc, disc, 'f')
    # print(dominance_matrix)
    return dominance_matrix

def destilacao_descendente(sw_d_m):
    l = 0
    ranking = {}    
    y = {i for i in range(sw_d_m.shape[0])}
    print(y)
    empty = set()
    while y != empty:
    # for _ in range(10):
        d = set() # Conjunto de alternativas que não são fortemente rankeadas
        u = set() # Alternativas do conjunto d, que possuem uma relação de dominânica fraca
        b = set() # Alternativas do conjunto u, que não são fracamente dominadas
        for alternative in y:
            if 'F' not in sw_d_m[:, alternative]:
                d.add(alternative)
                if 'f' in sw_d_m[:, alternative]:
                    u.add(alternative)
                    if 'f' not in sw_d_m[:, alternative]:
                        b.add(alternative)
        a = (d - u) | (b)
        # print('Y: ', y)
        # print('D: ', d)
        # print('U: ', u)
        # print('B: ', b)
        # print('A: ', a, '\n')        
        for alternative in a:
            ranking[alternative] = l + 1
            sw_d_m[:, alternative].fill('x')
            sw_d_m[alternative, :].fill('x')
        y = y - a
        l += 1
    for item in ranking:
        print(item, ranking[item])
    return ranking


def destilacao_ascendente(s_w_dominance_matrix):
    l = 0
    ranking = {}  
    ranking_final = {}  
    y = {i for i in range(s_w_dominance_matrix.shape[0])}
    alternatives = s_w_dominance_matrix.shape[0]
    print(y)
    empty = set()
    while y != empty:
    # for _ in range(10):
        d = set() # Conjunto de alternativas que não são fortemente rankeadas
        u = set() # Alternativas do conjunto d, que possuem uma relação de dominânica fraca
        b = set() # Alternativas do conjunto u, que não são fracamente dominadas
        for alternative in y:
            if 'F' not in s_w_dominance_matrix[alternative]:
                d.add(alternative)
                if 'f' in s_w_dominance_matrix[alternative]:
                    u.add(alternative)
                    if 'f' not in s_w_dominance_matrix[alternative]:
                        b.add(alternative)
        a = (d - u) | (b)
        print('Y: ', y)
        print('D: ', d)
        print('U: ', u)
        print('B: ', b)
        print('A: ', a, '\n')
        for alternative in a:
            ranking[alternative] = alternatives - l -1
            s_w_dominance_matrix[:, alternative].fill('x')
            s_w_dominance_matrix[alternative, :].fill('x')
        y = y - a
        l += 1
    # for item in ranking:
        # print(item, ranking[item])
    return ranking


def ordenacoes(rd, ra): #ranking descendente e ranking ascendente
    rank_final_media = {}
    rank_final = {}
    alternatives = len(rd)
    mat_final_rank = np.empty( (alternatives, alternatives), dtype=object )
    mat_final_rank.fill('--')
    for alt in rd:
        rank_final_media[alt] = (rd[alt] + ra[alt]) / 2
        print(alt, rank_final_media[alt])
    for (frst, scnd) in itertools.combinations(rd, 2):
        if (rd[frst] < rd[scnd] and ra[frst] < ra[scnd]) or (rd[frst] == rd[scnd] and ra[frst] < ra[scnd]) or (rd[frst] < rd[scnd] and ra[frst] == ra[scnd]):
            mat_final_rank[frst, scnd] = 'P+'
            mat_final_rank[scnd, frst] = 'P-'
        elif rd[frst] == rd[scnd] and ra[frst] == ra[scnd]:
            mat_final_rank[frst, scnd] = 'I '
            mat_final_rank[scnd, frst] = 'I '
        elif (rd[frst] < rd[scnd] and ra[frst] > ra[scnd]) or (rd[frst] > rd[scnd] and ra[frst] < ra[scnd]):
            mat_final_rank[frst, scnd] = 'R '
            mat_final_rank[scnd, frst] = 'R '
    for idx, line in enumerate(mat_final_rank):
        rank_final[idx] = alternatives - (line == 'P+').sum()
    for key in rank_final:
        print(key, rank_final[key])



def discordance_veto(data_dict, veto):
    discordance_matrix = np.zeros((len(data_dict), len(data_dict)))
    for pair in itertools.combinations(data_dict, 2):
        fst = list(data_dict.keys()).index(pair[0])
        scd = list(data_dict.keys()).index(pair[1])
        for i, v in enumerate(veto):
            if data_dict[pair[1]][i] - data_dict[pair[0]][i] >= v:
                discordance_matrix[fst, scd] = 1
            if data_dict[pair[0]][i] - data_dict[pair[1]][i] >= v:
                discordance_matrix[scd, fst] = 1
    return discordance_matrix

    
def normalize_weights(weights):# return weights summing up unity
    weigths = np.array(weights)
    weights = weights / np.sum(weights)
    return weights


def pprint(dictionary):
    for key in dictionary:
        # formatted = 
        print(key + ',' + ','.join([str(item) for item in dictionary[key]]) )

import networkx as nx
import matplotlib.pyplot as plt
def dominance(data_dict, concordance_matrix, discordance_matrix, p = 0.5, q = 0.4):
    G = nx.DiGraph()
    for n in data_dict:
        G.add_node(n)
    dominance_matrix = np.zeros((len(data_dict), len(data_dict)))
    for pair in itertools.permutations(data_dict, 2):        
        fst = list(data_dict.keys()).index(pair[0])
        scd = list(data_dict.keys()).index(pair[1])
        if concordance_matrix[fst, scd] >= p and discordance_matrix[fst, scd] <= q:
            dominance_matrix[fst, scd] = 1
            G.add_edge(pair[0], pair[1])
    fig = plt.figure()
    plt.title("P = " + str(p) + '      ' + 'Q = ' + str(q) )
    nx.draw_shell(G, with_labels = True, node_size=1000, node_color='g')
    fig.savefig('teste.png')
    return dominance_matrix
# calculate dominance matrix

def partial_concordance(header_dict, data_dict):
    Cab = collections.OrderedDict()    
    for i, c in enumerate(header_dict['criteria']):
        Ciab = np.empty((len(data_dict), len(data_dict)))
        Ciab.fill(None)
        for pair in itertools.permutations(data_dict, 2):        
            fst = list(data_dict.keys()).index(pair[0])
            scd = list(data_dict.keys()).index(pair[1])
            gia = data_dict[pair[0]][i]
            gib = data_dict[pair[1]][i]
            pi = header_dict['pref_estrita'][i]
            qi = header_dict['pref_fraca'][i]
            if gib - gia >= pi:
                Ciab[fst, scd] = 0
            elif gib - gia < qi:
                Ciab[fst, scd] = 1
            else:
                Ciab[fst, scd] = (pi - gib + gia) / (pi - qi)
        Cab[c] = Ciab
    return Cab

        