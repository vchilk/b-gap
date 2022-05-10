import numpy as np
import networkx as nx

def generate_adj_mats(states_path='states.txt'):
    with open(states_path, 'r') as f:
        states = f.readlines()
        for state in states:
            #make list of coordinates
            coords = []
            it = iter(state.split(', '))
            for x in it:
                x = float(x)
                y = float(next(it))
                coords.append([x,y])
            V = 5
            mat = np.zeros((V,V))
            for i in range(V):
                for j in range(V):
                    x1 = coords[i][0]
                    y1 = coords[i][1]

                    x2 = coords[j][0]
                    y2 = coords[j][1]

                    xd = x1-x2
                    yd = y1-y2

                    mat[i][j] = (xd**2 + yd**2) **0.5
            yield mat

def calculate_derivative(listnotkeyword):
    res = []
    for i in range(len(listnotkeyword)-1):
        res[i] = (listnotkeyword[i]+listnotkeyword[i+1])/2
    return res

def generate_cmetric_labels():
    current_episode = np.zeros((5, 0))
    frames_per_episode = 10
    prev_timestamp = None
    prev_diffs = None

    labels = []
    for i, adj_mat in enumerate(generate_adj_mats()):
        G = nx.from_numpy_matrix(adj_mat, create_using=nx.Graph)  
        closeness = nx.closeness_centrality(G, distance='weight')
        current_timestamp = np.array([closeness[index] for index in closeness])
        if prev_timestamp is not None:
            diffs = current_timestamp - prev_timestamp
            if prev_diffs is not None:
                second_diffs = diffs - prev_diffs
                labels.append((diffs, second_diffs))
            prev_diffs = diffs
        prev_timestamp = current_timestamp
    return labels

if __name__ == '__main__':
    labels = generate_cmetric_labels()
    print(labels)