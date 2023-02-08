import networkx as nx 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import csv

def generate_random_graph(n, s=0, t=1):
    connected = False
    while not connected:
        G = nx.erdos_renyi_graph(n, 0.2)
        connected = nx.is_connected(G)

    for i, e in enumerate(G.edges()):
        G[e[0]][e[1]]['i'] = i
        G[e[0]][e[1]]['r'] = np.random.randint(10,20)
    
    return G 

def edge_r(G, *edge):
    if len(edge) == 1:
        v1, v2 = edge[0][0], edge[0][1]
    else:
        v1, v2 = edge[0], edge[1]

    if G.get_edge_data(v1, v2) is None:
        raise ValueError("Either no such edge or it doesn't have any data")

    return G.get_edge_data(v1, v2)['r']

def edge_dir(G, *edge):
    if len(edge) == 1:
        v1, v2 = edge[0][0], edge[0][1]
    else:
        v1, v2 = edge[0], edge[1]

    return 1 if v1 < v2 else -1

def edge_i(G, *edge):
    if len(edge) == 1:
        v1, v2 = edge[0][0], edge[0][1]
    else:
        v1, v2 = edge[0], edge[1]

    if G.get_edge_data(v1, v2) is None:
        raise ValueError("Either no such edge or it doesn't have any data")

    return G.get_edge_data(v1, v2)['i']

def get_eqs_kirchhoff_2(G):
    A = []
    m = len(G.edges())
    for cycle in nx.cycle_basis(G):
        eq = [0]*m
        for i in range(len(cycle)):
            edge = cycle[i], cycle[(i+1)%len(cycle)]
            eq[edge_i(G, edge)] = edge_r(G, edge)*edge_dir(G, edge)
        A.append(eq)
    return np.array(A)

def get_eqs_kirchhoff_1(G, s, t):
    A = []
    m = len(G.edges())
    for node in G.nodes():
        if node in (s, t):
            continue
        eq = [0]*m
        for edge in G.edges(node):
            eq[edge_i(G, edge)] = edge_dir(G, edge)
        A.append(eq)
    return np.array(A)

def get_eq_path(G, s, t):
    eq = [0]*len(G.edges())
    path = nx.shortest_path(G, s, t)
    for i in range(len(path)-1):
        edge = path[i], path[i+1]
        eq[edge_i(G, edge)] = edge_r(G, edge)*edge_dir(G, edge)

    return np.array([eq])


def get_flow_graph(G, A, b):
    x = np.linalg.solve(A, b)

    G_flow = nx.DiGraph()
    for edge in G.edges():
        if (x[edge_i(G, edge)]<0 and edge_dir(G, edge)==1) or (x[edge_i(G, edge)]>0 and edge_dir(G, edge)==-1):
            edge = edge[::-1]


        G_flow.add_edge(edge[0], edge[1], i=edge_i(G, edge), x=np.abs(x[edge_i(G, edge)]) )

    return G_flow

def draw_resistance(G, s, t):
    pos = nx.nx_pydot.pydot_layout(G)
    options = {"edgecolors": "tab:gray", "node_size": 300, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), **options)
    nx.draw_networkx_nodes(G, pos, nodelist=[s, t], node_color="magenta")
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = nx.get_edge_attributes(G,'r'))
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    plt.show()

def draw_flow(G, G_flow, s, t):
    pos = nx.nx_pydot.pydot_layout(G)
    options = {"edgecolors": "tab:gray", "node_size": 300, "alpha": 0.9}
    nx.draw_networkx_nodes(G_flow, pos, nodelist=G_flow.nodes(), **options)
    nx.draw_networkx_nodes(G, pos, nodelist=[s, t], node_color="magenta")
    nx.draw_networkx_labels(G_flow, pos)

    vals = np.array(list(nx.get_edge_attributes(G_flow, 'x').values()))

    edges = nx.draw_networkx_edges(
        G_flow,
        pos,
        arrowstyle="->",
        arrowsize=20,
        width=2,
        edge_color = vals,
        alpha= np.tanh(0.1+vals/vals.max()),
        edge_cmap=plt.cm.copper,
        arrows=True
    )
    nx.draw_networkx_edge_labels(G_flow, pos, edge_labels = {k:round(v,2) for k,v in nx.get_edge_attributes(G_flow,'x').items()})

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.copper)
    pc.set_array(vals)
    plt.colorbar(pc)

    plt.show()

def graph_from_file(filepath):
    G = nx.Graph()
    with open(filepath, newline='') as f:
        for i, row in enumerate(csv.reader(f)):
            G.add_edge(int(row[0]), int(row[1]), r=int(row[2]), i=i)
            
    return G


if __name__ == '__main__':
    if len(sys.argv)<5:
        print("python lab2zad3.py <filepath> <s> <t> <E>")
        exit()
    else:
        print(sys.argv)

        s, t, E = [int(x) for x in sys.argv[2:]]

        G = graph_from_file(sys.argv[1])
        m = len(G.edges)
        print(G.edges)
        
        A = np.concatenate([
                get_eq_path(G, s, t),
                get_eqs_kirchhoff_1(G, s, t),
                get_eqs_kirchhoff_2(G)
                ])

        b = np.zeros(len(A))
        b[0] = E


        G_flow = get_flow_graph(G, A, b)

        print(G_flow.edges())
        draw_flow(G, G_flow, s, t)