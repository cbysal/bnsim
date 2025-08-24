import sys

import networkx as nx

node_num = int(sys.argv[1])
conn_num = int(sys.argv[2])
beta = float(sys.argv[3])
G = nx.watts_strogatz_graph(node_num, conn_num, beta)
print(list(G.edges()))
