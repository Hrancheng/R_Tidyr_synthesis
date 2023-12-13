import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

# 创建示例数据
data = {
    'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy'],
    'Humidity': ['High', 'Medium', 'Low', 'Medium', 'High'],
    'Temperature': ['0-10', '10-20', '20-30', '30-40', '20-30']
}

df = pd.DataFrame(data)

G = nx.Graph()

# 添加节点
for index, row in df.iterrows():
    weather = row['Weather']
    humidity= row['Humidity']
    temperature = row['Temperature']

    G.add_node(weather, feature=[1, 0, 0], type='Weather')
    G.add_node(humidity, feature=[0, 1, 0], type='Humidity')
    G.add_node(temperature, feature=[0, 0, 1], type='Temperature')

    G.add_edge(weather, humidity)
    G.add_edge(weather, temperature)
    G.add_edge(humidity, temperature)

    
    print(G)


    node_features = torch.tensor([G.nodes[node]['feature'] for node in G.nodes()], dtype=torch.float)
    edges = torch.tensor([list(e) for e in G.edges()], dtype=torch.long).t().contiguous()
    graph_data = Data(x=node_features, edge_index=edges)