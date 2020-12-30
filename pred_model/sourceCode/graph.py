# -*- coding: utf-8 -*-
"""
@Time   : 2020/9/29

@Author : Shen Fang
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as gnn


class GraphModel(nn.Module):
    def __init__(self, src_id, dst_id, in_c, hid_c, device):
        super(GraphModel, self).__init__()
        self.graph = dgl.graph((src_id, dst_id), device=device)
        self.linear = nn.Linear(2 * in_c, 2 * hid_c)
        self.out = nn.Linear(2 * hid_c, hid_c)

    def forward(self, features):
        self.graph.ndata["features"] = features
        self.graph.update_all(self.message_edge, self.message_reduce)
        new_features = self.graph.ndata.pop("new_features")

        return self.out(new_features)

    def message_edge(self, edge):
        features = torch.cat((edge.src["features"], edge.dst["features"]), dim=-1)  # [num_edges, 2 * num_features]

        attention = F.relu(self.linear(features))  # [num_edges, 1]

        return {"attention": attention, "features": features}

    def message_reduce(self, node):
        features = node.mailbox["features"]
        attention = node.mailbox["attention"]

        attention = attention.softmax(dim=1)
        new_features = torch.sum(attention * features, dim=1)

        return {"new_features": new_features}


class SGModel(nn.Module):
    def __init__(self, src_id, dst_id, in_c, hid_c, n_layers, device):
        super(SGModel, self).__init__()
        self.graph = dgl.graph((src_id, dst_id), device=device)

        self.gcn = nn.ModuleList([gnn.SGConv(in_c if i == 0 else hid_c, hid_c)
                                  for i in range(n_layers)])

        self.residual = nn.ModuleList([nn.Identity() if i !=0 else nn.Linear(in_c, hid_c) for i in range(n_layers)])

    def forward(self, features):
        input_features = features
        for i, conv in enumerate(self.gcn):
            output_features = F.relu(conv(self.graph, input_features)) + self.residual[i](input_features)
            input_features = output_features

        return input_features


class SAGEModel(nn.Module):
    def __init__(self, src_id, dst_id, in_c, hid_c, n_layers, device):
        super(SAGEModel, self).__init__()
        self.graph = dgl.graph((src_id, dst_id), device=device)

        self.gcn = nn.ModuleList([gnn.SAGEConv(in_c if i == 0 else hid_c, hid_c, "pool") for i in range(n_layers)])

        self.residual = nn.ModuleList([nn.Identity() if i != 0 else nn.Linear(in_c, hid_c) for i in range(n_layers)])

    def forward(self, features):
        input_features = features
        for i, conv in enumerate(self.gcn):
            output_features = F.relu(conv(self.graph, input_features)) + self.residual[i](input_features)
            input_features = output_features

        return input_features


if __name__ == '__main__':
    src_id = [0, 1, 2, 3, 4]
    dst_id = [1, 2, 3, 4, 0]

    model = GraphModel(src_id, dst_id, 6, 6, torch.device("cpu"))
    x = torch.rand(5, 32, 6)  # [num_nodes, batch_size, in_c]
    y = model(x)

    print(y.size())
