import dgl.nn as dglnn
from dgl import from_networkx

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        ### force to output fix dimensions
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        ### apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation
        
    def message_func(self, edges):
        """ Message passing function

        Args:
            edges (EdgeBatch): instance of EdgeBatch class.
                               During message passing, DGL generates
                               it internally to represent a batch of edges.
                               It has three members src, dst and data to access
                               features of source nodes, destination nodes, and edges, respectively.
        
            edges.src: features of the source nodes in the batch of edges provided in input
            edges.data: features of the edges in the batch

        Returns:
            _type_: _description_
        """
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 2))}
    
    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            
            """update_all() is a high-level API that merges message generation,
            message aggregation and node update in a single call,
            which leaves room for optimization as a whole.
            https://docs.dgl.ai/guide/message-api.html
            https://docs.dgl.ai/generated/dgl.DGLGraph.update_all.html
            """
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            
            
            # Final node embedding at depth K, now it includes edge features (different from GraphSAGE)
            g.ndata['h'] = F.relu(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']
        
        
class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        # 2 SAGE Layers
        # K = 2 layers --> neighbour information is aggregated from a two-hop neighbourhood
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                # Dropout: randomly zeroes some of the elements of the input tensor
                # with probability of 0.5 using Bernoulli Distribution
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)
        
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        
        # Final node embedding
        # Final output of the forward propagatiion stage in E-GraphSAGE
        # embedding of each edge 'uv' as concatenatiion of nodes 'u' and nodes 'v'
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}
    
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class EGraphSAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, 2)
        
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)