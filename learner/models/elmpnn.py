from .base_gnn import *
from torch_geometric.nn.conv import RGCNConv, FastRGCNConv#, CuGraphRGCNConv


class ELMPNNLayer(Module):
    def __init__(self, in_features: int, out_features: int, n_edge_labels: int):
      super(ELMPNNLayer, self).__init__()
      self.conv = FastRGCNConv(in_features, out_features, num_relations=n_edge_labels).jittable()
      self.n_edge_labels = n_edge_labels
      self.linear = Linear(in_features, out_features)
      return

    def forward(self, x: Tensor, list_of_edge_index: List[Tensor]) -> Tensor:
      edge_index = torch.hstack(list_of_edge_index)
      edge_type = []
      for i in range(len(list_of_edge_index)):
        edge_type.append(i * torch.ones(len(list_of_edge_index[i][0]), dtype=int))
      edge_type = torch.concat(edge_type).to(torch.device('cuda'))

      x_out = self.linear(x)
      x_out += self.conv(x, edge_index, edge_type)
      return x_out

# class ELMPNNLayer(Module):
#     def __init__(self, in_features: int, out_features: int, n_edge_labels: int):
#       super(ELMPNNLayer, self).__init__()
#       self.n_edge_labels = n_edge_labels
#       self.convs = torch.nn.ModuleList()
#       for _ in range(n_edge_labels):
#         self.convs.append(LinearMaxConv(in_features, out_features))
#         # self.convs.append(LinearMaxConv(in_features, out_features).jittable())
#       self.linear = Linear(in_features, out_features)
#       self.n_proc = min(self.n_edge_labels, torch.multiprocessing.cpu_count())
#       return
    
#     def forward(self, x: Tensor, list_of_edge_index: List[Tensor]) -> Tensor:

#       x_out = self.linear(x)
#       for i, conv in enumerate(self.convs):  # bottleneck
#         x_out += conv(x, list_of_edge_index[i])

#       return x_out
    

""" GNN with different weights for different edge labels """
class ELMPNN(BaseGNN):
  def __init__(self, params) -> None:
    super().__init__(params)
    if self.drop > 0:
      warnings.warn("dropout not implemented for ELGNN")
    return

  def create_layer(self):
    return ELMPNNLayer(self.nhid, self.nhid, n_edge_labels=self.n_edge_labels)

  def node_embedding(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
    """ overwrite (same semantics, different typing) for jit """
    x = self.emb(x)
    if self.vn:
      for layer, vn_layer in zip(self.layers, self.vn_layers):
        x = layer(x, list_of_edge_index) + vn_layer(self.pool(x, batch))[batch]
        x = F.relu(x)
    else:
      for layer in self.layers:
        x = layer(x, list_of_edge_index)
        x = F.relu(x)
    return x
  
  def graph_embedding(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
    """ overwrite (same semantics, different typing) for jit """
    x = self.node_embedding(x, list_of_edge_index, batch)
    x = self.pool(x, batch)
    return x

  def forward(self, x: Tensor, list_of_edge_index: List[Tensor], batch: Optional[Tensor]) -> Tensor:
    """ overwrite (same semantics, different typing) for jit """
    x = self.graph_embedding(x, list_of_edge_index, batch)
    x = self.mlp(x)
    x = x.squeeze(1)
    return x
  

class ELMPNNPredictor(BasePredictor):
  def __init__(self, params, jit=False) -> None:
    super().__init__(params, jit)
    return
  
  def create_model(self, params):
    self.model = ELMPNN(params)

  def h(self, state: FrozenSet[Proposition]) -> float:
    x, edge_index = self.rep.get_state_enc(state)
    x = x.to(self.device)
    for i in range(len(edge_index)):
      edge_index[i] = edge_index[i].to(self.device)
    h = self.model.forward(x, edge_index, None).item()
    h = round(h)
    return h

  def predict_action(self, state: FrozenSet[Proposition]):
    raise NotImplementedError
  