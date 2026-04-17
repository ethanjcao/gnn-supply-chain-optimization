"""
Graph Neural Network for Supply Chain Optimization

Author: Ethan (Jingtian) Cao
Affiliation: Johns Hopkins University

This script implements homogeneous and heterogeneous Graph Neural Network pipelines
for supply chain network modeling using DGL and PyTorch. It supports graph-level and
edge-level prediction, hyperparameter selection with 5-fold cross-validation, and
evaluation with both MAPE and SMAPE.

"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import SAGEConv
from dgl.nn import HeteroGraphConv


class SupplyChainHomoDataset(DGLDataset):   
    def __init__(
        self,
        network_config_path="network_config.csv",
        node_feature_path="node_feature.csv",
        arc_feature_path="arc_feature.csv",
        network_label_path="network_label.csv",
    ):
        self.network_config_path = network_config_path
        self.node_feature_path = node_feature_path
        self.arc_feature_path = arc_feature_path
        self.network_label_path = network_label_path
        super().__init__(name="supply_chain_homo")


    def process(self):
        
        edges = pd.read_csv(self.network_config_path)     
        nodes = pd.read_csv(self.node_feature_path)       
        arc = pd.read_csv(self.arc_feature_path)          
        props = pd.read_csv(self.network_label_path)      

        
        label_dict = {}
        for _, row in props.iterrows():           
            label_dict[int(row["id_g"])] = float(row["label"])
            
            
            

        
        edges_group = edges.groupby("id_g")
        nodes_group = nodes.groupby("id_g")
        arc_group = arc.groupby("id_g")

        self.graphs = []   
        self.labels = []   

        
        for gid in sorted(edges_group.groups.keys()):
            edges_of_g = edges_group.get_group(gid).reset_index(drop=True)
            nodes_of_g = nodes_group.get_group(gid).reset_index(drop=True)
            arc_of_g = arc_group.get_group(gid).reset_index(drop=True)

            
            src = torch.from_numpy(edges_of_g["src"].to_numpy()).long()
            dst = torch.from_numpy(edges_of_g["dst"].to_numpy()).long()
            
            g = dgl.graph((src, dst), num_nodes=33)

            
            
            
            assert len(nodes_of_g) == 33, f"id_g={gid}: node rows != 33"
            assert (nodes_of_g.loc[:11, "node_type"] == 1).all(), f"id_g={gid}: first 12 not source"
            assert (nodes_of_g.loc[12:, "node_type"] == 0).all(), f"id_g={gid}: last 21 not dest"

            supply = torch.from_numpy(nodes_of_g["supply"].to_numpy()).float()
            demand = torch.from_numpy(nodes_of_g["demand"].to_numpy()).float()

            node_feat = torch.zeros((33, 2), dtype=torch.float32) 
            node_feat[:12, 0] = supply[:12]   
            node_feat[12:, 1] = demand[12:]   
            g.ndata["feat"] = node_feat
            
            
            
            
            
            
            
            

            
            
            
            assert len(edges_of_g) == len(arc_of_g), f"id_g={gid}: edges and arc rows mismatch"

            g.edata["feat"] = torch.from_numpy(arc_of_g["arc_feat"].to_numpy()).float().unsqueeze(1)
            g.edata["label"] = torch.from_numpy(arc_of_g["arc_label"].to_numpy()).float().unsqueeze(1)

            
            self.graphs.append(g) 
            self.labels.append(label_dict[gid]) 

        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1) 

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]
    

    def __len__(self):
        return len(self.graphs)
    


class SupplyChainHeteroDataset(DGLDataset):
    def __init__(
        self,
        network_config_path="network_config.csv",
        node_feature_path="node_feature.csv",
        arc_feature_path="arc_feature.csv",
        network_label_path="network_label.csv",
    ):
        self.network_config_path = network_config_path
        self.node_feature_path = node_feature_path
        self.arc_feature_path = arc_feature_path
        self.network_label_path = network_label_path
        super().__init__(name="supply_chain_hetero")

    def process(self):
        
        edges = pd.read_csv(self.network_config_path)     
        nodes = pd.read_csv(self.node_feature_path)       
        arc = pd.read_csv(self.arc_feature_path)          
        props = pd.read_csv(self.network_label_path)      

        
        label_dict = {
            int(row["id_g"]): float(row["label"])
            for _, row in props.iterrows()
        }

        
        edges_group = edges.groupby("id_g")
        nodes_group = nodes.groupby("id_g")
        arc_group = arc.groupby("id_g")

        self.graphs = []
        self.labels = []

        
        for gid in sorted(edges_group.groups.keys()):
            edges_of_g = edges_group.get_group(gid).reset_index(drop=True)
            nodes_of_g = nodes_group.get_group(gid).reset_index(drop=True)
            arc_of_g = arc_group.get_group(gid).reset_index(drop=True)

            
            src_nodes = nodes_of_g.loc[nodes_of_g["node_type"] == 1].reset_index(drop=True)
            dst_nodes = nodes_of_g.loc[nodes_of_g["node_type"] == 0].reset_index(drop=True)

            assert len(src_nodes) == 12, f"id_g={gid}: source node count != 12"
            assert len(dst_nodes) == 21, f"id_g={gid}: destination node count != 21"

            
            src = torch.from_numpy(edges_of_g["src"].to_numpy()).long()
            dst = torch.from_numpy((edges_of_g["dst"] - 12).to_numpy()).long()

            
            g = dgl.heterograph(
                {
                    ("source", "ship", "destination"): (src, dst),
                    ("destination", "rev_ship", "source"): (dst, src),
                },
                num_nodes_dict={"source": 12, "destination": 21},
            )

            
            
            g.nodes["source"].data["feat"] = torch.from_numpy(
                src_nodes["supply"].to_numpy()
            ).float().unsqueeze(1)

            
            g.nodes["destination"].data["feat"] = torch.from_numpy(
                dst_nodes["demand"].to_numpy()
            ).float().unsqueeze(1)

            
            assert len(edges_of_g) == len(arc_of_g), f"id_g={gid}: edge mismatch"

            e_feat = torch.from_numpy(arc_of_g["arc_feat"].to_numpy()).float().unsqueeze(1)   
            e_lab  = torch.from_numpy(arc_of_g["arc_label"].to_numpy()).float().unsqueeze(1)  

            
            g.edges["ship"].data["feat"] = e_feat
            g.edges["ship"].data["label"] = e_lab

            g.edges["rev_ship"].data["feat"] = e_feat
            g.edges["rev_ship"].data["label"] = e_lab

            
            self.graphs.append(g)
            self.labels.append(label_dict[gid])

        self.labels = torch.tensor(self.labels, dtype=torch.float32).unsqueeze(1)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def mape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:

    y_pred = y_pred.detach().cpu().view(-1)
    y_true = y_true.detach().cpu().view(-1)
    denom = torch.clamp(torch.abs(y_true), min=eps) 
    return torch.mean(torch.abs((y_true - y_pred) / denom)).item()


def train_test_split_indices(n: int, test_ratio: float = 0.2, seed: int = 42):

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    test_size = int(round(n * test_ratio))
    test_idx = idx[:test_size].tolist()
    train_idx = idx[test_size:].tolist()
    return train_idx, test_idx    



def kfold_indices(indices, k: int = 5, seed: int = 42):

    rng = np.random.RandomState(seed)
    indices = np.array(indices)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)  
    for i in range(k):  
        val_idx = folds[i].tolist()  
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]).tolist()  
        yield train_idx, val_idx  




def collate_graph_level(samples):
    
    graphs, labels = map(list, zip(*samples))  
    bg = dgl.batch(graphs)  
    y = torch.stack(labels, dim=0)  
    return bg, y  


def collate_edge_level_homo(samples):
    
    graphs, _ = map(list, zip(*samples))  
    bg = dgl.batch(graphs)  
    y_edge = bg.edata["label"]  
    return bg, y_edge  


def collate_edge_level_hetero(samples):
    
    graphs, _ = map(list, zip(*samples))  
    bg = dgl.batch(graphs)  
    y_edge = bg.edges["ship"].data["label"]  
    return bg, y_edge  






class HomoSAGEEncoder(nn.Module):
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()  
        self.layers = nn.ModuleList()  
        self.dropout = nn.Dropout(dropout)  

        
        self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type="mean"))

        
        for _ in range(num_layers - 1):  
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type="mean"))

    
    def forward(self, g, x):
        
        h = x  
        for i, conv in enumerate(self.layers):  
            h = conv(g, h)  
            h = F.relu(h)  
            if i != len(self.layers) - 1:  
                h = self.dropout(h)  
        return h  




class HomoGraphRegressor(nn.Module):
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()  
        self.encoder = HomoSAGEEncoder(in_dim, hidden_dim, num_layers, dropout)  
        self.mlp = nn.Sequential(  
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Dropout(dropout),  
            nn.Linear(hidden_dim, 1),  
        )

    def forward(self, g):
        
        x = g.ndata["feat"]  
        h = self.encoder(g, x)  
        g.ndata["h"] = h  
        hg = dgl.mean_nodes(g, "h")  
        y = self.mlp(hg)  
        return y






class HomoEdgeRegressor(nn.Module):
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, use_edge_feat: bool = True):
        super().__init__()
        self.encoder = HomoSAGEEncoder(in_dim, hidden_dim, num_layers, dropout)  
        self.use_edge_feat = use_edge_feat  

        
        pred_in_dim = hidden_dim * 2 + (1 if use_edge_feat else 0)  

        self.mlp = nn.Sequential(  
            nn.Linear(pred_in_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  
        )

    def forward(self, g):
        
        x = g.ndata["feat"]  
        h = self.encoder(g, x)  

        src, dst = g.edges()  
        h_src = h[src]  
        h_dst = h[dst]  

        if self.use_edge_feat:  
            e_feat = g.edata["feat"]  
            z = torch.cat([h_src, h_dst, e_feat], dim=1)  
        else:  
            z = torch.cat([h_src, h_dst], dim=1)  

        y = self.mlp(z)  
        return y





class HeteroSAGEEncoder(nn.Module):
    
    def __init__(self, in_dim_dict: dict, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  
        self.layers = nn.ModuleList()  

        
        self.layers.append(
            HeteroGraphConv(
                {
                    ("source", "ship", "destination"): SAGEConv(in_dim_dict["source"], hidden_dim, "mean"),
                    ("destination", "rev_ship", "source"): SAGEConv(in_dim_dict["destination"], hidden_dim, "mean"),
                },
                aggregate="sum",
            )
        )

        
        for _ in range(num_layers - 1):
            self.layers.append(
                HeteroGraphConv(
                    {
                        ("source", "ship", "destination"): SAGEConv(hidden_dim, hidden_dim, "mean"),
                        ("destination", "rev_ship", "source"): SAGEConv(hidden_dim, hidden_dim, "mean"),
                    },
                    aggregate="sum",
                )
            )

    def forward(self, g, x_dict):
        
        h_dict = x_dict  
        for i, layer in enumerate(self.layers):
            
            h_dict = layer(g, h_dict)

            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

            if i != len(self.layers) - 1:
                h_dict = {k: self.dropout(v) for k, v in h_dict.items()}

        return h_dict



class HeteroGraphRegressor(nn.Module):
    
    def __init__(self, in_dim_dict: dict, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = HeteroSAGEEncoder(in_dim_dict, hidden_dim, num_layers, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, g):
        
        x_dict = {  
            "source": g.nodes["source"].data["feat"],  
            "destination": g.nodes["destination"].data["feat"],  
        }
        h_dict = self.encoder(g, x_dict)  

        g.nodes["source"].data["h"] = h_dict["source"]  
        g.nodes["destination"].data["h"] = h_dict["destination"]  

        hs = dgl.mean_nodes(g, "h", ntype="source")  
        hd = dgl.mean_nodes(g, "h", ntype="destination")  

        hg = torch.cat([hs, hd], dim=1)  
        y = self.mlp(hg)  
        return y  



class HeteroEdgeRegressor(nn.Module):
    
    def __init__(self, in_dim_dict: dict, hidden_dim: int, num_layers: int, dropout: float, use_edge_feat: bool = True):
        super().__init__()
        self.encoder = HeteroSAGEEncoder(in_dim_dict, hidden_dim, num_layers, dropout)
        self.use_edge_feat = use_edge_feat  

        pred_in_dim = hidden_dim * 2 + (1 if use_edge_feat else 0)  
        self.mlp = nn.Sequential(  
            nn.Linear(pred_in_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  
        )

    def forward(self, g):
        
        x_dict = {  
            "source": g.nodes["source"].data["feat"],  
            "destination": g.nodes["destination"].data["feat"],  
        }
        h_dict = self.encoder(g, x_dict)  

        src, dst = g.edges(etype="ship")  
        h_src = h_dict["source"][src]  
        h_dst = h_dict["destination"][dst]  

        if self.use_edge_feat:  
            e_feat = g.edges["ship"].data["feat"]  
            z = torch.cat([h_src, h_dst, e_feat], dim=1)  
        else:  
            z = torch.cat([h_src, h_dst], dim=1)  

        y = self.mlp(z)  
        return y  





def train_one_epoch_graph(model, loader, optimizer):
    
    model.train()  
    total_loss = 0.0  

    for bg, y in loader:  
        bg = bg.to(device)  
        y = y.to(device).float()  

        pred = model(bg)  
        loss = F.mse_loss(pred, y)  

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        total_loss += loss.item() * y.shape[0]  

    return total_loss / len(loader.dataset)  



@torch.no_grad()
def eval_graph(model, loader):
    
    
    model.eval()  
    preds = []  
    trues = []  

    for bg, y in loader:
        bg = bg.to(device)
        y = y.to(device).float()
        pred = model(bg)
        preds.append(pred)  
        trues.append(y)  

    preds = torch.cat(preds, dim=0)  
    trues = torch.cat(trues, dim=0)  
    return mape(preds, trues)  




def train_one_epoch_edge(model, loader, optimizer, hetero: bool = False):
    
    model.train()
    total_loss = 0.0
    total_edges = 0

    for bg, y_edge in loader:
        bg = bg.to(device)  
        y_edge = y_edge.to(device).float()  

        pred = model(bg)  
        loss = F.mse_loss(pred, y_edge)  

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        total_loss += loss.item() * y_edge.shape[0]  
        total_edges += y_edge.shape[0]  

    return total_loss / max(total_edges, 1)  



@torch.no_grad()
def eval_edge(model, loader):
    
    model.eval()
    preds = []  
    trues = []  

    for bg, y_edge in loader:
        bg = bg.to(device)
        y_edge = y_edge.to(device).float()
        pred = model(bg)  
        preds.append(pred)  
        trues.append(y_edge)  

    preds = torch.cat(preds, dim=0)  
    trues = torch.cat(trues, dim=0)  
    return mape(preds, trues)  




def cv_select_hparams_graph(dataset, train_indices, model_builder_fn, collate_fn, search_space, epochs: int = 30, batch_size: int = 32):
    
    best_cfg = None  
    best_score = float("inf")  

    for cfg in search_space:  
        fold_scores = []  

        for tr_idx, va_idx in kfold_indices(train_indices, k=5, seed=42):  
            tr_subset = torch.utils.data.Subset(dataset, tr_idx)  
            va_subset = torch.utils.data.Subset(dataset, va_idx)  

            tr_loader = GraphDataLoader(tr_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  
            va_loader = GraphDataLoader(va_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  

            model = model_builder_fn(cfg).to(device)  
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])  

            for _ in range(epochs):  
                train_one_epoch_graph(model, tr_loader, optimizer)  

            score = eval_graph(model, va_loader)  
            fold_scores.append(score)  

        avg_score = float(np.mean(fold_scores))  

        if avg_score < best_score:  
            best_score = avg_score  
            best_cfg = cfg  

    return best_cfg, best_score  



def cv_select_hparams_edge(dataset, train_indices, model_builder_fn, collate_fn, search_space, epochs: int = 30, batch_size: int = 32):
    
    best_cfg = None  
    best_score = float("inf")  

    for cfg in search_space:  
        fold_scores = []  

        for tr_idx, va_idx in kfold_indices(train_indices, k=5, seed=42):  
            tr_subset = torch.utils.data.Subset(dataset, tr_idx)  
            va_subset = torch.utils.data.Subset(dataset, va_idx)  

            tr_loader = GraphDataLoader(tr_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  
            va_loader = GraphDataLoader(va_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  

            model = model_builder_fn(cfg).to(device)  
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

            for _ in range(epochs):  
                train_one_epoch_edge(model, tr_loader, optimizer)  

            score = eval_edge(model, va_loader)  
            fold_scores.append(score)  

        avg_score = float(np.mean(fold_scores))  

        if avg_score < best_score:
            best_score = avg_score
            best_cfg = cfg  

    return best_cfg, best_score  




def fit_and_test_graph(dataset, train_idx, test_idx, model_builder_fn, collate_fn, cfg, epochs: int = 60, batch_size: int = 32):
    
    tr_subset = torch.utils.data.Subset(dataset, train_idx)  
    te_subset = torch.utils.data.Subset(dataset, test_idx)  

    tr_loader = GraphDataLoader(tr_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  
    te_loader = GraphDataLoader(te_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  

    model = model_builder_fn(cfg).to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    for _ in range(epochs):  
        train_one_epoch_graph(model, tr_loader, optimizer)  

    test_mape = eval_graph(model, te_loader)  
    return test_mape, model  


def fit_and_test_edge(dataset, train_idx, test_idx, model_builder_fn, collate_fn, cfg, epochs: int = 60, batch_size: int = 32):
    
    tr_subset = torch.utils.data.Subset(dataset, train_idx)  
    te_subset = torch.utils.data.Subset(dataset, test_idx)  

    tr_loader = GraphDataLoader(tr_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  
    te_loader = GraphDataLoader(te_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  

    model = model_builder_fn(cfg).to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])  

    for _ in range(epochs):  
        train_one_epoch_edge(model, tr_loader, optimizer)  

    test_mape = eval_edge(model, te_loader)  
    return test_mape, model  



if __name__ == "__main__":  

    
    
    


    homo_ds = SupplyChainHomoDataset(
        network_config_path="data/network_config.csv",
        node_feature_path="data/node_feature.csv",
        arc_feature_path="data/arc_feature.csv",
        network_label_path="data/network_label.csv",
    )


    hetero_ds = SupplyChainHeteroDataset(
        network_config_path="data/network_config.csv",
        node_feature_path="data/node_feature.csv",
        arc_feature_path="data/arc_feature.csv",
        network_label_path="data/network_label.csv",
    )

    
    _ = len(homo_ds)  
    _ = len(hetero_ds)  

    
    
    

    homo_train_idx, homo_test_idx = train_test_split_indices(len(homo_ds), test_ratio=0.2, seed=42)
    hetero_train_idx, hetero_test_idx = train_test_split_indices(len(hetero_ds), test_ratio=0.2, seed=42)

    
    
    

    search_space = [  
        {"hidden_dim": 32, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "wd": 1e-5},  
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "wd": 1e-5},  
        {"hidden_dim": 64, "num_layers": 3, "dropout": 0.2, "lr": 1e-3, "wd": 1e-5},  
        {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2, "lr": 5e-4, "wd": 1e-5},  
    ]

    
    
    

    def build_homo_graph_model(cfg):
        
        return HomoGraphRegressor(
            in_dim=2,  
            hidden_dim=cfg["hidden_dim"],  
            num_layers=cfg["num_layers"],  
            dropout=cfg["dropout"],  
        )

    def build_homo_edge_model(cfg):
        
        return HomoEdgeRegressor(
            in_dim=2,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            use_edge_feat=True,
        )

    def build_hetero_graph_model(cfg):
        
        return HeteroGraphRegressor(
            in_dim_dict={"source": 1, "destination": 1},  
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )

    def build_hetero_edge_model(cfg):
        
        return HeteroEdgeRegressor(
            in_dim_dict={"source": 1, "destination": 1},
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            use_edge_feat=True,
        )

    
    
    

    
    best_cfg_homo_graph, best_cv_homo_graph = cv_select_hparams_graph(
        dataset=homo_ds,  
        train_indices=homo_train_idx,  
        model_builder_fn=build_homo_graph_model,  
        collate_fn=collate_graph_level,  
        search_space=search_space,  
        epochs=30,  
        batch_size=32,
    )

    print(f"[HOMO GRAPH] Best CV cfg = {best_cfg_homo_graph}, CV MAPE = {best_cv_homo_graph:.4f}")

    
    best_cfg_homo_edge, best_cv_homo_edge = cv_select_hparams_edge(
        dataset=homo_ds,
        train_indices=homo_train_idx,
        model_builder_fn=build_homo_edge_model,
        collate_fn=collate_edge_level_homo,
        search_space=search_space,
        epochs=30,
        batch_size=32,
    )

    print(f"[HOMO EDGE]  Best CV cfg = {best_cfg_homo_edge}, CV MAPE = {best_cv_homo_edge:.4f}")

    
    best_cfg_hetero_graph, best_cv_hetero_graph = cv_select_hparams_graph(
        dataset=hetero_ds,
        train_indices=hetero_train_idx,
        model_builder_fn=build_hetero_graph_model,
        collate_fn=collate_graph_level,
        search_space=search_space,
        epochs=30,
        batch_size=32,
    )

    print(f"[HETERO GRAPH] Best CV cfg = {best_cfg_hetero_graph}, CV MAPE = {best_cv_hetero_graph:.4f}")

    
    best_cfg_hetero_edge, best_cv_hetero_edge = cv_select_hparams_edge(
        dataset=hetero_ds,
        train_indices=hetero_train_idx,
        model_builder_fn=build_hetero_edge_model,
        collate_fn=collate_edge_level_hetero,
        search_space=search_space,
        epochs=30,
        batch_size=32,
    )

    print(f"[HETERO EDGE]  Best CV cfg = {best_cfg_hetero_edge}, CV MAPE = {best_cv_hetero_edge:.4f}")

    
    
    

    
    homo_graph_test_mape, _ = fit_and_test_graph(
        dataset=homo_ds,
        train_idx=homo_train_idx,  
        test_idx=homo_test_idx,  
        model_builder_fn=build_homo_graph_model,  
        collate_fn=collate_graph_level,  
        cfg=best_cfg_homo_graph,  
        epochs=60,  
        batch_size=32,  
    )

    print(f"[HOMO GRAPH] Test MAPE = {homo_graph_test_mape:.4f}")

    
    homo_edge_test_mape, _ = fit_and_test_edge(
        dataset=homo_ds,
        train_idx=homo_train_idx,
        test_idx=homo_test_idx,
        model_builder_fn=build_homo_edge_model,
        collate_fn=collate_edge_level_homo,
        cfg=best_cfg_homo_edge,
        epochs=60,
        batch_size=32,
    )

    print(f"[HOMO EDGE]  Test MAPE = {homo_edge_test_mape:.4f}")

    
    hetero_graph_test_mape, _ = fit_and_test_graph(
        dataset=hetero_ds,
        train_idx=hetero_train_idx,
        test_idx=hetero_test_idx,
        model_builder_fn=build_hetero_graph_model,
        collate_fn=collate_graph_level,
        cfg=best_cfg_hetero_graph,
        epochs=60,
        batch_size=32,
    )

    print(f"[HETERO GRAPH] Test MAPE = {hetero_graph_test_mape:.4f}")

    
    hetero_edge_test_mape, _ = fit_and_test_edge(
        dataset=hetero_ds,
        train_idx=hetero_train_idx,
        test_idx=hetero_test_idx,
        model_builder_fn=build_hetero_edge_model,
        collate_fn=collate_edge_level_hetero,
        cfg=best_cfg_hetero_edge,
        epochs=60,
        batch_size=32,
    )

    print(f"[HETERO EDGE]  Test MAPE = {hetero_edge_test_mape:.4f}")




    def smape(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> float:

        y_pred = y_pred.detach().cpu().view(-1)
        y_true = y_true.detach().cpu().view(-1)

        numerator = 2 * torch.abs(y_true - y_pred)
        denominator = torch.abs(y_true) + torch.abs(y_pred) + eps

        return torch.mean(numerator / denominator).item()


    @torch.no_grad()
    def eval_edge_smape(model, loader):
        model.eval()
        preds = []
        trues = []

        for bg, y in loader:
            bg = bg.to(device)
            y = y.to(device).float()
            pred = model(bg)
            preds.append(pred)
            trues.append(y)

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        return smape(preds, trues)


    def cv_select_hparams_edge_smape(dataset, train_indices, model_builder_fn, collate_fn, search_space, epochs: int = 30, batch_size: int = 32):
        
        best_cfg = None  
        best_score = float("inf")  

        for cfg in search_space:  
            fold_scores = []  

            for tr_idx, va_idx in kfold_indices(train_indices, k=5, seed=42):  
                tr_subset = torch.utils.data.Subset(dataset, tr_idx)  
                va_subset = torch.utils.data.Subset(dataset, va_idx)  

                tr_loader = GraphDataLoader(tr_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  
                va_loader = GraphDataLoader(va_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  

                model = model_builder_fn(cfg).to(device)  
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

                for _ in range(epochs):  
                    train_one_epoch_edge(model, tr_loader, optimizer)  

                score = eval_edge_smape(model, va_loader)  
                fold_scores.append(score)  

            avg_score = float(np.mean(fold_scores))  

            if avg_score < best_score:
                best_score = avg_score
                best_cfg = cfg  

        return best_cfg, best_score  


    
    def fit_and_test_edge_smape(dataset, train_idx, test_idx, model_builder_fn, collate_fn, cfg, epochs: int = 60, batch_size: int = 32):
        
        tr_subset = torch.utils.data.Subset(dataset, train_idx)  
        te_subset = torch.utils.data.Subset(dataset, test_idx)  

        tr_loader = GraphDataLoader(tr_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  
        te_loader = GraphDataLoader(te_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  

        model = model_builder_fn(cfg).to(device)  
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])  

        for _ in range(epochs):  
            train_one_epoch_edge(model, tr_loader, optimizer)  

        test_smape = eval_edge_smape(model, te_loader)
        return test_smape, model


    

    
    best_cfg_homo_edge_smape, best_cv_homo_edge_smape = cv_select_hparams_edge_smape(
        dataset=homo_ds,
        train_indices=homo_train_idx,
        model_builder_fn=build_homo_edge_model,
        collate_fn=collate_edge_level_homo,
        search_space=search_space,
        epochs=30,
        batch_size=32
    )
    print("HOMO EDGE SMAPE CV:", best_cv_homo_edge_smape, "best cfg:", best_cfg_homo_edge_smape)

    
    homo_edge_smape_test, _ = fit_and_test_edge_smape(
        dataset=homo_ds,
        train_idx=homo_train_idx,
        test_idx=homo_test_idx,
        model_builder_fn=build_homo_edge_model,
        collate_fn=collate_edge_level_homo,
        cfg=best_cfg_homo_edge_smape,
        epochs=60,
        batch_size=32
    )
    print("HOMO EDGE SMAPE TEST:", homo_edge_smape_test)

    
    best_cfg_hetero_edge_smape, best_cv_hetero_edge_smape = cv_select_hparams_edge_smape(
        dataset=hetero_ds,
        train_indices=hetero_train_idx,
        model_builder_fn=build_hetero_edge_model,
        collate_fn=collate_edge_level_hetero,
        search_space=search_space,
        epochs=30,
        batch_size=32
    )
    print("HETERO EDGE SMAPE CV:", best_cv_hetero_edge_smape, "best cfg:", best_cfg_hetero_edge_smape)

    
    hetero_edge_smape_test, _ = fit_and_test_edge_smape(
        dataset=hetero_ds,
        train_idx=hetero_train_idx,
        test_idx=hetero_test_idx,
        model_builder_fn=build_hetero_edge_model,
        collate_fn=collate_edge_level_hetero,
        cfg=best_cfg_hetero_edge_smape,
        epochs=60,
        batch_size=32
    )
    print("HETERO EDGE SMAPE TEST:", hetero_edge_smape_test)
