# Graph Neural Network for Supply Chain Optimization

A research-driven machine learning project that formulates supply chain planning data as graphs and evaluates Graph Neural Networks for network-level and link-level prediction tasks.

## Executive Summary

This project was developed in the context of operations and analytics research, with a practical focus on how graph-based learning can improve prediction quality in supply chain systems.

Instead of treating the dataset as flat tabular records, the workflow represents each supply chain instance as a graph so the model can learn from both **entity attributes** and **network structure**. Two formulations were implemented and compared:

* **Homogeneous graphs**, where all nodes are embedded in a shared representation space
* **Heterogeneous graphs**, where source and destination nodes are modeled explicitly as distinct node types

The project evaluates both formulations on:

* **Graph-level prediction** for overall network outcomes
* **Edge-level prediction** for link-specific outcomes

## Problem Statement

Classical predictive approaches often underuse the structural information embedded in supply chain networks. In real operational settings, outcomes are shaped not only by isolated node-level variables, but also by how suppliers, destinations, and shipping relationships interact across the network.

The core question of this project is:

**Can structure-aware graph learning improve predictive performance and decision support in supply chain network modeling?**

## Project Framing

### Situation

Supply chain data was provided in structured CSV format, but the original representation did not directly expose relational dependencies in a way that standard tabular models could fully exploit.

### Task

Build a graph-based modeling pipeline that preserves network structure, supports both graph-level and edge-level prediction, and produces results relevant to cost estimation and operational analysis.

### Action

* Constructed custom datasets in DGL for both homogeneous and heterogeneous graph formulations
* Encoded node, edge, and graph labels from multi-table supply chain inputs
* Implemented GraphSAGE-based message passing architectures in PyTorch / DGL
* Performed 80/20 train-test splitting and 5-fold cross-validation for hyperparameter selection
* Diagnosed instability in edge-level MAPE caused by zero-valued targets and introduced SMAPE for more robust evaluation
* Compared homogeneous and heterogeneous formulations to assess the effect of role-aware structural representation

### Result

* Achieved strong performance on graph-level prediction, with graph-level MAPE reported below 0.10 in experimentation
* Stabilized edge-level evaluation by replacing MAPE with SMAPE in zero-sensitive settings
* Observed that heterogeneous graph modeling produced stronger predictive behavior than the homogeneous alternative, supporting the hypothesis that explicit role separation improves representation quality
* Built a reusable modeling framework for graph-based supply chain analytics rather than a one-off experiment

## Why This Matters

This project is relevant because supply chains are naturally graph-structured systems. Modeling them as graphs creates a more faithful representation of operational reality and opens the door to stronger decision support in areas such as:

* **Network cost estimation**
* **Link-level cost or flow prediction**
* **Allocation and routing analysis**
* **Structure-aware operational planning**
* **Role-sensitive modeling of source–destination interactions**

In practical terms, the project shows how graph learning can move beyond academic novelty and become useful in business settings where dependencies across entities matter.

## Technical Scope

### Data Modeling

The workflow constructs graph instances from four structured data sources:

* `network_config.csv`
* `node_feature.csv`
* `arc_feature.csv`
* `network_label.csv`

### Graph Formulations

#### Homogeneous Graph

* Shared node space across the full network
* Node features stored as a 2-dimensional vector: `[supply, demand]`
* Suitable as a simpler baseline representation

#### Heterogeneous Graph

* Separate node types for `source` and `destination`
* Supply stored on source nodes and demand stored on destination nodes
* Directional edge relations used to preserve bipartite structure more explicitly
* Designed to better capture type-aware interactions in the network

### Modeling Pipeline

* Graph construction with custom `DGLDataset` classes
* GraphSAGE-based encoders for homogeneous and heterogeneous settings
* Separate graph-level and edge-level regressors
* Hyperparameter search across hidden dimension, number of layers, dropout, learning rate, and weight decay
* Final training on the full training split using best-performing configurations

## Key Analytical Insight

The most important modeling insight from this work is that **evaluation design matters as much as architecture design**.

During edge-level prediction, MAPE became unreliable because many true edge labels were zero, which made a denominator-based percentage metric unstable. Replacing MAPE with SMAPE produced a more credible and decision-useful evaluation framework.

A second important finding is that **heterogeneous graph construction is not just a technical variation**. In this setting, it improved the model’s ability to preserve source–destination distinctions, which led to stronger predictive behavior and more meaningful operational interpretation.

## Business Implications

From a business perspective, this project demonstrates that graph-based machine learning can improve the quality of analytics for networked systems where dependencies matter. The practical implications include:

* Better visibility into **how network structure influences outcomes**
* More reliable estimation of **overall network behavior**
* Better support for **link-level operational decisions**
* A scalable framework for extending predictive analytics into **structure-aware decision systems**

For employers, the value of this project is not just that it uses GNNs. The value is that it connects model design, evaluation rigor, and business interpretation in a way that is directly relevant to real analytics work.

## Repository Contents

```text
.
├── gnn_supply_chain_model.py
├── network_config.csv
├── node_feature.csv
├── arc_feature.csv
├── network_label.csv
└── README.md
```

## Skills Demonstrated

* Graph data modeling
* Structured feature engineering
* Homogeneous and heterogeneous graph construction
* Graph Neural Network implementation in DGL / PyTorch
* Cross-validation and hyperparameter selection
* Evaluation metric redesign for zero-sensitive targets
* Translation of research workflows into business-relevant analytics framing

## Portfolio Positioning

This project is especially relevant to roles in:

* Data Science
* Machine Learning
* Supply Chain Analytics
* Operations Research
* Advanced Analytics
* Decision Science

## Author

**Ethan Cao**
Analytics professional with expertise in data products, machine learning, and decision-focused analytics.
