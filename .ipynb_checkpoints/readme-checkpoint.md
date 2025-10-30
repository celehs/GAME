# Representation Learning to Advance Multi-institutional Studies with Electronic Health Record Data

This repository contains the implementation of **GAME (Graph-based Alignment for Multi-institutional Embeddings)**, a framework designed to align multi-institutional codes within Electronic Health Records (EHR) data using a sophisticated knowledge graph system. Below is a detailed guide to help you understand and implement the GAME training process.

---

## Overview of the GAME Training Process

GAME comprises three main steps to align institutional embeddings from *M* institutions into a shared space. The process is illustrated below:

<p align="center">
  <img src="https://github.com/TongHan96/GAME/blob/main/pic/alg.png" alt="GAME Algorithm" title="GAME Algorithm" width="1000"/>
</p>

### **1. Aligning Institutions with PPMI Embedding**
- **Objective**: Align different institutions using Positive Pointwise Mutual Information (PPMI) embeddings.
- **Input**: PPMI embeddings \( V_i \) for each institution \( i \in \{1, 2, \ldots, M\} \).
- **Process**: Apply an institutional graph attention network to obtain aligned embeddings \( Y_i \).
- **Output**: A common representation \( Y \) that captures co-occurrence probabilities across institutions.

### **2. Generating Main Embeddings**
- **Objective**: Generate main embeddings using edges and a loss function.
- **Input**: Concatenation of SAPBERT embedding \( \mathbf{X} \) and aligned institutional embedding \( \mathbf{Y} \).
- **Process**: Optimize embeddings using a multi-similarity loss function.
- **Output**: A 256-dimensional embedding (\( r_{\text{max}} = 256 \)) suitable for similarity tasks.

### **3. Creating Relatedness Tail Embeddings**
- **Objective**: Generate relatedness tail embeddings \( \mathbf{Z}_{\mathcal{R}} \) for complex relatedness tasks.
- **Input**: All edges and main embeddings.
- **Process**: Concatenate relatedness tail embeddings with the main embedding.
- **Output**: A final 768-dimensional embedding \( \mathbb{Z} \).

---

## Repository Structure

The repository is organized as follows:

```terminal
GAME/
├── output/                 # Contains training results and outputs
├── readme.md               # This README file
├── src/                    # Source code for the GAME framework
│   ├── Attention.py        # Graph attention network implementation
│   ├── config.py           # Configuration settings
│   ├── data_structure.py   # Data structures and utilities
│   ├── evaluate.py         # Evaluation scripts
│   ├── load_data.py        # Data loading utilities
│   ├── main.py             # Main training script
│   └── utils.py            # Utility functions
└── supp_code/              # Supplementary code and downstream tasks
    └── feature_selection/  # Feature selection results and inputs
        ├── input/          # GPT scores for feature selection
```

---

## Implementation Guide

### **Step 1: Aligning Institutions with PPMI Embedding**

**Python Code:**
```python
%run main.py --path '~/GAME/' --epochs 500 --drop_out 0.5 --scale_sppmi 0.1 --lr 1e-4 --hidden_features 768 --DEVICE 'cuda' --path_origin 'align_NA'
```

### **Step 2: Obtaining the Similarity Embedding**

**Python Code:**
```python
%run main.py --path '~/GAME/' --rmax 256 --epochs 500 --scale_OTOL 30 --drop_out 0.5 --scale_sppmi 0.1 --lr 1e-6 --hidden_features 768 --DEVICE 'cuda' --align_path '[your_align_step_result_folder]'
```

### **Step 3: Acquiring the Relatedness Embedding**

**Python Code:**
```python
%run main.py --path '~/GAME/' --epochs 500 --drop_out 0.5 --scale_sppmi 1e-4 --lr 1e-6 --min_lr 5e-7 --hidden_features 768 --DEVICE 'cuda' --path_origin '[your_sim_step_result_folder]' --align_path '[your_align_step_result_folder]' 
```

---

## GAME Training Script Configuration

The GAME training script supports the following command-line arguments for customization:

### **Basic Settings**
- `--num_inst`: Number of institutions for training (default: `7`).
- `--path`: Path to the project or model (default: `config['path']`).
- `--input_dir`: Directory containing input data (default: `config['input_dir']`).
- `--path_origin`: Specifies the training step:
  - `align_NA`: Train the alignment PPMI step.
  - `None`: Train the similarity step.
  - Non-`None` value: Train the relatedness step (default: `config['path_origin']`).
- `--align_path`: Path to pre-trained aligned SPPMI embeddings (default: `None`).

### **Loss Function Parameters**
- `--AA`: Model-specific parameter (default: `1.0`).
- `--BB`: Model-specific parameter (default: `5.0`).
- `--lambd`: Lambda parameter for the loss function (default: `0.5`).

### **Scaling Factors for Loss Components**
- `--scale_hie`: Scaling factor for hierarchical alignment loss (default: `1`).
- `--scale_OTOL`: Scaling factor for one-to-one alignment loss (default: `30`).
- `--scale_REL`: Scaling factor for relevance alignment loss (default: `5`).
- `--scale_sppmi`: Scaling factor for PPMI feature selection loss (default: `0.1`).
- `--scale_align`: Scaling factor for alignment loss (default: `1`).

### **Dimensionality Specifications**
- `--rmax`: Maximum dimensionality for similarity embeddings (default: `256`).
- `--hidden_features`: Number of hidden features in the model (default: `768`).

### **Training Configuration**
- `--drop_out`: Drop edge probability during training (default: `0.0`).
- `--lr`: Learning rate for the optimizer (default: `1e-4`).
- `--min_lr`: Minimum learning rate for the optimizer (default: `5e-7`).
- `--epochs`: Number of training epochs (default: `500`).
- `--DEVICE`: Device for training, e.g., `'cuda:0'` for GPU (default: `'cuda:0'`).

### **Evaluation and Debugging**
- `--CHECK_ALL`: Enable checking all attention mechanisms during training (default: `False`).


---

## GAME Input

The input for GAME consists of several components, all of which are loaded in `load_data.py`. Below is a detailed description of each input component:


### **1. EHR Codes, Descriptions, and Institutional Index**
- **File**: `unique_name_desc.csv`
  - Contains EHR codes, their descriptions, and types.
  - Example:
    ```csv
    "code","desc","type"
    "CCS:1","incision and excision of cns","CCS"
    "CCS:10","thyroidectomy, partial or complete","CCS"
    "LOINC:15069-8","fructosamine serpl-scnc","LOINC"
    "LOINC:15189-4","kappa lc/lambda ser","LOINC"
    ...
    ```

- **File**: `inst_row.npz`
  - Contains institutional indices. The \(i\)-th list corresponds to the \(i\)-th institution's codes, as per `unique_name_desc.csv`.
  - Example:
![image](https://github.com/user-attachments/assets/40a6de64-4847-442a-b5f0-fbc342927040)
    


### **2. Edges and Pairs**
The relationships between codes are represented as edges and pairs. These are split into training and validation sets to ensure fairness and avoid data leakage.

#### **Splitting Pairs**
- For **similar hierarchical pairs**, the training and validation sets are split based on branches rather than individual pairs to maintain fairness (as explained in the paper appendix).
- For **related pairs** and **similar non-hierarchical pairs**, the split is performed randomly using the `load_data.py` script.

#### **Code for Splitting Pairs**
```python
# Split related pairs into train, validation, and test sets
train_rel_pairs, val_rel_pairs, test_rel_pairs = split_train_set(unique_name, REL_pairs=REL_pairs, scale=[0.7, 0.3])
test_rel_pairs = pd.concat([val_rel_pairs, test_rel_pairs])

# Save the split pairs
with open(f"{config['input_dir']}/similar_related_pairs/rel_pairs_0806.pkl", 'wb') as f:
    pickle.dump([train_rel_pairs, test_rel_pairs], f)

# Generate edges for related pairs
rel_edges = np.row_stack([match(train_rel_pairs.iloc[:, 0].values, unique_name), match(train_rel_pairs.iloc[:, 1].values, unique_name)])
np.save(f"{config['input_dir']}/edges/edges_rel.npy", rel_edges)
```

```python
# Split similar non-hierarchical pairs into train, validation, and test sets
train_sim_no_hie_pairs, val_sim_no_hie_pairs, test_sim_no_hie_pairs = split_train_set(unique_name, REL_pairs=SIM_no_hie_pairs, scale=[0.7, 0.3])
test_sim_no_hie_pairs = pd.concat([val_sim_no_hie_pairs, test_sim_no_hie_pairs])

# Save the split pairs
with open(f"{config['input_dir']}/similar_related_pairs/sim_no_hie_pairs_0806.pkl", 'wb') as f:
    pickle.dump([train_sim_no_hie_pairs, test_sim_no_hie_pairs], f)

# Generate edges for similar non-hierarchical pairs
sim_edges = np.row_stack([match(train_sim_no_hie_pairs.iloc[:, 0].values, unique_name), match(train_sim_no_hie_pairs.iloc[:, 1].values, unique_name)])
np.save(f"{config['input_dir']}/edges/edges_sim_no_hie.npy", sim_edges)
```

#### **Loading Pairs and Edges**
After splitting, the pairs and edges can be loaded directly:
- **Pairs**: These pairs (`REL_pairs`, `SIM_no_hie_pairs`, `test_sim_pairs`, etc) are stored in DataFrames with **5 columns**, structured as follows:
  - Example:
    ```plaintext
    code1       code2       type       similarity  relation
    "CCS:1"     "CCS:10"    "similar"  0.85       "hierarchical"
    "LOINC:15069-8" "LOINC:15189-4" "related" 0.92       "non-hierarchical"
    ```
- **Edges**: 2D tensors (with 2 rows) representing connections between codes. 

   GAME utilizes **multi-relation edges** to capture different types of relationships between codes. These edges include:

    - **Similarity Edges**: 
      - `edges_map`: Edges for mapping relationships (GPT-4).
      - `edges_hie`: Edges for hierarchical relationships.
      - `train_sim_no_hie_pairs`: Edges for similar but non-hierarchical relationships.
    - **Relatedness Edges**:
      - `train_rel_pairs`: Edges for relatedness relationships.
      - `pos_sppmi`: Edges for PPMI feature selection (GPT-4).

    All edges are loaded as **n × 2 csv** and will be processed as **2 × n tensors**, where each column indicates a connection between two codes.

### **3. Embeddings**
GAME uses multiple embeddings as input:
- **Institutional PPMI Embedding**: `inst_emb.pth`
  - A list of tensors, each representing the PPMI embedding for an institution.
  - Missing codes are imputed as zero vectors.
- **SAPBERT Embedding**: `sap_emb.pth` (or `coder_emb.pth`, `bge_emb.pth`)
  - A single tensor of shape \(N \times d\), where \(N\) is the number of codes and \(d\) is the embedding dimension (default: 768).
  - 
#### **Loading Embeddings**
```python
inst_emb = torch.load(f"{config['input_dir']}/inst_emb.pth")
sap_emb = torch.load(f"{config['input_dir']}/sap_emb.pth")
```

### **4. Loss Components**
GAME uses a multi-source loss function, which integrates information from hierarchies, positive/negative pairs, and training set relationships.

#### **Hierarchy Data**
- **File**: `hie_train.csv`
  - Contains hierarchical relationships for training, splitted by branches (real_grandpa).
  - Example:
   ![image](https://github.com/user-attachments/assets/ef8056af-37c4-45f7-a2c2-046a6337a426)
#### **Positive and Negative Code Mapping Pairs**
- **Positive Pairs (P_LTOL)**: Each row represents a positive standard code for a local code.
- **Negative Pairs (N_LTOL)**: Each row represents a negative standard code for a local code.
- Example:
![image](https://github.com/user-attachments/assets/689be0fa-277d-4631-a92d-6e4955b5454c)

#### **Positive and Negative Feature Selection Pairs**
- **Positive Pairs (pos_sppmi)**: **N × 2 csv** with postive pairs.
- **Negative Pairs (neg_sppmi)**: **M × 2 csv** with negative pairs.
  
#### **Integrating Loss Components**
The hierarchy, positive/negative code mapping pairs, and positive/negative feature selection pairs, training set pairs of related and non-hierarchical pairs are integrated into a single dictionary for efficient loss computation.

```python
# Generate my_objects using hierarchy and pairs
my_objects_new = origin_loss_set(unique_name, P_LTOL, N_LTOL, hie_loinc_rxn_phe, 
                          train_rel_pairs, train_sim_no_hie_pairs, pos_sppmi, neg_sppmi)

# Save the integrated loss components
np.save(f"{config['input_dir']}/edges/my_objects.npy", my_objects_new)
```

---
This concludes the description of the input data for GAME. For further details, refer to the `load_data.py` script or the paper appendix.
