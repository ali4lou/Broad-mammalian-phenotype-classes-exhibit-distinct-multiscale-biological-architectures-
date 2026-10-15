
# -*- coding: utf-8 -*-
"""
Optimized computation of KS and p-values 
Cell type and phenotype association

~13M cells y ~3000 phen
"""

import numpy as np
import pandas as pd
import warnings; warnings.simplefilter('ignore')
from sklearn.linear_model import LinearRegression
from scipy.stats import ks_2samp
from scipy import sparse
import gc
from joblib import Parallel, delayed

path = 'PATH'

# =====================
# 1) Load data
# =====================

# Gene list
df_gene = pd.read_csv(path+'df_gene.csv')
genes_sc = df_gene['gene_id'].to_numpy()

# Phenotype-related files
with open(path+'clean_genes_matrix.txt') as f:
    genes_associated_phen = np.array(f.read().splitlines())

with open(path+'clean_phen_matrix.txt') as f:
    phen = np.array([p for p in f.read().splitlines() if p != ''])

# Load phenotype matrix as sparse
dense_matrix = np.loadtxt(path+'clean_matrix.txt')
phen_matrix = sparse.csr_matrix(dense_matrix)
del dense_matrix

with open(path+'phen_filtro_20_500_genes_no_redundantes_final.txt') as f:
    phen_to_analyze = np.array(f.read().splitlines())

# List of expressed genes per cell
with open(path+'list_index_genes_per_single_cell.txt') as f:
    list_index_genes_per_single_cell = [
        np.fromstring(line, sep=',', dtype=np.int32) for line in f
    ]

# Genes per cell (total)
n_genes_total_per_cel_ind = np.loadtxt(path+'n_genes_total_per_cell.txt')
N_cell = len(n_genes_total_per_cel_ind)
print(f"N_cell = {N_cell:,}")

# Intersect genes
genes_final, ind_phen, ind_dev = np.intersect1d(
    genes_associated_phen, genes_sc, return_indices=True
)
submatrix_phen = phen_matrix[ind_phen, :]

X = n_genes_total_per_cel_ind.reshape(-1, 1)

# =====================
# 2) Cell types & times
# =====================

df_cell = pd.read_csv(path+'df_cell.csv')
m_types = np.genfromtxt(path+'m_types.txt', dtype=None, encoding=None)

cell_type_update = df_cell['celltype_update'].astype(str).to_numpy()
cell_type_update_unique = np.unique(cell_type_update)
cell_time = df_cell['day'].astype(str).to_numpy()
cell_time_unique = np.unique(cell_time)

N_cell = len(cell_type_update)
print(f"N_cell (df_cell) = {N_cell:,}")

# Sort times
numeric_part = [x for x in cell_time_unique if x != "P0"]
sorted_array = sorted(numeric_part, key=lambda x: float(x[1:]))
sorted_array.append("P0")
cell_time_unique = np.array(sorted_array, dtype=str)

# Sort cells
df_sort = pd.DataFrame({
    'index': np.arange(N_cell),
    'time': cell_time,
    'Category': pd.Categorical(cell_time, categories=cell_time_unique, ordered=True),
    'cell_type': cell_type_update
})
df_sort = df_sort.sort_values(by=["Category", "cell_type"]).to_numpy()

cell_type = df_sort[:, 3]
time = df_sort[:, 2]
cell_type_unique = np.array(list(dict.fromkeys(cell_type)))  # keep order

# Number of cells per type
n_cell_per_type = np.sum(m_types, axis=1)

# =====================
# 3) Core function
# =====================

def process_phenotype(p_idx):
    """Compute KS test + p-values for one phenotype"""
    chosen_phen = np.where(phen == phen_to_analyze[p_idx])[0]

    # Genes associated with this phenotype
    ind_not_null_genes_per_phen = np.where(
        submatrix_phen[:, chosen_phen].sum(axis=1) > 0
    )[0]

    # Vector (genes_final length) con 1 si el gen está asociado
    gene_mask = np.zeros(submatrix_phen.shape[0], dtype=np.int8)
    gene_mask[ind_not_null_genes_per_phen] = 1

    # Count associated genes per cell
    n_genes_specific_per_cel_ind = np.zeros(N_cell, dtype=np.int32)
    for i, indices in enumerate(list_index_genes_per_single_cell):
        if indices.size > 0:
            n_genes_specific_per_cel_ind[i] = np.sum(gene_mask[indices])

    # Linear regression
    y = n_genes_specific_per_cel_ind
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # KS test per cell type
    ks = []
    p_value_list = []
    for cel_type_k in cell_type_unique:
        mask_k = (cell_type_update == cel_type_k)
        res_bulk = residuals[~mask_k]
        res_specific = residuals[mask_k]
        ks_statistic, p_value = ks_2samp(
            res_bulk, res_specific, alternative='greater'
        )
        ks.append(ks_statistic)
        p_value_list.append(p_value)

    return ks, p_value_list


# =====================
# 4) Run in batches
# =====================

batch_size = 50
all_ks = []
all_p = []

for start in range(0, len(phen_to_analyze), batch_size):
    end = min(start + batch_size, len(phen_to_analyze))
    batch_idx = range(start, end)

    results = Parallel(n_jobs=8)(
        delayed(process_phenotype)(p) for p in batch_idx
    )

    for ks, p_values in results:
        all_ks.append(ks)
        all_p.append(p_values)

    print(f"Processed phenotypes {start}–{end-1}")
    gc.collect()

cell_type_ks_test = np.array(all_ks)
cell_type_p_value = np.array(all_p)

# =====================
# 5) Save results
# =====================

np.savetxt(path+'cell_type_ks_test.txt', cell_type_ks_test)
np.savetxt(path+'cell_type_p_value.txt', cell_type_p_value)
