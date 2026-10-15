# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:07:16 2025

@author: USUARIO
"""

# -*- coding: utf-8 -*-
"""

"""
import scanpy as sc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
import matplotlib as mpl
import os
import random
import warnings; warnings.simplefilter('ignore')
import os
import anndata
from umap import UMAP
from matplotlib import cm   
import matplotlib.pylab as pl
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib.pyplot import rc_context
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.io import mmread
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.stats import mstats, kstest, ttest_ind, t
from scipy import sparse
from sklearn.decomposition import NMF
from scipy.stats import ks_2samp
import scipy.sparse
import gc

# Define path
path = 'PATH'

output_file = path + "n_genes_total_per_cell.txt"
output_file_list = path + "list_index_genes_per_single_cell.txt"

# Both files are written in append mode, one block per atlas file, so a previous
# run must be removed first or its cells would be counted twice.
for f_out in (output_file, output_file_list):
    if os.path.exists(f_out):
        os.remove(f_out)

# Load gene list
df_gene = pd.read_csv(path+'df_gene.csv')
genes_sc = df_gene['gene_id'].to_numpy()

# Load phenotype-related files once (outside the loop)
with open(path+'clean_genes_matrix.txt', 'r') as f:
    genes_associated_phen = np.array(f.read().splitlines())

with open(path+'clean_phen_matrix.txt', 'r') as f:
    phen = np.array([p for p in f.read().splitlines() if p != ''])

phen_matrix = np.loadtxt(path+'clean_matrix.txt')

print("Processing datasets...")

for i in range(1, 5):
    # 1.) Load the dataset
    adata = sc.read_h5ad(path + f'adata_JAX_dataset_{i}.h5ad')
    print(f"adata shape: {adata.shape}")

    X = adata.X
    del adata
    gc.collect()

    # 1.7.) Find common genes and indices
    genes_final, ind_phen, ind_dev = np.intersect1d(genes_associated_phen, genes_sc, return_indices=True)
    submatrix_phen = phen_matrix[ind_phen, :]

    # Convert to sparse CSR if not already
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    
    # Select only relevant columns
    X_sub = X[:, ind_dev]
    del X
    gc.collect()

    # Count non-zero genes per cell efficiently
    n_genes_total_per_cell = np.diff(X_sub.indptr)
    
    # Append counts directly to file (fast)
    with open(output_file, "ab") as f:
        np.savetxt(f, n_genes_total_per_cell, fmt="%d")

    # Indices of the expressed genes per cell, one line per cell. The separator
    # must be the comma that computation_KS_faster.py reads with
    # np.fromstring(line, sep=','); any other separator is silently parsed as a
    # single gene per cell by older numpy, and raises in recent versions.
    with open(output_file_list, "a") as f:
        for c in range(X_sub.shape[0]):
            indices = X_sub[c].indices  # índices de elementos no cero
            f.write(",".join(map(str, indices)) + "\n")

    # Clean up
    del X_sub, n_genes_total_per_cell
    gc.collect()
    print(f"Dataset {i} processed.")

