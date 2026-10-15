#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 07:16:32 2026

@author: alicia
"""

import os
import pandas as pd
import networkx as nx
import obonet
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter
import math

# =============================================================================
# 0. PATHS AND ONTOLOGY LOADING
# =============================================================================
path = 'PATH'
path2 = 'PATH'
url = 'PATH/MPheno_OBO.ontology.obo'

print("Loading ontology graphs...")
mp_graph = obonet.read_obo(url)   # same file as `url`, read once
go_graph = obonet.read_obo(os.path.join(path2, "go.obo"))

id_to_name_mp = {id_: data.get('name', '').strip().lower() for id_, data in mp_graph.nodes(data=True)}

# =============================================================================
# 1-2. GENE-PHENOTYPE ASSIGNMENT
# =============================================================================
# Read from final_filtered_matrix.txt, the same gene-phenotype assignment used
# by the genetic layer and by the cell-type KS statistics, so that the three
# layers are built on identical gene sets.
#
# The matrix uses Ensembl identifiers and the GO annotation file uses MGI, so
# the rows are translated back with the same BioMart export applied upstream.
# =============================================================================
GENE_MATRIX = os.path.join(path2, 'final_filtered_matrix.txt')
MART_EXPORT = '/media/alicia/TOSHIBA EXT/DATOS_RATON/phenotype/mart_export.txt'

print("Reading gene-phenotype assignment from the filtered matrix...")

gm = pd.read_csv(GENE_MATRIX, sep='\t', index_col=0)
gm.index = gm.index.astype(str).str.strip()                    # Ensembl IDs
gm.columns = gm.columns.astype(str).str.strip().str.lower()    # phenotypes

ens2mgi = {}
with open(MART_EXPORT) as f:
    next(f)                                   # header: Gene stable ID,MGI ID
    for line in f:
        parts = line.strip().split(',')
        ens = [p for p in parts if p.startswith('ENSMUSG')]
        mgi = [p for p in parts if p.startswith('MGI:')]
        if ens and mgi:
            ens2mgi[ens[0]] = mgi[0]

mapped = gm.index.map(lambda g: ens2mgi.get(g))
n_unmapped = int(pd.isna(mapped).sum())
print(f"  Ensembl to MGI: {len(gm.index) - n_unmapped}/{len(gm.index)} mapped")

gm_mgi = gm.copy()
gm_mgi.index = mapped
gm_mgi = gm_mgi[~pd.isna(gm_mgi.index)]
gm_mgi = gm_mgi.groupby(level=0).max()        # collapse duplicate mappings

pheno_to_genes = {p: set(gm_mgi.index[gm_mgi[p] > 0]) for p in gm_mgi.columns}

# PHENOS replaces phen_filter as the phenotype list throughout this script.
PHENOS = sorted(pheno_to_genes)
phen_filter = set(PHENOS)                     # kept for backward compatibility
all_needed_genes = set().union(*pheno_to_genes.values())

empty = [p for p in PHENOS if not pheno_to_genes[p]]
if empty:
    raise ValueError(
        f"{len(empty)} phenotypes have no genes after mapping, e.g. {empty[:5]}. "
        "They would enter the matrices as empty vectors."
    )

_sizes = pd.Series({p: len(pheno_to_genes[p]) for p in PHENOS})
print(f"  {len(PHENOS)} phenotypes, {len(all_needed_genes)} unique genes")
print(f"  Genes per phenotype: median {_sizes.median():.0f}, "
      f"range {_sizes.min():.0f}-{_sizes.max():.0f}")

# =============================================================================
# 3. READ DIRECT GENE -> GO ANNOTATIONS (Only for needed genes)
# =============================================================================
print("Extracting direct GO annotations...")
gene_to_go_direct = defaultdict(set)
gaf_file = os.path.join(path2, "mgi.gaf")

with open(gaf_file, 'r') as f:
    for line in f:
        if line.startswith('!'): continue 
        parts = line.split('\t')
        if len(parts) > 4:
            mgi_id = parts[1].strip()
            go_id = parts[4].strip()
            
            full_mgi_id = mgi_id if mgi_id.startswith("MGI:") else f"MGI:{mgi_id}"
            
            # Save only if it is a gene belonging to our filtered phenotypes
            if full_mgi_id in all_needed_genes:
                gene_to_go_direct[full_mgi_id].add(go_id)

# =============================================================================
# 4. PREPARE THE "CELLULAR PROCESS" FILTER (GO:0009987)
# =============================================================================
CELLULAR_PROCESS_ID = "GO:0009987"
valid_cellular_processes = set()

# Pre-calculate which GO terms are "cellular processes" to speed up the loop
for go_id, data in go_graph.nodes(data=True):
    if data.get('namespace') == 'biological_process':
        # It's valid if it is the cellular process itself, or if cellular process is its ancestor
        if go_id == CELLULAR_PROCESS_ID or CELLULAR_PROCESS_ID in nx.descendants(go_graph, go_id):
            valid_cellular_processes.add(go_id)

# =============================================================================
# 5. PROPAGATE ANCESTORS AND CLASSIFY INTO 3 CATEGORIES
# =============================================================================
print("Propagating ancestors and splitting into CC, MF, and Cellular Process...")
gene_to_cc = defaultdict(set)
gene_to_mf = defaultdict(set)
gene_to_cp = defaultdict(set)

for mgi_id, direct_gos in gene_to_go_direct.items():
    propagated_gos = set(direct_gos)
    for go_id in direct_gos:
        if go_id in go_graph:
            # Add all ancestors
            propagated_gos.update(nx.descendants(go_graph, go_id))
            
    # Classify each GO term into its corresponding matrix
    for go in propagated_gos:
        if go not in go_graph: continue
        namespace = go_graph.nodes[go].get('namespace')
        
        if namespace == 'cellular_component':
            gene_to_cc[mgi_id].add(go)
        elif namespace == 'molecular_function':
            gene_to_mf[mgi_id].add(go)
        elif namespace == 'biological_process':
            # Only add if it is a verified Cellular Process
            if go in valid_cellular_processes:
                gene_to_cp[mgi_id].add(go)


# =============================================================================
# 5.1. CALCULATE INFORMATION CONTENT (IC) FOR EACH GO TERM (CORRECTED)
# =============================================================================
print("Calculating Information Content (IC) weights to penalize redundancy...")

def calculate_ic_weights(gene_to_go_dict):
    """
    Calculates the IC weight: IC(t) = -ln(p(t))
    where p(t) = (number of unique genes with term t) / (total unique genes)
    """
    go_to_genes = defaultdict(set)
    
    # EL FIX: El denominador correcto es el total de genes ÚNICOS en tu diccionario
    total_genes = len(gene_to_go_dict) 
    
    # Invert the dictionary to count genes per GO term
    for gene, gos in gene_to_go_dict.items():
        for go in gos:
            go_to_genes[go].add(gene)
            
    ic_weights = {}
    for go, genes in go_to_genes.items():
        # Probabilidad = (Número de genes anotados a este GO) / (Total de genes únicos)
        prob = len(genes) / total_genes 
        ic_weights[go] = -math.log(prob)
        
    return ic_weights

ic_cc = calculate_ic_weights(gene_to_cc)
ic_mf = calculate_ic_weights(gene_to_mf)
ic_cp = calculate_ic_weights(gene_to_cp)

# # =============================================================================
# # 6. CREATE THE 3 FINAL MATRICES (Phenotype -> List of GOs)
# # =============================================================================
# pheno_cc_matrix = []
# pheno_mf_matrix = []
# pheno_cp_matrix = []

# for pheno in phen_filter:
#     genes = pheno_to_genes.get(pheno, set())
    
#     cc_set, mf_set, cp_set = set(), set(), set()
    
#     for g in genes:
#         cc_set.update(gene_to_cc.get(g, set()))
#         mf_set.update(gene_to_mf.get(g, set()))
#         cp_set.update(gene_to_cp.get(g, set()))
        
#     pheno_cc_matrix.append({'Phenotype': pheno, 'GO_Cellular_Component': list(cc_set)})
#     pheno_mf_matrix.append({'Phenotype': pheno, 'GO_Molecular_Function': list(mf_set)})
#     pheno_cp_matrix.append({'Phenotype': pheno, 'GO_Cellular_Process': list(cp_set)})

# # Convert to DataFrames for easy export or visualization
# df_cc = pd.DataFrame(pheno_cc_matrix)
# df_mf = pd.DataFrame(pheno_mf_matrix)
# df_cp = pd.DataFrame(pheno_cp_matrix)

# # Save to separate files
# df_cc.to_csv(os.path.join(path2, 'matrix_GO_Cellular_Component.csv'), index=False)
# df_mf.to_csv(os.path.join(path2, 'matrix_GO_Molecular_Function.csv'), index=False)
# df_cp.to_csv(os.path.join(path2, 'matrix_GO_Cellular_Process.csv'), index=False)

# print("\nProcess completed!")
# print(f"Cellular Component rows saved: {len(df_cc)}")
# print(f"Molecular Function rows saved: {len(df_mf)}")
# print(f"Cellular Process (strict filter) rows saved: {len(df_cp)}")

# =============================================================================
# 6. CREATE THE 3 WEIGHTED DATA LISTS
# =============================================================================
print("Calculating GO term weights per phenotype...")

pheno_cc_data = []
pheno_mf_data = []
pheno_cp_data = []

# PHENOS is sorted, so the column order of the matrices is reproducible.
# Iterating over a set, as before, changes it between runs because of Python's
# hash randomization.
for pheno in PHENOS:
    genes = pheno_to_genes[pheno]
    
    # Use Counter to tally how many genes contribute to each GO term
    cc_counter = Counter()
    mf_counter = Counter()
    cp_counter = Counter()
    
    for g in genes:
        cc_counter.update(gene_to_cc.get(g, set()))
        mf_counter.update(gene_to_mf.get(g, set()))
        cp_counter.update(gene_to_cp.get(g, set()))
        
    # Append to lists as dictionaries containing the counts 
    # e.g., {'Phenotype': 'tail', 'GO_Cellular_Component': {'GO:0005575': 3, 'GO:0005622': 1}}
    pheno_cc_data.append({'Phenotype': pheno, 'GO_Cellular_Component': dict(cc_counter)})
    pheno_mf_data.append({'Phenotype': pheno, 'GO_Molecular_Function': dict(mf_counter)})
    pheno_cp_data.append({'Phenotype': pheno, 'GO_Cellular_Process': dict(cp_counter)})


#correlation between genes
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MultiLabelBinarizer

# print("1. Identificando genes comunes con fenotipos asociados...")

# # 1A. Extraer todos los genes que tienen al menos un fenotipo en tu filtro
# genes_con_fenotipo = set()
# for pheno in phen_filter:
#     genes_con_fenotipo.update(pheno_to_genes.get(pheno, set()))

# # 1B. Extraer genes que tienen al menos un término GO en cada espacio
# genes_cc = {gene for gene, terms in gene_to_cc.items() if terms}
# genes_mf = {gene for gene, terms in gene_to_mf.items() if terms}
# genes_cp = {gene for gene, terms in gene_to_cp.items() if terms}

# # 1C. Intersección de oro: Tienen fenotipo Y están en los 3 espacios GO
# common_genes = list(
#     genes_cc.intersection(genes_mf)
#     .intersection(genes_cp)
#     .intersection(genes_con_fenotipo)
# )
# common_genes.sort() # Ordenar es CRÍTICO para alinear las filas de las 3 matrices

# print("=" * 60)
# print(f"Genes con fenotipos: {len(genes_con_fenotipo)}")
# print(f"Genes comunes en CC, MF, CP: {len(genes_cc.intersection(genes_mf).intersection(genes_cp))}")
# print(f"-> GENES FINALES (Comunes + Fenotipo): {len(common_genes)}")
# print("=" * 60 + "\n")

# if len(common_genes) < 2:
#     raise ValueError("No hay suficientes genes para calcular similitudes.")

# # --- 2. Función para crear matriz, calcular similitud y extraer vector ---
# def build_space_vector(gene_dict, gene_list):
#     # Crear lista de listas de GO terms manteniendo el orden de gene_list
#     go_lists = [list(gene_dict[gene]) for gene in gene_list]
    
#     # Construir la matriz binaria Gen x GO
#     mlb = MultiLabelBinarizer()
#     binary_matrix = mlb.fit_transform(go_lists)
    
#     # Calcular similitud pairwise entre genes (Pearson)
#     # Nota: Si prefieres similitud Coseno, cambia np.corrcoef por cosine_similarity
#     sim_matrix = np.corrcoef(binary_matrix)
    
#     # Extraer triángulo superior (sin diagonal)
#     upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
#     return sim_matrix[upper_tri_indices]

# print("2. Construyendo matrices y calculando similitudes pairwise...")
# vec_cc = build_space_vector(gene_to_cc, common_genes)
# vec_mf = build_space_vector(gene_to_mf, common_genes)
# vec_cp = build_space_vector(gene_to_cp, common_genes)

# # Limpiar posibles NaNs (ocurre si algún gen termina sin varianza en la matriz binaria)
# mask = ~np.isnan(vec_cc) & ~np.isnan(vec_mf) & ~np.isnan(vec_cp)
# vec_cc, vec_mf, vec_cp = vec_cc[mask], vec_mf[mask], vec_cp[mask]

# # --- 3. Crear DataFrame y calcular el Pearson entre espacios ---
# df_pairwise = pd.DataFrame({
#     'Cellular Component': vec_cc,
#     'Molecular Function': vec_mf,
#     'Cellular Process': vec_cp
# })

# space_corr = df_pairwise.corr(method='pearson')

# # --- 4. Imprimir Resultados por Pantalla ---
# median_cc = df_pairwise['Cellular Component'].median()
# median_mf = df_pairwise['Molecular Function'].median()
# median_cp = df_pairwise['Cellular Process'].median()

# print("\n" + "#" * 60)
# print("     MEDIANAS DE SIMILITUD ENTRE GENES (Solo genes filtrados)")
# print("#" * 60)
# print(f"• Cellular Component : {median_cc:.6f}")
# print(f"• Molecular Function : {median_mf:.6f}")
# print(f"• Cellular Process   : {median_cp:.6f}")
# print("#" * 60 + "\n")

# print("#" * 60)
# print("       CORRELACIÓN DE PEARSON ENTRE ESPACIOS DE GO")
# print("#" * 60)
# print(space_corr.to_string())
# print("#" * 60 + "\n")


# =============================================================================
# 7. BUILD THE MATRICES
# =============================================================================
# 7.1 Map IDs to Names
go_id_to_name = {id_: data.get('name', 'unknown') for id_, data in go_graph.nodes(data=True)}

# 7.2 Function to build a WEIGHTED matrix from our list of dictionaries
def build_weighted_matrix(pheno_go_list, go_col_name, ic_weights):
    data_dict = {}
    
    for row in pheno_go_list:
        pheno = row['Phenotype']
        # Extract the dictionary of {go_id: count} we made in step 6
        go_counts = row[go_col_name] 
        
        go_names_with_weights = {}
        for go, count in go_counts.items():
            # Get the IC weight calculated in step 5.5
            weight = ic_weights.get(go, 0)
            
            # Apply the formula: (genes in common) * IC weight
            final_value = count * weight
            
            go_names_with_weights[f"{go} | {go_id_to_name.get(go, 'unknown')}"] = final_value
            
        data_dict[pheno] = go_names_with_weights
        
    # Convert dictionary to DataFrame
    # Fill missing with 0 and cast to float (since weights are decimals)
    df_matrix = pd.DataFrame(data_dict).fillna(0).astype(float)
    
    return df_matrix


# =============================================================================
# 7.3. GENERATE THE 3 MATRICES
# =============================================================================
print("Building IC-weighted matrices for correlation...")

# Pass the respective IC dictionaries to the function
matrix_cc = build_weighted_matrix(pheno_cc_data, 'GO_Cellular_Component', ic_cc)
matrix_mf = build_weighted_matrix(pheno_mf_data, 'GO_Molecular_Function', ic_mf)
matrix_cp = build_weighted_matrix(pheno_cp_data, 'GO_Cellular_Process', ic_cp)

# =============================================================================
# 7.4. PHENOTYPE GO ANNOTATION STATISTICS
# =============================================================================
print("\nCalculating missing annotation statistics...")

total_phenotypes = len(PHENOS)

# Count how many phenotypes have a sum of 0 (no GO terms associated).
# An all-zero column becomes a NaN cosine, which is then filled with 0 and
# therefore gives that phenotype zero similarity against every other one,
# contributing ~n spurious pairs to every downstream statistic. With the gene
# sets now taken from the same matrix as the other layers, this should be zero
# in all three spaces; if it is not, stop before saving the matrices.
unassociated_cc = int((matrix_cc.sum(axis=0) == 0).sum())
unassociated_mf = int((matrix_mf.sum(axis=0) == 0).sum())
unassociated_cp = int((matrix_cp.sum(axis=0) == 0).sum())

# Calculate percentages
pct_cc = (unassociated_cc / total_phenotypes) * 100
pct_mf = (unassociated_mf / total_phenotypes) * 100
pct_cp = (unassociated_cp / total_phenotypes) * 100

print("\n" + "="*75)
print(f"      PHENOTYPE ANNOTATION STATISTICS (Total Phenotypes: {total_phenotypes})")
print("="*75)
print(f"Cellular Component: {unassociated_cc}/{total_phenotypes} unassociated phenotypes ({pct_cc:.2f}%)")
print(f"Molecular Function: {unassociated_mf}/{total_phenotypes} unassociated phenotypes ({pct_mf:.2f}%)")
print(f"Cellular Process:   {unassociated_cp}/{total_phenotypes} unassociated phenotypes ({pct_cp:.2f}%)")
print("="*75 + "\n")

if max(unassociated_cc, unassociated_mf, unassociated_cp) > 0:
    for label, mat in [('CC', matrix_cc), ('MF', matrix_mf), ('CP', matrix_cp)]:
        bad = mat.columns[mat.sum(axis=0) == 0].tolist()
        if bad:
            print(f"  {label}: {len(bad)} unassociated, e.g. {bad[:5]}")
    raise ValueError(
        "Some phenotypes have no GO annotation and would enter the similarity "
        "matrices as all-zero columns. Resolve before saving: either exclude "
        "them explicitly and state the number in the Methods, or find why their "
        "genes carry no GO annotation in that space."
    )


# # =============================================================================
# # 7.5. NORMALIZE MATRICES BY PHENOTYPE (Column fractions)
# # =============================================================================
# print("Normalizing matrices by phenotype...")

# matrix_cc_norm = matrix_cc.div(matrix_cc.sum(axis=0), axis=1).fillna(0)
# matrix_mf_norm = matrix_mf.div(matrix_mf.sum(axis=0), axis=1).fillna(0)
# matrix_cp_norm = matrix_cp.div(matrix_cp.sum(axis=0), axis=1).fillna(0)

# #Save them
# matrix_cc_norm.to_csv(os.path.join(path2, 'weighted_matrix_GO_Cellular_Component_norm.csv'), index=True)
# matrix_mf_norm.to_csv(os.path.join(path2, 'weighted_matrix_GO_Molecular_Function_norm.csv'), index=True)
# matrix_cp_norm.to_csv(os.path.join(path2, 'weighted_matrix_GO_Cellular_Process_norm.csv'), index=True)

# =============================================================================
# 7.6. COMPUTE COSINE SIMILARITY BETWEEN PHENOTYPES (Handling NaNs)
# =============================================================================
print("Computing Cosine Similarity matrices (filling empty vector NaNs with 0)...")

# Cellular Component
dist_cc = pdist(matrix_cc.T, metric='cosine')
cosine_sim_cc_array=1-dist_cc
cosine_sim_cc_array = np.nan_to_num(cosine_sim_cc_array, nan=0.0)
sim_matrix_cc = pd.DataFrame(
    1 - squareform(dist_cc), 
    index=matrix_cc.columns, 
    columns=matrix_cc.columns
).fillna(0)  # <-- .fillna(0) replaces NaNs from division-by-zero with 0 similarity

# Molecular Function
dist_mf = pdist(matrix_mf.T, metric='cosine')
cosine_sim_mf_array=1-dist_mf
cosine_sim_mf_array = np.nan_to_num(cosine_sim_mf_array, nan=0.0)
sim_matrix_mf = pd.DataFrame(
    1 - squareform(dist_mf), 
    index=matrix_mf.columns, 
    columns=matrix_mf.columns
).fillna(0)

# Cellular Process
dist_cp = pdist(matrix_cp.T, metric='cosine')
cosine_sim_cp_array=1-dist_cp
cosine_sim_cp_array = np.nan_to_num(cosine_sim_cp_array, nan=0.0)
sim_matrix_cp = pd.DataFrame(
    1 - squareform(dist_cp), 
    index=matrix_cp.columns, 
    columns=matrix_cp.columns
).fillna(0)

# =============================================================================
# 7.7. SAVE SIMILARITY MATRICES TO TEXT FILES (Tab-separated)
# =============================================================================
sim_matrix_cc.to_csv(os.path.join(path2, 'similarity_cosine_GO_Cellular_Component.txt'), sep='\t')
sim_matrix_mf.to_csv(os.path.join(path2, 'similarity_cosine_GO_Molecular_Function.txt'), sep='\t')
sim_matrix_cp.to_csv(os.path.join(path2, 'similarity_cosine_GO_Cellular_Process.txt'), sep='\t')

# =============================================================================
# 7.8. Figures
# =============================================================================
# 7.8.1.) CELLULAR PROCESS
#histogram
x1 = np.percentile(cosine_sim_cp_array, 50) # 33rd percentile (Tertile 1)
x2 = np.percentile(cosine_sim_cp_array, 75) # 66th percentile (Tertile 2)
x3 = np.percentile(cosine_sim_cp_array, 95) # 66th percentile (Tertile 2)
plt.figure(figsize=(4, 3), dpi=600)
plt.hist(cosine_sim_cp_array, bins=100, color='lightgray', alpha=1, log=True)
plt.axvline(x=x1, color='mediumslateblue', linestyle='--', linewidth=1.5, label='Median')
plt.axvline(x=x2, color='darkorange', linestyle='--', linewidth=1.5, label='75th Percentile')
plt.axvline(x=x3, color='crimson', linestyle='--', linewidth=1.5, label='95th Percentile')
plt.xlabel('Cosine similarity', fontsize=12)
plt.ylabel('# pairwise phenotype', fontsize=12)
plt.title('GO term-Phenotype associations\n(Cellular Process)', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=8, loc='lower left')
plt.tight_layout()
plt.grid(False)
plt.savefig(path2 + 'histogram_cosine_Sim_phen_cellular_process.png', dpi=600, bbox_inches='tight')
plt.show()


# 7.5.2.) CELLULAR COMPONENT
#histogram
x1 = np.percentile(cosine_sim_cc_array, 50) # 33rd percentile (Tertile 1)
x2 = np.percentile(cosine_sim_cc_array, 75) # 66th percentile (Tertile 2)
x3 = np.percentile(cosine_sim_cc_array, 95) # 66th percentile (Tertile 2)
plt.figure(figsize=(4, 3), dpi=600)
plt.hist(cosine_sim_cc_array, bins=100, color='lightgray', alpha=1, log=True)
plt.axvline(x=x1, color='mediumslateblue', linestyle='--', linewidth=1.5, label='Median')
plt.axvline(x=x2, color='darkorange', linestyle='--', linewidth=1.5, label='75th Percentile')
plt.axvline(x=x3, color='crimson', linestyle='--', linewidth=1.5, label='95th Percentile')
plt.xlabel('Cosine similarity', fontsize=12)
plt.ylabel('# pairwise phenotype', fontsize=12)
plt.title('GO term-Phenotype associations\n(Cellular Component)', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=8, loc='lower left')
plt.tight_layout()
plt.grid(False)
plt.savefig(path2 + 'histogram_cosine_Sim_phen_cellular_comp.png', dpi=600, bbox_inches='tight')
plt.show()


#7.5.3.) MOLECULAR FUNCTION
#histogram
x1 = np.percentile(cosine_sim_mf_array, 50) # 33rd percentile (Tertile 1)
x2 = np.percentile(cosine_sim_mf_array, 75) # 66th percentile (Tertile 2)
x3 = np.percentile(cosine_sim_mf_array, 95) # 66th percentile (Tertile 2)
plt.figure(figsize=(4, 3), dpi=600)
plt.hist(cosine_sim_mf_array, bins=100, color='lightgray', alpha=1, log=True)
plt.axvline(x=x1, color='mediumslateblue', linestyle='--', linewidth=1.5, label='Median')
plt.axvline(x=x2, color='darkorange', linestyle='--', linewidth=1.5, label='75th Percentile')
plt.axvline(x=x3, color='crimson', linestyle='--', linewidth=1.5, label='95th Percentile')
plt.xlabel('Cosine similarity', fontsize=12)
plt.ylabel('# pairwise phenotype', fontsize=12)
plt.title('GO term-Phenotype associations\n(Molecular Function)', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=8, loc='lower left')
plt.tight_layout()
plt.grid(False)
plt.savefig(path2 + 'histogram_cosine_Sim_phen_molec_func.png', dpi=600, bbox_inches='tight')
plt.show()


#7.6. Pearson correaltion between similarities
print(pearsonr(cosine_sim_cp_array, cosine_sim_cc_array))
print(pearsonr(cosine_sim_cp_array, cosine_sim_mf_array))
print(pearsonr(cosine_sim_mf_array, cosine_sim_cc_array))



# =============================================================================
# 8. PERMUTATION TEST (GENE SHUFFLING) - COMBO VALIDATION
# =============================================================================
import random

print("\n" + "="*75)
print(" STARTING PERMUTATION TEST (GENE SHUFFLING)")
print("="*75)
print("Running dual validation: Intra-space similarity & Inter-space correlation...")

n_permutations = 100
SEED = 42
random.seed(SEED)   # the null resamples gene sets, so the seed must be fixed

# 1. Distributions for the Artifact Test (Pairwise Similarities)
null_medians_cc = []
null_medians_mf = []
null_medians_cp = []

# 2. Distributions for the Pattern Test (Pearson between spaces)
null_corrs_cp_cc = []
null_corrs_cp_mf = []
null_corrs_mf_cc = []

gene_pool = list(all_needed_genes)

for i in range(n_permutations):
    if (i + 1) % 10 == 0:
        print(f"  Running permutation {i + 1}/{n_permutations}...")
        
    shuffled_pheno_to_genes = {}
    for pheno, genes in pheno_to_genes.items():
        shuffled_pheno_to_genes[pheno] = set(random.sample(gene_pool, len(genes)))
        
    temp_cc_data, temp_mf_data, temp_cp_data = [], [], []
    # PHENOS, not phen_filter: iterating the set would change the column order
    # of the three matrices from one permutation to the next.
    for pheno in PHENOS:
        shuffled_genes = shuffled_pheno_to_genes.get(pheno, set())
        cc_c, mf_c, cp_c = Counter(), Counter(), Counter()
        
        for g in shuffled_genes:
            cc_c.update(gene_to_cc.get(g, set()))
            mf_c.update(gene_to_mf.get(g, set()))
            cp_c.update(gene_to_cp.get(g, set()))
            
        temp_cc_data.append({'Phenotype': pheno, 'GO_Cellular_Component': dict(cc_c)})
        temp_mf_data.append({'Phenotype': pheno, 'GO_Molecular_Function': dict(mf_c)})
        temp_cp_data.append({'Phenotype': pheno, 'GO_Cellular_Process': dict(cp_c)})
        
    temp_matrix_cc = build_weighted_matrix(temp_cc_data, 'GO_Cellular_Component', ic_cc)
    temp_matrix_mf = build_weighted_matrix(temp_mf_data, 'GO_Molecular_Function', ic_mf)
    temp_matrix_cp = build_weighted_matrix(temp_cp_data, 'GO_Cellular_Process', ic_cp)
    
    # Normalise by phenotype. Cosine similarity is invariant to scaling each
    # vector, so this yields exactly the same similarities as the unnormalised
    # observed matrices of section 7.6; it is kept only for readability.
    temp_matrix_cc_norm = temp_matrix_cc.div(temp_matrix_cc.sum(axis=0), axis=1).fillna(0)
    temp_matrix_mf_norm = temp_matrix_mf.div(temp_matrix_mf.sum(axis=0), axis=1).fillna(0)
    temp_matrix_cp_norm = temp_matrix_cp.div(temp_matrix_cp.sum(axis=0), axis=1).fillna(0)
    
    # Compute temporary Cosine Similarities
    temp_sim_cc = np.nan_to_num(1 - pdist(temp_matrix_cc_norm.T, metric='cosine'), nan=0.0)
    temp_sim_mf = np.nan_to_num(1 - pdist(temp_matrix_mf_norm.T, metric='cosine'), nan=0.0)
    temp_sim_cp = np.nan_to_num(1 - pdist(temp_matrix_cp_norm.T, metric='cosine'), nan=0.0)
    
    # Save Medians (Artifact Check)
    null_medians_cc.append(np.median(temp_sim_cc))
    null_medians_mf.append(np.median(temp_sim_mf))
    null_medians_cp.append(np.median(temp_sim_cp))
    
    # Save Pearson Correlations (Pattern Check)
    null_corrs_cp_cc.append(pearsonr(temp_sim_cp, temp_sim_cc)[0])
    null_corrs_cp_mf.append(pearsonr(temp_sim_cp, temp_sim_mf)[0])
    null_corrs_mf_cc.append(pearsonr(temp_sim_mf, temp_sim_cc)[0])

# =============================================================================
# 9. EVALUATE PERMUTATION RESULTS
# =============================================================================
# Get observed medians
obs_median_cc = np.median(cosine_sim_cc_array)
obs_median_mf = np.median(cosine_sim_mf_array)
obs_median_cp = np.median(cosine_sim_cp_array)

# Get observed Pearsons (make sure these variables are defined in your step 7.7)
obs_cp_cc = pearsonr(cosine_sim_cp_array, cosine_sim_cc_array)[0]
obs_cp_mf = pearsonr(cosine_sim_cp_array, cosine_sim_mf_array)[0]
obs_mf_cc = pearsonr(cosine_sim_mf_array, cosine_sim_cc_array)[0]

# Calculate Empirical P-values
# (1 + count) / (n + 1), so the p-value is never exactly 0 and the floor it can
# reach is explicit. With 100 permutations that floor is 0.0099.
def emp_p(null, obs):
    return (1 + sum(1 for x in null if x >= obs)) / (n_permutations + 1)

pval_med_cc = emp_p(null_medians_cc, obs_median_cc)
pval_med_mf = emp_p(null_medians_mf, obs_median_mf)
pval_med_cp = emp_p(null_medians_cp, obs_median_cp)

pval_corr_cp_cc = emp_p(null_corrs_cp_cc, obs_cp_cc)
pval_corr_cp_mf = emp_p(null_corrs_cp_mf, obs_cp_mf)
pval_corr_mf_cc = emp_p(null_corrs_mf_cc, obs_mf_cc)

print("\n" + "="*75)
print(" FINAL VALIDATION REPORT")
print("="*75)
print(f"Empirical p-values from {n_permutations} permutations; "
      f"smallest attainable value = {1/(n_permutations+1):.4f}")
print("TEST 1: ARE THE SIMILARITIES BIOLOGICALLY REAL? (Intra-space Medians)")
print(f"  CC : Obs = {obs_median_cc:.4f} | Null = {np.mean(null_medians_cc):.4f} | p-value = {pval_med_cc:.4f}")
print(f"  MF : Obs = {obs_median_mf:.4f} | Null = {np.mean(null_medians_mf):.4f} | p-value = {pval_med_mf:.4f}")
print(f"  CP : Obs = {obs_median_cp:.4f} | Null = {np.mean(null_medians_cp):.4f} | p-value = {pval_med_cp:.4f}")

print("\nTEST 2: ARE THE PATTERNS CONSERVED ACROSS SPACES? (Inter-space Pearson)")
print(f"  CP vs CC : Obs = {obs_cp_cc:.4f} | Null = {np.mean(null_corrs_cp_cc):.4f} | p-value = {pval_corr_cp_cc:.4f}")
print(f"  CP vs MF : Obs = {obs_cp_mf:.4f} | Null = {np.mean(null_corrs_cp_mf):.4f} | p-value = {pval_corr_cp_mf:.4f}")
print(f"  MF vs CC : Obs = {obs_mf_cc:.4f} | Null = {np.mean(null_corrs_mf_cc):.4f} | p-value = {pval_corr_mf_cc:.4f}")
print("="*75 + "\n")


# =============================================================================
# 10. PLEIOTROPY ANALYSIS (PHEN VS GO)
# =============================================================================
# 10.1. Pleiotropy per gene to phenotype
gene_to_phenotypes_count = Counter()
for pheno, genes_set in pheno_to_genes.items():
    for g in genes_set:
        gene_to_phenotypes_count[g] += 1

# 10.2. Pleiotropy per gene to all the three GO ontologies
# Propagte he categories
all_genes_in_analysis = set(gene_to_phenotypes_count.keys())

gene_to_go_total_count = {}
for g in all_genes_in_analysis:
    union_gos = gene_to_cc.get(g, set()) | gene_to_mf.get(g, set()) | gene_to_cp.get(g, set())
    gene_to_go_total_count[g] = len(union_gos)

# 10.3. Pleiotropy DataFrame
df_pleiotropy = pd.DataFrame({
    'Phen_Degree': pd.Series(gene_to_phenotypes_count),
    'GO_Degree': pd.Series(gene_to_go_total_count)
}).fillna(0) 

# =============================================================================
# Pleio in GO vs pleio in phenotypes
# =============================================================================
rho, p_val = pearsonr(df_pleiotropy['Phen_Degree'], df_pleiotropy['GO_Degree'])

plt.figure(figsize=(3.9, 3.1), dpi=600)
sns.scatterplot(data=df_pleiotropy, x='GO_Degree', y='Phen_Degree', alpha=0.5, color='darkcyan', edgecolor='w', s=30)
plt.title(f'Pearson r: {rho:.3f} (p-value: {p_val:.2e})', fontsize=10)
plt.xlabel('# associated GO terms', fontsize=12, fontweight='bold')
plt.ylabel('# associated phenotypes', fontsize=12, fontweight='bold')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(path2, 'diagnostico_pleiotropia_go_vs_phen.png'), dpi=600, bbox_inches='tight')
plt.show()

print(f"-> Correlación de Pearson entre grados: r = {rho:.3f} (p = {p_val:.2e})")


# =============================================================================
# 11. DROP-OUT: recomputing without most pleiotropic genes
# =============================================================================
# =============================================================================
# DROP-OUT (We eliminate the top 25% most pleiotropic genes
# =============================================================================
df_pleiotropy['Total_Score'] = df_pleiotropy['Phen_Degree'] * df_pleiotropy['GO_Degree']

# The drop-out removes the top QUARTILE of genes by Total_Score (phenotype
# degree x GO degree), not a fixed number of genes. The number actually removed
# is printed below and is the one to quote in the text.
q_cutoff = df_pleiotropy['Total_Score'].quantile(0.75)
genes_to_keep = set(df_pleiotropy[df_pleiotropy['Total_Score'] < q_cutoff].index)

print(f"\n[Drop-out Test]: El percentil 75 del Total_Score es {q_cutoff:.2f}")
print(f"Si filtramos los genes promiscuos, mantendremos {len(genes_to_keep)} genes de los {len(df_pleiotropy)} totales.")
print(f"Se eliminarán {len(df_pleiotropy) - len(genes_to_keep)} genes omnipresentes "
      f"({100*(len(df_pleiotropy)-len(genes_to_keep))/len(df_pleiotropy):.1f}% del total). "
      f"<-- este es el numero a citar en el texto")


# 11.1. Filter genes
pheno_to_genes_filtered = {}
for pheno, genes_set in pheno_to_genes.items():
    pheno_to_genes_filtered[pheno] = genes_set.intersection(genes_to_keep)

# 11.2. Recompute weights to GO per phen (filtered genes)
print("Recalculando anotaciones GO sin los super-genes...")
pheno_cc_data_filt, pheno_mf_data_filt, pheno_cp_data_filt = [], [], []

for pheno in PHENOS:
    genes = pheno_to_genes_filtered.get(pheno, set())
    
    cc_counter, mf_counter, cp_counter = Counter(), Counter(), Counter()
    
    for g in genes:
        cc_counter.update(gene_to_cc.get(g, set()))
        mf_counter.update(gene_to_mf.get(g, set()))
        cp_counter.update(gene_to_cp.get(g, set()))
        
    pheno_cc_data_filt.append({'Phenotype': pheno, 'GO_Cellular_Component': dict(cc_counter)})
    pheno_mf_data_filt.append({'Phenotype': pheno, 'GO_Molecular_Function': dict(mf_counter)})
    pheno_cp_data_filt.append({'Phenotype': pheno, 'GO_Cellular_Process': dict(cp_counter)})

# 11.3. Compute new matrices
print("Construyendo nuevas matrices ponderadas...")
matrix_cc_filt = build_weighted_matrix(pheno_cc_data_filt, 'GO_Cellular_Component', ic_cc)
matrix_mf_filt = build_weighted_matrix(pheno_mf_data_filt, 'GO_Molecular_Function', ic_mf)
matrix_cp_filt = build_weighted_matrix(pheno_cp_data_filt, 'GO_Cellular_Process', ic_cp)

# 11.4. Recompute similarity
print("Calculando nuevas matrices de similitud del coseno...")
def get_cosine_sim_array(matrix):
    dist = pdist(matrix.T, metric='cosine')
    sim_array = 1 - dist
    return np.nan_to_num(sim_array, nan=0.0)

cosine_sim_cc_filt_array = get_cosine_sim_array(matrix_cc_filt)
cosine_sim_mf_filt_array = get_cosine_sim_array(matrix_mf_filt)
cosine_sim_cp_filt_array = get_cosine_sim_array(matrix_cp_filt)

# =============================================================================
# 11.5. Figure: ORIGINAL VS DROP-OUT
# =============================================================================
print("Generando gráficos comparativos...")

def plot_comparative_histogram(original_array, filtered_array, title, filename_suffix):
    plt.figure(figsize=(5, 4), dpi=600)
    
    plt.hist(original_array, bins=100, color='lightgray', alpha=0.6, log=True, label='Original (all genes)')
    plt.hist(filtered_array, bins=100, color='mediumseagreen', alpha=0.7, log=True, label='Without most pleiotropic genes (top 25%)')
    med = np.percentile(filtered_array, 50)
    plt.axvline(x=med, color='darkgreen', linestyle='--', linewidth=1.5, label=f'New median: {med:.2f}')
    plt.xlabel('Cosine similarity', fontsize=18)
    plt.ylabel('# pairwise phenotype', fontsize=18)
    plt.title(f'{title}', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(os.path.join(path2, f'histogram_comparative_{filename_suffix}.png'), dpi=600, bbox_inches='tight')
    plt.show()

plot_comparative_histogram(cosine_sim_cp_array, cosine_sim_cp_filt_array, 
                           "GO: Cellular Process", "cellular_process")

plot_comparative_histogram(cosine_sim_cc_array, cosine_sim_cc_filt_array, 
                           "GO: Cellular Component", "cellular_comp")

plot_comparative_histogram(cosine_sim_mf_array, cosine_sim_mf_filt_array, 
                           "GO: Molecular Function", "molec_func")

print("\n¡Experimento Drop-out completado con éxito! Revisa los gráficos generados.")

pearsonr(cosine_sim_cp_filt_array, cosine_sim_cc_filt_array)
pearsonr(cosine_sim_cp_filt_array, cosine_sim_mf_filt_array)
pearsonr(cosine_sim_cc_filt_array, cosine_sim_mf_filt_array)

# =============================================================================
# END OF THE SCRIPT
# =============================================================================
# The three similarity matrices written in section 7.7 are the GO layers used by
# section1_layer_comparison.py, section2_modularity.py and
# disjoint_integration.py. Everything after section 7.7 validates them: the
# gene-shuffling null (sections 8-9), the relation between GO and phenotype
# degree per gene (section 10) and the drop-out of the most pleiotropic genes
# (section 11).
#
# The within- versus between-category comparison of each layer is NOT here: it
# is done on the breadth-residualised matrices, with the probability of
# superiority, in section2_modularity.py. The effect of annotation breadth on
# each layer is in section1_layer_comparison.py (Block 2).
print("\nDone.")
