"""
Phenotypic correlation across the genes 
"""
import numpy as np
import pandas as pd
import os
import warnings
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import obonet
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import networkx as nx
import obonet
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import LogNorm
import itertools

warnings.simplefilter('ignore')

# Set Paths
path = 'PATH'
path2 = 'PATH'
url = 'PATH/MPheno_OBO.ontology.obo'

# =============================================================================
# 1. LOAD SIGNIFICANT KS-TEST RESULTS
# =============================================================================
m_types = np.loadtxt(os.path.join(path, 'm_types.txt'))
n_cells_per_type = np.sum(m_types, axis=1)

with open(os.path.join(path, 'cell_type_unique_mouse.txt'), 'r') as f:
    cell_type_unique = np.array(f.read().split('\n')[:-1])

with open(os.path.join(path, 'phen_filtro_20_500_genes_no_redundantes_final.txt'), 'r') as f:
    phen = np.array(f.read().split('\n')[:-1])

cell_type_ks_test = np.loadtxt(os.path.join(path, 'cell_type_ks_test.txt'), dtype=float)
cell_type_p_value = np.loadtxt(os.path.join(path, 'cell_type_p_value.txt'), dtype=float)

# Filter clusters with > 50 cells
ind_good_cells = np.where(n_cells_per_type > 50)[0]
filt_cell_type = cell_type_unique[ind_good_cells]
filt_ks_test = cell_type_ks_test[:, ind_good_cells]
filt_p_value = cell_type_p_value[:, ind_good_cells]

# Get indices for significance P < 0.0001
sig_phen_idx, sig_cell_idx = np.where(filt_p_value < 0.0001)

df_significant = pd.DataFrame({
    'Phenotype': phen[sig_phen_idx],
    'Query_Cluster': filt_cell_type[sig_cell_idx],
    'KS': filt_ks_test[sig_phen_idx, sig_cell_idx],
    'P_Value': filt_p_value[sig_phen_idx, sig_cell_idx]
})

# 1.1. Safely read the files: keep spaces inside names, but remove blank lines and trailing spaces
# clean_matrix.txt, clean_genes_matrix.txt and clean_phen_matrix.txt come from
# phen_matrix_construction.py; copy them into path2 before running this script.
with open(path2 + 'clean_genes_matrix.txt', 'r') as f:
    genes = [line.strip() for line in f if line.strip()]

with open(path2 + 'clean_phen_matrix.txt', 'r') as f:
    phenotypes = [line.strip() for line in f if line.strip()]

with open(path2 + 'phen_filtro_20_500_genes_no_redundantes_final.txt', 'r') as f:
    phen_filter = [line.strip() for line in f if line.strip()]

# 1.2. Read the main matrix 
# \s+ handles the spaces between your matrix numbers, regardless of how many there are
matrix_data = pd.read_csv(path2 + 'clean_matrix.txt', sep=r'\s+', header=None)

# 1.3. Assign genes as index (rows) and phenotypes as columns
matrix_data.index = genes
matrix_data.columns = phenotypes

# 1.4. Filter the matrix
# This will now perfectly match "abnormal adipose tissue morphology" between both lists
common_phenotypes = matrix_data.columns.intersection(phen_filter)
filtered_matrix = matrix_data[common_phenotypes]

# 1.5. Save the resulting matrix
# We save it using tabs so the spaces inside the phenotype names don't break your columns!
filtered_matrix.to_csv(path2 + 'final_filtered_matrix.txt', sep='\t')

print(f"The filtered matrix has {filtered_matrix.shape[0]} genes and {filtered_matrix.shape[1]} phenotypes.")

genes_per_phen=np.sum(filtered_matrix, axis=0)
genes_per_phen.to_csv(path2 + 'genes_per_phenotype.txt', sep='\t')


#============================================================================================
#2.) COMPUTE THE JACCARD CORRELATION BETWEEN FILTERED PHENOTYPES
#============================================================================================
jaccard_dist_array = pdist(filtered_matrix.T, metric='jaccard')
jaccard_sim_array=1-jaccard_dist_array

plt.hist(jaccard_sim_array, bins=100, log=True)
plt.show()

x1 = np.percentile(jaccard_sim_array, 50) # 33rd percentile (Tertile 1)
x2 = np.percentile(jaccard_sim_array, 75) # 66th percentile (Tertile 2)
x3 = np.percentile(jaccard_sim_array, 95) # 66th percentile (Tertile 2)

plt.figure(figsize=(4, 3), dpi=600)
plt.hist(jaccard_sim_array, bins=100, color='lightgray', alpha=1, log=True)
plt.axvline(x=x1, color='mediumslateblue', linestyle='--', linewidth=1.5, label=f'Median')
plt.axvline(x=x2, color='darkorange', linestyle='--', linewidth=1.5, label=f'75th Percentile')
plt.axvline(x=x3, color='crimson', linestyle='--', linewidth=1.5, label=f'95th Percentile')
plt.xlabel('Jaccard similarity', fontsize=12)
plt.ylabel('# pairwise phenotypes', fontsize=12)
plt.title('Gene-Phenotype associations', fontsize=12, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.grid(False)
plt.savefig(path2 + 'histogram_jaccard_Sim_phen.png', dpi=600, bbox_inches='tight')
plt.show()

# 2.1. Convert the 1D array back into a readable 2D dataframe
jaccard_distance_matrix = pd.DataFrame(
    squareform(jaccard_dist_array),
    index=filtered_matrix.columns,
    columns=filtered_matrix.columns
)

# 2.2. (Optional) Convert to Jaccard SIMILARITY
# If you prefer 1 to mean "identical" and 0 to mean "no overlap", just subtract from 1:
jaccard_similarity_matrix = 1 - jaccard_distance_matrix

# 2.3. Save the similarity matrix
jaccard_similarity_matrix.to_csv(path2 + 'phenotype_jaccard_similarity.txt', sep='\t')

print(f"Jaccard matrix calculated: {jaccard_similarity_matrix.shape[0]}x{jaccard_similarity_matrix.shape[1]}")



# ============================================================================================
# 2.4.) TASK A: MEDIAN SIMILARITY PER PHENOTYPE VS NUMBER OF ASSOCIATED GENES
# ============================================================================================

# 2.4.1. To avoid skewing the median, we replace the diagonal (which is 1.0) with NaNs
jaccard_sim_no_diag = jaccard_similarity_matrix.copy()
np.fill_diagonal(jaccard_sim_no_diag.values, np.nan)

# 2.4.2. Calculate the median similarity for each phenotype (ignoring the NaNs on the diagonal)
median_sim_per_phen = jaccard_sim_no_diag.median(axis=1, skipna=True)

# 2.4.3. Calculate the Pearson correlation
# Note: genes_per_phen was computed earlier in your code
pearson_r_A, p_val_A = pearsonr(genes_per_phen, median_sim_per_phen)

# 2.4.4. Scatter plot for Task A
plt.figure(figsize=(5, 4), dpi=600)
plt.scatter(genes_per_phen, median_sim_per_phen, alpha=0.7, edgecolors='w', s=50, color='mediumslateblue')
plt.xlabel('Number of Genes per Phenotype', fontsize=10)
plt.ylabel('Median Jaccard Similarity', fontsize=10)
plt.title(f'Median Similarity vs. Gene Count\nPearson r: {pearson_r_A:.3f} (p-val: {p_val_A:.2e})', fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(path2 + 'scatter_median_sim_vs_genes.png', dpi=600, bbox_inches='tight')
plt.show()

print(f"Task A - Pearson Correlation (Median Sim vs Genes): r = {pearson_r_A:.3f}, p = {p_val_A:.2e}")

#sliding window figure
df_A = pd.DataFrame({'X': genes_per_phen.values, 'Y': median_sim_per_phen.values})
df_A = df_A.sort_values(by='X').reset_index(drop=True)

window_size_A = 100 
df_A['X_smooth'] = df_A['X'].rolling(window=window_size_A, center=True).mean()
df_A['Y_smooth'] = df_A['Y'].rolling(window=window_size_A, center=True).mean()

plt.figure(figsize=(4.1, 3.1), dpi=600)
plt.scatter(df_A['X'], df_A['Y'], alpha=0.5, edgecolors='w', s=32, color='mediumslateblue')
plt.plot(df_A['X_smooth'], df_A['Y_smooth'], color='indigo', linewidth=1.5, label=f'Trend (sw={window_size_A})')

plt.xlabel('# genes per phenotype', fontsize=12, fontweight='bold')
plt.ylabel('Median Jaccard similarity', fontsize=12, fontweight='bold')
plt.title(f'Pearson r: {pearson_r_A:.3f} (p-value: {p_val_A:.2e})', fontsize=10)
plt.legend(fontsize=9)
plt.tight_layout()
plt.grid(False)
plt.savefig(path2 + 'scatter_median_sim_vs_genes_sliding.png', dpi=600, bbox_inches='tight')
plt.show()

# ============================================================================================
# 2.5.) TASK B: PAIRWISE SIMILARITY VS AVERAGE NUMBER OF GENES PER PAIR
# ============================================================================================

# 2.5.1. Get the gene counts as a flat numpy array
gene_counts_array = genes_per_phen.values

# 2.5.2. scipy's pdist creates pairwise arrays in the exact combination order of (0,1), (0,2)...(n-2, n-1).
# We use itertools.combinations to replicate this exact order and calculate the average genes for each pair.
avg_genes_per_pair = np.array([
    (gene_counts_array[i] + gene_counts_array[j]) / 2.0
    for i, j in itertools.combinations(range(len(gene_counts_array)), 2)
])

# 2.5.3. Calculate the Pearson correlation between the pairwise averages and the pairwise Jaccard similarities
# Note: jaccard_sim_array was calculated earlier in your code
pearson_r_B, p_val_B = pearsonr(avg_genes_per_pair, jaccard_sim_array)

print(f"Task B - Pearson Correlation (Pairwise Sim vs Avg Pair Genes): r = {pearson_r_B:.3f}, p = {p_val_B:.2e}")

#sliding window figure
df_B = pd.DataFrame({'X': avg_genes_per_pair, 'Y': jaccard_sim_array})
df_B = df_B.sort_values(by='X').reset_index(drop=True)

window_size_B = 500  
df_B['X_smooth'] = df_B['X'].rolling(window=window_size_B, center=True).mean()
df_B['Y_smooth'] = df_B['Y'].rolling(window=window_size_B, center=True).mean()

plt.figure(figsize=(4, 3.1), dpi=600)
plt.scatter(df_B['X'], df_B['Y'], alpha=0.05, s=4, color='darkorange')
plt.plot(df_B['X_smooth'], df_B['Y_smooth'], color='crimson', linewidth=1.5, label=f'Trend (sw={window_size_B})')
plt.xlabel('Average number of genes per pair', fontsize=12, fontweight='bold')
plt.ylabel('Pairwise Jaccard similarity', fontsize=12, fontweight='bold')
plt.title(f'Pearson r: {pearson_r_B:.3f} (p-value: {p_val_B:.2e})', fontsize=10)
plt.grid(False)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(path2 + 'scatter_pairwise_sim_vs_avg_genes_sliding.png', dpi=600, bbox_inches='tight')
plt.show()




#=========================================================================================
#3. CLUSTER AND PAINT THE CORRELATION MATRIX
#=========================================================================================
# # 3.1. linkage
# # jaccard_dist_array que calculamos en el paso de Jaccard.
# # 'average' (UPGMA) -> standard method for Jaccard
# enlaces = linkage(jaccard_dist_array, method='average')
# sorted_indices = leaves_list(enlaces)
# sorted_phenotypes = jaccard_similarity_matrix.columns[sorted_indices].tolist()

# matriz_ordenada = jaccard_similarity_matrix.iloc[sorted_indices, sorted_indices]
# mi_paleta = plt.get_cmap("gnuplot_r").copy()
# mi_paleta.set_under("white")

# # 3.2. Heatmap 
# plt.figure(figsize=(16, 36)) 
# ax = sns.heatmap(
#     matriz_ordenada,
#     cmap=mi_paleta,
#     vmin=1e-6,               
#     xticklabels=False,       # Quitamos las del eje X para limpiar la imagen
#     yticklabels=10,           # y axis with 1 label in each 10 steps
#     cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.5} # shrink hace la barra de color un poco más pequeña
# )
# plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
# plt.title("Jaccard similarity (between phenotypes) trough genes", fontsize=16, pad=20)
# plt.savefig(path2 + 'phenotype_jaccard_heatmap_genes.png', dpi=600, bbox_inches='tight')
# plt.show()

# #3.4.) Save the phen
# with open(path2 + 'phenotypes_sorted_by_clustering.txt', 'w') as f:
#     for p in sorted_phenotypes:
#         f.write(f"{p}\n")



#logscale figure
# 3.1. linkage
# jaccard_dist_array calculated in the Jaccard step.
# 'average' (UPGMA) -> standard method for Jaccard
enlaces = linkage(jaccard_dist_array, method='average')
sorted_indices = leaves_list(enlaces)
sorted_phenotypes = jaccard_similarity_matrix.columns[sorted_indices].tolist()

matriz_ordenada = jaccard_similarity_matrix.iloc[sorted_indices, sorted_indices]
mi_paleta = plt.get_cmap("viridis").copy()
mi_paleta.set_under("white")

# 3.2. Heatmap 
plt.figure(figsize=(16, 36)) 
ax = sns.heatmap(
    matriz_ordenada,
    cmap=mi_paleta,
    norm=LogNorm(vmin=1e-6, vmax=1.0),       # --- NEW: Logarithmic scale (0 remains white due to vmin)
    xticklabels=False,                       # Remove X axis labels to clean the image
    yticklabels=10,                          # Y axis with 1 label every 10 steps
    cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.5} # Shrink makes the color bar slightly smaller
)

# --- NEW: Increase the size of the colorbar numbers and label ---
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)            # Increases the numbers on the colorbar
cbar.set_label('Jaccard Similarity', size=16) # Increases the 'Jaccard Similarity' text

plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.title("Jaccard similarity (between phenotypes) trough genes", fontsize=16, pad=20)
plt.savefig(path2 + 'phenotype_jaccard_heatmap_genes.png', dpi=600, bbox_inches='tight')
plt.show()

#3.4.) Save the phen
with open(path2 + 'phenotypes_sorted_by_clustering.txt', 'w') as f:
    for p in sorted_phenotypes:
        f.write(f"{p}\n")

#=======================================================================================
#4.) Compare the jaccard distances between families of broad phenotypes
#=======================================================================================
graph = obonet.read_obo(url)
# Create basic mappings
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}

# --- 4.1. Dynamically Find Top Terms (Broad Categories) ---
root_name = 'mammalian phenotype'
root_id = name_to_id.get(root_name) 

all_broad_ids = set()
meta_dict = {}

if root_id and root_id in graph:
    children_ids = list(graph.predecessors(root_id))
    
    print(f"Direct children '{root_name}' ({root_id}):")
    print("-" * 50)
    
    for child_id in children_ids:
        child_name = id_to_name.get(child_id, "Unknown Name")
        print(f"{child_id} : {child_name}")
        
        # Populate our tracking sets dynamically
        all_broad_ids.add(child_id)
        meta_dict[child_id] = child_name  # e.g., 'MP:0001186': 'pigmentation phenotype'
        
    print("-" * 50)
    print(f"Principal categories: {len(children_ids)}")
else:
    print(f"The term '{root_name}' wasn't in the ontology.")

# --- 4.2. Create a Robust Dictionary (Lowercase + Synonyms) ---
robust_name_to_id = {}
for id_, data in graph.nodes(data=True):
    if 'name' in data:
        official_name = data['name'].strip().lower()
        robust_name_to_id[official_name] = id_
    
    if 'synonym' in data:
        for syn in data['synonym']:
            if '"' in syn:
                syn_text = syn.split('"')[1].strip().lower()
                robust_name_to_id[syn_text] = id_

# --- 4.3. Function to Retrieve ALL Broad IDs ---
def get_broad_mp_ids_all(phen_name, graph, robust_dict, broad_id_set):
    clean_name = phen_name.strip().lower()
    phen_id = robust_dict.get(clean_name)
    
    if not phen_id or phen_id not in graph:
        return [] 
        
    ancestors = nx.descendants(graph, phen_id)
    reachable_nodes = ancestors.union({phen_id})
    matches = reachable_nodes.intersection(broad_id_set)
    
    return list(matches) 

# --- 4.4. Map Phenotypes to their Broad Families ---
# Assuming 'jaccard_similarity_matrix' is already loaded in your workspace from previous steps
filtered_phens = jaccard_similarity_matrix.columns.tolist()

# Initialize dictionary with the actual names of the 28 categories
family_to_phens = {meta: [] for meta in set(meta_dict.values())}

for p_name in filtered_phens:
    broad_ids = get_broad_mp_ids_all(p_name, graph, robust_name_to_id, all_broad_ids)
    
    # Translate those MP IDs into the broad category names
    meta_categories = set(meta_dict[mp_id] for mp_id in broad_ids if mp_id in meta_dict)
    
    # If it belongs to 3 categories, it gets appended to all 3 lists
    for meta in meta_categories:
        family_to_phens[meta].append(p_name)

# --- 4.5. Extract Similarities, Count Phenotypes & Statistical Testing ---
plot_data = []
stats_results = []

for family, phens in family_to_phens.items():
    phens_in = [p for p in phens if p in jaccard_similarity_matrix.columns]
    phens_out = [p for p in jaccard_similarity_matrix.columns if p not in phens_in]
    
    n_phens = len(phens_in)
    
    # Skip families with less than 2 phenotypes in the filtered matrix
    if n_phens < 2 or len(phens_out) == 0:
        continue
        
    # --- NEW: Append the number of phenotypes to the family label ---
    family_label = f"{family} (n={n_phens})"
        
    # A) Within-Family
    sub_matrix_in = jaccard_similarity_matrix.loc[phens_in, phens_in].values
    row_idx, col_idx = np.triu_indices(n_phens, k=1)
    within_sims = sub_matrix_in[row_idx, col_idx]
    
    # B) Between-Family
    sub_matrix_out = jaccard_similarity_matrix.loc[phens_in, phens_out].values
    between_sims = sub_matrix_out.flatten() 
    
    for val in within_sims:
        plot_data.append({'Family': family_label, 'Comparison': 'Within category', 'Similarity': val})
    for val in between_sims:
        plot_data.append({'Family': family_label, 'Comparison': 'Between categories', 'Similarity': val})
        
    # C) Mann-Whitney U test (Within > Between)
    stat, p_val = mannwhitneyu(within_sims, between_sims, alternative='greater')
    
    stats_results.append({
        'Family': family_label,
        'N_Phenotypes': n_phens,
        'Within_Median': np.median(within_sims),
        'Between_Median': np.median(between_sims),
        'P-Value': p_val
    })

# --- 4.6. Sorting, Results & Plotting ---
df_plot = pd.DataFrame(plot_data)
df_stats = pd.DataFrame(stats_results)

df_stats = df_stats.sort_values(by='Within_Median', ascending=False)
sorted_families = df_stats['Family'].tolist()

print("\n--- Mann-Whitney U Test Results (Sorted by Within Median) ---")
print(df_stats.to_string(index=False))

def get_significance(p):
    if p < 0.0001: return '****'
    elif p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns' # ns = no significativo

# figure
plt.figure(figsize=(14, 8))
ax = sns.boxplot(
    data=df_plot, 
    x='Family', 
    y='Similarity', 
    hue='Comparison', 
    palette=['deepskyblue', 'deeppink'],
    showfliers=False, # Oculta los outliers extremos en el dibujo
    order=sorted_families
)

y_min, y_max = ax.get_ylim()
y_range = y_max - y_min
ax.set_ylim(y_min, y_max + y_range * 0.15) # Añadimos un 15% de espacio arriba

for i, family in enumerate(sorted_families):
    p_val = df_stats.loc[df_stats['Family'] == family, 'P-Value'].values[0]
    sig_symbol = get_significance(p_val)
    if sig_symbol != 'ns':
        family_data = df_plot[df_plot['Family'] == family]
        local_max = 0
        
        # CORRECCIÓN: Usar .unique() evita fallos por mayúsculas/minúsculas en el texto
        for comp in family_data['Comparison'].unique():
            comp_data = family_data[family_data['Comparison'] == comp]['Similarity']
            if not comp_data.empty:
                q1 = comp_data.quantile(0.25)
                q3 = comp_data.quantile(0.75)
                iqr = q3 - q1
                whisker_max = q3 + 1.5 * iqr
                actual_whisker_max = comp_data[comp_data <= whisker_max].max()
                if pd.notna(actual_whisker_max) and actual_whisker_max > local_max:
                    local_max = actual_whisker_max
                    
        # CORRECCIÓN: Aumentamos un pelín el margen (0.05) para despegarlo bien de la caja
        y = local_max + y_range * 0.05  
        h = y_range * 0.02              
        
        x1, x2 = i - 0.2, i + 0.2    
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, color='black')
        plt.text((x1+x2)*.5, y+h, sig_symbol, ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')

# plt.title('Jaccard Similarity: Within vs. Between Broad Phenotype Families', fontsize=16, pad=15)
plt.ylabel('Jaccard similarity\n(Gene level)', fontsize=20)
plt.xlabel('Broad phenotype category', fontsize=20)
plt.xticks(rotation=90, ha='center', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper right', fontsize=16, title_fontsize=16)
plt.grid(False)
plt.tight_layout()
plt.savefig(path2 + 'family_similarity_boxplot_sorted_annotated.png', dpi=600, bbox_inches='tight')
plt.show()
