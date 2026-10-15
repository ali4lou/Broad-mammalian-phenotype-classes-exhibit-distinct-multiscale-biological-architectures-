#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALIDATION OF THE PHENOTYPE x CELL TYPE SPACE AGAINST THE MP ONTOLOGY

The Cell Ontology (CL) files link some of their cell-type terms to Mammalian
Phenotype (MP) terms. Those curated MP-CL pairs (mp_cl_associations.tsv) are
used here as an external standard for the significant phenotype x cell-type
associations of the KS analysis (p < KS_PVALUE_CUTOFF in cell types with more
than MIN_CELLS_PER_TYPE cells).

The atlas cluster names are mapped to CL identifiers with a combined automated
and manual approach, modifying or removing the associations that do not align
(mapeo_manual_IA.xlsx); the CL identifier of each curated name is then read from
cl.obo by exact name match. An association counts as validated when its cell
type is one the ontology links to that phenotype, or a descendant of one. Only
'is_a' edges between CL terms are followed.

The script then compares the KS statistics of the validated and non-validated
associations, and reports how often the validated cell type is the one with the
highest KS of its phenotype.

"""

import inspect
import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
import obonet
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp

warnings.simplefilter('ignore')

# Set Paths
path = 'PATH'
manual_mapping_file = os.path.join(path, 'mapeo_manual_IA.xlsx')
obo_path = os.path.join(path, 'cl.obo')

MIN_CELLS_PER_TYPE = 50
KS_PVALUE_CUTOFF = 0.0001

# =============================================================================
# 1. LOAD THE SIGNIFICANT KS ASSOCIATIONS
# =============================================================================
m_types = np.loadtxt(os.path.join(path, 'm_types.txt'))
n_cells_per_type = np.sum(m_types, axis=1)

with open(os.path.join(path, 'cell_type_unique_mouse.txt'), 'r') as f:
    cell_type_unique = np.array(f.read().split('\n')[:-1])

with open(os.path.join(path, 'phen_filtro_20_500_genes_no_redundantes_final.txt'), 'r') as f:
    phen = np.array(f.read().split('\n')[:-1])

cell_type_ks_test = np.loadtxt(os.path.join(path, 'cell_type_ks_test.txt'), dtype=float)
cell_type_p_value = np.loadtxt(os.path.join(path, 'cell_type_p_value.txt'), dtype=float)

ind_good_cells = np.where(n_cells_per_type > MIN_CELLS_PER_TYPE)[0]
filt_cell_type = cell_type_unique[ind_good_cells]
filt_ks_test = cell_type_ks_test[:, ind_good_cells]
filt_p_value = cell_type_p_value[:, ind_good_cells]

sig_phen_idx, sig_cell_idx = np.where(filt_p_value < KS_PVALUE_CUTOFF)

df_significant = pd.DataFrame({
    'Phenotype': phen[sig_phen_idx],
    'Query_Cluster': filt_cell_type[sig_cell_idx],
    'KS': filt_ks_test[sig_phen_idx, sig_cell_idx],
    'P_Value': filt_p_value[sig_phen_idx, sig_cell_idx]
})

# =============================================================================
# 2. CL IDENTIFIER OF EVERY CELL-TYPE NAME, BY EXACT MATCH ON cl.obo
# =============================================================================
id_map = {}

# Descargar cl.obo si no lo tienes
if not os.path.exists(obo_path):
    print("Descargando cl.obo oficial...")
    urllib.request.urlretrieve("http://purl.obolibrary.org/obo/cl.obo", obo_path)

print("Leyendo cl.obo (extrayendo estrictamente 'id' y 'name')...")

with open(obo_path, 'r', encoding='utf-8') as f:
    current_id = None
    for line in f:
        line = line.strip()
        
        # Capturamos el ID de la célula
        if line.startswith('id: CL:'):
            # Nos quedamos solo con la parte del ID, ignorando comentarios
            current_id = line.split('id: ')[1].split(' ')[0].strip()
            
        # Capturamos el Nombre
        elif line.startswith('name: ') and current_id:
            # Extraemos el nombre y lo pasamos a minúsculas para que el cruce no falle por mayúsculas
            name = line.split('name: ')[1].strip().lower()
            id_map[name] = current_id
            
            # Reseteamos current_id para no arrastrarlo por error a otras líneas
            current_id = None 

print(f"Diccionario listo. Se cargaron {len(id_map)} mapeos exactos Nombre -> ID.")

def get_cl_id(cell_name):
    if pd.isna(cell_name):
        return "ID_NOT_FOUND"
    
    # Limpiamos el nombre del excel (minúsculas y sin espacios a los lados)
    clean_name = str(cell_name).strip().lower()
    return id_map.get(clean_name, "ID_NOT_FOUND")

# =============================================================================
# 3. MERGE WITH THE MANUAL CURATION OF THE ATLAS CLUSTER NAMES
# =============================================================================
print("Aplicando mapeos al DataFrame...")

# Cargar tu Excel
df_manual = pd.read_excel(manual_mapping_file)


# -----------------------------------------------------------------------------
# 3b. PROVENANCE OF THE MAPPING, FOR THE METHODS
# -----------------------------------------------------------------------------
# The cluster -> CL mapping was produced in two steps: an automatic proposal for
# every atlas cluster name, and a manual review of each proposal, labelled in
# the spreadsheet as 'Mantener' (accepted), 'Cambiar' (corrected) or 'Eliminar'
# (the cluster is not a cell type and is dropped). The counts below are the ones
# to report, and the three checks after them are what makes the curation
# auditable: a duplicated cluster, or one with no label, would silently change
# which associations survive.
COL_QUERY = 'Query Original (Tu clúster)'
COL_STATE = 'Estado del Mapeo'

state = df_manual[COL_STATE].astype(str).str.lower()
n_total = len(df_manual)
n_kept = int(state.str.contains('mantener').sum())
n_changed = int(state.str.contains('cambiar').sum())
n_removed = int(state.str.contains('eliminar').sum())

print("\n  Manual curation of the atlas cluster names:")
print(f"    clusters reviewed : {n_total}")
if n_total:
    print(f"    accepted          : {n_kept:3d}  ({100 * n_kept / n_total:.1f}%)")
    print(f"    corrected         : {n_changed:3d}  ({100 * n_changed / n_total:.1f}%)")
    print(f"    dropped           : {n_removed:3d}  ({100 * n_removed / n_total:.1f}%)")

unlabelled = n_total - (n_kept + n_changed + n_removed)
if unlabelled:
    missing = df_manual.loc[~(state.str.contains('mantener') |
                              state.str.contains('cambiar') |
                              state.str.contains('eliminar')), COL_QUERY]
    print(f"    WARNING: {unlabelled} clusters carry none of the three labels: "
          f"{missing.tolist()[:10]}")

dups = df_manual[COL_QUERY].value_counts()
dups = dups[dups > 1]
if len(dups):
    print(f"    WARNING: {len(dups)} cluster names appear more than once in the "
          f"spreadsheet: {dups.index.tolist()[:10]}")

# Clusters that reach the curation but are absent from it, and the other way
# round. Both lists must be empty for the curation to cover the whole atlas.
curated = set(df_manual[COL_QUERY].dropna().astype(str).str.strip())
in_data = set(pd.Series(filt_cell_type).astype(str).str.strip())
only_data = sorted(in_data - curated)
if only_data:
    print(f"    WARNING: {len(only_data)} cell types of the atlas are not in the "
          f"curation: {only_data[:10]}")

# Limpiar las columnas clave de tu Excel para evitar fallos por espacios invisibles
df_manual['Query Original (Tu clúster)'] = df_manual['Query Original (Tu clúster)'].astype(str).str.strip()
df_manual['Mapeo Definitivo (Nuestra corrección)'] = df_manual['Mapeo Definitivo (Nuestra corrección)'].astype(str).str.strip()

# Hacer el Merge entre tus resultados significativos y tu Excel manual
df_merged = df_significant.merge(
    df_manual, 
    left_on='Query_Cluster', 
    right_on='Query Original (Tu clúster)', 
    how='left'
)

n_unmatched = int(df_merged['Query Original (Tu clúster)'].isna().sum())
if n_unmatched:
    print(f"  {n_unmatched} significant associations belong to clusters absent from "
          f"the manual curation; they are kept and end up as ID_NOT_FOUND.")

# Filtrar las células que marcaste como "eliminar"
is_deleted = df_merged['Estado del Mapeo'].astype(str).str.lower().str.contains('eliminar')
df_final = df_merged[~is_deleted].copy()

# APLICAR EL MAPEO desde la columna definitiva que curaste
df_final['Cell_Type_Final'] = df_final['Mapeo Definitivo (Nuestra corrección)']
df_final['CL_ID'] = df_final['Cell_Type_Final'].apply(get_cl_id)

# Guardar los datos finales limpios
output_cols = ['Phenotype', 'Query_Cluster', 'Cell_Type_Final', 'CL_ID', 'KS', 'P_Value']
df_to_save = df_final[output_cols].rename(columns={
    'Query_Cluster': 'Cell_Type',
    'Cell_Type_Final': 'label',
    'CL_ID': 'cl_id'
})

df_to_save.to_csv(os.path.join(path, 'df_integrated_clean.csv'), index=False)

# Comprobación de errores final
missing_ids = df_final[df_final['CL_ID'] == 'ID_NOT_FOUND']['Cell_Type_Final'].unique()
if len(missing_ids) > 0:
    print("\n⚠️ Células de tu Excel que no hicieron match exacto con el 'name' del OBO:")
    for cell in missing_ids: 
        print(f" - {cell}")
else:
    print("\n🎉 ¡Todo mapeado perfectamente!")
    

print("\nCargando ontología CL como grafo para validación jerárquica...")
cl_graph = obonet.read_obo(obo_path)
print(f"¡Grafo CL cargado! Contiene {len(cl_graph)} nodos.")

# Crear diccionario inverso rápido para mapear CL_ID -> Nombre de la célula
cl_id_to_name = {n: data.get('name', n) for n, data in cl_graph.nodes(data=True)}

df_mp_cl = pd.read_csv(os.path.join(path, 'mp_cl_associations.tsv'), sep='\t')
mp_to_cl_dict = df_mp_cl.groupby('mp_name')['cl_id'].apply(set).to_dict()

# FUNCIÓN 1: El filtro purificador (Ignora ratones, procesos GO y anatomía UBERON)
# FUNCIÓN 1 CORREGIDA: El filtro purificador
def get_true_cl_ancestors(graph, start_node):
    ancestors = set()
    if start_node not in graph:
        return ancestors
        
    queue = [start_node]
    visited = set([start_node])
    
    while queue:
        current_node = queue.pop(0)
        
        # obonet guarda las flechas de salida en out_edges.
        # El tercer valor (key) es directamente el nombre de la relación biológica
        for _, parent, key in graph.out_edges(current_node, keys=True):
            
            # Si la relación es estrictamente 'is_a' y el destino es una célula 'CL:'
            if key == 'is_a' and parent.startswith('CL:') and parent not in visited:
                visited.add(parent)
                ancestors.add(parent)
                queue.append(parent)
                
    return ancestors


# FUNCIÓN 2: La validación (Devuelve ancestros SOLO en los True)
def get_cl_hierarchy_details(row):
    query_cl = row['CL_ID']
    phenotype = row['Phenotype']
    
    # Cell types the ontology associates with this phenotype
    target_cls = list(mp_to_cl_dict.get(phenotype, set()))
    target_names = [cl_id_to_name.get(c, c) for c in target_cls]
    
    # Extraemos el linaje PURO de la célula
    all_ancestors_ids = set()
    all_ancestors_names = []
    if pd.notna(query_cl) and query_cl != "ID_NOT_FOUND" and query_cl in cl_graph:
        all_ancestors_ids = get_true_cl_ancestors(cl_graph, query_cl)
        all_ancestors_names = [cl_id_to_name.get(c, c) for c in all_ancestors_ids]
    
    # Fallo 1: Falta ID
    if query_cl == "ID_NOT_FOUND" or pd.isna(query_cl):
        return pd.Series([False, target_names, [], [], "Fallo: Falta ID de célula en tu dataset"])
        
    # Fallo 2: El fenotipo no está en MP
    if not target_cls:
        return pd.Series([False, [], [], [], "Fallo: Fenotipo no existe en base de datos MP"])
        
    # A. Match Directo
    if query_cl in target_cls:
        match_name = cl_id_to_name.get(query_cl, query_cl)
        # ÉXITO: Guardamos la lista purificada de ancestros
        return pd.Series([True, target_names, [match_name], all_ancestors_names, "Match Directo (Exacto)"])
        
    # B. Búsqueda Jerárquica
    if all_ancestors_ids:
        overlap = all_ancestors_ids.intersection(set(target_cls))
        
        if len(overlap) > 0:
            overlap_names = [cl_id_to_name.get(c, c) for c in overlap]
            # ÉXITO: Guardamos la lista purificada de ancestros
            return pd.Series([True, target_names, overlap_names, all_ancestors_names, "Match Jerárquico (Es descendiente puro)"])
            
    # C. Fallo total
    # FALLO: Dejamos la lista vacía [] para no ensuciar el reporte
    return pd.Series([False, target_names, [], [], "Fallo: No hay coincidencia directa ni linaje 'is_a'"])

# Aplicamos la función
print("Validando jerárquicamente (Filtrado estricto 'is_a')...")
df_final[['is_validated', 'MP_Target_Cells', 'Matched_Ancestors', 'All_Query_Ancestors', 'Match_Description']] = df_final.apply(get_cl_hierarchy_details, axis=1)

# =============================================================================
# 4. REPORT TABLES
# =============================================================================

# Guardar el reporte detallado
report_cols = [
    'Phenotype', 'Cell_Type_Final', 'CL_ID', 
    'is_validated', 'Match_Description', 'MP_Target_Cells', 'Matched_Ancestors', 'All_Query_Ancestors'
]
# df_detailed_report = df_final[report_cols].copy()

# Guardar el DataFrame validado limpio
df_validated = df_final[df_final['is_validated'] == True].copy()
report_file = os.path.join(path, 'df_validation1_detailed_report.csv')
df_validated.to_csv(report_file, index=False)
print(f"\n✅ Reporte detallado guardado en: {report_file}")

val_output_cols = ['Phenotype', 'Query_Cluster', 'Cell_Type_Final', 'CL_ID', 'KS', 'P_Value']
df_val_unique = df_validated[val_output_cols].drop_duplicates(subset=['Phenotype', 'Query_Cluster'])
df_val_unique.to_csv(os.path.join(path, 'df_validation1_curated_hierarchical.csv'), index=False)


# =============================================================================
# 5. METRICS AND STATISTICS
# =============================================================================
total_signif = len(df_final)
total_val = len(df_validated)
percentage = (total_val / total_signif) * 100 if total_signif > 0 else 0

print("\n" + "=" * 60)
print(" VALIDATION SUMMARY (HIERARCHICAL)")
print("=" * 60)
print(f"Total significant associations: {total_signif}")
print(f"Total valid associations:       {total_val}")
print(f"Validation rate:                {percentage:.2f}%")
print("-" * 60)

# 2. DataFrame de únicos basados en el Tipo Celular Final (Biológico)
df_final_unique_bio = df_final.drop_duplicates(subset=['Phenotype', 'Query_Cluster'])
df_val_unique_bio = df_val_unique # Ya lo deduplicamos arriba

total_signif_bio = len(df_final_unique_bio)
total_val_bio = len(df_val_unique_bio)
pct_bio = (total_val_bio / total_signif_bio) * 100 if total_signif_bio > 0 else 0

print("🧬 Basado en Biología (Phenotype + cell type único):")
print(f"   Initial unique associations : {total_signif_bio}")
print(f"   Valid unique associations   : {total_val_bio}")
print(f"   Validation rate             : {pct_bio:.2f}%")
print("-" * 60)

# # Compare KS distributions: Validated vs Non-Validated
# ks_validated = df_final[df_final['is_validated'] == True]['KS']
# ks_not_validated = df_final[df_final['is_validated'] == False]['KS']

# Código corregido (CORRECTO para únicos)
ks_validated = df_final_unique_bio[df_final_unique_bio['is_validated'] == True]['KS']
ks_not_validated = df_final_unique_bio[df_final_unique_bio['is_validated'] == False]['KS']

# Evitar fallos estadísticos si algún grupo está vacío
if len(ks_validated) > 0 and len(ks_not_validated) > 0:
    stat_mw, p_value_mw = mannwhitneyu(ks_validated, ks_not_validated, alternative='two-sided')
    stat_ks, p_value_ks = ks_2samp(ks_validated, ks_not_validated)
    print(f"Mann-Whitney U P-value:     {p_value_mw:.2e}")
    print(f"Kolmogorov-Smirnov P-value: {p_value_ks:.2e}")
else:
    print("No hay suficientes datos en ambos grupos para hacer el cálculo estadístico (MW / KS).")
print("="*60) 

# =============================================================================
# 6. FIGURES
# =============================================================================
data = [ks_not_validated, ks_validated]

# --- BOXPLOT ---
fig, ax = plt.subplots(figsize=(4, 4), dpi=600)
# `labels` was renamed to `tick_labels` in matplotlib 3.9 and removed in 3.11
_lab = 'tick_labels' if 'tick_labels' in inspect.signature(ax.boxplot).parameters else 'labels'
box = ax.boxplot(
    data,
    **{_lab: ["Not Validated", "Validated\nMP ontology"]},
    widths=0.45, 
    patch_artist=True, 
    flierprops=dict(marker='o', color='gray', markersize=0.5), 
    boxprops=dict(edgecolor="none")
)

colors = ["lightgray", "royalblue"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.setp(box["medians"], color="black", linewidth=1)
plt.setp(box["whiskers"], color="black", linestyle="--", linewidth=0.5)
plt.setp(box["caps"], color="black", linewidth=0.5)

plt.ylabel("KS Statistic", fontsize=16)
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=13)

# Significance label, derived from the Mann-Whitney p-value rather than typed in
def stars(p):
    if p < 1e-4:
        return '****'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


label_signif = stars(p_value_mw)
y_max = max([max(d) for d in data])
h = 0.02 * y_max  
y_line = y_max + 0.05 * y_max 
x1, x2 = 1, 2
ax.plot([x1, x1, x2, x2], [y_line, y_line+h, y_line+h, y_line], lw=0.5, c='black')
ax.text((x1 + x2) / 2, y_line + 0.01 * y_max, label_signif, ha='center', fontsize=10, c='red')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(path, 'ks_boxplot_validation1.png'), dpi=600, bbox_inches='tight')
plt.close()

# --- HISTOGRAM ---
plt.figure(figsize=(7, 5), dpi=600)
sns.histplot(ks_not_validated, color="gray", label="Not Validated", kde=True, stat="density", bins=40, alpha=0.4)
sns.histplot(ks_validated, color="royalblue", label="Validated", kde=True, stat="density", bins=40, alpha=0.6)

plt.xlabel('KS Statistic', fontsize=17)
plt.ylabel('Associations (density)', fontsize=17)
plt.legend(frameon=False, fontsize=11)
plt.grid(False)
sns.despine()

plt.tight_layout()
plt.savefig(os.path.join(path, 'ks_histogram_validation1.png'), dpi=600, bbox_inches='tight')
plt.close()

# =============================================================================
# 7. PHENOTYPES WITH AT LEAST ONE VALIDATED ASSOCIATION: STRONGEST CELL TYPE
# =============================================================================
print("\n" + "="*60)
print(" MAX KS CELL TYPES FOR VALIDATED PHENOTYPES")
print("="*60)

# 1. Identify which phenotypes have AT LEAST ONE validated association
validated_phenotypes = df_val_unique['Phenotype'].unique()
print(f"Found {len(validated_phenotypes)} phenotypes with at least one MP validation.")

# 2. Filter the main dataset to ONLY include these validated phenotypes
df_phenos_val = df_final[df_final['Phenotype'].isin(validated_phenotypes)].copy()

# 3. Add an 'Is_Validated' flag to this filtered dataset
validated_pairs = set(zip(df_val_unique['Phenotype'], df_val_unique['Cell_Type_Final']))
df_phenos_val['Is_Validated'] = df_phenos_val.apply(
    lambda row: (row['Phenotype'], row['Cell_Type_Final']) in validated_pairs, 
    axis=1
)

# 4. Find the cell type with the maximum KS for each of these phenotypes
# Group by phenotype and find the index of the row with the max KS
idx_max_ks = df_phenos_val.groupby('Phenotype')['KS'].idxmax()

# Filter the dataframe using these indices and sort by KS descending
top_cells_df = df_phenos_val.loc[idx_max_ks].sort_values(by='KS', ascending=False)

# Keep only the relevant columns for a clean table
top_table_cols = ['Phenotype', 'Query_Cluster', 'Cell_Type_Final', 'CL_ID', 'KS', 'P_Value', 'Is_Validated']
top_cells_clean = top_cells_df[top_table_cols]

# Save the results to a new CSV file
output_file_top = os.path.join(path, 'max_ks_in_validated_phenotypes.csv')
top_cells_clean.to_csv(output_file_top, index=False)

# Print a preview
print("\nTop 15 Strongest Cell Types (Max KS) within Validated Phenotypes:")
print(top_cells_clean.head(15).to_string(index=False))

# Calculate how many times the Max KS cell type IS the validated one
match_count = top_cells_clean['Is_Validated'].sum()
match_percent = (match_count / len(top_cells_clean)) * 100
print("-" * 60)
print(f"🎯 In {match_count} out of {len(top_cells_clean)} phenotypes ({match_percent:.1f}%),")
print("the cell type with the absolute highest KS score is the one validated by the ontology.")
print("="*60)


# =============================================================================
# 8. KS DISTRIBUTION WITHIN THE VALIDATED PHENOTYPES
# =============================================================================

# Sort phenotypes by their max KS to make the plot cascade nicely
ordered_phenos = top_cells_clean['Phenotype'].tolist()

# Apply the order to the dataframe for plotting
df_phenos_val['Phenotype'] = pd.Categorical(df_phenos_val['Phenotype'], categories=ordered_phenos, ordered=True)

# Dynamically adjust plot height based on the number of phenotypes
plot_height = max(6, len(ordered_phenos) * 0.4)
plt.figure(figsize=(13, plot_height), dpi=600)

# A) Boxplot for background KS distribution of that phenotype
sns.boxplot(
    data=df_phenos_val, 
    x='KS', 
    y='Phenotype',
    color='whitesmoke', 
    fliersize=0, # Hide default outliers
    width=0.5,
    linewidth=1
)

# B) Stripplot for individual associations (dots)
sns.stripplot(
    data=df_phenos_val, 
    x='KS', 
    y='Phenotype',
    hue='Is_Validated',
    palette={True: 'red', False: 'darkgray'},
    size=5, 
    alpha=0.7, 
    jitter=0.2, 
    dodge=False 
)

plt.xlabel("KS Statistic", fontsize=16, fontweight='bold')
plt.ylabel("")#, fontsize=16, fontweight='bold')
plt.yticks(fontsize=15)
plt.xticks(fontsize=14)
# plt.title("KS Distribution Within Validated Phenotypes", fontsize=16, pad=15)
# Fix legend positioning and labels
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['Not Validated', 'Validated (MP ontology)'], frameon=False, loc='lower right')

sns.despine()
plt.tight_layout()

# Save and show
plt.savefig(os.path.join(path, 'ks_boxplot_validated_phenotypes.png'), dpi=600, bbox_inches='tight')
plt.close()


#top 10 KS cell types per validated phenotype
df_top10_ks = df_phenos_val.groupby('Phenotype', observed=False).apply(
    lambda x: x.nlargest(10, 'KS')
).reset_index(drop=True)

print(df_top10_ks['Phenotype'].value_counts())
df_top10_ks.to_csv(path+'df_top10_ks_validation1_phen.csv', index=False)

# =============================================================================
# 9. VALIDATED PHENOTYPES: KS DISTRIBUTION AND TOP CELL TYPES
# =============================================================================

print("\n" + "="*60)
print("🏆 TOP CELL TYPES PER VALIDATED PHENOTYPE")
print("="*60)

# 1. General distribution of KS values for validated associations
print("KS Distribution of Validated Associations:")
print(df_val_unique['KS'].describe())
print("-" * 60)

# 2. Find the cell type with the maximum KS for EACH validated phenotype
# idxmax() finds the index of the maximum KS value within each Phenotype group
idx_max_ks = df_val_unique.groupby('Phenotype')['KS'].idxmax()

# Filter the dataframe using these indices
top_cells_per_pheno = df_val_unique.loc[idx_max_ks]

# Sort by KS value descending to see the absolute strongest associations at the top
top_cells_per_pheno = top_cells_per_pheno.sort_values(by='KS', ascending=False)

# Keep only the relevant columns for a clean table
top_table_cols = ['Phenotype', 'Cell_Type_Final', 'CL_ID', 'KS', 'P_Value']
top_cells_clean = top_cells_per_pheno[top_table_cols]

# Save the results to a new CSV file
output_file_top = os.path.join(path, 'top_validated_cells_per_phenotype.csv')
top_cells_clean.to_csv(output_file_top, index=False)

# Print the top 15 absolute strongest associations
print("\nTop 15 Strongest Validated Associations (Max KS):")
print(top_cells_clean.head(15).to_string(index=False))

# 3. Overall Maximum KS in the entire validated dataset
global_max_row = top_cells_clean.iloc[0]
print("-" * 60)
print("GLOBAL MAX: the strongest validated association overall is:")
print(f"Phenotype : {global_max_row['Phenotype']}")
print(f"Cell Type : {global_max_row['Cell_Type_Final']} ({global_max_row['CL_ID']})")
print(f"KS Score  : {global_max_row['KS']:.4f}")
print("="*60)













