# -*- coding: utf-8 -*-
"""
CELL-TYPE ORDER AND NUMBER OF CELLS PER CELL TYPE AND DEVELOPMENTAL STAGE

Reads the cell metadata of the embryonic atlas and fixes the order of the cell
types used by every later script: sorted by developmental stage and then
alphabetically by cell type. It then counts how many cells each cell type has at
each stage.

The two files written here must stay aligned: the rows of m_types.txt follow the
order of cell_type_unique_mouse.txt, and the columns of the KS matrices built by
computation_KS_faster.py follow that same order, because that script rebuilds it
with the same code.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
path = 'PATH'

# =============================================================================
# 1. CELL METADATA
# =============================================================================
df_cell = pd.read_csv(path + 'df_cell.csv')
N_cell = df_cell.shape[0]
print(f"Cells in the atlas: {N_cell:,}")

cell_type_update = np.array(df_cell['celltype_update'], dtype='str')
cell_time = np.array(df_cell['day'], dtype='str')
cell_time_unique = np.unique(cell_time)

# Developmental stages in biological order: E8.0, E8.5 ... then P0 last
numeric_part = [x for x in cell_time_unique if x != "P0"]
sorted_array = sorted(numeric_part, key=lambda x: float(x[1:]))
sorted_array.append("P0")
cell_time_unique = np.array(sorted_array, dtype=str)

# =============================================================================
# 2. CELL-TYPE ORDER: BY STAGE, THEN BY CELL-TYPE NAME
# =============================================================================
df = pd.DataFrame({
    'index': np.arange(N_cell, dtype=int),
    'time': cell_time,
    'Category': pd.Categorical(cell_time, categories=cell_time_unique, ordered=True),
    'cell_type': cell_type_update,
})
a = df.sort_values(by=["Category", "cell_type"]).to_numpy()
del df

cell_type = a[:, 3]
time = a[:, 2]

# Unique cell types KEEPING that order (not np.unique, which sorts alphabetically)
cell_type_unique = []
seen = set()
for value in cell_type:
    if value not in seen:
        cell_type_unique.append(value)
        seen.add(value)
cell_type_unique = np.array(cell_type_unique)

print(f"Cell types: {len(cell_type_unique)}")

# =============================================================================
# 3. NUMBER OF CELLS PER CELL TYPE AND STAGE
# =============================================================================
n_cell_type_time = np.zeros((len(cell_type_unique), len(cell_time_unique)))
for i in range(len(cell_type_unique)):
    ind_type = np.where(cell_type == cell_type_unique[i])[0]
    for k in range(len(ind_type)):
        ind_time = np.where(cell_time_unique == cell_time[int(ind_type[k])])[0]
        n_cell_type_time[i][int(ind_time)] = n_cell_type_time[i][int(ind_time)] + 1

n_cell_type_time = np.array(n_cell_type_time, dtype=int)

# =============================================================================
# 4. SAVE
# =============================================================================
# Both files come out of the same loop, so their order cannot drift apart.
np.savetxt(path + 'm_types.txt', n_cell_type_time, fmt='%f')
np.savetxt(path + 'cell_type_unique_mouse.txt', cell_type_unique, fmt='%s')
print(f"Saved m_types.txt {n_cell_type_time.shape} and cell_type_unique_mouse.txt")

# =============================================================================
# 5. FIGURE: CELL TYPE x STAGE MAP
# =============================================================================
# Labels as 'major cell type - subtype', from the atlas annotation table. Only
# used for the figure; skipped if the file is not available.
try:
    df_sub = pd.read_excel(path + 'Cell types and subtypes.xlsx', engine='openpyxl')
    major_cell_types = df_sub.iloc[:, 0].to_numpy()
    cell_subtype = df_sub.iloc[:, 1].to_numpy()

    final_cell_type = []
    for i in range(len(cell_type_unique)):
        ind_c = np.where(cell_subtype == cell_type_unique[i])[0]
        final_cell_type.append(major_cell_types[ind_c][0] + ' - ' + cell_type_unique[i])
    final_cell_type = np.array(final_cell_type)
except FileNotFoundError:
    print("  'Cell types and subtypes.xlsx' not found; using the plain cell-type names.")
    final_cell_type = cell_type_unique

plt.figure(figsize=(10, 20), dpi=600)
plt.imshow(np.log(n_cell_type_time + 1), cmap='cool', aspect='auto')
cbar = plt.colorbar()
cbar.set_label('# cells (log scale)', size=20)
cbar.ax.set_position([0.85, 0.1, 0.03, 0.8])
plt.grid(False)
plt.ylabel('Cell type', fontsize=20)
plt.xlabel('Embryo stage', fontsize=20)
plt.xticks(np.arange(len(cell_time_unique)), cell_time_unique, fontsize=10, rotation=90)
plt.yticks(np.arange(len(final_cell_type)), final_cell_type, fontsize=6)
plt.savefig(path + 'm_types.png', dpi=600, bbox_inches='tight')
plt.close()
