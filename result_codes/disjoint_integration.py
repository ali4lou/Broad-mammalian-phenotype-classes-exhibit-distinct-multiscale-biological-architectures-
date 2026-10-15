#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISJOINT INTEGRATION: PS (AUC) and permutation p-values, excluding phenotypes
shared between the two categories.

Standalone script -- reloads and rebuilds everything it needs (similarity
matrices, residuals, category mapping) independently of section2_modularity.py,
so it can be run on its own.

For each pair of categories (A, B) and each layer, this computes the PS of
similarity linking A-only phenotypes (in A but not B) to B-only phenotypes
(in B but not A), i.e. with phenotypes shared between A and B excluded from
both sides. This isolates the integration signal from any inflation caused
by phenotype pairs that are trivially similar because they belong to BOTH
categories at once.

The null redraws groups of the same reduced sizes (|A_only|, |B_only|) from
outside the union of the two full categories, and recomputes the same
cross-group PS, so the test is matched to the disjoint group sizes rather
than to the original (unreduced) category sizes.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import obonet
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, norm
from statsmodels.stats.multitest import multipletests

# =============================================================================
# CONFIGURATION -- identical to section2_modularity.py
# =============================================================================
path = 'PATH'
path2 = 'PATH'
url = 'PATH/MPheno_OBO.ontology.obo'
OUTDIR = os.path.join('PATH')
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42

QUICK_TEST = False
if QUICK_TEST:
    N_PERM_DISJ = 200
else:
    # Same resolution rationale as N_PERM_INT in section2_modularity.py:
    # ~350 category pairs per layer -> normal approximation for the FDR,
    # 2,000 permutations to estimate the null mean/sd.
    N_PERM_DISJ = 2000

MIN_CATEGORY_SIZE = 5
FDR_ALPHA = 0.05
MIN_CELLS_PER_TYPE = 50
KS_PVALUE_CUTOFF = 0.0001

GENE_LAYER = 'Genes'
CT_LAYER = 'Cell types'
GO_LAYERS = ['GO (Cellular Process)', 'GO (Cellular Component)',
             'GO (Molecular Function)']
LAYER_ORDER = [GENE_LAYER] + GO_LAYERS + [CT_LAYER]

MIN_DISJOINT_GROUP = 2  # a_only / b_only must have at least this many phenotypes


# =============================================================================
# BLOCK 1. LOAD AND ALIGN THE SIMILARITY MATRICES (identical to the main script)
# =============================================================================
print("=" * 78)
print(" BLOCK 1: loading similarity matrices")
print("=" * 78)

with open(os.path.join(path, 'phen_filtro_20_500_genes_no_redundantes_final.txt')) as f:
    phen_filter = [ln.strip() for ln in f if ln.strip()]

m_types = np.loadtxt(os.path.join(path, 'm_types.txt'))
with open(os.path.join(path, 'cell_type_unique_mouse.txt')) as f:
    cell_type_unique = np.array(f.read().split('\n')[:-1])

ks_test = np.loadtxt(os.path.join(path, 'cell_type_ks_test.txt'), dtype=float)
ks_pval = np.loadtxt(os.path.join(path, 'cell_type_p_value.txt'), dtype=float)

good = np.where(m_types.sum(axis=1) > MIN_CELLS_PER_TYPE)[0]
significant_ks = np.where(ks_pval[:, good] < KS_PVALUE_CUTOFF, ks_test[:, good], 0)
df_ks_full = pd.DataFrame(significant_ks, index=phen_filter, columns=cell_type_unique[good])
df_ks_similarity = pd.DataFrame(1 - squareform(pdist(df_ks_full.values, metric='cosine')),
                                index=df_ks_full.index, columns=df_ks_full.index)

matrix_dict = {
    GENE_LAYER: pd.read_csv(os.path.join(path2, 'phenotype_jaccard_similarity.txt'),
                            sep='\t', index_col=0),
    'GO (Cellular Process)': pd.read_csv(
        os.path.join(path2, 'similarity_cosine_GO_Cellular_Process.txt'), sep='\t', index_col=0),
    'GO (Cellular Component)': pd.read_csv(
        os.path.join(path2, 'similarity_cosine_GO_Cellular_Component.txt'), sep='\t', index_col=0),
    'GO (Molecular Function)': pd.read_csv(
        os.path.join(path2, 'similarity_cosine_GO_Molecular_Function.txt'), sep='\t', index_col=0),
    CT_LAYER: df_ks_similarity,
}

sorted_phenos = sorted({str(p).strip().lower() for p in phen_filter})

for name, M in matrix_dict.items():
    idx = {str(i).strip().lower() for i in M.index}
    missing = set(sorted_phenos) - idx
    if missing:
        raise ValueError(f"{len(missing)} phenotypes absent from '{name}', "
                         f"e.g. {sorted(missing)[:5]}")

aligned = {}
for name, M in matrix_dict.items():
    M = M.copy()
    M.index = M.index.astype(str).str.strip().str.lower()
    M.columns = M.columns.astype(str).str.strip().str.lower()
    aligned[name] = M.loc[sorted_phenos, sorted_phenos]

gm = pd.read_csv(os.path.join(path2, 'final_filtered_matrix.txt'), sep='\t', index_col=0)
gm.index = gm.index.astype(str).str.strip()
gm.columns = gm.columns.astype(str).str.strip().str.lower()
breadth_all = (gm > 0).sum(axis=0)

PHENOS = [p for p in sorted_phenos if p in breadth_all.index]
n = len(PHENOS)
iu = np.triu_indices(n, k=1)
sizes = breadth_all.reindex(PHENOS).values.astype(float)
pheno_idx = {p: i for i, p in enumerate(PHENOS)}

print(f"  {n} phenotypes, {len(iu[0]):,} pairs")


# =============================================================================
# BLOCK 2. RESIDUAL MATRICES (identical to the main script)
# =============================================================================
print("\n" + "=" * 78)
print(" BLOCK 2: residualising against annotation breadth")
print("=" * 78)


def rank01(v):
    return (rankdata(v) - 0.5) / len(v)


C_min = np.minimum(sizes[iu[0]], sizes[iu[1]])
C_max = np.maximum(sizes[iu[0]], sizes[iu[1]])
X = np.column_stack([np.ones(len(iu[0])), rank01(C_min), rank01(C_max),
                     rank01(C_min) * rank01(C_max)])

RES = {}
for name in LAYER_ORDER:
    v = rank01(aligned[name].loc[PHENOS, PHENOS].values[iu])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    res = v - X @ beta
    R = np.zeros((n, n))
    R[iu] = res
    RES[name] = R + R.T
    print(f"  {name:<26} R2 removed = {1 - res.var() / v.var():.4f}")




# =============================================================================
# BLOCK 3. MAP PHENOTYPES TO TOP-LEVEL MP CATEGORIES (identical to the main script)
# =============================================================================
print("\n" + "=" * 78)
print(" BLOCK 3: mapping phenotypes to broad MP categories")
print("=" * 78)

graph = obonet.read_obo(url)
id_to_name = {i: d.get('name') for i, d in graph.nodes(data=True)}
name_to_id = {d['name']: i for i, d in graph.nodes(data=True) if 'name' in d}

root_id = name_to_id.get('mammalian phenotype')
if root_id is None:
    raise ValueError("'mammalian phenotype' not found in the ontology")

broad_ids = set(graph.predecessors(root_id))
meta = {i: id_to_name[i] for i in broad_ids}

robust = {}
for i, d in graph.nodes(data=True):
    if 'name' in d:
        robust[d['name'].strip().lower()] = i
    for syn in d.get('synonym', []):
        if '"' in syn:
            robust[syn.split('"')[1].strip().lower()] = i


def broad_of(pheno):
    pid = robust.get(pheno.strip().lower())
    if not pid or pid not in graph:
        return set()
    reach = nx.descendants(graph, pid) | {pid}
    return {meta[i] for i in reach & broad_ids}


cat_members = defaultdict(list)
for p in PHENOS:
    for c in broad_of(p):
        cat_members[c].append(p)

CATEGORIES = sorted(c for c in cat_members if len(cat_members[c]) >= MIN_CATEGORY_SIZE)
cat_idx = {c: np.array([pheno_idx[p] for p in cat_members[c]]) for c in CATEGORIES}

print(f"  {len(CATEGORIES)} categories with at least {MIN_CATEGORY_SIZE} phenotypes")


# =============================================================================
# HELPER FUNCTIONS (identical logic to section2_modularity.py Block 5)
# =============================================================================
N_PAIRS = len(iu[0])

POS = np.full((n, n), -1, dtype=np.int64)
POS[iu] = np.arange(N_PAIRS)
POS[(iu[1], iu[0])] = np.arange(N_PAIRS)

RANKS = {name: rankdata(RES[name][iu]) for name in LAYER_ORDER}


def auc_of(ranks, p):
    n1 = len(p)
    n2 = N_PAIRS - n1
    return (ranks[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def cross_positions(ia, ib):
    p = POS[np.ix_(ia, ib)]
    return np.unique(p[p >= 0])



# =============================================================================
# CONSISTENCY CHECK
# =============================================================================
# This script rebuilds the residuals from scratch, so before anything else we
# verify that they reproduce the PS values of the main script. If the two
# differ (a different phenotype ordering, a regenerated input file), the
# disjoint values would not be comparable with the non-disjoint ones.
prev_path = os.path.join(OUTDIR, 'TableS_integration.txt')
if os.path.exists(prev_path):
    prev = pd.read_csv(prev_path, sep='\t')
    diffs = []
    for _, r in prev.sample(min(20, len(prev)), random_state=SEED).iterrows():
        ia, ib = cat_idx[r.Category_A], cat_idx[r.Category_B]
        diffs.append(abs(auc_of(RANKS[r.Layer], cross_positions(ia, ib)) - r.AUC))
    worst = max(diffs)
    print(f"\n  Consistency check against the main script: "
          f"largest difference over 20 pairs = {worst:.2e}")
    assert worst < 1e-9, ("The residuals differ from those of "
                          "section2_modularity.py; do not use these values.")
else:
    print(f"\n  WARNING: {prev_path} not found, consistency check skipped.")


# =============================================================================
# DISJOINT INTEGRATION: PS + permutation test + FDR
# =============================================================================
print("\n" + "=" * 78)
print(f" DISJOINT INTEGRATION: PS and permutation p-values "
      f"({N_PERM_DISJ:,} permutations per pair and layer)")
print("=" * 78)

rows = []
for name in LAYER_ORDER:
    ranks = RANKS[name]
    for i, ca in enumerate(CATEGORIES):
        for cb in CATEGORIES[i + 1:]:
            ia, ib = cat_idx[ca], cat_idx[cb]
            a_only = np.setdiff1d(ia, ib)
            b_only = np.setdiff1d(ib, ia)
            n_shared = len(np.intersect1d(ia, ib))
            k1, k2 = len(a_only), len(b_only)

            if k1 < MIN_DISJOINT_GROUP or k2 < MIN_DISJOINT_GROUP:
                rows.append(dict(Layer=name, Category_A=ca, Category_B=cb,
                                 N_A_only=k1, N_B_only=k2, N_shared=n_shared,
                                 Frac_shared=n_shared / min(len(ia), len(ib)),
                                 AUC_disjoint=np.nan, Cliffs_delta=np.nan,
                                 Null_mean=np.nan, Null_sd=np.nan,
                                 p_empirical=np.nan, p_normal=np.nan,
                                 Null_skew=np.nan))
                continue

            obs = auc_of(ranks, cross_positions(a_only, b_only))

            # Null: redraw groups of size k1, k2 from OUTSIDE the union of the
            # two full (non-reduced) categories -- same exclusion logic as
            # the non-disjoint integration null in section2_modularity.py,
            # but matched to the disjoint (reduced) group sizes.
            pool = np.setdiff1d(np.arange(n), np.union1d(ia, ib))
            rng = np.random.default_rng(SEED + abs(hash(ca + cb + 'disjoint')) % 10000)
            null = np.empty(N_PERM_DISJ)
            for j in range(N_PERM_DISJ):
                a = rng.choice(pool, size=k1, replace=False)
                b = rng.choice(pool, size=k2, replace=False)
                null[j] = auc_of(ranks, cross_positions(a, b))

            mu, sd = null.mean(), null.std()
            z = (obs - mu) / sd if sd > 0 else np.nan
            rows.append(dict(Layer=name, Category_A=ca, Category_B=cb,
                             N_A_only=k1, N_B_only=k2, N_shared=n_shared,
                             Frac_shared=n_shared / min(len(ia), len(ib)),
                             AUC_disjoint=obs, Cliffs_delta=2 * obs - 1,
                             Null_mean=mu, Null_sd=sd,
                             p_empirical=(1 + np.sum(null >= obs)) / (N_PERM_DISJ + 1),
                             p_normal=norm.sf(z) if sd > 0 else np.nan,
                             Null_skew=float(pd.Series(null).skew()) if sd > 0 else np.nan))
    print(f"  {name}: done")

df_disj_full = pd.DataFrame(rows)

# FDR within each layer, on the pairs that had enough disjoint phenotypes
df_disj_full['FDR'] = np.nan
for name in LAYER_ORDER:
    m = (df_disj_full.Layer == name) & df_disj_full.p_normal.notna()
    if m.sum() > 0:
        df_disj_full.loc[m, 'FDR'] = multipletests(df_disj_full.loc[m, 'p_normal'],
                                                    method='fdr_bh')[1]

out_path = os.path.join(OUTDIR, 'TableS_integration_disjoint.txt')
df_disj_full.to_csv(out_path, sep='\t', index=False)

n_skipped = df_disj_full.AUC_disjoint.isna().sum()
print(f"\n  Saved: {out_path}")
print(f"  {len(df_disj_full)} (layer, category pair) rows total; "
      f"{n_skipped} skipped for having fewer than {MIN_DISJOINT_GROUP} "
      f"disjoint phenotypes on one side.")
print(f"  Significant disjoint integration at FDR < {FDR_ALPHA}: "
      f"{(df_disj_full.FDR < FDR_ALPHA).sum()} of {df_disj_full.FDR.notna().sum()}")
print(df_disj_full[df_disj_full.FDR < FDR_ALPHA].groupby('Layer').size()
      .reindex(LAYER_ORDER).fillna(0).astype(int).to_string())

print("\nDone.")

# =============================================================================
# RANKED SHEET: disjoint PS per layer, wide format, ranked by weakest layer
# =============================================================================
wide = df_disj_full.pivot_table(index=['Category_A', 'Category_B'],
                                columns='Layer', values='AUC_disjoint')[LAYER_ORDER]

wide['min_AUC'] = wide.min(axis=1)
wide['max_AUC'] = wide.max(axis=1)
wide['avg_AUC'] = wide.mean(axis=1)

# N_shared and Frac_shared are the same for every layer, so take them once
extra = (df_disj_full.groupby(['Category_A', 'Category_B'])[['N_shared', 'Frac_shared']]
         .first())
wide = wide.join(extra)

wide = wide.sort_values('min_AUC', ascending=False).reset_index()
wide.to_csv(os.path.join(OUTDIR, 'integration_disjoint_pairs_ranked.txt'),
           sep='\t', index=False)

print(f"\n  Saved integration_disjoint_pairs_ranked.txt "
      f"({len(wide)} pairs, ranked by min AUC across layers)")
print(wide.head(10).to_string(index=False,
                              float_format=lambda x: f'{x:.3f}'))

# =============================================================================
# CROSS-CHECK AGAINST THE EARLIER OVERLAP ANALYSIS
# =============================================================================
# integration_overlap_check.txt reported the minimum disjoint PS across layers
# without permutations. The PS is computed identically here, so the minimum of
# the per-layer values must reproduce it.
chk_path = os.path.join(OUTDIR, 'integration_overlap_check.txt')
if os.path.exists(chk_path):
    old = pd.read_csv(chk_path, sep='\t').set_index(['Category_A', 'Category_B'])
    new_min = (df_disj_full.pivot_table(index=['Category_A', 'Category_B'],
                                        columns='Layer', values='AUC_disjoint')
               .min(axis=1).rename('new_min'))
    cmp = old[['min_AUC_disjoint', 'N_shared']].join(new_min, how='inner')
    cmp['abs_diff'] = (cmp.min_AUC_disjoint - cmp.new_min).abs()

    print(f"\n  Cross-check against integration_overlap_check.txt "
          f"({len(cmp)} pairs):")
    print(f"    largest difference = {cmp.abs_diff.max():.2e}")
    
    # --- Replace the N_shared check with an explicit index alignment ---
    new_shared = (df_disj_full.groupby(['Category_A', 'Category_B'])
                  .N_shared.first())
    shared_check = cmp['N_shared'].eq(new_shared.reindex(cmp.index))
    
    print(f"\n  Cross-check against integration_overlap_check.txt "
          f"({len(cmp)} pairs):")
    print(f"    largest AUC difference = {cmp.abs_diff.max():.2e}")
    print(f"    N_shared matches       = {shared_check.all()} "
          f"({shared_check.sum()} of {len(shared_check)})")
    if not shared_check.all():
        mism = cmp.index[~shared_check]
        print(f"    Mismatched pairs: {list(mism)[:5]}")
    
    if cmp.abs_diff.max() > 1e-9:
        print("    Pairs that differ:")
        print(cmp[cmp.abs_diff > 1e-9].to_string(float_format=lambda x: f'{x:.6f}'))
else:
    print(f"\n  {chk_path} not found, cross-check skipped.")
    

with pd.ExcelWriter(os.path.join(OUTDIR, 'TableS_integration_disjoint_matrices.xlsx')) as xl:
    for name in LAYER_ORDER:
        sub = df_disj_full[df_disj_full.Layer == name]
        M = pd.DataFrame(np.nan, index=CATEGORIES, columns=CATEGORIES)
        for _, r in sub.iterrows():
            M.loc[r.Category_A, r.Category_B] = M.loc[r.Category_B, r.Category_A] = r.AUC_disjoint
        sheet = name.replace('GO (', '').replace(')', '')[:31]  # Excel limita a 31 caracteres
        M.to_excel(xl, sheet_name=sheet)
print("  Saved TableS_integration_disjoint_matrices.xlsx")

"""
BLOCK 8: DISJOINT INTEGRATION, FIGURES AND RESIDUALS AGAINST THE GENETIC LAYER

PART A  Figure 3 (disjoint version): one heatmap per layer of the PS linking
        A-only phenotypes to B-only phenotypes. Shared phenotypes are excluded
        from both sides.
 
PART B  Residual analysis against the genetic layer, same logic as
        genetic_layer_correlations.py (Part B) but on the disjoint PS.
 
        STEP 1  Size-correct the genetic PS: residual of PS_Genes on
                rank(min), rank(max) and their interaction.
        STEP 2  Spearman between each other layer's PS (uncorrected) and the
                size-corrected genetic PS, with a Mantel-style permutation of
                category labels (pairs are not independent).
        STEP 3  Residual of each other layer's PS regressed linearly on the
                size-corrected genetic PS.
 
DESIGN DECISIONS TO REPORT
  - Size covariates are N_A_only and N_B_only (the groups actually compared),
    not the full category sizes N_A and N_B used in the joint version.
  - Category pairs with fewer than MIN_DISJOINT_GROUP phenotypes on one side
    have no disjoint PS. They appear as grey cells in the figures, are
    dropped from the regressions, and the Mantel statistic is computed on the
    pairs present in both matrices after each permutation.
"""
 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
 
N_PERM_MANTEL = 999          # floor p = 1.0e-3
OTHER_LAYERS = [l for l in LAYER_ORDER if l != GENE_LAYER]
TAG_RES = {'GO (Cellular Process)': 'GO_CP', 'GO (Cellular Component)': 'GO_CC',
           'GO (Molecular Function)': 'GO_MF', 'Cell types': 'Cell_types'}
 
 
def _tag(name):
    return name.replace(' ', '_').replace('(', '').replace(')', '')
 
 
# =============================================================================
# PART A. FIGURES
# =============================================================================
print("\n" + "=" * 78)
print(" BLOCK 8A: disjoint integration heatmaps")
print("=" * 78)
 
 
def _matrix(layer):
    sub = df_disj_full[df_disj_full.Layer == layer]
    M = pd.DataFrame(np.nan, index=CATEGORIES, columns=CATEGORIES)
    for r in sub.itertuples():
        M.loc[r.Category_A, r.Category_B] = r.AUC_disjoint
        M.loc[r.Category_B, r.Category_A] = r.AUC_disjoint
    return M
 
 
# Diagonal = modularity AUC, taken from the main script's output
mod_path = os.path.join(OUTDIR, 'modularity_auc.txt')
AUC_MOD_FILE = (pd.read_csv(mod_path, sep='\t', index_col=0)
                if os.path.exists(mod_path) else None)
if AUC_MOD_FILE is None:
    print("  modularity_auc.txt not found: diagonal left empty.")
 
MATS = {}
for name in LAYER_ORDER:
    M = _matrix(name)
    if AUC_MOD_FILE is not None:
        for c in CATEGORIES:
            if c in AUC_MOD_FILE.index and name in AUC_MOD_FILE.columns:
                M.loc[c, c] = AUC_MOD_FILE.loc[c, name]
    MATS[name] = M
 
# One colour scale for all layers, so intensities are comparable between panels
glim = max(np.nanmax(np.abs(M.values - 0.5)) for M in MATS.values())
vmin, vmax = 0.5 - glim, 0.5 + glim
print(f"  Global AUC scale: {vmin:.3f} to {vmax:.3f}")
 
 
def _order(M):
    """
    Same clustering as the joint figure: cosine distance, average linkage, on
    Cliff's delta (2 PS - 1) with the diagonal at zero. Pairs without a
    disjoint PS enter as 0 (no excess), only for ordering.
    """
    arr = np.nan_to_num(2 * M.values - 1, nan=0.0)
    np.fill_diagonal(arr, 0.0)
    dist = np.nan_to_num(pdist(arr, metric='cosine'), nan=1.0)
    return sch.leaves_list(sch.linkage(dist, method='average'))
 
 
fdr_lookup = {(r.Layer, r.Category_A, r.Category_B): r.FDR
              for r in df_disj_full.itertuples()}
 
for name in LAYER_ORDER:
    M = MATS[name]
    o = _order(M)
    ordr = M.iloc[o, o]
 
    star = pd.DataFrame('', index=ordr.index, columns=ordr.columns)
    for a in ordr.index:
        for b in ordr.columns:
            if a != b:
                f = fdr_lookup.get((name, a, b), fdr_lookup.get((name, b, a), np.nan))
                if pd.notna(f) and f < FDR_ALPHA:
                    star.loc[a, b] = '*'
 
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#e6e6e6')          # grey = no disjoint AUC for that pair
    sns.heatmap(ordr, cmap='PuOr_r', center=0.5, vmin=vmin, vmax=vmax,
                annot=star, fmt='',
                annot_kws={'size': 10, 'color': 'black', 'ha': 'center', 'va': 'center'},
                linewidths=.4, linecolor='lightgray', ax=ax,
                cbar_kws={'label': 'PS', 'shrink': 0.82})
    
    cbar = ax.collections[0].colorbar
    cbar.set_label('PS', fontsize=14)  # Tamaño de la etiqueta 'PS'
    cbar.ax.tick_params(labelsize=11)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.suptitle(f'{name}', fontsize=16, fontweight='bold', y=0.92)
    out = os.path.join(OUTDIR, f'Figure3_integration_disjoint_{_tag(name)}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(out)}")
 


