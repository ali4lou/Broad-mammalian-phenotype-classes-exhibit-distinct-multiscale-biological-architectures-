#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SECTION 2: MODULARITY AND INTEGRATION OF BROAD MP CATEGORIES

Modularity (cohesion within a category) and integration (connection between two
categories) are both measured by one quantity, the AUC, computed on the
RESIDUAL similarity matrices, and tested by permutation of phenotypes.

  BLOCK 1  Load and align the five similarity matrices
  BLOCK 2  Build the residual matrices
  BLOCK 3  Map phenotypes to top-level MP categories
  BLOCK 4  Diagnostic: is each category enriched in broadly annotated phenotypes?
  BLOCK 5  Modularity: PS (AUC), permutation p-value, FDR
  BLOCK 6  Integration: PS (AUC), permutation p-value, FDR
  BLOCK 7  Figures: Figure 2 (modularity), OLD Figure 3 (integration, one per layer)

"""

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import obonet
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests

# =============================================================================
# CONFIGURATION
# =============================================================================
path = 'PATH'
path2 = os.path.join('PATH')
url = 'PATH/MPheno_OBO.ontology.obo'
OUTDIR = os.path.join('PATH')
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42

# Run once with QUICK_TEST = True to check the pipeline and time it.
# The numbers it produces are NOT reportable.
QUICK_TEST = False

if QUICK_TEST:
    N_PERM_MOD, N_PERM_INT = 200, 100
else:
    # Modularity: ~27 categories per layer, so BH needs p resolution better than
    # 0.05/27 = 1.9e-3. With 10,000 permutations the floor is 1.0e-4.
    N_PERM_MOD = 10000
    # Integration: ~350 category pairs per layer, needing resolution better than
    # 1.4e-4, which no feasible number of permutations provides. The 2,000
    # permutations estimate the mean and standard deviation of the null AUC, and
    # p-values come from the normal approximation, checked in Block 6.
    N_PERM_INT = 2000

MIN_CATEGORY_SIZE = 5      # below this the null is too discrete to trust
FDR_ALPHA = 0.05

MIN_CELLS_PER_TYPE = 50
KS_PVALUE_CUTOFF = 0.0001

GENE_LAYER = 'Genes'
CT_LAYER = 'Cell types'
GO_LAYERS = ['GO (Cellular Process)', 'GO (Cellular Component)',
             'GO (Molecular Function)']
LAYER_ORDER = [GENE_LAYER] + GO_LAYERS + [CT_LAYER]

rng_global = np.random.default_rng(SEED)


# =============================================================================
# BLOCK 1. LOAD AND ALIGN THE SIMILARITY MATRICES
# =============================================================================
print("=" * 78)
print(" BLOCK 1: loading similarity matrices")
print("=" * 78)

with open(os.path.join(path, 'phen_filtro_20_500_genes_no_redundantes_final.txt')) as f:
    phen_filter = [ln.strip() for ln in f if ln.strip()]

# --- cell-type layer, built here from the KS statistics ---------------------
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

# Strict check: filling an absent phenotype with zeros would give it zero
# similarity against everything and bias every statistic below.
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

# --- annotation breadth ------------------------------------------------------
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
print(f"  Gene-set size: median {np.median(sizes):.0f}, "
      f"range {sizes.min():.0f}-{sizes.max():.0f}")


# =============================================================================
# BLOCK 2. RESIDUAL MATRICES
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
    print(f"  {name:<26} R2 removed = {1 - res.var() / v.var():.4f}, "
          f"residual sd = {res.std():.4f}")

print("\n  Residual standard deviations differ between layers only in proportion")
print("  to (1 - R2), so z-scores computed on them are comparable across layers.")


# =============================================================================
# BLOCK 3. MAP PHENOTYPES TO TOP-LEVEL MP CATEGORIES
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
dropped = [c for c in cat_members if c not in CATEGORIES]
if dropped:
    print(f"  Excluded for being too small: {dropped}")
for c in CATEGORIES:
    print(f"    {c:<45} n = {len(cat_idx[c])}")


# =============================================================================
# BLOCK 4. DIAGNOSTIC: ARE CATEGORIES ENRICHED IN BROADLY ANNOTATED PHENOTYPES?
# =============================================================================
# Residualisation handles this uniformly, so this is not used to decide which
# categories to correct. It is evidence that the correction was needed, and it
# identifies the categories whose uncorrected scores would have been most
# inflated or most depressed.
print("\n" + "=" * 78)
print(" BLOCK 4: annotation-breadth diagnostic per category")
print("=" * 78)

rows = []
for c in CATEGORIES:
    inside = sizes[cat_idx[c]]
    mask = np.ones(n, bool)
    mask[cat_idx[c]] = False
    outside = sizes[mask]
    U = mannwhitneyu(inside, outside, alternative='two-sided')
    rows.append(dict(Category=c, N=len(inside),
                     Median_genes_in=np.median(inside),
                     Median_genes_out=np.median(outside),
                     Cliffs_delta=2 * U.statistic / (len(inside) * len(outside)) - 1,
                     p=U.pvalue))

df_diag = pd.DataFrame(rows).sort_values('Cliffs_delta', key=abs, ascending=False)
print(df_diag.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
df_diag.to_csv(os.path.join(OUTDIR, 'TableS_category_breadth_bias.txt'),
               sep='\t', index=False)
print(f"\n  |delta| > 0.15 in {(df_diag.Cliffs_delta.abs() > 0.15).sum()} of "
      f"{len(df_diag)} categories.")


# =============================================================================
# BLOCK 5. MODULARITY
# =============================================================================
print("\n" + "=" * 78)
print(f" BLOCK 5: modularity AUC ({N_PERM_MOD:,} permutations per category and layer)")
print("=" * 78)

N_PAIRS = len(iu[0])

# Position of every unordered pair (i, j), i != j, in the upper-triangle vector
POS = np.full((n, n), -1, dtype=np.int64)
POS[iu] = np.arange(N_PAIRS)
POS[(iu[1], iu[0])] = np.arange(N_PAIRS)

# Global ranks of every pair in each layer, ties averaged. Computed once.
RANKS = {name: rankdata(RES[name][iu]) for name in LAYER_ORDER}

_triu_cache = {}


def triu_of(k):
    if k not in _triu_cache:
        _triu_cache[k] = np.triu_indices(k, k=1)
    return _triu_cache[k]


def auc_of(ranks, p):
    """AUC of the pairs at positions p against all remaining pairs of the matrix."""
    n1 = len(p)
    n2 = N_PAIRS - n1
    return (ranks[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def within_positions(idx):
    r, c = triu_of(len(idx))
    return POS[np.ix_(idx, idx)][r, c]


def cross_positions(ia, ib):
    """
    Positions of the pairs linking two groups. Self-pairs (a phenotype present
    in both groups) are dropped, and each unordered pair is counted once: when
    the groups overlap, a pair of shared phenotypes appears twice in the
    rectangle, as (i, j) and (j, i).
    """
    p = POS[np.ix_(ia, ib)]
    return np.unique(p[p >= 0])


rows = []
for name in LAYER_ORDER:
    ranks = RANKS[name]
    for c in CATEGORIES:
        idx = cat_idx[c]
        k = len(idx)
        obs = auc_of(ranks, within_positions(idx))

        # Surrogate groups exclude the category's own members, which would
        # otherwise make up part of each random group and inflate the null.
        pool = np.setdiff1d(np.arange(n), idx)
        rng = np.random.default_rng(SEED + abs(hash(c)) % 10000)
        null = np.empty(N_PERM_MOD)
        for i in range(N_PERM_MOD):
            sel = rng.choice(pool, size=k, replace=False)
            null[i] = auc_of(ranks, within_positions(sel))

        rows.append(dict(Layer=name, Category=c, N=k,
                         AUC=obs, Cliffs_delta=2 * obs - 1,
                         Null_mean=null.mean(), Null_sd=null.std(),
                         p_empirical=(1 + np.sum(null >= obs)) / (N_PERM_MOD + 1),
                         p_empirical_below=(1 + np.sum(null <= obs)) / (N_PERM_MOD + 1)))
    print(f"  {name}: done")

df_mod = pd.DataFrame(rows)

# FDR within each layer, on the one-sided test for cohesion
df_mod['FDR'] = np.nan
for name in LAYER_ORDER:
    m = df_mod.Layer == name
    df_mod.loc[m, 'FDR'] = multipletests(df_mod.loc[m, 'p_empirical'], method='fdr_bh')[1]
df_mod['p_floor'] = 1 / (N_PERM_MOD + 1)
df_mod.to_csv(os.path.join(OUTDIR, 'TableS_modularity.txt'), sep='\t', index=False)

AUC_MOD = df_mod.pivot(index='Category', columns='Layer', values='AUC')[LAYER_ORDER]
FDR_MOD = df_mod.pivot(index='Category', columns='Layer', values='FDR')[LAYER_ORDER]
AUC_MOD.to_csv(os.path.join(OUTDIR, 'modularity_auc.txt'), sep='\t')

print("\n  Null mean AUC per layer (should sit close to 0.5):")
print(df_mod.groupby('Layer').Null_mean.describe()[['min', '50%', 'max']]
      .loc[LAYER_ORDER].to_string(float_format=lambda x: f'{x:.4f}'))

sig = df_mod[df_mod.FDR < FDR_ALPHA]
print(f"\n  Significant cohesion at FDR < {FDR_ALPHA}: {len(sig)} of {len(df_mod)}")
print(f"  Categories cohesive in at least one layer: {sig.Category.nunique()} "
      f"of {len(CATEGORIES)}")
print(f"  Categories cohesive in three or more layers: "
      f"{(sig.groupby('Category').size() >= 3).sum()}")
print("  Significant counts per layer:")
print(sig.groupby('Layer').size().reindex(LAYER_ORDER).fillna(0).astype(int)
      .to_string())

below = df_mod[df_mod.p_empirical_below * 2 < FDR_ALPHA]
if len(below):
    print("\n  Below chance (two-sided, uncorrected), for the text:")
    print(below[['Layer', 'Category', 'AUC']].to_string(
        index=False, float_format=lambda x: f'{x:.3f}'))


# =============================================================================
# BLOCK 6. INTEGRATION
# =============================================================================
print("\n" + "=" * 78)
print(f" BLOCK 6: integration AUC ({N_PERM_INT:,} permutations per pair and layer)")
print("=" * 78)

rows = []
for name in LAYER_ORDER:
    ranks = RANKS[name]
    for i, ca in enumerate(CATEGORIES):
        for cb in CATEGORIES[i + 1:]:
            ia, ib = cat_idx[ca], cat_idx[cb]
            obs = auc_of(ranks, cross_positions(ia, ib))

            pool = np.setdiff1d(np.arange(n), np.union1d(ia, ib))
            rng = np.random.default_rng(SEED + abs(hash(ca + cb)) % 10000)
            k1, k2 = len(ia), len(ib)
            null = np.empty(N_PERM_INT)
            for j in range(N_PERM_INT):
                a = rng.choice(pool, size=k1, replace=False)
                b = rng.choice(pool, size=k2, replace=False)
                null[j] = auc_of(ranks, cross_positions(a, b))

            mu, sd = null.mean(), null.std()
            z = (obs - mu) / sd if sd > 0 else np.nan
            rows.append(dict(Layer=name, Category_A=ca, Category_B=cb,
                             N_A=k1, N_B=k2,
                             N_shared=len(np.intersect1d(ia, ib)),
                             AUC=obs, Cliffs_delta=2 * obs - 1,
                             Null_mean=mu, Null_sd=sd,
                             p_empirical=(1 + np.sum(null >= obs)) / (N_PERM_INT + 1),
                             p_normal=norm.sf(z),
                             Null_skew=float(pd.Series(null).skew())))
    print(f"  {name}: done")

df_int = pd.DataFrame(rows)

print(f"\n  Skewness of the null AUC across all pairs: median "
      f"{df_int.Null_skew.median():.3f}, 95th percentile of |skew| "
      f"{df_int.Null_skew.abs().quantile(.95):.3f}")
print("  Values near zero support the normal approximation used for the FDR.")

df_int['FDR'] = np.nan
for name in LAYER_ORDER:
    m = df_int.Layer == name
    df_int.loc[m, 'FDR'] = multipletests(df_int.loc[m, 'p_normal'], method='fdr_bh')[1]
df_int.to_csv(os.path.join(OUTDIR, 'TableS_integration.txt'), sep='\t', index=False)

print(f"  Significant integration at FDR < {FDR_ALPHA}: "
      f"{(df_int.FDR < FDR_ALPHA).sum()} of {len(df_int)}")
print(df_int[df_int.FDR < FDR_ALPHA].groupby('Layer').size()
      .reindex(LAYER_ORDER).fillna(0).astype(int).to_string())

# Square matrices. The diagonal holds each category's modularity AUC, since the
# integration of a category with itself is its modularity.
for name in LAYER_ORDER:
    sub = df_int[df_int.Layer == name]
    Ma = pd.DataFrame(np.nan, index=CATEGORIES, columns=CATEGORIES)
    for _, r in sub.iterrows():
        Ma.loc[r.Category_A, r.Category_B] = Ma.loc[r.Category_B, r.Category_A] = r.AUC
    for c in CATEGORIES:
        Ma.loc[c, c] = AUC_MOD.loc[c, name]
    tag = name.replace(' ', '_').replace('(', '').replace(')', '')
    Ma.to_csv(os.path.join(OUTDIR, f'integration_auc_{tag}.csv'))

# Pairs connected in every layer, ranked by their weakest one: the systems
# integrated at all levels, which are the ones most worth describing.
wide = df_int.pivot_table(index=['Category_A', 'Category_B'],
                          columns='Layer', values='AUC')[LAYER_ORDER]
ranked = wide.assign(min_AUC=wide.min(axis=1)).sort_values('min_AUC', ascending=False)
ranked.to_csv(os.path.join(OUTDIR, 'integration_pairs_ranked.txt'), sep='\t')
print("\n  Most integrated pairs, ranked by their weakest layer:")
print(ranked.head(10).to_string(float_format=lambda x: f'{x:.3f}'))


# =============================================================================
# BLOCK 7. FIGURES
# =============================================================================
print("\n" + "=" * 78)
print(" BLOCK 7: figures")
print("=" * 78)

# --- Figure 2: modularity ----------------------------------------------------
# Clustered on the AUC profiles directly: they are bounded and on a common
# scale, so no transform is needed and no column dominates the distances.
link = sch.linkage(pdist(AUC_MOD.values, metric='euclidean'), method='ward')
A = AUC_MOD.iloc[sch.leaves_list(link)]
F = FDR_MOD.loc[A.index]
annot = A.round(2).astype(str) + np.where(F < FDR_ALPHA, '*', '')

lim = np.nanmax(np.abs(A.values - 0.5))
plt.figure(figsize=(9, 12), dpi=300)
ax = sns.heatmap(A, cmap='PuOr_r', center=0.5, vmin=0.5 - lim, vmax=0.5 + lim,
                 annot=annot, fmt='', annot_kws={'size': 8}, linewidths=.5,
                 cbar_kws={'shrink': .45,
                           'label': 'P(within-category pair > random pair)'})
ax.set_ylabel('Broad MP phenotype category', fontsize=13, fontweight='bold')
ax.set_xlabel('')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Figure2_modularity.png'), dpi=600, bbox_inches='tight')
plt.close()

labels = sch.fcluster(link, t=4, criterion='maxclust')
groups = defaultdict(list)
for cat, lab in zip(AUC_MOD.index, labels):
    groups[lab].append(cat)
print("  Figure2_modularity.png   (* marks FDR < 0.05)")
print("\n  Modularity clusters:")
for lab, mem in sorted(groups.items()):
    print(f"    Group {lab} ({len(mem)}): {', '.join(sorted(mem))}")
pd.Series(dict(zip(AUC_MOD.index, labels)), name='cluster').to_csv(
    os.path.join(OUTDIR, 'modularity_clusters.txt'), sep='\t')

# --- Figure 3: integration, one clustermap per layer -------------------------
# Clustering uses Cliff's delta (2 AUC - 1): cosine distance on raw AUC values,
# which all lie around 0.5-0.9, would find nearly every profile alike, whereas
# centring on zero makes the sign count, so categories connected to the same
# systems, and disconnected from the same ones, group together. The diagonal
# (modularity) is set to zero for clustering so it does not dominate, and is
# shown in the plotted matrix.
fdr_int = df_int.set_index(['Layer', 'Category_A', 'Category_B'])['FDR']

for name in LAYER_ORDER:
    tag = name.replace(' ', '_').replace('(', '').replace(')', '')
    Ma = pd.read_csv(os.path.join(OUTDIR, f'integration_auc_{tag}.csv'), index_col=0)

    D = 2 * Ma - 1
    np.fill_diagonal(D.values, 0.0)
    D = D.fillna(0.0)

    lim = np.nanmax(np.abs(Ma.values - 0.5))
    g = sns.clustermap(D, method='average', metric='cosine',
                       cmap='PuOr_r', center=0, vmin=-2 * lim, vmax=2 * lim,
                       linewidths=.4, linecolor='lightgray', figsize=(10.5, 10),
                       cbar_pos=(0.02, 0.83, 0.025, 0.14),
                       dendrogram_ratio=(.15, .15))

    ordr = Ma.iloc[g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind]
    star = pd.DataFrame('', index=ordr.index, columns=ordr.columns)
    for a in ordr.index:
        for b in ordr.columns:
            if a != b:
                f = fdr_int.get((name, a, b), fdr_int.get((name, b, a), np.nan))
                if f < FDR_ALPHA:
                    star.loc[a, b] = '*'

    g.ax_heatmap.clear()
    sns.heatmap(ordr, cmap='PuOr_r', center=0.5, vmin=0.5 - lim, vmax=0.5 + lim,
                annot=star, fmt='', annot_kws={'size': 7, 'color': 'black'},
                linewidths=.4, linecolor='lightgray', ax=g.ax_heatmap,
                cbar_ax=g.cax,
                cbar_kws={'label': 'AUC'})
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=9)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')
    g.figure.suptitle(name, fontsize=15, fontweight='bold', y=1.01)
    g.savefig(os.path.join(OUTDIR, f'Figure3_integration_{tag}.png'),
              dpi=600, bbox_inches='tight')
    plt.close('all')

    labels = sch.fcluster(g.dendrogram_row.linkage, t=4, criterion='maxclust')
    groups = defaultdict(list)
    for cat, lab in zip(D.index, labels):
        groups[lab].append(cat)
    print(f"\n  [{name}] integration clusters:")
    for lab, mem in sorted(groups.items()):
        print(f"    Group {lab} ({len(mem)}): {', '.join(sorted(mem))}")

print(f"""
>>> READING THE FIGURES
    0.5 means no excess similarity. Above 0.5, pairs within a category (Figure 2)
    or linking two categories (Figure 3, off-diagonal) are more alike than a
    random pair; below 0.5, less. In Figure 3 the diagonal shows each category's
    modularity for reference. * marks FDR < 0.05 from the permutation test.

All outputs written to:
  {OUTDIR}
""")


#common figures
# --- Figure 3: integration, ordered by clustering but without dendrogram ---

fdr_int = df_int.set_index(['Layer', 'Category_A', 'Category_B'])['FDR']

# 1. Leer todas las matrices para fijar la escala global y preparar el clustering
matrices = {}
matrices_D = {}
global_lim = 0.0

for name in LAYER_ORDER:
    tag = name.replace(' ', '_').replace('(', '').replace(')', '')
    Ma = pd.read_csv(os.path.join(OUTDIR, f'integration_auc_{tag}.csv'), index_col=0)
    
    # Calcular D (Cliff's delta) puramente para el clustering
    D = 2 * Ma - 1
    np.fill_diagonal(D.values, 0.0)
    D = D.fillna(0.0)
    
    matrices[name] = Ma
    matrices_D[name] = D
    
    # Buscar el límite global para la barra de color
    lim = np.nanmax(np.abs(Ma.values - 0.5))
    if lim > global_lim:
        global_lim = lim

# Límites globales unificados
global_vmin = 0.5 - global_lim
global_vmax = 0.5 + global_lim

print(f"Escala global de AUC establecida: vmin={global_vmin:.3f}, vmax={global_vmax:.3f}")

# 2. Dibujar los heatmaps limpios usando el orden del clustering
for name in LAYER_ORDER:
    tag = name.replace(' ', '_').replace('(', '').replace(')', '')
    Ma = matrices[name]
    D = matrices_D[name]

    # TRUCO: Clustermap fantasma solo para robarle el orden
    g_dummy = sns.clustermap(D, method='average', metric='cosine')
    row_order = g_dummy.dendrogram_row.reordered_ind
    col_order = g_dummy.dendrogram_col.reordered_ind
    plt.close(g_dummy.figure)  # Destruimos la figura antes de que se muestre/guarde

    # Reordenar la matriz original con el índice del dendrograma
    ordr = Ma.iloc[row_order, col_order]

    # Matriz de asteriscos (ahora sobre la matriz ya ordenada)
    star = pd.DataFrame('', index=ordr.index, columns=ordr.columns)
    for a in ordr.index:
        for b in ordr.columns:
            if a != b:
                f = fdr_int.get((name, a, b), fdr_int.get((name, b, a), np.nan))
                if f < FDR_ALPHA:
                    star.loc[a, b] = '*'

    # Crear figura limpia
    fig, ax = plt.subplots(figsize=(11, 9))

    # Heatmap final
    sns.heatmap(ordr, cmap='PuOr_r', center=0.5, vmin=global_vmin, vmax=global_vmax,
                annot=star, fmt='', annot_kws={'size': 10, 'color': 'black', 'ha': 'center', 'va': 'center'},
                linewidths=.4, linecolor='lightgray', ax=ax,
                cbar_kws={'label': 'AUC', 'shrink': 0.82})
    
    # Ajustes estéticos
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.suptitle(name, fontsize=15, fontweight='bold', y=0.92) # y ajustado por quitar el dendrograma
    
    # Guardar figura
    fig.savefig(os.path.join(OUTDIR, f'Figure3_integration_{tag}_no_dendo.png'),
                dpi=600, bbox_inches='tight')
    plt.close(fig)

    # Imprimir los grupos en la consola (manteniendo tu lógica original usando scipy)
    dist_matrix = sch.distance.pdist(D.values, metric='cosine')
    linkage_matrix = sch.linkage(dist_matrix, method='average')
    labels = sch.fcluster(linkage_matrix, t=4, criterion='maxclust')
    groups = defaultdict(list)
    for cat, lab in zip(D.index, labels):
        groups[lab].append(cat)
    
    print(f"\n  [{name}] integration clusters:")
    for lab, mem in sorted(groups.items()):
        print(f"    Group {lab} ({len(mem)}): {', '.join(sorted(mem))}")
        
        
        
rows = []
for name in LAYER_ORDER:
    ranks = RANKS[name]
    for i, ca in enumerate(CATEGORIES):
        for cb in CATEGORIES[i + 1:]:
            ia, ib = cat_idx[ca], cat_idx[cb]
            a_only, b_only = np.setdiff1d(ia, ib), np.setdiff1d(ib, ia)
            shared = len(np.intersect1d(ia, ib))
            auc = (auc_of(ranks, cross_positions(a_only, b_only))
                   if len(a_only) >= 2 and len(b_only) >= 2 else np.nan)
            rows.append(dict(Layer=name, Category_A=ca, Category_B=cb,
                             N_shared=shared,
                             Frac_shared=shared / min(len(ia), len(ib)),
                             AUC_disjoint=auc))
df_disj = pd.DataFrame(rows)
w = df_disj.pivot_table(index=['Category_A', 'Category_B'],
                        columns='Layer', values='AUC_disjoint')[LAYER_ORDER]
chk = ranked[['min_AUC']].join(w.min(axis=1).rename('min_AUC_disjoint')).join(
      df_disj.groupby(['Category_A', 'Category_B'])[['N_shared', 'Frac_shared']].first())
chk.to_csv(os.path.join(OUTDIR, 'integration_overlap_check.txt'), sep='\t')
print(chk.head(20).to_string(float_format=lambda x: f'{x:.3f}'))


# Phenotypes shared between each pair of categories: the ones excluded from the
# disjoint integration analysis.
rows = []
for i, ca in enumerate(CATEGORIES):
    for cb in CATEGORIES[i + 1:]:
        shared = np.intersect1d(cat_idx[ca], cat_idx[cb])
        for j in shared:
            rows.append(dict(Category_A=ca, Category_B=cb, Phenotype=PHENOS[j]))

df_shared = pd.DataFrame(rows)
df_shared.to_csv(os.path.join(OUTDIR, 'TableS_shared_phenotypes.txt'),
                 sep='\t', index=False)

# The pairs that changed most in the check
for ca, cb in [('craniofacial phenotype', 'skeleton phenotype'),
               ('craniofacial phenotype', 'growth/size/body region phenotype'),
               ('limbs/digits/tail phenotype', 'skeleton phenotype'),
               ('craniofacial phenotype', 'digestive/alimentary phenotype'),
               ('endocrine/exocrine gland phenotype', 'neoplasm')]:
    a, b = sorted([ca, cb])
    sub = df_shared[(df_shared.Category_A == a) & (df_shared.Category_B == b)]
    print(f"\n{a}  x  {b}  ({len(sub)} shared)")
    for p in sorted(sub.Phenotype):
        print(f"   {p}")
        
        
        
#correlation of modularity with size category
from scipy.stats import spearmanr

auc = pd.read_csv(os.path.join(OUTDIR, 'modularity_auc.txt'), sep='\t', index_col=0)
mod = pd.read_csv(os.path.join(OUTDIR, 'TableS_modularity.txt'), sep='\t')
N = mod.drop_duplicates('Category').set_index('Category')['N']

print("Spearman rho entre AUC y N, por capa")
for L in auc.columns:
    rho, p = spearmanr(auc[L], N.reindex(auc.index))
    print(f"  {L:<26} rho = {rho:+.3f}   p = {p:.3f}")


import matplotlib.pyplot as plt
plt.scatter(N.reindex(auc.index), auc['Genes'])
for cat in auc.index:
    plt.annotate(cat[:15], (N[cat], auc.loc[cat, 'Genes']), fontsize=6)
plt.xlabel('N'); plt.ylabel('AUC (Genes)')
plt.show()
    

        
