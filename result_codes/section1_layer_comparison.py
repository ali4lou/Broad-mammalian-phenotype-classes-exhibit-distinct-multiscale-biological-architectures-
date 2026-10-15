#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SECTION 1: PHENOTYPIC SIMILARITY AND CONSERVATION ACROSS BIOLOGICAL LAYERS

  BLOCK 1  Build and align the five similarity matrices: developmental cell-type layer, built here from the KS statistics         
  BLOCK 2  Annotation breadth per layer                          -> Note S2, Figure S2
  BLOCK 3  Are the strongest genetic pairs size-driven?          
  BLOCK 4  Partial Spearman between layers                       -> Note S3, Figure S3
  BLOCK 5  Functional similarity beyond shared genes             -> Note S5
  BLOCK 6  Convergent pairs and annotation-depth robustness      -> Note S5

NOT IN THIS SCRIPT. The gene-shuffling null on the GO matrices (Note S4) belongs
to the matrix construction pipeline, since it must rebuild the phenotype-GO
matrices from scratch at every permutation. It lives in the GO construction
script.

METHODOLOGICAL NOTES

Node level, not pair level. Annotation breadth is measured against each
phenotype's median similarity to all others, because the permutation tests
resample phenotypes rather than pairs. Where a pair-level summary is needed, the
minimum and maximum gene-set size are used rather than the mean: set-overlap
measures depend on the relation between the two sizes, not on their average.

Mantel, not parametric. With ~10^6 non-independent pairs any parametric p-value
is meaningless. Permuting phenotype labels preserves the dependence structure.

"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, pearsonr, spearmanr, mannwhitneyu
from matplotlib.colors import LogNorm
from scipy.stats import rankdata, pearsonr, spearmanr

# =============================================================================
# CONFIGURATION
# =============================================================================
path = 'PATH'
path2 = os.path.join('PATH')
OUTDIR = os.path.join('PATH')
os.makedirs(OUTDIR, exist_ok=True)

SEED = 42
N_MANTEL = 999            # smallest attainable p = 1/1000
N_PERM_SMALL = 1000       # nulls for median-rank tests
N_TOP_PAIRS = 500
TOP_FRACTIONS = [0.001, 0.01, 0.05]
MIN_GENES_GRID = [0, 20, 50, 100]

MIN_CELLS_PER_TYPE = 50
KS_PVALUE_CUTOFF = 0.0001

# Layers must contain identical phenotype sets. If they do not, the script stops
# rather than filling the gaps with zeros, which would inject artificial rows of
# zero similarity and bias every downstream statistic. Set this to True only as
# a deliberate fallback: it restricts the analysis to the intersection, which
# changes the phenotype universe and must be reported in the Methods.
RESTRICT_TO_INTERSECTION = False

GENE_LAYER = 'Genes'
CT_LAYER = 'Cell types'
GO_LAYERS = ['GO (Cellular Process)',
             'GO (Cellular Component)',
             'GO (Molecular Function)']
LAYER_ORDER = [GENE_LAYER] + GO_LAYERS + [CT_LAYER]

rng_global = np.random.default_rng(SEED)


# =============================================================================
# SHARED HELPERS
# =============================================================================
def rank01(v):
    """Rank-transform to (0, 1). Makes every regression a Spearman-type analysis."""
    return (rankdata(v) - 0.5) / len(v)


def upper(M):
    return M[np.triu_indices_from(M, k=1)]


def median_sim_per_phenotype(S):
    """Node-level propensity: median similarity of each phenotype against all others."""
    S = S.copy()
    np.fill_diagonal(S, np.nan)
    return np.nanmedian(S, axis=1)


def cliffs_delta(x, y):
    """Non-parametric effect size in [-1, 1]. |d| < 0.147 is conventionally negligible."""
    U = mannwhitneyu(x, y, alternative='two-sided').statistic
    return 2 * U / (len(x) * len(y)) - 1


def eta_squared(sizes, values, n_bins=10):
    """
    Variance of `values` explained by the decile of `sizes`.

    Shape-agnostic, unlike a correlation coefficient: it captures monotone and
    non-monotone dependence alike.
    """
    values, sizes = np.asarray(values, float), np.asarray(sizes, float)
    ok = np.isfinite(values) & np.isfinite(sizes)
    values, sizes = values[ok], sizes[ok]
    ranks = np.argsort(np.argsort(sizes))
    bins = np.minimum((ranks * n_bins) // len(ranks), n_bins - 1)
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total == 0:
        return np.nan
    ss_between = sum(((values[bins == b].mean() - grand) ** 2) * (bins == b).sum()
                     for b in range(n_bins) if (bins == b).any())
    return ss_between / ss_total


def mantel(Ra, Rb, n_perm=N_MANTEL, seed=SEED):
    """
    Correlate two square matrices, with significance from permuting phenotype
    labels rather than pairs. Works on similarity or on residual matrices.
    """
    rng = np.random.default_rng(seed)
    iu_ = np.triu_indices(Ra.shape[0], k=1)
    a = Ra[iu_]
    obs = pearsonr(a, Rb[iu_])[0]
    count = sum(abs(pearsonr(a, Rb[np.ix_(p, p)][iu_])[0]) >= abs(obs)
                for p in (rng.permutation(Ra.shape[0]) for _ in range(n_perm)))
    return obs, (1 + count) / (n_perm + 1)


def to_square(res, n, iu):
    R = np.zeros((n, n))
    R[iu] = res
    return R + R.T


# =============================================================================
# BLOCK 1. BUILD AND ALIGN THE FIVE SIMILARITY MATRICES      
# =============================================================================
print("=" * 78)
print(" BLOCK 1: building and aligning similarity matrices")
print("=" * 78)

with open(os.path.join(path, 'phen_filtro_20_500_genes_no_redundantes_final.txt')) as f:
    phen_filter = [ln.strip() for ln in f if ln.strip()]

# --- Cell-type layer, built here from the KS statistics ---------------------
m_types = np.loadtxt(os.path.join(path, 'm_types.txt'))
n_cells_per_type = m_types.sum(axis=1)
with open(os.path.join(path, 'cell_type_unique_mouse.txt')) as f:
    cell_type_unique = np.array(f.read().split('\n')[:-1])

ks_test = np.loadtxt(os.path.join(path, 'cell_type_ks_test.txt'), dtype=float)
ks_pval = np.loadtxt(os.path.join(path, 'cell_type_p_value.txt'), dtype=float)

good = np.where(n_cells_per_type > MIN_CELLS_PER_TYPE)[0]
significant_ks = np.where(ks_pval[:, good] < KS_PVALUE_CUTOFF, ks_test[:, good], 0)
df_ks_full = pd.DataFrame(significant_ks, index=phen_filter, columns=cell_type_unique[good])

ks_dist = pdist(df_ks_full.values, metric='cosine')
df_ks_similarity = pd.DataFrame(1 - squareform(ks_dist),
                                index=df_ks_full.index, columns=df_ks_full.index)

# --- Layers loaded from disk ------------------------------------------------
df_gene_sim = pd.read_csv(os.path.join(path2, 'phenotype_jaccard_similarity.txt'),
                          sep='\t', index_col=0)
df_cp_sim = pd.read_csv(os.path.join(path2, 'similarity_cosine_GO_Cellular_Process.txt'),
                        sep='\t', index_col=0)
df_cc_sim = pd.read_csv(os.path.join(path2, 'similarity_cosine_GO_Cellular_Component.txt'),
                        sep='\t', index_col=0)
df_mf_sim = pd.read_csv(os.path.join(path2, 'similarity_cosine_GO_Molecular_Function.txt'),
                        sep='\t', index_col=0)

matrix_dict = {
    GENE_LAYER: df_gene_sim,
    'GO (Cellular Process)': df_cp_sim,
    'GO (Cellular Component)': df_cc_sim,
    'GO (Molecular Function)': df_mf_sim,
    CT_LAYER: df_ks_similarity,
}

sorted_phenos = sorted({str(p).strip().lower() for p in phen_filter})

# --- Strict alignment check -------------------------------------------------
# Every layer must contain exactly the same phenotypes. Reindexing a missing
# phenotype and filling with zero would inject an artificial row of zero
# similarity, which biases every downstream statistic: it lowers that
# phenotype's node-level propensity, distorts the breadth regression, and
# contributes ~n spurious pairs to every correlation. Silent filling is
# therefore not acceptable here.
print("\n  Phenotype coverage per layer:")
coverage, missing_by_layer = [], {}
for name, M in matrix_dict.items():
    idx = set(M.index.astype(str).str.strip().str.lower())
    cols = set(M.columns.astype(str).str.strip().str.lower())
    missing = set(sorted_phenos) - idx
    missing_by_layer[name] = missing
    coverage.append(dict(Layer=name,
                         N_in_matrix=len(idx),
                         N_expected=len(sorted_phenos),
                         N_missing=len(missing),
                         Index_equals_columns=(idx == cols)))
    flag = "OK" if not missing else f"MISSING {len(missing)}"
    print(f"    {name:<26} {len(idx):>6} phenotypes   {flag}")

df_cov = pd.DataFrame(coverage)
df_cov.to_csv(os.path.join(OUTDIR, 'TableS_phenotype_coverage.txt'), sep='\t', index=False)

bad_square = df_cov.loc[~df_cov.Index_equals_columns, 'Layer'].tolist()
if bad_square:
    raise ValueError(
        f"Index and columns differ in: {bad_square}. "
        "These matrices must be square and symmetric in the same phenotype order."
    )

any_missing = {k: v for k, v in missing_by_layer.items() if v}
if any_missing:
    with open(os.path.join(OUTDIR, 'missing_phenotypes_per_layer.txt'), 'w') as fh:
        for name, miss in any_missing.items():
            fh.write(f"# {name} ({len(miss)} missing)\n")
            fh.writelines(f"{p}\n" for p in sorted(miss))
            fh.write("\n")
    report = "\n".join(
        f"  {name}: {len(miss)} missing, e.g. {sorted(miss)[:5]}"
        for name, miss in any_missing.items())

    if not RESTRICT_TO_INTERSECTION:
        raise ValueError(
            "Phenotype sets do not match across layers.\n" + report +
            "\n\nFull lists written to missing_phenotypes_per_layer.txt in the "
            "output directory.\nResolve this before continuing: either "
            "regenerate the affected matrices with the same phenotype filter, "
            "or set RESTRICT_TO_INTERSECTION = True to analyse the "
            "intersection instead."
        )

    common = set(sorted_phenos)
    for M in matrix_dict.values():
        common &= set(M.index.astype(str).str.strip().str.lower())
    print(f"\n  RESTRICT_TO_INTERSECTION is on:\n{report}")
    print(f"    Analysing the intersection: {len(common)} of {len(sorted_phenos)} "
          f"phenotypes. Report this in the Methods.")
    sorted_phenos = sorted(common)
else:
    print("    All layers contain the same phenotype set.")

aligned_matrices = {}
for name, M in matrix_dict.items():
    M = M.copy()
    M.index = M.index.astype(str).str.strip().str.lower()
    M.columns = M.columns.astype(str).str.strip().str.lower()
    aligned_matrices[name] = M.loc[sorted_phenos, sorted_phenos]
    assert not aligned_matrices[name].isna().any().any(), f"NaNs remain in {name}"

# --- Annotation breadth -----------------------------------------------------
gene_matrix = pd.read_csv(os.path.join(path2, 'final_filtered_matrix.txt'),
                          sep='\t', index_col=0)
gene_matrix.index = gene_matrix.index.astype(str).str.strip().str.lower()
gene_matrix.columns = gene_matrix.columns.astype(str).str.strip().str.lower()

# Orientation is detected rather than assumed
if len(set(gene_matrix.columns) & set(sorted_phenos)) > len(set(gene_matrix.index) & set(sorted_phenos)):
    breadth_all = (gene_matrix > 0).sum(axis=0)      # genes x phenotypes
else:
    breadth_all = (gene_matrix > 0).sum(axis=1)      # phenotypes x genes
print(f"  Gene matrix orientation detected: {len(breadth_all)} phenotypes with a gene count")

# The gene matrix must also cover every phenotype: the breadth value is the
# covariate in every regression below, so a phenotype without one would be
# dropped silently and change the universe between blocks.
no_breadth = [p for p in sorted_phenos if p not in breadth_all.index]
if no_breadth:
    msg = (f"{len(no_breadth)} phenotypes have no gene count in the gene matrix, "
           f"e.g. {no_breadth[:5]}")
    if not RESTRICT_TO_INTERSECTION:
        raise ValueError(
            msg + "\nThe gene matrix must use the same phenotype filter as the "
                  "similarity matrices. Regenerate it, or set "
                  "RESTRICT_TO_INTERSECTION = True."
        )
    print(f"  {msg}\n  Excluding them.")

PHENOS = [p for p in sorted_phenos if p in breadth_all.index]
n = len(PHENOS)
iu = np.triu_indices(n, k=1)
sizes = breadth_all.reindex(PHENOS).values.astype(float)

print(f"  {n} phenotypes retained  ->  {len(iu[0]):,} pairs")
print(f"  Gene-set size: median {np.median(sizes):.0f}, range {sizes.min():.0f}-{sizes.max():.0f}")


def flat(name):
    return aligned_matrices[name].loc[PHENOS, PHENOS].values[iu]


def square(name):
    return aligned_matrices[name].loc[PHENOS, PHENOS].values


# --- Figure S2: similarity distributions ------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(19, 3.2), dpi=300)
for ax, name in zip(axes, LAYER_ORDER):
    v = flat(name)
    ax.hist(v, bins=100, color='lightgray', log=True)
    for q, col, lab in [(50, 'mediumslateblue', 'Median'),
                        (75, 'darkorange', '75th Percentile'),
                        (95, 'crimson', '95th Percentile')]:
        ax.axvline(np.percentile(v, q), color=col, ls='--', lw=1.4, label=lab)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Jaccard similarity' if name == GENE_LAYER else 'Cosine similarity',
                  fontsize=9)
    ax.grid(False)
axes[0].set_ylabel('# pairwise phenotype', fontsize=9)
axes[0].legend(fontsize=7, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'FigS7_similarity_distributions.png'),
            dpi=600, bbox_inches='tight')
plt.close()

pd.DataFrame([dict(Layer=name,
                   Median=np.median(flat(name)),
                   P75=np.percentile(flat(name), 75),
                   P95=np.percentile(flat(name), 95),
                   Pct_zero=100 * np.mean(flat(name) == 0))
              for name in LAYER_ORDER]).to_csv(
    os.path.join(OUTDIR, 'TableS_similarity_quantiles.txt'), sep='\t', index=False)


# =============================================================================
# BLOCK 2. ANNOTATION BREADTH PER LAYER    
# =============================================================================
print("\n" + "=" * 78)
print(" BLOCK 2: annotation breadth")
print("=" * 78)

C_min = np.minimum(sizes[iu[0]], sizes[iu[1]])
C_max = np.maximum(sizes[iu[0]], sizes[iu[1]])

# Design matrix used for every residualisation in this script
X_breadth = np.column_stack([
    np.ones(len(iu[0])),
    rank01(C_min),
    rank01(C_max),
    rank01(C_min) * rank01(C_max),
])


def r2_of(v, design):
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return 1 - (v - design @ beta).var() / v.var()


rows = []
for name in LAYER_ORDER:
    v = rank01(flat(name))
    node = median_sim_per_phenotype(square(name))
    rows.append(dict(
        Layer=name,
        R2_breadth_pairwise=r2_of(v, X_breadth),
        Pearson_node=pearsonr(sizes, node)[0],
        Spearman_node=spearmanr(sizes, node)[0],
        Eta2_node=eta_squared(sizes, node),
    ))

df_breadth = pd.DataFrame(rows)
print(df_breadth.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
df_breadth.to_csv(os.path.join(OUTDIR, 'TableS_breadth_per_layer.txt'), sep='\t', index=False)
print("\n  R2_breadth_pairwise is the value quoted in the main text and plotted in Figure 1a.")
print("  Spearman is reported alongside Pearson because the relationship saturates.")

# --- Figure 
plt.figure(figsize=(4.6, 3.2), dpi=300)
colors = ['steelblue'] + ['indianred'] * 3 + ['mediumseagreen']
plt.barh(df_breadth.Layer, df_breadth.R2_breadth_pairwise, color=colors)
plt.xlabel('Variance explained by annotation breadth ($R^2$)', fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Fig1a_breadth_variance.png'), dpi=600, bbox_inches='tight')
plt.close()

# --- Figure
fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), dpi=300)
for ax, name in zip(axes, LAYER_ORDER):
    node = median_sim_per_phenotype(square(name))
    ax.scatter(sizes, node, s=9, alpha=.35, color='mediumslateblue', edgecolors='none')
    o = np.argsort(sizes)
    ax.plot(pd.Series(sizes[o]).rolling(30, center=True).mean(),
            pd.Series(node[o]).rolling(30, center=True).mean(),
            color='indigo', lw=1.3)
    r = df_breadth.loc[df_breadth.Layer == name].iloc[0]
    ax.set_title(f"{name}\nr={r.Pearson_node:.3f}  rho={r.Spearman_node:.3f}", fontsize=9)
    ax.set_xlabel('# genes per phenotype', fontsize=9)
    ax.grid(False)
axes[0].set_ylabel('Median similarity', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'FigS8_S9_breadth_vs_similarity.png'),
            dpi=600, bbox_inches='tight')
plt.close()

# --- Decile profile: shows the saturation as numbers ------------------------
ranks = np.argsort(np.argsort(sizes))
dec = np.minimum((ranks * 10) // len(ranks), 9)
prof = []
for name in LAYER_ORDER:
    node = median_sim_per_phenotype(square(name))
    r = dict(Layer=name)
    for b in range(10):
        r[f'D{b+1}'] = np.nanmean(node[dec == b])
    prof.append(r)
df_prof = pd.DataFrame(prof)
print("\n  Median similarity by decile of gene-set size (saturation):")
print(df_prof.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
df_prof.to_csv(os.path.join(OUTDIR, 'TableS_decile_profile.txt'), sep='\t', index=False)

edges = pd.DataFrame([(b + 1, int(sizes[dec == b].min()), int(sizes[dec == b].max()),
                       int((dec == b).sum())) for b in range(10)],
                     columns=['Decile', 'Min_genes', 'Max_genes', 'N_phenotypes'])
edges.to_csv(os.path.join(OUTDIR, 'TableS_decile_ranges.txt'), sep='\t', index=False)


# =============================================================================
# BLOCK 3. ARE THE STRONGEST GENETIC PAIRS SIZE-DRIVEN?        
# =============================================================================
print("\n" + "=" * 78)
print(" BLOCK 3: gene-set size of the strongest genetic pairs")
print("=" * 78)

jac = flat(GENE_LAYER)
summaries = {
    'mean_size': (C_min + C_max) / 2,     # what Note S5 currently uses
    'min_size': C_min,                    # Jaccard is capped by the smaller set
    'max_size': C_max,
    'size_ratio': C_min / C_max,          # 1 = equally sized sets
}

rows = []
for frac in TOP_FRACTIONS:
    k = max(int(round(frac * len(jac))), 10)
    top_idx = np.argpartition(jac, -k)[-k:]
    for label, vals in summaries.items():
        rows.append(dict(Top_fraction=frac, N_top=k, Summary=label,
                         Median_top=np.median(vals[top_idx]),
                         Median_all=np.median(vals),
                         Cliffs_delta=cliffs_delta(vals[top_idx], vals)))

df_top = pd.DataFrame(rows)
print(df_top.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
df_top.to_csv(os.path.join(OUTDIR, 'TableS_top_pairs_size.txt'), sep='\t', index=False)
print("""
  Report Cliff's delta, not a p-value: with this many pairs every p is tiny.
    |delta| < 0.147 on all summaries -> the Note S5 claim holds as written.
    delta positive on max_size       -> the top pairs do involve broadly
                                        annotated phenotypes; soften the claim.
    delta positive on size_ratio     -> the top pairs are pairs of similarly
                                        sized sets, a mechanical property of
                                        Jaccard. Worth reporting either way.
""")

print("\n" + "=" * 78)
print(" BLOCK 3b: Figure S8")
print("=" * 78)

jac = flat(GENE_LAYER)                                             # noqa: F821
node_med = median_sim_per_phenotype(square(GENE_LAYER))            # noqa: F821
size_ratio = C_min / C_max                                         # noqa: F821

TOP_FRAC = 0.01
k = int(round(TOP_FRAC * len(jac)))
top_idx = np.argpartition(jac, -k)[-k:]
print(f"  Highlighting the top {TOP_FRAC:.0%} of pairs (n = {k:,})")

r_node = df_breadth.loc[df_breadth.Layer == GENE_LAYER].iloc[0]     # noqa: F821

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=300)

# --- Panel A: node level ----------------------------------------------------
ax = axes[0]
ax.scatter(sizes, node_med, s=10, alpha=.35,                        # noqa: F821
           color='mediumslateblue', edgecolors='none')
o = np.argsort(sizes)                                               # noqa: F821
ax.plot(pd.Series(sizes[o]).rolling(50, center=True).mean(),        # noqa: F821
        pd.Series(node_med[o]).rolling(50, center=True).mean(),
        color='indigo', lw=1.4, label='Moving average')
ax.set_xlabel('# genes per phenotype', fontsize=10)
ax.set_ylabel('Median Jaccard similarity', fontsize=10)
ax.set_title(f'A\nPearson r = {r_node.Pearson_node:.3f}, '
             f'Spearman rho = {r_node.Spearman_node:.3f}',
             fontsize=9, loc='left')
ax.legend(fontsize=8)
ax.grid(False)

# --- Panel B: the constraint, size ratio ------------------------------------
ax = axes[1]
hb = ax.hexbin(size_ratio, jac, gridsize=60, cmap='Greys',
               norm=LogNorm(), mincnt=1, edgecolors='none')
sub = np.random.default_rng(42).choice(top_idx, size=min(5000, k), replace=False)
ax.scatter(size_ratio[sub], jac[sub], s=3, alpha=.5, color='crimson',
           edgecolors='none', label=f'Top {TOP_FRAC:.0%} of pairs')
# The upper bound of Jaccard given the size ratio, reached when the smaller set
# is fully contained in the larger one
x = np.linspace(0.01, 1, 200)
ax.plot(x, x, color='black', ls='--', lw=1, label='Upper bound (nested sets)')
ax.set_xlabel('Gene-set size ratio (min / max)', fontsize=10)
ax.set_ylabel('Pairwise Jaccard similarity', fontsize=10)
ax.set_ylim(0, np.percentile(jac, 99.99) * 1.1)
ax.set_title('B', fontsize=9, loc='left')
ax.legend(fontsize=7, loc='upper left')
cb = plt.colorbar(hb, ax=ax, shrink=.8)
cb.set_label('# pairs', fontsize=8)
ax.grid(False)

# --- Panel C: are the top pairs more broadly annotated? ---------------------
ax = axes[2]
bins = np.linspace(0, np.percentile(C_max, 99.5), 60)              # noqa: F821
ax.hist(C_max, bins=bins, density=True, color='lightgray',          # noqa: F821
        label='All pairs')
ax.hist(C_max[top_idx], bins=bins, density=True, color='crimson',   # noqa: F821
        alpha=.55, label=f'Top {TOP_FRAC:.0%} of pairs')
delta_max = df_top.loc[(df_top.Top_fraction == 0.01) &              # noqa: F821
                       (df_top.Summary == 'max_size'), 'Cliffs_delta'].values[0]
ax.set_xlabel('Larger gene set of the pair', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title(f"C\nCliff's delta = {delta_max:+.3f}", fontsize=9, loc='left')
ax.legend(fontsize=8)
ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'FigS8_genetic_layer_size.png'),   # noqa: F821
            dpi=600, bbox_inches='tight')
plt.close()

print(f"  Written to {os.path.join(OUTDIR, 'FigS8_genetic_layer_size.png')}")  # noqa: F821
print(f"""
  Panel B: the dashed line is the maximum Jaccard attainable for a given size
  ratio, reached when the smaller gene set is entirely contained in the larger
  one. High similarities are therefore confined to the right-hand side of the
  panel by arithmetic, not by biology.

  Panel C: Cliff's delta = {delta_max:+.3f} for the larger gene set of the pair.
  Values below 0.147 in absolute terms are conventionally negligible, so the
  strongest pairs are not enriched in broadly annotated phenotypes.
""")

# =============================================================================
# BLOCK 4. PARTIAL SPEARMAN BETWEEN LAYERS   
# =============================================================================
print("=" * 78)
print(f" BLOCK 4: partial correlations ({N_MANTEL} Mantel permutations per pair)")
print("=" * 78)

layer_vec, layer_resid = {}, {}
for name in LAYER_ORDER:
    v = rank01(flat(name))
    layer_vec[name] = v
    beta, *_ = np.linalg.lstsq(X_breadth, v, rcond=None)
    layer_resid[name] = to_square(v - X_breadth @ beta, n, iu)

rows = []
for a, b in itertools.combinations(LAYER_ORDER, 2):
    raw = spearmanr(layer_vec[a], layer_vec[b])[0]
    part, p = mantel(layer_resid[a], layer_resid[b])
    rows.append(dict(Layer_A=a, Layer_B=b, Spearman_raw=raw, Spearman_partial=part,
                     Pct_change=100 * (part / raw - 1) if raw else np.nan, Mantel_p=p))
    print(f"  {a} vs {b}: {raw:+.3f} -> {part:+.3f}  (p = {p:.4f})")

df_part = pd.DataFrame(rows)
df_part['p_floor'] = 1 / (N_MANTEL + 1)
df_part.to_csv(os.path.join(OUTDIR, 'TableS_partial_correlations.txt'), sep='\t', index=False)


def square_from(col):
    M = pd.DataFrame(np.eye(len(LAYER_ORDER)), index=LAYER_ORDER, columns=LAYER_ORDER)
    for _, r in df_part.iterrows():
        M.loc[r.Layer_A, r.Layer_B] = M.loc[r.Layer_B, r.Layer_A] = r[col]
    return M


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
for ax, col, title in zip(axes, ['Spearman_raw', 'Spearman_partial'],
                          ['Uncontrolled',
                           'Controlling for annotation breadth']):
    sns.heatmap(square_from(col), annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, square=True, linewidths=.5,
                cbar_kws={'shrink': .7}, ax=ax)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Fig1bc_raw_vs_partial.png'), dpi=600, bbox_inches='tight')
plt.close()

print("""
  Correlations that INCREASE after control are a suppression effect, not an
  artefact: with no breadth signal in one layer there is no shared confounder to
  remove, and controlling the other layer strips variance unrelated to the
  comparison, so the shared signal emerges more clearly.
""")


# =============================================================================
# BLOCK 5. FUNCTIONAL SIMILARITY BEYOND SHARED GENES   
# =============================================================================
print("=" * 78)
print(" BLOCK 5: functional similarity beyond shared genes")
print("=" * 78)

gene_sim = rank01(flat(GENE_LAYER))
ct_sim = rank01(flat(CT_LAYER))

# Shared genes AND breadth, since removing only one leaves the other inside
X_full = np.column_stack([X_breadth, gene_sim])
X_genes_only = np.column_stack([np.ones(len(iu[0])), gene_sim])


def residualise(v, design=X_full):
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    return v - design @ beta


rows, go_res = [], {}
for name in GO_LAYERS:
    v = layer_vec[name]
    go_res[name] = residualise(v)
    rows.append(dict(Layer=name,
                     R2_genes_only=r2_of(v, X_genes_only),
                     R2_breadth_only=r2_of(v, X_breadth),
                     R2_both=r2_of(v, X_full),
                     Residual_variance=1 - r2_of(v, X_full)))

df_dec = pd.DataFrame(rows)
print(df_dec.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
df_dec.to_csv(os.path.join(OUTDIR, 'TableSx_variance_decomposition.txt'), sep='\t', index=False)

ct_res = residualise(ct_sim)

print("\n  Residual functional similarity vs cell types (external validation):")
rows = []
for name in GO_LAYERS:
    raw = spearmanr(layer_vec[name], ct_sim)[0]
    part, p = mantel(to_square(go_res[name], n, iu), to_square(ct_res, n, iu))
    rows.append(dict(GO_layer=name, Spearman_raw=raw, Partial=part, Mantel_p=p))
    print(f"    {name}: {raw:+.3f} -> {part:+.3f}  (p = {p:.4f})")

df_ct = pd.DataFrame(rows)
df_ct.to_csv(os.path.join(OUTDIR, 'TableS_residual_go_vs_celltypes.txt'), sep='\t', index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)
df_dec.set_index('Layer')[['R2_genes_only', 'R2_breadth_only', 'Residual_variance']].plot(
    kind='barh', ax=axes[0], color=['steelblue', 'darkorange', 'mediumseagreen'])
axes[0].set_xlabel('Fraction of variance', fontsize=9)
axes[0].set_ylabel('')
axes[0].legend(['Shared genes', 'Annotation breadth', 'Residual'], fontsize=8)
axes[0].set_title('What drives GO similarity', fontsize=10, fontweight='bold')

w, x = 0.35, np.arange(len(df_ct))
axes[1].bar(x - w / 2, df_ct.Spearman_raw, w, label='Raw', color='lightgray')
axes[1].bar(x + w / 2, df_ct.Partial, w, color='crimson',
            label='Residual (genes + breadth removed)')
axes[1].axhline(0, color='k', lw=.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([l.replace('GO (', '').replace(')', '') for l in df_ct.GO_layer],
                        fontsize=8)
axes[1].set_ylabel('Spearman vs cell types', fontsize=9)
axes[1].legend(fontsize=8)
axes[1].set_title('Does residual function predict cell type?', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'Fig1d_functional_beyond_genes.png'),
            dpi=600, bbox_inches='tight')
plt.close()

print("""
  Cell-type similarity is built from KS statistics rather than shared-element
  counts and carries essentially no breadth signal, so it is the layer that can
  arbitrate. A partial clearly above zero means the convergence is biological
  rather than residual noise.
""")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BLOCK 5b: VALIDATION RESTRICTED TO PAIRS SHARING NO GENES AT ALL

Append after Block 5 of section1_layer_comparison.py. Uses the variables already
defined there: PHENOS, n, iu, sizes, aligned_matrices, layer_vec, GO_LAYERS,
GENE_LAYER, CT_LAYER, rank01, flat, OUTDIR, N_MANTEL, SEED.

"""

print("\n" + "=" * 78)
print(" BLOCK 5b: pairs sharing no genes at all")
print("=" * 78)

# --- Define the subset ------------------------------------------------------
gene_raw = flat(GENE_LAYER)                                        # noqa: F821
zero_mask = gene_raw == 0
n_zero = int(zero_mask.sum())
print(f"  Pairs with Jaccard exactly 0: {n_zero:,} of {len(gene_raw):,} "
      f"({100 * n_zero / len(gene_raw):.1f}%)")

if n_zero < 1000:
    raise ValueError("Too few zero-overlap pairs for this analysis.")

# Row and column indices of those pairs, needed to re-read them after permuting
rows_z = iu[0][zero_mask]                                          # noqa: F821
cols_z = iu[1][zero_mask]

# --- Breadth covariates within the subset -----------------------------------
cmin_z = np.minimum(sizes[rows_z], sizes[cols_z])                  # noqa: F821
cmax_z = np.maximum(sizes[rows_z], sizes[cols_z])                  # noqa: F821
Xz = np.column_stack([
    np.ones(n_zero),
    rank01(cmin_z),                                                # noqa: F821
    rank01(cmax_z),                                                # noqa: F821
    rank01(cmin_z) * rank01(cmax_z),                               # noqa: F821
])


def residualise_z(v):
    beta, *_ = np.linalg.lstsq(Xz, v, rcond=None)
    return v - Xz @ beta


# --- Observed statistics ----------------------------------------------------
ct_square = aligned_matrices[CT_LAYER].loc[PHENOS, PHENOS].values   # noqa: F821
ct_z = rank01(ct_square[rows_z, cols_z])                            # noqa: F821
ct_z_res = residualise_z(ct_z)

go_z_res = {}
for name in GO_LAYERS:                                              # noqa: F821
    go_square = aligned_matrices[name].loc[PHENOS, PHENOS].values    # noqa: F821
    v = rank01(go_square[rows_z, cols_z])                            # noqa: F821
    go_z_res[name] = residualise_z(v)


def mantel_fixed_pairs(go_res_vec, n_perm=N_MANTEL, seed=SEED):      # noqa: F821
    """
    Permute phenotype labels of the cell-type matrix, read the same pair
    positions, recompute. The pair subset stays fixed; only the cell-type
    layer's association with phenotype identity is broken.
    """
    rng = np.random.default_rng(seed)
    obs = pearsonr(go_res_vec, ct_z_res)[0]
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(PHENOS))                             # noqa: F821
        perm = ct_square[np.ix_(p, p)]
        v = residualise_z(rank01(perm[rows_z, cols_z]))               # noqa: F821
        if abs(pearsonr(go_res_vec, v)[0]) >= abs(obs):
            count += 1
    return obs, (1 + count) / (n_perm + 1)


print(f"\n  Residual functional similarity vs cell types, within zero-overlap pairs")
print(f"  ({N_MANTEL} permutations; smallest attainable p = {1/(N_MANTEL+1):.1e})\n")   # noqa: F821

rows = []
for name in GO_LAYERS:                                               # noqa: F821
    go_square = aligned_matrices[name].loc[PHENOS, PHENOS].values     # noqa: F821
    raw_z = spearmanr(go_square[rows_z, cols_z], ct_square[rows_z, cols_z])[0]
    part, p = mantel_fixed_pairs(go_z_res[name])
    # Full-data value for reference, computed in Block 5
    rows.append(dict(GO_layer=name,
                     N_pairs=n_zero,
                     Spearman_raw_zero_overlap=raw_z,
                     Partial_zero_overlap=part,
                     Mantel_p=p))
    print(f"    {name}: raw {raw_z:+.3f}, partial {part:+.3f}  (p = {p:.4f})")

df_zero = pd.DataFrame(rows)
df_zero['p_floor'] = 1 / (N_MANTEL + 1)                              # noqa: F821
df_zero.to_csv(os.path.join(OUTDIR, 'TableS_zero_overlap_validation.txt'),  # noqa: F821
               sep='\t', index=False)


# --- Combined score within the subset, and its top pairs --------------------
combined_z = np.mean([go_z_res[k] for k in GO_LAYERS], axis=0)       # noqa: F821
k_top = min(500, n_zero)
order_z = np.argsort(combined_z)[::-1][:k_top]

rng = np.random.default_rng(SEED)                                    # noqa: F821
med_top = np.median(ct_z[order_z])
null = np.array([np.median(rng.choice(ct_z, size=k_top, replace=False))
                 for _ in range(1000)])
p_top = (1 + np.sum(null >= med_top)) / 1001

print(f"\n  Top {k_top} convergent pairs within the zero-overlap subset:")
print(f"    median cell-type rank = {med_top:.3f}  (null {null.mean():.3f}, "
      f"p = {p_top:.4f})")

ph = np.array(PHENOS)                                                # noqa: F821
pd.DataFrame({
    'Phenotype_A': ph[rows_z[order_z]],
    'Phenotype_B': ph[cols_z[order_z]],
    'Residual_functional_score': combined_z[order_z],
    'CellType_similarity_rank': ct_z[order_z],
    'Genes_A': sizes[rows_z[order_z]],                               # noqa: F821
    'Genes_B': sizes[cols_z[order_z]],                               # noqa: F821
}).to_csv(os.path.join(OUTDIR, 'TableS_zero_overlap_top_pairs.txt'),  # noqa: F821
          sep='\t', index=False)


# --- Figure -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.6), dpi=300)
x = np.arange(len(df_zero))
w = 0.35
ax.bar(x - w / 2, df_zero.Spearman_raw_zero_overlap, w,
       color='lightgray', label='Raw')
ax.bar(x + w / 2, df_zero.Partial_zero_overlap, w,
       color='crimson', label='Residual (breadth removed)')
ax.axhline(0, color='k', lw=.8)
ax.set_xticks(x)
ax.set_xticklabels([l.replace('GO (', '').replace(')', '') for l in df_zero.GO_layer],
                   fontsize=8)
ax.set_ylabel('Spearman vs cell types', fontsize=9)
ax.set_title(f'Pairs sharing no genes (n = {n_zero:,})', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'FigS_zero_overlap_validation.png'),   # noqa: F821
            dpi=600, bbox_inches='tight')
plt.close()

print("""
>>> HOW TO READ IT
    Within this subset no pair shares a single gene, so circularity through
    shared gene membership is impossible by construction.
    
    Compare these values against the full-data ones from Block 5. A similar
    magnitude means nothing was riding on shared gene membership.
""")

# =============================================================================
# BLOCK 6. CONVERGENT PAIRS AND ANNOTATION DEPTH  
# =============================================================================
print("=" * 78)
print(" BLOCK 6: convergent pairs and robustness across annotation depth")
print("=" * 78)


def analyse_subset(min_genes, n_top=N_TOP_PAIRS, seed=SEED):
    """
    Residualisation is recomputed WITHIN each subset: both the rank transform
    and the regression coefficients depend on which phenotypes are included, so
    reusing the global residuals would be wrong.
    """
    keep = [p for p in PHENOS if breadth_all[p] > min_genes]
    m = len(keep)
    if m < 50:
        return None, None
    iu_s = np.triu_indices(m, k=1)
    sz = breadth_all.reindex(keep).values.astype(float)

    def fl(name):
        return aligned_matrices[name].loc[keep, keep].values[iu_s]

    g_raw = fl(GENE_LAYER)
    g, c = rank01(g_raw), rank01(fl(CT_LAYER))
    cmin = np.minimum(sz[iu_s[0]], sz[iu_s[1]])
    cmax = np.maximum(sz[iu_s[0]], sz[iu_s[1]])
    Xs = np.column_stack([np.ones(len(iu_s[0])), rank01(cmin), rank01(cmax),
                          rank01(cmin) * rank01(cmax), g])

    res = []
    for name in GO_LAYERS:
        v = rank01(fl(name))
        beta, *_ = np.linalg.lstsq(Xs, v, rcond=None)
        res.append(v - Xs @ beta)
    combined = np.mean(res, axis=0)

    k = min(n_top, len(combined))
    order = np.argsort(combined)[::-1][:k]
    med_ct = np.median(c[order])

    rng = np.random.default_rng(seed)
    null = np.array([np.median(rng.choice(c, size=k, replace=False))
                     for _ in range(N_PERM_SMALL)])

    summary = dict(Min_genes=min_genes, N_phenotypes=m, N_pairs=len(combined),
                   Median_genes=np.median(sz),
                   Top_median_CT_rank=med_ct, Null_median_CT_rank=null.mean(),
                   Pct_top_zero_genes=100 * np.mean(g_raw[order] == 0),
                   Pct_all_zero_genes=100 * np.mean(g_raw == 0),
                   p_empirical=(1 + np.sum(null >= med_ct)) / (N_PERM_SMALL + 1))

    ph = np.array(keep)
    pairs = pd.DataFrame({
        'Min_genes_filter': min_genes,
        'Phenotype_A': ph[iu_s[0][order]], 'Phenotype_B': ph[iu_s[1][order]],
        'Residual_functional_score': combined[order],
        'Gene_similarity_raw': g_raw[order],
        'CellType_similarity_rank': c[order],
        'Genes_A': sz[iu_s[0][order]], 'Genes_B': sz[iu_s[1][order]],
    })
    return summary, pairs


summaries, all_pairs = [], []
for mg in MIN_GENES_GRID:
    print(f"\n  Subset: more than {mg} genes")
    s, pairs = analyse_subset(mg)
    if s is None:
        print("    too few phenotypes, skipped")
        continue
    summaries.append(s)
    all_pairs.append(pairs)
    print(f"    {s['N_phenotypes']} phenotypes, {s['N_pairs']:,} pairs")
    print(f"    Top pairs: median cell-type rank {s['Top_median_CT_rank']:.3f} "
          f"(null {s['Null_median_CT_rank']:.3f}, p = {s['p_empirical']:.4f})")
    print(f"    Sharing zero genes: {s['Pct_top_zero_genes']:.1f}% of top pairs "
          f"vs {s['Pct_all_zero_genes']:.1f}% of all")

df_rob = pd.DataFrame(summaries)
df_rob.to_csv(os.path.join(OUTDIR, 'TableS_convergence_robustness.txt'), sep='\t', index=False)
pd.concat(all_pairs).to_csv(os.path.join(OUTDIR, 'TableSy_convergent_pairs.txt'),
                            sep='\t', index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=300)
axes[0].plot(df_rob.Min_genes, df_rob.Top_median_CT_rank, 'o-', color='crimson',
             label='Top convergent pairs')
axes[0].plot(df_rob.Min_genes, df_rob.Null_median_CT_rank, 'o--', color='gray',
             label='Random pairs')
axes[0].axhline(0.5, color='k', lw=.6, ls=':')
axes[0].set_ylim(0, 1)
axes[0].set_xlabel('Minimum genes per phenotype', fontsize=9)
axes[0].set_ylabel('Median cell-type similarity rank', fontsize=9)
axes[0].legend(fontsize=8)
axes[0].set_title('Convergence across annotation depth', fontsize=10, fontweight='bold')

axes[1].plot(df_rob.Min_genes, df_rob.Pct_top_zero_genes, 'o-', color='steelblue',
             label='Top pairs')
axes[1].plot(df_rob.Min_genes, df_rob.Pct_all_zero_genes, 'o--', color='gray',
             label='All pairs')
axes[1].set_xlabel('Minimum genes per phenotype', fontsize=9)
axes[1].set_ylabel('% of pairs sharing zero genes', fontsize=9)
axes[1].legend(fontsize=8)
axes[1].set_title('Gene sharing among convergent pairs', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'FigS12_convergence_robustness.png'),
            dpi=600, bbox_inches='tight')
plt.close()

print(f"""
  The enrichment in zero gene overlap is what defends the convergence claim
  against the objection that convergent pairs simply have small gene sets.
  Consider promoting that panel to the main figure.

All outputs written to:
  {OUTDIR}
""")
