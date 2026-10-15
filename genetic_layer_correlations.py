#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENETIC LAYER vs OTHER LAYERS: WHICH CATEGORIES DEPART FROM WHAT THE GENETIC
LAYER PREDICTS?

Question: does a category (modularity) or category pair (integration) that
scores high in the genetic layer also score high in the other four layers, and
which ones depart from that relationship?

Only the genetic layer is corrected for size. It was the only layer whose PS
depended on the number of phenotypes, whereas the other four layers showed no
relationship (|rho| < 0.24, all p > 0.25). 

  STEP 1  Size-correct the genetic AUC (PS): residual of AUC_Genes regressed on the
          rank-transformed size covariate(s).
            modularity   rank(N), N = number of phenotypes in the category
            integration  rank(min(N_A, N_B)), rank(max(N_A, N_B)) and their
                         interaction, the min/max design used for annotation
                         breadth

  STEP 2  Correlate each other layer's PS, uncorrected, with the size-corrected
          genetic PS (Spearman).
            modularity   one point per category; categories are independent
                         observations, so the ordinary p-value is valid.
            integration  one point per category PAIR. Pairs are NOT independent
                         (each category appears in many of them), so
                         significance comes from a Mantel-style permutation. 
                         Same logic as the partial Mantel tests
                         used elsewhere in this paper.

  STEP 3  Residual of each other layer's PS regressed linearly on the
          size-corrected genetic PS.
            positive  more modular / integrated in that layer than the
                      genetic layer predicts
            negative  less

The fit in step 3 is linear on the PS scale, left unranked so that residuals
stay interpretable as differences in PS; the correlation in step 2 is Spearman
and therefore robust to the shape of the relationship.

OUTPUTS
  TableS_genetic_layer_correlations.txt   correlation summary
  TableS_genetic_layer_residuals.xlsx     residuals per category / category
                                          pair
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# =============================================================================
# CONFIGURATION
# =============================================================================
path2 = 'PATH'
OUTDIR = os.path.join('PATH')

SEED = 42
N_PERM = 999   # Mantel-style permutations for integration; floor p = 1.0e-3

DISJOINT = True   # True: integración con AUC disjoint; False: la versión joint de siempre
SUFFIX = '_disjoint' if DISJOINT else ''


GENE_LAYER = 'Genes'
OTHER_LAYERS = ['GO (Cellular Process)', 'GO (Cellular Component)',
                'GO (Molecular Function)', 'Cell types']
LAYER_ORDER = [GENE_LAYER] + OTHER_LAYERS
TAG = {'GO (Cellular Process)': 'GO_CP', 'GO (Cellular Component)': 'GO_CC',
       'GO (Molecular Function)': 'GO_MF', 'Cell types': 'Cell_types'}


def rank01(v):
    """Rank transform to the unit interval, as used throughout this project."""
    return (rankdata(v) - 0.5) / len(v)


def ols_residual(y, X):
    """Residual of y after ordinary least squares on the design matrix X."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def square_from(df, value_col, categories):
    M = pd.DataFrame(np.nan, index=categories, columns=categories)
    for _, r in df.iterrows():
        M.loc[r.Category_A, r.Category_B] = M.loc[r.Category_B, r.Category_A] = r[value_col]
    return M


def mantel_spearman(M1, M2, n_perm, seed):
    """
    Spearman correlation between the upper triangles of two square matrices,
    with significance from permuting the row/column order of M2 jointly.
    """
    cats = M1.index.tolist()
    iu = np.triu_indices(len(cats), k=1)
    v1 = M1.values[iu]
    obs, _ = spearmanr(v1, M2.values[iu])

    rng = np.random.default_rng(seed)
    idx = np.arange(len(cats))
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(idx)
        null[i], _ = spearmanr(v1, M2.values[np.ix_(perm, perm)][iu])
    p = (1 + np.sum(np.abs(null) >= np.abs(obs))) / (n_perm + 1)
    return obs, p


# =============================================================================
# PART A. MODULARITY
# =============================================================================
print("=" * 78)
print(" PART A: modularity")
print("=" * 78)

mod = pd.read_csv(os.path.join(OUTDIR, 'TableS_modularity.txt'), sep='\t')
N_mod = mod.drop_duplicates('Category').set_index('Category')['N']
auc_mod = mod.pivot(index='Category', columns='Layer', values='AUC')[LAYER_ORDER]
CATEGORIES_MOD = auc_mod.index.tolist()
N_vec = N_mod.reindex(CATEGORIES_MOD).values.astype(float)

X_size = np.column_stack([np.ones(len(N_vec)), rank01(N_vec)])

print("\n  Dependence of each layer's AUC on category size (diagnostic):")
for layer in LAYER_ORDER:
    rho, p = spearmanr(N_vec, auc_mod[layer].values)
    print(f"    {layer:<26} Spearman rho with N = {rho:+.3f}   p = {p:.3f}")

# STEP 1: size-correct the genetic layer only
g_corr = ols_residual(auc_mod[GENE_LAYER].values, X_size)

mod_out = pd.DataFrame(index=CATEGORIES_MOD)
mod_out.index.name = 'Category'
mod_out['N'] = N_vec.astype(int)
mod_out['PS_Genes'] = auc_mod[GENE_LAYER].values
mod_out['PS_Genes_residuals_size_corrected'] = g_corr

X_g = np.column_stack([np.ones(len(g_corr)), g_corr])
mod_corr_rows = []
for layer in OTHER_LAYERS:
    v = auc_mod[layer].values
    # STEP 2: correlation with the size-corrected genetic AUC
    rho, p = spearmanr(g_corr, v)
    mod_corr_rows.append(dict(Analysis='Modularity', Layer_B=layer,
                              N=len(CATEGORIES_MOD), Spearman_rho=rho,
                              p_value=p, Method='spearmanr'))
    # STEP 3: residual of this layer given the genetic layer
    mod_out[f'PS_{TAG[layer]}'] = v
    mod_out[f'Residual_{TAG[layer]}'] = ols_residual(v, X_g)

print("\n  Other layers vs size-corrected genetic AUC (modularity):")
print(pd.DataFrame(mod_corr_rows).to_string(index=False,
                                            float_format=lambda x: f'{x:.4f}'))


# =============================================================================
# PART B. INTEGRATION
# =============================================================================
print("\n" + "=" * 78)
print(" PART B: integration")
print("=" * 78)

if DISJOINT:
    integ = pd.read_csv(os.path.join(OUTDIR, 'TableS_integration_disjoint.txt'), sep='\t')
    integ = (integ.rename(columns={'AUC_disjoint': 'AUC',
                                   'N_A_only': 'N_A', 'N_B_only': 'N_B'})
                  .dropna(subset=['AUC']))
    # keep only pairs with a finite AUC in ALL layers
    n_layers = integ.groupby(['Category_A', 'Category_B']).Layer.transform('nunique')
    integ = integ[n_layers == len(LAYER_ORDER)]
else:
    integ = pd.read_csv(os.path.join(OUTDIR, 'TableS_integration.txt'), sep='\t')

CATEGORIES_INT = sorted(set(integ.Category_A) | set(integ.Category_B))


def pair_design(df):
    m = np.minimum(df.N_A.values, df.N_B.values).astype(float)
    M = np.maximum(df.N_A.values, df.N_B.values).astype(float)
    rm, rM = rank01(m), rank01(M)
    return np.column_stack([np.ones(len(df)), rm, rM, rm * rM])


print("\n  Dependence of each layer's AUC on pair size (diagnostic), "
      "R2 of AUC ~ rank(min), rank(max), interaction:")
for layer in LAYER_ORDER:
    sub = integ[integ.Layer == layer]
    res = ols_residual(sub.AUC.values, pair_design(sub))
    print(f"    {layer:<26} R2 = {1 - res.var() / sub.AUC.values.var():.4f}")

# STEP 1: size-correct the genetic layer only
gene = (integ[integ.Layer == GENE_LAYER]
        .sort_values(['Category_A', 'Category_B']).reset_index(drop=True))
gene['PS_Genes_residuals_size_corrected'] = ols_residual(gene.AUC.values, pair_design(gene))
gene = gene.rename(columns={'AUC': 'PS_Genes'})[
    ['Category_A', 'Category_B', 'N_A', 'N_B', 'PS_Genes', 'PS_Genes_residuals_size_corrected']]

M_gene = square_from(gene, 'PS_Genes_residuals_size_corrected', CATEGORIES_INT)
n_pairs = len(CATEGORIES_INT) * (len(CATEGORIES_INT) - 1) // 2

int_corr_rows, int_out_rows = [], []
for layer in OTHER_LAYERS:
    other = (integ[integ.Layer == layer][['Category_A', 'Category_B', 'AUC']]
             .rename(columns={'AUC': 'PS_Layer'}))
    d = gene.merge(other, on=['Category_A', 'Category_B'], how='inner',
                   validate='one_to_one')
    if len(d) != len(gene):
        raise ValueError(f"{layer}: {len(gene) - len(d)} category pairs are "
                         f"missing from this layer")

    # STEP 2: Mantel-style permutation test
    rho, p = mantel_spearman(M_gene, square_from(d, 'PS_Layer', CATEGORIES_INT),
                             N_PERM, SEED)
    int_corr_rows.append(dict(Analysis='Integration', Layer_B=layer,
                              N=n_pairs, Spearman_rho=rho, p_value=p,
                              Method=f'Mantel permutation (n={N_PERM})'))

    # STEP 3: residual of this layer given the genetic layer
    Xg = np.column_stack([np.ones(len(d)), d.PS_Genes_residuals_size_corrected.values])
    d['Residual'] = ols_residual(d.PS_Layer.values, Xg)
    d.insert(0, 'Layer', layer)
    int_out_rows.append(d)

int_out = pd.concat(int_out_rows, ignore_index=True)

print("\n  Other layers vs size-corrected genetic AUC (integration):")
print(pd.DataFrame(int_corr_rows).to_string(index=False,
                                            float_format=lambda x: f'{x:.4f}'))


# =============================================================================
# SAVE
# =============================================================================
summary = pd.DataFrame(mod_corr_rows + int_corr_rows)
summary_path = os.path.join(OUTDIR, f'TableS_genetic_layer_correlations{SUFFIX}.txt')
summary.to_csv(summary_path, sep='\t', index=False)
print(f"\nSaved correlation summary to {summary_path}")

readme = [
    "Table S_genetic_layer_residuals. For each category (Modularity sheet) or category pair "
    "(Integration sheet), how much more or less modular or integrated it is in each non-genetic "
    "layer than its genetic-layer PS predicts, once the genetic PS is corrected for category size.",
    "",
    ("Integration is the disjoint version: for each pair of categories, the AUC compares the "
     "phenotypes exclusive to A against those exclusive to B, so phenotypes belonging to both "
     "categories are excluded from both sides (TableS_integration_disjoint.txt)."
     if DISJOINT else
     "Integration is the joint version: every phenotype of each category takes part, and only "
     "the pairs of two shared phenotypes are dropped (TableS_integration.txt)."),
    "PS is the probability of superiority, the effect size used for modularity and integration (equivalent to the AUC): the probability that a pair of interest has a higher residual similarity than a pair drawn from the rest of the matrix, ties counted as one half. 0.5 means no excess.",
    "",
    "Only the genetic layer is corrected for size: it was the only layer whose PS depended on the "
    "number of phenotypes (genetic modularity against N, Spearman rho = -0.42; the other layers "
    "showed no relationship).",
    "",
    "Columns",
    "N (Modularity): number of phenotypes in the category.",
    ("N_A, N_B (Integration): number of phenotypes that belong to category A (or B) but not to "
     "the other category of the pair; phenotypes shared by both categories are excluded from the "
     "comparison and from these counts."
     if DISJOINT else
     "N_A, N_B (Integration): number of phenotypes in each category of the pair."),
    "PS_Genes: PS in the genetic layer.",
    "PS_Genes_residuals_size_corrected: residual of PS_Genes regressed on rank(N) (Modularity), or on "
    "rank(min(N_A,N_B)), rank(max(N_A,N_B)) and their interaction (Integration).",
    "PS_GO_CP, PS_GO_CC, PS_GO_MF, PS_Cell_types (Modularity); PS_Layer (Integration): PS "
    "in the Gene Ontology Cellular Process, Cellular Component and Molecular Function spaces "
    "and in the cell-type layer. Not size-corrected.",
    "Residual_GO_CP, Residual_GO_CC, Residual_GO_MF, Residual_Cell_types (Modularity); Residual "
    "(Integration): the PS of that layer minus the value predicted from PS_Genes_residuals_size_corrected "
    "by linear regression. Positive: more modular or integrated in that layer than the genetic "
    "layer predicts. Negative: less.",
    "Layer (Integration): the non-genetic layer to which PS_Layer and Residual refer.",
    "",
    "The correlations between each layer and the size-corrected genetic PS are in "
    f"TableS_genetic_layer_correlations{SUFFIX}.txt: ordinary Spearman correlation for modularity "
    "(categories are independent points) and a Mantel permutation of category labels for "
    "integration (category pairs are not independent, since each category appears in several).",
]

xlsx_path = os.path.join(OUTDIR, f'TableS_genetic_layer_residuals{SUFFIX}.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as xl:
    pd.DataFrame({'': readme}).to_excel(xl, sheet_name='README', index=False, header=False)
    mod_out.reset_index().to_excel(xl, sheet_name='Modularity', index=False)
    int_out.to_excel(xl, sheet_name='Integration', index=False)

print(f"Saved residuals to {xlsx_path}")
