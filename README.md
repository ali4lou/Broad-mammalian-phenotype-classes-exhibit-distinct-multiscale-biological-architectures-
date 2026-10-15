# Broad-mammalian-phenotype-classes-exhibit-distinct-multiscale-biological-architectures-
Source code, datasets, and instructions for reproducing the results of our paper 'Broad mammalian phenotype classes exhibit distinct multiscale biological architectures'. 

The code builds five independent representations of 2,399 Mammalian Phenotype (MP) terms (shared genes, three Gene Ontology spaces, and developmental cell types), compares them while controlling for annotation breadth, and tests the modularity and integration of the broad MP categories.

## Reproducing the Results

This repository is organized into two main folders:

- **`result_codes/`** – Python scripts used to reproduce each analysis presented in the paper.
- **`files/`** – Required datasets and annotation files (**provided as compressed files**).

Edit the paths in the `CONFIGURATION` block at the top of each script before running. The scripts must be run in the order of the table below, since each one reads the files written by the previous ones.

---

## Overview of Scripts and Outputs

| Script | Purpose | Output (Figures/Tables) | Required Files (from `files/`) |
|---|---|---|---|
| `phen_matrix_construction.py` | Builds the gene–phenotype matrix, propagating every annotation to all ancestor terms of the MP ontology, and selects the phenotypes annotated to 20–500 genes that are not ancestors of another selected one | Phenotype set used in all analyses | `MPheno_OBO.ontology.obo`, `MGI_PhenoGenoMP.rpt.txt`, `mart_export.txt` |
| `datos_raton.py` | Fixes the cell-type order (by developmental stage, then by name) and counts the cells of each cell type at each stage | — | Atlas cell metadata (`df_cell.csv`, `Cell types and subtypes.xlsx`) |
| `phen_CellType_Steps.py` | Lists which phenotype-associated genes are expressed in each cell of the atlas | — | `adata_JAX_dataset_1-4.h5ad`, `df_gene.csv` |
| `computation_KS_faster.py` | KS test of each phenotype against each cell type, on the residuals of the number of phenotype genes per cell regressed on the total number of genes per cell | Cell-type layer | — |
| `correlation_phenotypes_across_genes.py` | Genetic layer: pairwise Jaccard similarity between phenotypes | Genetic layer | — |
| `GO_phenotype_similarity.py` | Three Gene Ontology layers: phenotype × GO matrices weighted by information content, and their cosine similarity. Validated with a gene-shuffling null and a drop-out of the most pleiotropic genes | GO layers; Note **S7** | `go.obo`, `mgi.gaf`, `mart_export.txt` |
| `validation_celltype_ontology.py` | Checks the significant phenotype × cell-type associations against the curated MP–CL pairs, following the `is_a` lineage of each cell type, and compares the KS statistics of validated and non-validated associations | Figure **[FILL]** | `mp_cl_associations.tsv`, `mapeo_manual_IA.xlsx`, `cl.obo` |
| `section1_layer_comparison.py` | Section 1: annotation-breadth effects, partial Spearman correlations between layers, functional similarity beyond shared genes, validation on pairs sharing no genes, and the annotation-depth sweep | Figures **1a–d**, **S7–S9**, **S12**; Table **S1** | — |
| `make_figure1.py` | Assembles Figure 1 | Figure **1** | — |
| `section2_modularity.py` | Section 2: modularity and integration of each broad MP category, as the probability of superiority (PS) on the breadth-residualised matrices | Figures **2**, **3**; Tables of modularity and integration | `MPheno_OBO.ontology.obo` |
| `disjoint_integration.py` | Integration recomputed excluding, for each pair of categories, the phenotypes that belong to both | Figure **3** (disjoint version) | — |
| `genetic_layer_correlations.py` | Compares each layer with the genetic one after correcting the genetic PS for category size, and computes the residuals | Table of genetic-layer residuals | — |

A dash means the script only passes its output to the next ones. `genetic_layer_correlations.py` has a `DISJOINT` switch that selects which integration table it reads; the residuals reported in the paper use the disjoint one.

The cell-type layer has no script of its own: it is rebuilt from the KS statistics, with the same few lines, by `section1_layer_comparison.py`, `section2_modularity.py` and `disjoint_integration.py`, so the layer has a single definition.

---

## Input data

| File | Description | Source |
|---|---|---|
| `MPheno_OBO.ontology.obo` | Mammalian Phenotype Ontology (release 2024-08-08) | Mouse Genome Informatics (MGI), https://www.informatics.jax.org/downloads/reports/index.html#pheno |
| `MGI_PhenoGenoMP.rpt.txt` | Genotype–phenotype associations | Mouse Genome Informatics (MGI), https://www.informatics.jax.org/downloads/reports/index.html#pheno |
| `mart_export.txt` | MGI to Ensembl gene identifier map | Ensembl BioMart, https://www.ensembl.org/biomart |
| `go.obo`, `mgi.gaf` | Gene Ontology and mouse gene annotations | Gene Ontology Consortium, https://geneontology.org [FILL: release date] |
| `cl.obo` | Cell Ontology; downloaded automatically on the first run | OBO Foundry, http://purl.obolibrary.org/obo/cl.obo |
| `mp_cl_associations.tsv` | Phenotype (MP term) and cell type (CL identifier) pairs, the gold standard of the validation | Extracted from the Cell Ontology files, where cell-type terms carry their associated Mammalian Phenotype terms |
| `mapeo_manual_IA.xlsx` | Manual curation mapping each atlas cluster name to a Cell Ontology cell type | Curated for this work |
| Embryonic single-cell atlas | Cells × genes, with cell type and developmental stage | [FILL: atlas citation and URL] |

---

## Statistical conventions

- **Partial Spearman correlation.** Similarities are rank-transformed, annotation breadth (rank of the smaller and larger gene set of the pair, and their interaction) is regressed out of each layer, and the Pearson correlation of the residuals is reported.
- **Significance.** Phenotype pairs are not independent, so all p-values come from permutations of phenotype or category labels (Mantel-style), never from parametric tests. With 999 permutations the smallest attainable p-value is 0.001.
- **Modularity and integration.** Quantified by the probability of superiority (PS = AUC = Mann–Whitney U / n₁n₂) on the breadth-residualised similarity matrices. 0.5 means no excess.
- **Multiple testing.** Benjamini–Hochberg FDR within each layer.
- **Reproducibility.** The PS values, and therefore every figure, are deterministic. The permutation p-values are estimated from a finite number of draws and vary in the third or fourth decimal between runs.

---

## References

[FILL: references of the datasets above, in the format of the template]
