# Broad mammalian phenotype classes exhibit distinct multiscale biological architectures
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
| `phen_matrix_construction.py` | Builds the gene–phenotype matrix (MP ontology) | Phenotype set used in all analyses | `MPheno_OBO.ontology.obo`, `MGI_PhenoGenoMP.rpt.txt`, `mart_export.txt` |
| `data_mouse.py` | Fixes the cell-type order (by developmental stage, then by name) and counts the cells of each cell type at each stage | — | Atlas cell metadata (`df_cell.csv`, `Cell types and subtypes.xlsx`) |
| `phen_CellType_Steps.py` | Lists which phenotype-associated genes are expressed in each cell of the atlas, and how many of them per cell | — | `adata_JAX_dataset_1-4.h5ad`, `df_gene.csv` |
| `computation_KS_faster.py` | KS test of each phenotype against each cell type, on the residuals of the number of phenotype genes per cell regressed on the total number of genes per cell | Phenotype x Embryo cell-type Matrix | — |
| `correlation_phenotypes_across_genes.py` | Genetic layer: pairwise Jaccard similarity between phenotypes | Genetic layer | — |
| `GO_phenotype_similarity.py` | Three Gene Ontology layers: builds phenotype × GO matrices weighted by information content, and compute their cosine similarity | GO layers; Note **S4** | `go.obo`, `mgi.gaf`, `mart_export.txt` |
| `validation_celltype_ontology.py` | Checks the significant phenotype × cell-type associations against the curated MP–CL pairs | Note **S1** | `mp_cl_associations.tsv`, `curated_mapping_CL.xlsx`, `cl.obo` |
| `section1_layer_comparison.py` | Annotation-breadth effects, partial Spearman correlations between layers, functional similarity beyond shared genes | Notes **S2, S3, S5**; Table **S1** | — |
| `section2_modularity.py` | Modularity of each broad MP category, as the probability of superiority (PS) on the breadth-residualised matrices | Figure **2**; Table **S2** | `MPheno_OBO.ontology.obo` |
| `disjoint_integration.py` | Integration excluding, for each pair of MP categories, the phenotypes that belong to both | Figure **3**, Table **S3** | — |
| `genetic_layer_correlations.py` | Compares the PS values of each layer with the genetic one after correcting the genetic PS for category size, and computes the residuals | Note **S7**; Table **S4** | — |

The cell-type layer has no script of its own: it is rebuilt from the KS statistics, with the same few lines, by `section1_layer_comparison.py`, `section2_modularity.py` and `disjoint_integration.py`, so the layer has a single definition.

#### Reproducibility

The PS values, and therefore every figure, are deterministic. The permutation p-values are estimated from a finite number of draws and vary in the third or fourth decimal between runs.

---

## Input data

 File | Description | Source |
|---|---|---|
| `MPheno_OBO.ontology.obo` | Mammalian Phenotype Ontology (release 2024-08-08) | Mouse Genome Informatics (MGI), https://www.informatics.jax.org/downloads/reports/index.html#pheno |
| `MGI_PhenoGenoMP.rpt.txt` | Genotype–phenotype associations | Mouse Genome Informatics (MGI), https://www.informatics.jax.org/downloads/reports/index.html#pheno |
| `mart_export.txt` | MGI to Ensembl gene identifier map | Ensembl BioMart, https://www.ensembl.org/biomart |
| `go.obo` | Gene Ontology (release 2025-02-06) | Gene Ontology Consortium, https://geneontology.org |
| `mgi.gaf` | Mouse GO annotations (GAF 2.2, generated 2026-03-28) | Gene Ontology Consortium, http://current.geneontology.org/annotations/mgi.gaf.gz |
| `cl.obo` | Cell Ontology | OBO Foundry, http://purl.obolibrary.org/obo/cl.obo |
| `mp_cl_associations.tsv` | Phenotype (MP term) and cell type (CL identifier) pairs | Extracted from the Cell Ontology files, where cell-type terms carry their associated Mammalian Phenotype terms |
| `curated_mapping_CL.xlsx` | Automatic and manual curation mapping each atlas cell type name to a Cell Ontology cell type | Curated for this work |
| Embryonic single-cell atlas: `adata_JAX_dataset_1-4.h5ad` | Cells × genes, with cell type and developmental stage (Qiu et al., 2024) | https://omg.gs.washington.edu/jax/public/about.html |


---

## References
 
Qiu, C., et al. (2024). A single-cell time-lapse of mouse prenatal development from gastrula to birth. Nature, 626(8001), 1084–1093.
https://doi.org/10.1038/s41586-024-07069-w
 
Baldarelli, R. M., et al. (2024). Mouse Genome Informatics: an integrated knowledgebase system for the laboratory mouse. Genetics, 227(1), iyae031.
https://doi.org/10.1093/genetics/iyae031
 
Smith, C. L., & Eppig, J. T. (2009). The Mammalian Phenotype Ontology: enabling robust annotation and comparative analysis. Wiley Interdisciplinary Reviews: Systems Biology and Medicine, 1(3), 390–399.
https://doi.org/10.1002/wsbm.44
 
Ashburner, M., et al. (2000). Gene Ontology: tool for the unification of biology. Nature Genetics, 25(1), 25–29.
https://doi.org/10.1038/75556
 
The Gene Ontology Consortium. (2025). The Gene Ontology knowledgebase in 2026. Nucleic Acids Research, gkaf1292.
https://doi.org/10.1093/nar/gkaf1292
 
Diehl, A. D., et al. (2016). The Cell Ontology 2016: enhanced content, modularization, and ontology interoperability. Journal of Biomedical Semantics, 7, 44.
https://doi.org/10.1186/s13326-016-0088-7

