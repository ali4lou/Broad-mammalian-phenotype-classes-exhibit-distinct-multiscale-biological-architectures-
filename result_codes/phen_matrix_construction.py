
import obonet
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib as mpl
from scipy.cluster.hierarchy import dendrogram
from matplotlib import cm 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from scipy.stats import pearsonr


def convert_dictionary_to_array(dictionary):
    # pairs in the dictionary
    result = dictionary.items()
    # Convert object to a list
    data = list(result)
    # Convert list to an array
    array = np.array(data)
    return array


        
def find_path_mouse(graph, inicio, name_to_id): 
    paths = nx.all_simple_paths(
        graph,
        source=name_to_id[inicio],
        target=name_to_id['mammalian phenotype']
    )
    innerlist = []
    for path in paths:
        innerlist.append(path)
        
    return innerlist

    


url='PATH/MPheno_OBO.ontology.obo'


graph = obonet.read_obo(url)

#create mappings
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
phen_description = {data['def']: id_ for id_, data in graph.nodes(data=True) if 'def' in data}

del url



phenotypes=convert_dictionary_to_array(id_to_name)
phen_description=convert_dictionary_to_array(phen_description)


pathlist=[]
for i in range(len(phenotypes)):
    inicio=id_to_name[phenotypes[i][0]]
    camino=find_path_mouse(graph, inicio, name_to_id)
    pathlist.append(camino)
del camino, inicio
    
    


#Read the associations genes-phenotypes
path='PATH'

with open(path+'MGI_PhenoGenoMP.rpt.txt', 'r') as file:
    data = file.read()

# Dividir el texto por el término "term" para crear nuevas filas
rows = data.split('\n')

# Eliminar cualquier espacio en blanco inicial/final de cada fila
rows = [row.strip() for row in rows if row.strip()]

del data, file

# Array 
phen_association = []
gene_association=[]

for elemento in rows:
    partes = elemento.split("\t")
    relation=[]

    for i in range(len(partes)):
    
        if partes[i].startswith("MP"):
            phen_association.append(partes[i])
        
        if partes[i].startswith("MGI"):
            gene_association.append(partes[i])
    
del partes, elemento


elementos_con_barra = [elemento for elemento in gene_association if "|" in elemento]

# Filtrar los elementos que contienen "|"
new_gene_association=[]
new_phen_association=[]
i=0
for elemento in gene_association:
    if "|" in elemento:
        partes = elemento.split("|")
        new_gene_association.extend(partes)
        for j in range(len(partes)):
            new_phen_association.append(phen_association[i])

        i=i+1
    else:
        # Si no contiene '|', simplemente agregar el elemento al nuevo array
        new_gene_association.append(elemento)
        new_phen_association.append(phen_association[i])
        i=i+1

del gene_association, phen_association
new_gene_association=np.array(new_gene_association)
new_phen_association=np.array(new_phen_association)

genes_id_unicos=np.unique(new_gene_association)
phenotypes_unicos=np.unique(new_phen_association)

# =============================================================================
# BUILD THE GENE x PHENOTYPE MATRIX
# =============================================================================
# Each association is propagated to every ancestor term of the MP ontology, so a
# gene annotated to a specific phenotype also counts for all the broader
# phenotypes above it. This is the slow step; it is run once and the result is
# read back below. Set BUILD_MATRIX = False to reuse an existing matrix.
BUILD_MATRIX = True

if BUILD_MATRIX:
    matriz = np.zeros((len(genes_id_unicos), len(phenotypes[:, 0])))
    for i in range(len(genes_id_unicos)):
        ind_gen = np.where(new_gene_association == genes_id_unicos[i])[0]
        for j in range(len(ind_gen)):
            busca_ind = np.where(phenotypes[:, 0] == new_phen_association[int(ind_gen[j])])
            busca_ind = int(busca_ind[0])
            # every phenotype of the hierarchy above this one
            for n in range(len(pathlist[busca_ind])):
                for dentro in range(len(pathlist[busca_ind][n])):
                    ind_matrix = np.where(phenotypes[:, 0] == pathlist[busca_ind][n][dentro])
                    ind_matrix = int(ind_matrix[0])
                    matriz[i][ind_matrix] = 1

    np.savetxt(path + "matriz_gene_phen.txt", matriz, delimiter="\t")

    phen_guardar = np.array(phenotypes[:, 1])
    with open(path + 'phen.txt', 'w') as file:
        for item in phen_guardar:
            file.write(f"{item}\n")

    with open(path + 'genes_unique_matrix.txt', 'w') as file:
        for item in genes_id_unicos:
            file.write(f"{item}\n")
    




path='PATH'
matriz=np.loadtxt(path+"matriz_gene_phen.txt", delimiter="\t")

with open(path+"phen.txt", 'r') as file:
    phen_guardar = file.readlines()
phen_guardar=np.array(phen_guardar, dtype=str)

# with open(path+"genes_unique_matrix.txt", 'r') as file:
#     genes_unique_matrix = file.readlines()
# genes_unique_matrix=np.array(genes_unique_matrix, dtype=str)


#ahora relaciono los genes del MGI con ENSMUS gracias a biomart ensemble


with open(path+'mart_export.txt', 'r') as file:
    data = file.read()

# Dividir el texto por el término "term" para crear nuevas filas
rows = data.split('\n')

# Eliminar cualquier espacio en blanco inicial/final de cada fila
rows = [row.strip() for row in rows if row.strip()]


rows=np.delete(rows, 0)

del data, file

# Array donde almacenaremos los elementos sin el prefijo "id"
gene_ensmus = []
gene_mgi=[]


for elemento in rows:
    existe=0

    # Dividimos el elemento por tabulador
    partes = elemento.split(",")

    relation=[]

    for i in range(len(partes)):
        if partes[i].startswith("MGI"):
            gene_mgi.append(partes[i])
            existe=1
            
    if existe==1:
        for i in range(len(partes)):
            if partes[i].startswith("ENS"):
                gene_ensmus.append(partes[i])

gene_ensmus=np.array(gene_ensmus, dtype=str)
gene_mgi=np.array(gene_mgi, dtype=str)
#genes de la matriz:
genes_id_unicos_ens=[]
final_matrix=[]
for i in range(len(genes_id_unicos)):
    index_gene=np.where(gene_mgi==genes_id_unicos[i])[0]
    if len(index_gene)==1:
        genes_id_unicos_ens.append(gene_ensmus[int(index_gene)])
        final_matrix.append(matriz[i])

genes_id_unicos_ens=np.array(genes_id_unicos_ens)
with open(path+'genes.txt', 'w') as file:
    for item in genes_id_unicos_ens:
        file.write(f"{item}\n")

genes_id_unicos_ens=np.array(genes_id_unicos_ens)
final_matrix=np.array(final_matrix)

print('phen with null genes:', len(np.where(np.sum(final_matrix, axis=0)==0)[0]))
ind_null_phen=np.where(np.sum(final_matrix, axis=0)==0)[0]

clean_matrix=np.delete(final_matrix, ind_null_phen, axis=1)
clean_phen=np.delete(phen_guardar, ind_null_phen)

np.savetxt(path+'clean_genes_matrix.txt', genes_id_unicos_ens, fmt='%s') 
np.savetxt(path+'clean_phen_matrix.txt', clean_phen, fmt='%s') 
np.savetxt(path+'clean_matrix.txt', clean_matrix, fmt='%d') 

len(np.sum(clean_matrix, axis=0))
print(np.median(np.sum(clean_matrix, axis=0)))
print(np.mean(np.sum(clean_matrix, axis=0)))
print(np.percentile(np.sum(clean_matrix, axis=0), 10))
print(np.percentile(np.sum(clean_matrix, axis=0), 90))
print(np.std(np.sum(clean_matrix, axis=0)))

plt.figure(figsize=(4, 3), dpi=600)
plt.hist(np.sum(clean_matrix, axis=0), color='darkorange', log=True, bins=100)
plt.ylabel('# phenotypes', fontsize=18)
plt.xlabel('# genes associated', fontsize=18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=13)
plt.grid(False)
plt.tight_layout()
plt.savefig(path+'n_genes_per_phen.png', dpi=600, bbox_inches='tight')
plt.show()


#Specific phen with between 20 and 500 genes

papas=[]
for i in range(len(pathlist)):
    papas_inner=[]
    for j in range(len(pathlist[i])):
        for k in range(1, len(pathlist[i][j])):
            papas_inner.append(pathlist[i][j][k])
    papas.append(np.unique(papas_inner))

papas_descrip=[]
for i in range(len(papas)):
    papas_inner=[]
    for j in range(len(papas[i])):
        ind=np.where(phenotypes[:, 0]==papas[i][j])[0]
        papas_inner.append(phenotypes[int(ind)][1])
    papas_descrip.append(papas_inner)
        
del papas_inner, papas




n_genes_por_phen=[]
for i in range(len(phen_guardar)):
    n_genes_por_phen.append(np.sum(matriz[:, i]))
    

descrip_all_phenotypes_filtro=[]
indices_filtro=[]
path_filtro=[]
max_num_genes=500
for i in range(len(phen_guardar)):
    if (n_genes_por_phen[i]>20) & (n_genes_por_phen[i]<max_num_genes):
        descrip_all_phenotypes_filtro.append(phen_guardar[i])
        indices_filtro.append(i)
        for j in range(len(papas_descrip[i])):
            path_filtro.append(papas_descrip[i][j])
descrip_all_phenotypes_filtro=np.array(descrip_all_phenotypes_filtro)
indices_filtro=np.array(indices_filtro)
path_filtro=np.unique(path_filtro)



descrip_all_phenotypes_filtro_check=np.array([s.strip() for s in descrip_all_phenotypes_filtro])
del descrip_all_phenotypes_filtro

descrip_all_phenotypes_filtro_no_redundantes=[]
for i in range(len(descrip_all_phenotypes_filtro_check)):
    ind_phen_redundant=np.where(path_filtro==descrip_all_phenotypes_filtro_check[i])[0]
    if len(ind_phen_redundant)==0:
        descrip_all_phenotypes_filtro_no_redundantes.append(descrip_all_phenotypes_filtro_check[i])
        
descrip_all_phenotypes_filtro_no_redundantes=np.array(descrip_all_phenotypes_filtro_no_redundantes)


np.savetxt(path+'phen_filtro_20_500_genes_no_redundantes_final.txt', descrip_all_phenotypes_filtro_no_redundantes, fmt='%s') 















