import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
file1 = 'TyrosineProteinKinaseABL1_P00519_fingerprints_dim10_jaccard_UMAP_similarity_space_COEMBED.csv'
file2 = 'TyrosineProteinKinaseABL1_P00519_fingerprints_dim10_similarity_space.csv'

# Load the data
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Identify UMAP columns - assuming 2 dimensions based on typical UMAP usage
umap_cols = [f'UMAP-Jaccard-{i}' for i in range(1, 3)]

# Extract UMAP projections and identify CHEMBL molecules
is_chembl1 = df1['MOLECULE ID'].str.startswith('CHEMBL', na=False)
chembl_proj1 = df1.loc[is_chembl1, umap_cols].values
other_proj1 = df1.loc[~is_chembl1, umap_cols].values

is_chembl2 = df2['MOLECULE ID'].str.startswith('CHEMBL', na=False)
chembl_proj2 = df2.loc[is_chembl2, umap_cols].values
other_proj2 = df2.loc[~is_chembl2, umap_cols].values

# Create the plot for file 1 (Coembedding)
plt.figure(figsize=(48, 40))
plt.scatter(other_proj1[:, 0], other_proj1[:, 1], label='Coembedding (Other)', alpha=0.1, s=2, marker='o', color='blue')
plt.scatter(chembl_proj1[:, 0], chembl_proj1[:, 1], label='Coembedding (CHEMBL)', alpha=0.8, s=250, marker='+', color='black')
plt.title('UMAP Coembedding')
plt.xlabel(umap_cols[0])
plt.ylabel(umap_cols[1])
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename1 = 'umap_coembedding.png'
plt.savefig(output_filename1)
print(f"Plot saved to {output_filename1}")
plt.close()

# Create the plot for file 2 (Projections)
plt.figure(figsize=(48, 40))
plt.scatter(other_proj2[:, 0], other_proj2[:, 1], label='Projections (Other)', alpha=0.1, s=2, marker='o', color='orange')
plt.scatter(chembl_proj2[:, 0], chembl_proj2[:, 1], label='Projections (CHEMBL)', alpha=0.8, s=250, marker='+', color='black')
plt.title('UMAP Projections')
plt.xlabel(umap_cols[0])
plt.ylabel(umap_cols[1])
plt.legend()
plt.grid(True)
plt.tight_layout()
output_filename2 = 'umap_projections.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")
plt.close()

# plt.show()

