# UMMBAS_screening_experiments

### **Experimental Design: Evaluating Molecular Function-Guided Similarity Spaces for Ligand-Based Virtual Screening**

#### **Abstract**

A significant challenge in ligand-based drug discovery is the scarcity of known active compounds for novel or understudied protein targets. To address this, we developed and validated a computational framework that leverages the target's broader Molecular Function (MF) to enrich the pool of relevant chemical matter for virtual screening. This study employs a rigorous "leave-one-target-out" experimental design to assess the hypothesis that a similarity space, built from a general MF chemical landscape, can effectively prioritize true active ligands for a specific, held-out protein. We systematically compared two molecular representations (physicochemical features and ECFP4 fingerprints), three dimensionality reduction (DR) algorithms (PCA, UMAP, t-SNE), and two distinct embedding strategies (Projection vs. Co-embedding). Performance was evaluated over five replicate runs by ranking held-out actives against a large decoy set using rank-based metrics, including ROC-AUC, PR-AUC, and Enrichment Factor at 1% (EF@1%). 

#### **Methodology**

**1. Experimental Hypothesis and Design**

The central hypothesis of this study is that ligands active against a specific protein will exhibit measurable proximity to a "cloud" of compounds known to modulate other proteins that share the same molecular function. To test this, we implemented a leave-one-target-out experimental design. For a given target protein (e.g., ABL1 Kinase), its known ligands were sequestered as a "held-out active set". A reference chemical space was then constructed using two populations: (1) an "MF Cloud," comprising all ligands from the ChEMBL database known to be active against any *other* human protein with the same MF (e.g., "Protein kinase inhibitor"), explicitly excluding the held-out actives; and (2) a "Decoy Set," consisting of over 1.2 million presumed inactive compounds from the ZINC database, from which any molecules identical to the held-out actives were also removed.

The primary evaluation task was to determine if a scoring function based on proximity within this MF-guided space could effectively rank the held-out actives significantly higher than the decoys. This entire process was repeated in quintuplicate (N=5) using different random seeds to ensure statistical robustness.

**2. Molecular Representations**

Two orthogonal molecular representations were evaluated to capture different aspects of chemical similarity:

*   **Physicochemical Features:** A set of 42 pre-selected 1D, 2D, and 3D descriptors were calculated using the Mordred library and RDKit. These features, including properties like molecular weight, atom counts, topological indices, and dipole moment, represent molecules in a continuous property space.
*   **Structural Fingerprints:** Molecules were encoded as 2048-bit Extended-Connectivity Fingerprints with a radius of 2 (ECFP4). This high-dimensional binary representation captures discrete, localized structural motifs.

**3. Similarity Space Construction and Embedding Strategies**

Low-dimensional similarity spaces were generated using three dimensionality reduction (DR) algorithms: Principal Component Analysis (PCA), Uniform Manifold Approximation and Projection (UMAP), and t-Distributed Stochastic Neighbor Embedding (t-SNE). Two distinct strategies for incorporating the held-out actives were compared:

*   **Co-embedding:** The held-out actives were combined with the MF Cloud and Decoy Set *before* the DR algorithm was applied. This creates a unique similarity space where the actives themselves influence the final layout.
*   **Projection:** A baseline similarity space and its corresponding DR model were first generated using only the MF Cloud and Decoy Set. The held-out actives were then transformed into this pre-existing space using the saved, fitted model. This simulates a true prospective screening scenario where the model is built without any knowledge of the query compounds.

For features, a preprocessing step was performed, scaling the features by removing the mean and scaling to unit variance.

For fingerprints, which are high-dimensional and sparse, a preliminary PCA reduction to 50 dimensions was applied before running UMAP (with Euclidean distance) and t-SNE to ensure computational stability and reduce noise. For UMAP with binary-native metrics (e.g., Jaccard), the full 2048-bit unscaled data was used. No scaling was performed.


**4. Performance Evaluation**

The performance of each experimental configuration was assessed by scoring all held-out actives and decoys based on their negative minimum Euclidean distance to any point in the MF Cloud within the generated similarity space. The resulting ranked list was evaluated using three standard metrics:

*   **ROC-AUC:** Measures the global ability of the model to discriminate between actives and decoys across all ranking thresholds.
*   **PR-AUC:** Measures performance on imbalanced datasets, reflecting the trade-off between precision and recall, and is reported against the random baseline.
*   **Enrichment Factor at 1% (EF@1%):** Measures the concentration of actives in the top 1% of the ranked list compared to random selection, quantifying the model's utility for early enrichment.

Results were aggregated across the five replicate runs to report the mean and standard deviation for each metric, providing a statistically sound basis for comparing the different experimental arms.