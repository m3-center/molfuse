

# **Expert Report: Virtual Screening Methods and Contextualization of EF Benchmarks for Latent Space Similarity Search**

## **1\. Introduction: The Imperative of Early Enrichment in Modern Drug Discovery**

The landscape of drug discovery is increasingly reliant on computational methods to maximize the efficiency of compound prioritization, a process known as Virtual Screening (VS).1 Modern VS strategies have fundamentally shifted, moving beyond purely physics-based simulations towards sophisticated machine learning (ML) and deep learning (DL) models that seek to learn predictive representations of chemical space.

### **1.1. Virtual Screening Paradigms: A Spectrum of Approaches**

VS methodologies are broadly categorized into structure-based (SBVS) and ligand-based (LBVS) methods. SBVS, relying on a known protein target structure, utilizes molecular docking to predict ligand binding poses and calculate affinity scores.2 LBVS, conversely, relies on the similarity principle: compounds chemically similar to a known active ligand are expected to also possess activity.3

The methodology employed by the user—ligand-based similarity search using dimensionality reduction (PCA/UMAP) on molecular descriptors—is a classical, non-learned precursor to the state-of-the-art vector-based LBVS. In both the classical approach and modern deep learning models, the primary objective is to define interaction likelihood based on proximity within a dense vector space, thereby accelerating the screening process significantly compared to traditional docking.4

### **1.2. The Metrics of Success: Quantifying Early Enrichment**

Selecting the appropriate metric is crucial for accurately benchmarking VS performance. While the Area Under the Receiver Operating Characteristic (AUC) is frequently reported, it measures the global classification ability across the entire dataset.5 However, the industrial goal of VS is to minimize experimental costs by identifying active compounds within the top ranks of the scored list—known as the "early recognition problem".6

The Enrichment Factor ($\\mathrm{EF}\_{\\chi}$) is the gold standard for evaluating this early recognition capability.6 $\\mathrm{EF}\_{\\chi}$ is defined as the ratio of the percentage of true actives retrieved in the top $\\chi$ percent of the ranked list to the percentage of true actives present in the entire database:

$$\\mathrm{EF}\_{\\chi} \= \\frac{\\frac{n\_{\\chi}}{N\_{\\chi}}}{\\frac{n}{N}}$$  
where $n\_{\\chi}$ is the number of actives found in the top $N\_{\\chi}$ compounds (corresponding to the top $\\chi$ fraction of the total library $N$), and $n$ is the total number of actives in the library.

Metrics such as AUC are mathematically proven to be poor choices for comparing VS methods focused on early ranking.6 Alternative, more robust metrics that reward early detection include the Robust Initial Enhancement (RIE) and the Boltzmann-enhanced discrimination of ROC (BEDROC) 6, or the ROC-Enrichment Factor (ROC-EF), which quantifies the ratio of the True Positive Rate (TPR) to the False Positive Rate (FPR) at a very low FPR cutoff (e.g., $\\text{FPF} \= 0.001$).5 For the purpose of contextualizing the user's results, the focus remains on $\\mathrm{EF}\_{1\\%}, \\mathrm{EF}\_{5\\%}$, and $\\mathrm{EF}\_{10\\%}$.

## **2\. State-of-the-Art Virtual Screening Architectures**

The latest advances in virtual screening reflect a move toward highly scalable, interpretable, and generalized predictive models, often leveraging graph representations and sophisticated AI algorithms.

### **2.1. Structure-Based Virtual Screening (SBVS) Evolution**

Traditional SBVS tools, such as the DOCK suite and Glide, rely on scoring functions derived from molecular mechanics or empirical observations.2 The performance of these methods is highly contingent upon meticulous target preparation, including the correct handling of specific chemical environments, such as the redistribution of partial atomic charges in metalloenzymes like Adenosine Deaminase (ADA).8 A successful conventional SBVS run may show good early enrichment (e.g., $\\mathrm{EF}\_{1} \= 38$ for Aldose reductase) but performance can weaken rapidly at later stages of screening.8

Recent breakthroughs demonstrate the superior potential of large, general AI models in complex SBVS tasks. For instance, models such as AlphaFold3 have been rigorously assessed for VS performance, achieving near-perfect classification (average AUC $= 98.3\\%$) when identifying covalent active binders over property-matched decoys.9 This performance dramatically exceeds that of classical covalent docking tools.9 Deep learning models are also increasingly used to refine or replace traditional scoring functions, as seen in frameworks like GNINA and improved DeepDTA, which are extensively benchmarked on rigorous datasets.10

### **2.2. Ligand-Based Virtual Screening (LBVS) and Vector-Based Approaches**

LBVS has transitioned from reliance on high-dimensional fixed descriptors (e.g., 2D fingerprints) and Tanimoto similarity to advanced vector-based co-embedding techniques.11 These methods embed both molecular and protein data into a shared vector space, where the Euclidean distance directly correlates with the interaction likelihood.12

Modern vector-based screening solutions like **ConPLex** and **DrugCLIP** have overcome the inherent limitations of traditional SBVS, particularly speed, which prevents proteome-scale analysis.12 By computing Drug-Target Interactions (DTIs) via a simple dot product in the co-embedding space, these methods enable the screening of millions of molecules against targets across thousands of microbial or human proteomes rapidly, often within 24 hours.12

The user’s method—descriptor $\\rightarrow$ PCA/UMAP $\\rightarrow$ similarity search—can be viewed as a computationally efficient, non-learned parallel to these modern vector architectures. If the chosen molecular descriptors and subsequent dimensionality reduction (PCA/UMAP) effectively capture the underlying chemical relationships relevant to biological activity, the resulting low-dimensional space should facilitate effective nearest-neighbor searching that approaches the performance of complex learned models.

## **3\. Dimensionality Reduction in Chemical Space: PCA, UMAP, and Nearest Neighbors**

The success of the user’s methodology hinges on the ability of Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP) to map high-dimensional molecular descriptors onto a latent manifold that preserves chemically relevant structure while mitigating the effects of the "curse of dimensionality".4

### **3.1. Role and Performance of PCA and UMAP**

High-dimensional fingerprints or descriptor vectors often contain redundancy, making similarity comparisons based on simple distance metrics (e.g., Euclidean distance) unreliable. Dimensionality reduction addresses this by identifying the intrinsic, lower-dimensional structure of the data.

PCA, a linear method focused on maximizing variance, has shown benefits when applied to rule-based fingerprints, often improving their virtual screening performance.13 However, UMAP, a non-linear manifold learning technique, offers distinct advantages. UMAP is faster than its predecessor (t-SNE) and is specifically designed to preserve both the global structure and local neighborhoods of the high-dimensional data.14

UMAP constructs its low-dimensional mapping based on k-nearest neighbor (k-NN) graphs.15 This structural foundation makes it intrinsically well-suited for similarity-based VS, which is fundamentally a k-NN search task in chemical space.16 The proximity of compounds in the UMAP-reduced space is expected to reliably reflect chemical similarity, allowing for rapid retrieval of actives.

### **3.2. UMAP for Robust Validation and Computational Efficiency**

The validity of UMAP as a meaningful metric for chemical diversity is strongly supported by its application in rigorous benchmarking protocols. Studies evaluating ML models for VS have demonstrated that using UMAP-based clustering to split data yields more challenging and realistic benchmarks compared to traditional random, scaffold, or Butina clustering splits.17

These traditional splits often result in artificially high performance because structurally similar molecules leak between training and test sets.17 Because UMAP clustering excels at defining and separating structurally dissimilar compounds for held-out test sets, it confirms that the resulting latent space effectively disentangles genuine chemical diversity from spurious correlations.18 This robust structural mapping capability provides confidence that the UMAP space is an excellent foundation for executing high-performance similarity *search*—the opposite operation of splitting—by accurately grouping chemically similar known actives together.

Furthermore, using low-dimensional embeddings generated by PCA or UMAP drastically improves computational efficiency. This framework allows for the employment of fast data structures, such as k-d trees, enabling nearest-neighbor searches across massive chemical databases (e.g., over a billion chemicals) in fractions of a second, representing a speed improvement of up to five orders of magnitude over brute-force methods.4

### **3.3. Learned Dimensionality Reduction Alternatives**

For comparative purposes, it is valuable to consider *learned* dimensionality reduction techniques, such as those employing Variational Autoencoders (VAEs).19 VAEs compress input data (like molecular fingerprints or SMILES strings) into a latent vector in a bottleneck layer, optimizing the representation based on reconstruction accuracy.11 This optimization process can potentially capture richer chemical feature relationships than non-learned linear (PCA) or manifold (UMAP) methods operating on fixed descriptors.11 If the $\\mathrm{EF}\_{\\chi}$ results obtained using PCA/UMAP are marginal, adopting an Autoencoder framework for generating the low-dimensional molecular representation might yield superior performance for similarity searches.19

## **4\. Quantitative Benchmarks for Enrichment Factor**

To contextualize the user's results, it is necessary to establish quantitative benchmarks for $\\mathrm{EF}\_{\\chi}$, particularly focusing on $\\mathrm{EF}\_{1\\%}$, which measures hit identification in the most critical top 1 percent of the screened library.

### **4.1. Interpreting $\\mathrm{EF}\_{1\\%}$ Performance**

An $\\mathrm{EF}$ value of 1.0 indicates no enrichment, corresponding to random selection.20 The maximum achievable $\\mathrm{EF}\_{\\chi}$ is highly dependent on the active-to-decoy ratio ($N/n$) in the database.6 For most modern benchmarks, $N/n$ is around 50 or higher, meaning theoretical maximum $\\mathrm{EF}$ values can be very high. However, practical results reflect the ability of the ranking method to distinguish true actives.

The following table provides contextual guidelines derived from peer-reviewed literature for interpreting $\\mathrm{EF}\_{1\\%}$ values obtained on typical, albeit often biased, virtual screening benchmarks:

Table 1: Interpretive Guidelines for $\\mathrm{EF}\_{1\\%}$ Performance (Contextual)

| EF1%​ Range (Approximate) | Contextual Interpretation | Achieved By (Examples) |
| :---- | :---- | :---- |
| $\< 5$ | Poor / Worse than HTS | Non-optimized approaches. |
| $5 \- 20$ | Fair / Intermediate | Single-query LBVS (2D/3D integration).21 |
| $20 \- 35$ | Good / Competitive Baseline | Optimized traditional SBVS (e.g., Glide, PLANTS).22 |
| $35 \- 55+$ | Excellent / SOTA Benchmark | Advanced consensus methods (ENS-VS), Modern DL models.24 |

### **4.2. Benchmarking Quantitative Results**

Direct comparison against established tools provides necessary context. For structure-based methods evaluated on the DUD-E dataset (102 targets):

1. **Traditional SBVS Baseline:** A robust commercial docking tool like Glide achieves a mean $\\mathrm{EF}\_{1\\%}$ of $21.253$ and a mean AUC of $0.795$.25 This $\\mathrm{EF}\_{1\\%}$ value serves as a critical performance floor: if the rapid LBVS+PCA/UMAP method achieves comparable enrichment ($\\approx 20-25$), it represents a significant success, given the vast speed advantage gained by converting the problem into a rapid nearest-neighbor lookup.4  
2. **Advanced Scoring Functions:** State-of-the-art scoring functions demonstrate superior performance, often leveraging consensus or machine learning. For instance, the SIEVE-Score method reports a mean $\\mathrm{EF}\_{1\\%}$ of $43.913$ (mean AUC $0.899$) on DUD-E, while the consensus method ENS-VS achieved a mean $\\mathrm{EF}\_{1\\%}$ of $52.77$ (mean AUC $0.982$).24

For ligand-based methods, the performance can be strong, but typically lower than the highest-performing SBVS models. Integrated 2D and 3D similarity methods, prior to deep learning adoption, reported $\\mathrm{EF}\_{1\\%}$ values around $17.52$ to $19.96$.21 Specialized re-scoring models applied to specific protein families, such as DHFR, have reached $\\mathrm{EF}\_{1\\%}$ values of $28$ to $31$.23

## **5\. Critical Analysis of Virtual Screening Benchmark Datasets**

The numerical $\\mathrm{EF}\_{\\chi}$ values reported by the user must be interpreted solely within the context of the dataset used, as the most common benchmarks are known to suffer from severe, performance-inflating biases.

### **5.1. Limitations of DUD-E and MUV**

The Directory of Useful Decoys: Enhanced (DUD-E) remains a standard historical benchmark, featuring 22,886 actives across 102 targets, with an average of 50 decoys per active.26 The decoys were selected to match physicochemical properties (e.g., molecular weight, calculated $\\log P$, hydrogen bond features) while ensuring 2D topological dissimilarity.27

However, DUD-E (and its predecessor DUD and MUV) is plagued by two critical issues: analogue bias and decoy bias.10

1. **Analogue Bias:** The active compounds for many targets are structurally highly similar.29 This allows any method relying on similarity, including the user’s PCA/UMAP nearest-neighbor search 16, to achieve artificially high $\\mathrm{EF}$ values simply by exploiting these correlations rather than demonstrating genuine generalization to novel chemical scaffolds.29  
2. **Decoy Bias:** The specific generation protocol for decoys—matching 1D properties but enforcing 2D dissimilarity—introduces artifacts that deep learning models can learn to exploit as a shortcut for classification.29 This results in highly inflated AUCs (e.g., $\>0.9$) and $\\mathrm{EF}$ values that do not translate accurately to prospective screening success.29

### **5.2. The Data Integrity Crisis in LIT-PCBA**

The LIT-PCBA (Literature-derived PubChem BioAssay) benchmark was introduced specifically to address the known biases of DUD-E and MUV.10 It includes 15 protein targets with X-ray structures and curates 7844 confirmed actives and over 400,000 confirmed inactives from high-confidence PubChem bioassays.31 It utilizes the Asymmetric Validation Embedding (AVE) procedure designed to reduce spurious correlations and ensure similar molecular property distributions between actives and inactives.30

Despite its rigorous design, recent critical audits have exposed severe flaws in the LIT-PCBA dataset.33 These flaws include egregious data leakage, rampant duplication, and pervasive analog redundancy, collectively invalidating the benchmark for fair model evaluation.33 Key data integrity failures include:

* Thousands of inactives duplicated across training and validation sets.33  
* Critical leakage of query ligands (meant to represent unseen test cases) into both the training and validation sets.10  
* High structural redundancy, with over $80\\%$ of query ligands for some targets being near duplicates (Tanimoto similarity $\\geq 0.9$).33

The consequence of these flaws is that models trained on LIT-PCBA often demonstrate *memorization* rather than *generalization*. Audits showed that a trivial memorization-based baseline, exploiting these artifacts, can outperform sophisticated state-of-the-art deep neural networks like CHEESE on the LIT-PCBA validation split.33

The existence of fundamental data issues in both DUD-E and LIT-PCBA creates a significant benchmarking paradox. High EF values reported on DUD-E are likely inflated by implicit similarity bias, while high EF values reported on LIT-PCBA are suspect due to explicit data leakage. Rigorous validation of any VS methodology, including the user's LBVS approach, must prioritize techniques that explicitly enforce chemical diversity between training and testing sets.

Table 2: Critical Comparison of Primary Virtual Screening Benchmark Datasets

| Feature | DUD-E | LIT-PCBA (Original Intent) | LIT-PCBA (Audit Findings) |
| :---- | :---- | :---- | :---- |
| **Number of Targets** | 102 27 | 15 32 | 15 33 |
| **Active/Inactive Source** | Curated literature/DUD decoys 26 | High-confidence PubChem Bioassays 31 | Compromised by Duplication/Leakage 33 |
| **Primary Bias** | Analog and Decoy Bias 29 | Designed to be unbiased (AVE process) 30 | **Severe Data Leakage & Redundancy** 33 |
| **Highest Reported Mean $\\mathrm{EF}\_{1\\%}$** | $\\sim 52.77$ (ENS-VS) 24 | Varies (SOTA deep learning pipelines claimed superior $\\mathrm{EF}\_{1\\%}$) 10 | High $\\mathrm{EF}\_{1\\%}$ reflects memorization of artifacts 33 |

## **6\. Advanced Strategies: Multi-Target and Transfer Learning**

Beyond single-target enrichment, the complexity of modern drug discovery demands methodologies capable of handling polypharmacology and maximizing generalizability across targets.

### **6.1. Multi-Target Directed Ligands (MTDLs)**

The search for Multi-Target Directed Ligands (MTDLs)—compounds that selectively modulate multiple targets—is a growing strategy, moving away from the traditional "one drug, one target" paradigm.34 However, computationally identifying effective MTDLs requires navigating an intractable search space that exceeds the capacity of brute-force experimental or traditional docking methods.35

A critical consideration in MTDL discovery is the management of Pan-Assay Interference Compounds (PAINS). These are compounds prone to causing non-specific, artifactual activity in biochemical assays.34 While *in silico* PAINS filters exist, excessive filtering based solely on computational prediction may inappropriately discard valuable scaffolds with complex mechanisms of action.34 Consequently, computational identification of MTDLs must be coupled with rigorous biochemical validation using a "Fair Trial Strategy" to discriminate between truly useful polypharmacology and experimental artifacts.34

### **6.2. Transfer Learning and Data Augmentation**

Transfer learning has emerged as a powerful technique to improve the performance and generalizability of VS models, especially when data for a specific target is sparse. By pre-training models on vast datasets covering diverse protein families (data set augmentation), the model learns fundamental structural comparisons and binding interactions common across related targets.37

Applying this knowledge to a novel or data-poor target class significantly improves generalization to unseen chemical space.37 For example, deep learning models applied to kinase inhibitor prediction demonstrated improved Enrichment Factors when utilizing transfer learning techniques.7 The mechanism is based on leveraging the shared structural mechanisms inherent to protein families, such as the conserved kinase domain fold.20

For the user’s LBVS+PCA/UMAP methodology, the principle of transfer learning can be applied by employing the dimensionality reduction step on a massive, chemically diverse database (such as ChEMBL). This process would allow the PCA/UMAP pipeline to learn a generalized, chemically rich feature representation of the entire drug-like chemical space. This established manifold structure could then be applied to specific targets, potentially followed by target-specific weighting or fine-tuning in the reduced space, leading to superior $\\mathrm{EF}\_{\\chi}$ results on novel targets compared to a PCA/UMAP reduction trained only on limited single-target data.

## **7\. Conclusions and Recommendations**

The methodology of performing ligand-based similarity search using dimensionality reduction (PCA/UMAP) on molecular descriptors represents a highly valuable and computationally efficient approach for virtual screening. Its alignment with modern vector-based screening architectures leverages the speed benefits of nearest-neighbor searching in low-dimensional space.4

### **7.1. Contextualizing $\\mathrm{EF}\_{\\chi}$ Results**

To interpret the user's $\\mathrm{EF}\_{\\chi}$ values:

1. **Performance Baseline:** If the LBVS+PCA/UMAP method achieves an $\\mathrm{EF}\_{1\\%}$ in the **$20-30$ range**, the performance is considered **Good** and is competitive with robust traditional structure-based docking methods (e.g., Glide mean $\\mathrm{EF}\_{1\\%} \\approx 21$).25 This confirms that the PCA/UMAP reduction successfully maintains the chemical manifold structure necessary for effective similarity retrieval.  
2. **Trade-off:** If the results fall below $\\mathrm{EF}\_{1\\%} \= 10$, the selection of descriptors or the parameters used for dimensionality reduction (PCA components retained, UMAP parameters $n\\\_neighbors$, $\\text{min\\\_dist}$) may be failing to preserve relevant chemical relationships.

### **7.2. Recommendations for Rigorous Validation**

Given the confirmed data integrity issues in standard benchmarks, rigorous validation is paramount to ensure the reported $\\mathrm{EF}\_{\\chi}$ values reflect true generalization:

1. **Avoid Biased Benchmarks:** Any high $\\mathrm{EF}\_{\\chi}$ values reported on DUD-E ($\\mathrm{EF}\_{1\\%} \> 40$) should be treated with skepticism due to known analogue and decoy biases.29 Similarly, results from LIT-PCBA must be treated with extreme caution due to confirmed data leakage and structural redundancy.33  
2. **Implement Structural Validation Splits:** The most robust way to validate the LBVS methodology is to ensure the test data is truly dissimilar from the training data. It is strongly recommended to use **UMAP clustering splits** 17 or other robust scaffold-dissimilar protocols to partition the dataset. High $\\mathrm{EF}\_{\\chi}$ values on a UMAP-split test set provide the strongest evidence that the model is learning to generalize chemical recognition, not merely memorizing known analogues.  
3. **Supplement with Alternative Metrics:** Relying solely on $\\mathrm{EF}\_{\\chi}$ can be misleading due to its sensitivity to the active-to-decoy ratio. The use of $\\mathrm{BEDROC}$ or $\\mathrm{ROC}$-$\\mathrm{EF}$ (at low FPR cutoffs) is recommended to provide a balanced and statistically robust measure of early enrichment performance.5

#### **Works cited**

1. Practical Model Selection for Prospective Virtual Screening \- ACS Publications, accessed on November 7, 2025, [https://pubs.acs.org/doi/10.1021/acs.jcim.8b00363](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00363)  
2. Virtual Ligand Screening Against Comparative Protein Structure Models \- PMC \- NIH, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3386294/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3386294/)  
3. Ligand-based structural hypotheses for virtual screening \- PubMed \- NIH, accessed on November 7, 2025, [https://pubmed.ncbi.nlm.nih.gov/14761196/](https://pubmed.ncbi.nlm.nih.gov/14761196/)  
4. \[2402.07970\] Utilizing Low-Dimensional Molecular Embeddings for Rapid Chemical Similarity Search \- arXiv, accessed on November 7, 2025, [https://arxiv.org/abs/2402.07970](https://arxiv.org/abs/2402.07970)  
5. Enhancing Virtual Screening Performance of Protein Kinases with Molecular Dynamics Simulations \- PMC \- NIH, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC5323360/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5323360/)  
6. Evaluating Virtual Screening Methods: Good and Bad Metrics for the “Early Recognition” Problem | Journal of Chemical Information and Modeling \- ACS Publications, accessed on November 7, 2025, [https://pubs.acs.org/doi/10.1021/ci600426e](https://pubs.acs.org/doi/10.1021/ci600426e)  
7. Improving the Virtual Screening Ability of Target-Specific Scoring Functions Using Deep Learning Methods \- PubMed Central, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6713720/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6713720/)  
8. Benchmarking Sets for Molecular Docking \- PMC \- NIH, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3383317/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3383317/)  
9. State-of-the-art covalent virtual screening with AlphaFold3 \- bioRxiv, accessed on November 7, 2025, [https://www.biorxiv.org/content/10.1101/2025.03.19.642201v1.full.pdf](https://www.biorxiv.org/content/10.1101/2025.03.19.642201v1.full.pdf)  
10. Data Leakage and Redundancy in the LIT-PCBA Benchmark \- arXiv, accessed on November 7, 2025, [https://arxiv.org/html/2507.21404v2](https://arxiv.org/html/2507.21404v2)  
11. Feature Reduction for Molecular Similarity Searching Based on Autoencoder Deep Learning, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9029813/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9029813/)  
12. Scaling Structure Aware Virtual Screening to Billions of Molecules with SPRINT \- arXiv, accessed on November 7, 2025, [https://arxiv.org/html/2411.15418v2](https://arxiv.org/html/2411.15418v2)  
13. Comparative analysis of molecular fingerprints in prediction of drug combination effects | Briefings in Bioinformatics | Oxford Academic, accessed on November 7, 2025, [https://academic.oup.com/bib/article/22/6/bbab291/6353238](https://academic.oup.com/bib/article/22/6/bbab291/6353238)  
14. Understanding UMAP, accessed on November 7, 2025, [https://pair-code.github.io/understanding-umap/](https://pair-code.github.io/understanding-umap/)  
15. A Quantitative Framework for Evaluating Single-Cell Data Structure Preservation by Dimensionality Reduction Techniques \- PubMed Central, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7305633/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7305633/)  
16. A Small Step Toward Generalizability: Training a Machine Learning Scoring Function for Structure-Based Virtual Screening | Journal of Chemical Information and Modeling, accessed on November 7, 2025, [https://pubs.acs.org/doi/10.1021/acs.jcim.3c00322](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00322)  
17. UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines \- PubMed, accessed on November 7, 2025, [https://pubmed.ncbi.nlm.nih.gov/40495205/](https://pubmed.ncbi.nlm.nih.gov/40495205/)  
18. UMAP-based clustering split for rigorous evaluation of AI models for virtual screening on cancer cell lines\* \- PMC \- NIH, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12153141/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12153141/)  
19. VAE-Sim: A Novel Molecular Similarity Measure Based on a Variational Autoencoder \- PMC, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7435890/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7435890/)  
20. An Improved Metric and Benchmark for Assessing the Performance of Virtual Screening Models \- NIH, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10980085/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10980085/)  
21. Maximizing the Performance of Similarity-Based Virtual Screening Methods by Generating Synergy from the Integration of 2D and 3D Approaches \- PubMed Central, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9322642/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9322642/)  
22. Best EF1% values reached by the various consensus models developed... \- ResearchGate, accessed on November 7, 2025, [https://www.researchgate.net/figure/Best-EF1-values-reached-by-the-various-consensus-models-developed-using-the-PLANTS\_tbl2\_349046152](https://www.researchgate.net/figure/Best-EF1-values-reached-by-the-various-consensus-models-developed-using-the-PLANTS_tbl2_349046152)  
23. Benchmarking the Structure-Based Virtual Screening Performance of Wild-Type and Resistant PfDHFR Using Docking and Machine Learning Re-Scoring \- PMC \- NIH, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12363558/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12363558/)  
24. Improved method of structure-based virtual screening based on ensemble learning, accessed on November 7, 2025, [https://pubs.rsc.org/en/content/articlehtml/2020/ra/c9ra09211k](https://pubs.rsc.org/en/content/articlehtml/2020/ra/c9ra09211k)  
25. Improved Method of Structure-Based Virtual Screening via Interaction-Energy-Based Learning | Journal of Chemical Information and Modeling, accessed on November 7, 2025, [https://pubs.acs.org/doi/10.1021/acs.jcim.8b00673](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00673)  
26. DUD-E: A Database of Useful (Docking) Decoys — Enhanced, accessed on November 7, 2025, [https://dude.docking.org/](https://dude.docking.org/)  
27. FAQ | DUD-E: A Database of Useful (Docking) Decoys — Enhanced, accessed on November 7, 2025, [https://dude.docking.org/faq](https://dude.docking.org/faq)  
28. Maximum Unbiased Validation (MUV) Data Sets for Virtual Screening Based on PubChem Bioactivity Data | Journal of Chemical Information and Modeling \- ACS Publications, accessed on November 7, 2025, [https://pubs.acs.org/doi/10.1021/ci8002649](https://pubs.acs.org/doi/10.1021/ci8002649)  
29. Hidden bias in the DUD-E dataset leads to misleading performance of deep learning in structure-based virtual screening \- NIH, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6701836/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6701836/)  
30. LIT-PCBA: An Unbiased Data Set for Machine Learning and Virtual Screening \- PubMed, accessed on November 7, 2025, [https://pubmed.ncbi.nlm.nih.gov/32282202/](https://pubmed.ncbi.nlm.nih.gov/32282202/)  
31. MF-PCBA: Multifidelity High-Throughput Screening Benchmarks for Drug Discovery and Machine Learning | Journal of Chemical Information and Modeling \- ACS Publications, accessed on November 7, 2025, [https://pubs.acs.org/doi/10.1021/acs.jcim.2c01569](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01569)  
32. LIT-PCBA: A dataset, accessed on November 7, 2025, [https://drugdesign.unistra.fr/LIT-PCBA/](https://drugdesign.unistra.fr/LIT-PCBA/)  
33. Data Leakage and Redundancy in the LIT-PCBA Benchmark \- arXiv, accessed on November 7, 2025, [https://arxiv.org/html/2507.21404v1](https://arxiv.org/html/2507.21404v1)  
34. Gains from no real PAINS: Where 'Fair Trial Strategy' stands in the development of multi-target ligands \- PMC \- PubMed Central, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8642439/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8642439/)  
35. Machine Learning for Multi-Target Drug Discovery: Challenges and Opportunities in Systems Pharmacology \- PubMed Central, accessed on November 7, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12473769/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473769/)  
36. Dealing with frequent hitters in drug discovery: a multidisciplinary view on the issue of filtering compounds on biological screenings \- PubMed, accessed on November 7, 2025, [https://pubmed.ncbi.nlm.nih.gov/31416369/](https://pubmed.ncbi.nlm.nih.gov/31416369/)  
37. Protein Family-Specific Models Using Deep Neural Networks and Transfer Learning Improve Virtual Screening and Highlight the Need for More Data \- ACS Publications, accessed on November 7, 2025, [https://pubs.acs.org/doi/abs/10.1021/acs.jcim.8b00350](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.8b00350)  
38. DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design, accessed on November 7, 2025, [https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334)