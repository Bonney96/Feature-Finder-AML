AML Histone Modifications ML Concept Map
Concept Map: AML Histone Modifications ML Concept Map
GitHub Repository
Project Repository: Feature-Finder-AML
Requirements
Infrastructure and Software:
HPC Cluster with Docker support and sufficient storage for large genomic datasets.
Software and Tools:
bowtie2, bwa, STAR, methylKit, MACS2, deepTools, MEME, HOMER.
TensorFlow, PyTorch, PyTorch Geometric (PyG), scikit-learn, pandas, numpy, Optuna, SHAP.
Visualization: IGV, UCSC Genome Browser, matplotlib, seaborn.
Data:
Histone Modifications: CUT&Tag data (H3K27ac, H3K4me1).
Methylation Data: Whole-genome bisulfite sequencing (WGBS) for DMRs.
Transcriptomics: RNA-seq from AML patients.
Mutational Data: WGS/Exome for AML mutations.
Control Data: Healthy blood stem cells (CD34+).
Machine Learning Models:
Baseline: 1D CNN for motif detection.
Advanced: GNN for spatial interactions, Transformers for long-range dependencies, Multi-Modal architecture combining all approaches.
