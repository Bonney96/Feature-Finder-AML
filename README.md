# Feature-Finder-AML
A deep learning model for predicting methylation in AML, with the primary focus being on understanding the model's feature weighting and identifying the key features associated with AML.

AML Histone Modifications ML Concept Map:

https://lucid.app/lucidchart/8f9cda5f-7753-4761-a9dd-4e08d61c37d2/edit?viewport_loc=-1083%2C-1158%2C4619%2C2226%2C0_0&invitationId=inv_1b0bc6d4-359a-4f1b-9018-e0c8575d44ca


Infrastructure and Software:

HPC Cluster with Docker support and sufficient storage for large genomic datasets.


Software and Tools:

bowtie2, bwa, STAR, methylKit, MACS2, deepTools, MEME, HOMER.
TensorFlow, PyTorch, PyTorch Geometric (PyG), scikit-learn, pandas, numpy, Optuna, SHAP.


Visualization: 

IGV, UCSC Genome Browser, matplotlib, seaborn.


Data:

Histone Modifications: CUT&Tag data (H3K27ac, H3K4me1).
Methylation Data: Whole-genome bisulfite sequencing (WGBS) for DMRs.
Transcriptomics: RNA-seq from AML patients.
Mutational Data: WGS/Exome for AML mutations.
Control Data: Healthy blood stem cells (CD34+).


Machine Learning Models:

Baseline: 1D CNN for motif detection.
Advanced: GNN for spatial interactions, Transformers for long-range dependencies, Multi-Modal architecture combining all approaches.
