# interpretability/explainability.py

import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import LayerIntegratedGradients, IntegratedGradients
import logging
import pandas as pd
from collections import defaultdict
from data.dataset import MethylationDataset
from utils.helpers import encode_sequences
from evaluation.visualization import plot_gene_saliency_maps

def explain_with_integrated_gradients(model, data_loader, device, folder_path, max_samples=10):
    model.eval()
    model.to(device)
    embedding_layer = model.dna_module.embedding

    def forward_func(sequence, sequence_lengths, features):
        return model(sequence, sequence_lengths, features)

    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max_samples:
            break

        sequences = batch['sequence'].to(device)
        features = batch['features'].to(device)
        sequence_lengths = batch['sequence_length'].to(device)

        attributions_ig = lig.attribute(
            inputs=sequences,
            baselines=torch.zeros_like(sequences),
            additional_forward_args=(sequence_lengths, features),
            n_steps=50
        )

        attributions_np = attributions_ig.detach().cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.hist(attributions_np.flatten(), bins=50, color='blue', alpha=0.7)
        plt.title(f'Integrated Gradients Attributions for Sample {batch_idx + 1}')
        plt.xlabel('Attribution Value')
        plt.ylabel('Frequency')
        plt.tight_layout()

        plot_filename = os.path.join(folder_path, f'ig_attribution_sample_{batch_idx + 1}.png')
        plt.savefig(plot_filename)
        plt.close()

        logging.info(f"Attribution plot saved to: {plot_filename}")

def run_explainability_methods(model, test_loader, device, folder_path):
    logging.info("Running Integrated Gradients explainability...")
    explain_with_integrated_gradients(model, test_loader, device, folder_path, max_samples=10)

def compute_histone_feature_importance(model, data_loader, histone_feature_names, device):
    model.eval()
    model.to(device)

    ig = IntegratedGradients(lambda features, sequences, sequence_lengths: model(sequences, sequence_lengths, features))

    all_attributions = []

    for batch in data_loader:
        sequences = batch['sequence'].to(device)
        features = batch['features'].to(device)
        sequence_lengths = batch['sequence_length'].to(device)

        features_baseline = torch.zeros_like(features)

        attributions = ig.attribute(
            inputs=features,
            baselines=features_baseline,
            target=None,
            n_steps=50,
            additional_forward_args=(sequences, sequence_lengths)
        )

        feature_attributions = attributions.detach().cpu().numpy()
        all_attributions.append(feature_attributions)

    all_attributions = np.vstack(all_attributions)
    mean_attributions = np.mean(np.abs(all_attributions), axis=0)
    feature_importance = pd.Series(mean_attributions, index=histone_feature_names)

    return feature_importance

def compute_dmr_saliency_maps(model, data_loader, device):
    model.eval()
    embedding_layer = model.dna_module.embedding

    def forward_func(sequence, sequence_lengths, features):
        return model(sequence, sequence_lengths, features)

    lig = LayerIntegratedGradients(forward_func, embedding_layer)
    sequences_list = []
    dmr_saliency_maps = []

    for batch in data_loader:
        sequence = batch['sequence'].to(device)
        features = batch['features'].to(device)
        sequence_lengths = batch['sequence_length'].to(device)

        attributions, delta = lig.attribute(
            inputs=sequence,
            baselines=torch.zeros_like(sequence),
            additional_forward_args=(sequence_lengths, features),
            return_convergence_delta=True
        )
        attributions = attributions.detach().cpu().numpy()
        dmr_saliency_maps.append(attributions[0])
        sequences_list.append(sequence.cpu().numpy()[0])

    logging.info("DMR-level saliency maps computed.")
    return sequences_list, dmr_saliency_maps

def compute_gene_saliency_maps(model, data_with_genes, genes_of_interest, histone_marks, max_seq_len, device, folder_path):
    data_filtered = data_with_genes[data_with_genes['gene'].apply(lambda x: any(gene in x for gene in genes_of_interest))]

    sequences_of_interest = data_filtered['sequence'].values
    other_features_of_interest = data_filtered[histone_marks].values
    labels_of_interest = data_filtered['avg_methylation'].values
    gene_names_of_interest = data_filtered['gene'].apply(lambda x: [gene for gene in x if gene in genes_of_interest]).values

    gene_names_flat = [genes[0] if genes else 'Unknown' for genes in gene_names_of_interest]

    encoded_sequences_of_interest = encode_sequences(sequences_of_interest, maxlen=max_seq_len)

    dataset_of_interest = MethylationDataset(
        encoded_sequences_of_interest,
        other_features_of_interest,
        labels_of_interest
    )

    loader_of_interest = DataLoader(dataset_of_interest, batch_size=1, shuffle=False)

    model.eval()
    embedding_layer = model.dna_module.embedding

    def forward_func(sequences, sequence_lengths, features):
        return model(sequences, sequence_lengths, features)

    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    sequences_list = []
    attributions_list = []
    positions_list = []

    for idx, batch in enumerate(loader_of_interest):
        sequences = batch['sequence'].to(device)
        features = batch['features'].to(device)
        sequence_lengths = batch['sequence_length'].to(device)

        attributions, delta = lig.attribute(
            inputs=sequences,
            baselines=torch.zeros_like(sequences),
            additional_forward_args=(sequence_lengths, features),
            return_convergence_delta=True
        )
        attributions = attributions.detach().cpu().numpy()
        sequence = sequences.cpu().numpy()[0]
        sequences_list.append(sequence)
        attributions_list.append(attributions[0])

        chrom = data_filtered.iloc[idx]['Chromosome']
        start = data_filtered.iloc[idx]['Start']
        end = data_filtered.iloc[idx]['End']
        positions_list.append(f"{chrom}:{start}-{end}")

    gene_attributions_dict = defaultdict(lambda: {'sequences': [], 'attributions': []})
    gene_positions_dict = defaultdict(list)

    for idx, gene_name in enumerate(gene_names_flat):
        gene_attributions_dict[gene_name]['sequences'].append(sequences_list[idx])
        gene_attributions_dict[gene_name]['attributions'].append(attributions_list[idx])
        gene_positions_dict[gene_name].append(positions_list[idx])

    plot_gene_saliency_maps(gene_attributions_dict, gene_positions_dict, folder_path)
