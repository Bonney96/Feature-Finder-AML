# evaluation/visualization.py

import matplotlib.pyplot as plt
import os
import logging
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def plot_predictions_vs_actual(predictions, labels, dmrs, dataset_name, folder_path):
    df = pd.DataFrame({
        'Actual Methylation Levels': labels,
        'Predicted Methylation Levels': predictions,
        'DMR': dmrs
    })

    fig = px.scatter(
        df,
        x='Actual Methylation Levels',
        y='Predicted Methylation Levels',
        hover_data=['DMR'],
        title=f'Predicted vs Actual Methylation Levels ({dataset_name})'
    )

    fig.add_shape(
        type='line',
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color='Red', dash='dash')
    )

    fig.update_layout(
        xaxis_title='Actual Methylation Levels',
        yaxis_title='Predicted Methylation Levels',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )

    html_filename = f'pred_vs_actual_{dataset_name}.html'
    html_path = os.path.join(folder_path, html_filename)
    fig.write_html(html_path)
    logging.info(f"Interactive plot saved to: {html_path}")

def plot_training_validation_loss(train_losses, val_losses, num_epochs, folder_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_filename = os.path.join(folder_path, 'training_validation_loss.png')
    plt.savefig(loss_plot_filename)
    plt.close()
    logging.info(f"Training and validation loss plot saved to: {loss_plot_filename}")

def plot_confusion_matrix(predictions, labels, folder_path):
    # Define bins for methylation levels
    bins = [0.0, 0.33, 0.66, 1.0]
    labels_bins = ['Low', 'Medium', 'High']

    # Discretize labels
    labels_binned = pd.cut(labels, bins=bins, labels=labels_bins, include_lowest=True)
    predictions_binned = pd.cut(predictions, bins=bins, labels=labels_bins, include_lowest=True)

    # Convert to numpy arrays
    labels_binned = np.array(labels_binned)
    predictions_binned = np.array(predictions_binned)

    # Handle NaN values
    labels_binned = np.where(pd.isnull(labels_binned), 'Unknown', labels_binned)
    predictions_binned = np.where(pd.isnull(predictions_binned), 'Unknown', predictions_binned)

    # Convert arrays to strings
    labels_binned = labels_binned.astype(str)
    predictions_binned = predictions_binned.astype(str)

    # Create confusion matrix
    cm_labels = labels_bins + ['Unknown']
    cm = confusion_matrix(labels_binned, predictions_binned, labels=cm_labels)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    disp.plot()
    plt.title('Confusion Matrix (Test Set)')
    # Save the confusion matrix plot in the new folder
    cm_filename = os.path.join(folder_path, 'confusion_matrix_test.png')
    plt.savefig(cm_filename)
    plt.close()
    logging.info(f"Confusion matrix plot saved to: {cm_filename}")

def plot_histone_feature_importance(feature_importance, folder_path):
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title('Feature Importance for Histone Features')
    plt.ylabel('Mean Absolute Attribution')
    plt.xlabel('Histone Marks')
    plt.tight_layout()

    # Save the plot
    histone_importance_plot_path = os.path.join(folder_path, 'histone_feature_importance.png')
    plt.savefig(histone_importance_plot_path)
    plt.close()
    logging.info(f"Histone feature importance plot saved to: {histone_importance_plot_path}")

def plot_saliency_maps(sequences_list, dmr_saliency_maps, folder_path):
    for i in range(min(10, len(dmr_saliency_maps))):
        sequence = sequences_list[i]
        attribution = dmr_saliency_maps[i]
        # Convert the sequence from numerical indices to nucleotide characters
        reverse_mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
        nucleotides = [reverse_mapping.get(int(idx), 'N') for idx in sequence]

        # Compute the norm of attribution for each nucleotide position
        attribution_per_nucleotide = np.linalg.norm(attribution, axis=-1)

        # Plot the saliency map
        plt.figure(figsize=(15, 4))
        plt.bar(range(len(nucleotides)), attribution_per_nucleotide, color='blue')
        plt.xlabel('Sequence Position')
        plt.ylabel('Attribution Value')
        plt.title(f'Saliency Map for DMR Index {i}')
        plt.xticks(range(len(nucleotides)), nucleotides, fontsize=5)
        plt.tight_layout()

        # Save the plot
        plot_filename = os.path.join(folder_path, f'dmr_saliency_map_{i}.png')
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Saliency map for DMR index {i} saved to: {plot_filename}")

def plot_gene_saliency_maps(gene_saliency_dict, gene_positions_dict, folder_path):
    for gene_name, attributions in gene_saliency_dict.items():
        sequences = gene_saliency_dict[gene_name]['sequences']
        attributions = gene_saliency_dict[gene_name]['attributions']
        positions = gene_positions_dict[gene_name]

        # Plot heatmap
        num_sequences = len(sequences)
        max_length = max([len(seq) for seq in sequences])
        heatmap_data = np.zeros((num_sequences, max_length))
        for i, attribution in enumerate(attributions):
            attribution_per_nucleotide = np.linalg.norm(attribution, axis=-1)
            heatmap_data[i, :len(attribution_per_nucleotide)] = attribution_per_nucleotide

        plt.figure(figsize=(20, max(6, num_sequences * 0.5)))
        plt.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Attribution Value')
        plt.xlabel('Sequence Position')
        plt.ylabel('Sequences')
        plt.yticks(range(len(positions)), positions, fontsize=8)
        plt.title(f'Saliency Map Heatmap for {gene_name}')
        plt.tight_layout()

        # Save the heatmap plot
        sanitized_gene_name = gene_name.replace('/', '_').replace('\\', '_')
        plot_filename = os.path.join(folder_path, f'saliency_map_heatmap_{sanitized_gene_name}.png')
        plt.savefig(plot_filename)
        plt.close()
        logging.info(f"Saliency map heatmap for {gene_name} saved to: {plot_filename}")
