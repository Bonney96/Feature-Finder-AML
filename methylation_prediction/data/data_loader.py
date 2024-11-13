# data/data_loader.py
import pandas as pd
import pyranges as pr
import logging

from .preprocess import standardize_chromosome_names

def load_data_files(config):
    # Histone marks columns
    histone_marks = config['data']['histone_marks']

    # Load CSV data
    data = pd.read_csv(config['data']['csv_file_path'])
    logging.info("CSV data loaded successfully.")

    # Add sequence_id column
    data['sequence_id'] = data.index
    data = data.rename(columns={'chrom': 'Chromosome', 'start': 'Start', 'end': 'End'})

    # Standardize chromosome names
    data = standardize_chromosome_names(data, 'Chromosome')

    # Load BED file (Labels)
    bed_df = pd.read_csv(
        config['data']['bed_file_path'],
        sep='\t',
        header=None,
        names=['Chromosome', 'Start', 'End', 'average_methylation']
    )
    logging.info("BED file with average methylation loaded successfully.")

    # Load gene annotations BED file
    bed_columns = [
        'Chromosome', 'Start', 'End', 'gene', 'score', 'strand',
        'thickStart', 'thickEnd', 'itemRgb', 'blockCount',
        'blockSizes', 'blockStarts'
    ]

    gene_annotations_df = pd.read_csv(
        config['data']['gene_annotations_bed_file_path'],
        sep='\t',
        header=None,
        names=bed_columns
    )
    logging.info("Gene annotations loaded successfully.")

    # Convert 'Start' and 'End' to numeric values, coercing errors
    gene_annotations_df['Start'] = pd.to_numeric(gene_annotations_df['Start'], errors='coerce')
    gene_annotations_df['End'] = pd.to_numeric(gene_annotations_df['End'], errors='coerce')

    # Drop rows with NaN values in 'Start' or 'End'
    gene_annotations_df.dropna(subset=['Start', 'End'], inplace=True)

    # Convert 'Start' and 'End' to integers
    gene_annotations_df['Start'] = gene_annotations_df['Start'].astype(int)
    gene_annotations_df['End'] = gene_annotations_df['End'].astype(int)

    gene_annotations_df = standardize_chromosome_names(gene_annotations_df, 'Chromosome')

    return data, bed_df, gene_annotations_df
