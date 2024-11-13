# data/preprocess.py
import pandas as pd
import logging
import pyranges as pr

def standardize_chromosome_names(df, chrom_column):
    df[chrom_column] = df[chrom_column].astype(str)
    df.loc[~df[chrom_column].str.startswith('chr'), chrom_column] = 'chr' + df[chrom_column]
    return df

def preprocess_data(data, bed_df, gene_annotations_df, config):
    histone_marks = config['data']['histone_marks']

    # Proceed with PyRanges conversion
    csv_columns = ['Chromosome', 'Start', 'End', 'sequence_id', 'sequence'] + histone_marks + ['Sample', 'avg_methylation']
    csv_pr = pr.PyRanges(data[csv_columns])
    gene_annotations_pr = pr.PyRanges(gene_annotations_df)

    # Join operation
    joined = csv_pr.join(gene_annotations_pr, how='left')

    # Convert back to Pandas DataFrame
    data_with_genes = joined.df
    logging.info("Data after joining with gene annotations.")

    # Handle multiple gene overlaps
    group_columns = ['sequence_id']
    data_with_genes = data_with_genes.groupby(group_columns).agg({
        'Chromosome': 'first',
        'Start': 'first',
        'End': 'first',
        'sequence': 'first',
        **{hm: 'first' for hm in histone_marks},
        'Sample': 'first',
        'avg_methylation': 'first',
        'gene': lambda x: list(x)
    }).reset_index()

    # Replace empty lists with ['Unknown']
    data_with_genes['gene'] = data_with_genes['gene'].apply(lambda x: x if x and not pd.isna(x[0]) else ['Unknown'])

    logging.info("Data after handling multiple gene overlaps.")

    return data_with_genes
