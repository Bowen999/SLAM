from rdkit import Chem
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



def parse_mgf(file_path):
    # Convert MGF file to CSV
    records = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    block = None
    ms2_list = []
    
    for line in lines:
        line = line.strip()
        # Start a new ion block
        if line == 'BEGIN IONS':
            block = {}
            ms2_list = []
            continue
        # End of an ion block: add the MS2 data and save the block
        if line == 'END IONS':
            if block is not None:
                block['MS2'] = ms2_list
                records.append(block)
            block = None
            continue
        # If we are inside a block, process the line
        if block is not None:
            # Lines with key=value pairs (like PEPMASS, CHARGE, etc.)
            if '=' in line:
                key, value = line.split('=', 1)
                block[key] = value
            # Lines without '=' are assumed to be the m/z intensity pairs
            else:
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz = float(parts[0])
                            intensity = float(parts[1])
                            ms2_list.append([mz, intensity])
                        except ValueError:
                            # If conversion fails, skip the line
                            continue
    # Create a DataFrame from the records
    return pd.DataFrame(records)





def smiles_to_inchikey(smiles):
    """
    Convert a SMILES string to an InChI Key using RDKit.
    
    Parameters:
        smiles (str): The SMILES representation of the molecule.
    
    Returns:
        str or None: The InChI Key of the molecule, or None if the SMILES string is invalid.
    """
    # Create a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Generate and return the InChI Key
    inchi_key = Chem.MolToInchiKey(mol)
    return inchi_key





def post_process_df(df):
    """
    Process a DataFrame of chemical compound data with structural and metadata fields.

    Transformations:
    - SMILES --> INCHIKEY: Generate unique structure identifiers using `smiles_to_inchikey`.
    - Filter --> Drop rows where INCHIKEY is missing or empty.
    - LIBRARYQUALITY --> Convert to numeric and keep only values [1, 2, 3].
    - INCHIKEY --> INCHIKEY MAIN BLOCK: Extract first 14 characters (connectivity layer).
    - NAME --> ADDUCT: Extract last word from NAME as presumed adduct form (e.g., [M+H]+).

    Parameters:
    df (pandas.DataFrame): Input DataFrame with required columns:
        - 'SMILES': SMILES strings representing chemical structures.
        - 'LIBRARYQUALITY': Quality metric (1, 2, or 3 expected).
        - 'NAME': Compound name, often ending with adduct information.

    Returns:
    pandas.DataFrame: Cleaned and enriched DataFrame.


    Notes:
    - Assumes a helper function `smiles_to_inchikey(smiles)` is available.
    - Helps standardize and filter compound data for downstream cheminformatics tasks.

    Example:
    >>> processed_df = post_process_df(raw_df)
    >>> processed_df[['INCHIKEY', 'INCHIKEY MAIN BLOCK', 'ADDUCT']].head()
    """

    # Convert SMILES to INCHIKEY using the provided function.
    df['INCHIKEY'] = df['SMILES'].apply(smiles_to_inchikey)
    
    # Filter out rows where 'INCHIKEY' is None or empty.
    df = df[df['INCHIKEY'].notna() & (df['INCHIKEY'] != '')]
    
    # Convert LIBRARYQUALITY to a numeric type and filter for values 1, 2, or 3.
    df['LIBRARYQUALITY'] = pd.to_numeric(df['LIBRARYQUALITY'], errors='coerce')
    df = df[df['LIBRARYQUALITY'].isin([1, 2, 3])]
    
    # Create a new column for INCHIKEY MAIN BLOCK: the first 14 characters of 'INCHIKEY'
    df['INCHIKEY MAIN BLOCK'] = df['INCHIKEY'].str[:14]
    
    # Create a new column for ADDUCT: characters after the last space in 'NAME'
    df['ADDUCT'] = df['NAME'].str.split().str[-1]
    
    return df




def clean_column(column):
    # Remove any text outside square brackets, leaving only the inner value.
    return column.astype(str).str.replace(r'^.*\[|\].*$', '', regex=True)




def process_adduct_df(df):
    # First, clean the 'ADDUCT' column.
    df['ADDUCT'] = clean_column(df['ADDUCT'])
    
    # Define the mapping dictionary.
    adduct_mapping = {
        'M+H': '[M+H]+',
        'M-H': '[M-H]-',
        'M+Na': '[M+Na]+',
        'M+HCOO': '[M+HCOO]-',
        'M': '[M]',
        'M+CH3COOH-H': '[M+CH3COOH-H]-',
        'M+CH3COO': '[M+CH3COO]-',
        'M-H1': '[M-H1]-',
        'M-H2O+H': '[M-H2O+H]+',
        'M+NH4': '[M+NH4]+',
        'M-2H': '[M-2H]2-',
        '2M+Na': '[2M+Na]+',
        'M+K': '[M+K]+',
        '2M+H': '[2M+H]+',
        'M-2H2O+H': '[M-2H2O+H]+',
        'M+2H': '[M+2H]2+',
        'M+H-H2O': '[M+H-H2O]+',
        'M+': '[M+]+',
        '2M-H': '[2M-H]-'}
    
    
    # Map the cleaned 'ADDUCT' values using the mapping dictionary.
    df['ADDUCT'] = df['ADDUCT'].map(adduct_mapping)
    
    # Drop rows where the mapping was not found.
    df = df.dropna(subset=['ADDUCT']).reset_index(drop=True)
    
    return df
