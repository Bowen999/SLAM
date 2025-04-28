## Download
training_set, test_set, test_set2, can be downloaded from: https://huggingface.co/datasets/Bowen999/lipids_ms2/tree/main

## Data Source
- **Training Set**:  
  The training set contains 959,031 entries of lipid MS/MS spectra and corresponding structural information.  
  It is constructed by integrating data from multiple sources, including **MS DIAL Lipidome Atlas**, **GNPS**, **MassBank**, **MoNA**, **PNNL-Lipids**, **HCE**, and **IOBA_NHC**.

- **Test Set**:  
  The test set is sourced from **MassSpecGym**, containing 4,631 MS/MS spectra.  
  Among them, 1,372 spectra are **novel** (not present in the training set).

- **Test Set2**:  
MS2 data of standards from Li Lab and Nova Medical Testing (non-public data)"

*Preprocessing for **comprehensive MS2 databases** (like MassSpecGym) involved retaining only lipids by filtering the **InChI Key main block** using references from the MS DIAL Lipidome Atlas and Swiss Lipids*.


## Dataset Description
| Column # | Column Name     | Description |
|:---------|:----------------|:------------|
| 1 | **name** | Full lipid name including class and detailed acyl-chain notation (e.g. “PC 18:1/16:0”) |
| 2 | **simple_name** | Simplified lipid name summarizing total carbons and double bonds (e.g. “PC 34:1”) |
| 3 | **class** | Lipid class abbreviation (e.g. PC, PE, TAG) |
| 4 | **chain** | Acyl-chain composition string (e.g. “18:1/16:0”) |
| 5 | **num_c_1** | Number of carbons in chain 1; **Note: `num_c_1` can be equal to 0, but the `num_c` values for all four chains cannot all be 0 at the same time.**|
| 3 | **class** | Lipid class abbreviation (e.g. PC, PE, TAG) | 
| 6 | **num_db_1** | Number of C=C in chain 1|
| 7 | **extra_1** | Extra modifications on chain 1 (e.g. (2OH)), if not present, then None |
| 8 | **num_c_2** | Number of carbons in chain 2 |
| 9 | **num_db_2** | Number of C=C in chain 2 |
| 10 | **extra_2** | Extra modifications on chain 2 |
| 11 | **num_c_3** | Number of carbons in chain 3 |
| 12 | **num_db_3** | Number of C=C in chain 3 |
| 13 | **extra_3** | Extra modifications on chain 3 |
| 14 | **num_c_4** | Number of carbons in chain 4 |
| 15 | **num_db_4** | Number of C=C in chain 4 |
| 16 | **extra_4** | Extra modifications on chain 4 |
| 17 | **precursor_mz** | Mass-to-charge ratio (m/z) of the precursor ion |
| 18 | **adduct** | Adduct ion type detected (e.g. [M+H]+, [M+Na]+) |
| 19 | **ion_mode** | Ionization mode (Positive or Negative) |
| 20 | **charge** | Observed charge state of the ion |
| 21 | **smiles** | SMILES string representation of the molecule |
| 22 | **inchi** | Full IUPAC InChI identifier |
| 23 | **inchikey** | InChIKey |
| 24 | **inchikey_main** | Main layer (first 14 characters) of the InChIKey |
| 25 | **exact_mass** | Theoretical monoisotopic neutral mass |
| 26 | **formula** | Chemical formula (e.g. C₃₆H₇₀O₈P) |
| 27 | **synonyms** | Alternative names |
| 28 | **retention_time** | Liquid chromatographic retention time |
| 29 | **lib_quality** | Library match quality score or confidence (1 is best) |
| 30 | **source** | Origin of the entry (database name) |
| 31 | **spectrum_id** | Unique identifier for the MS/MS spectrum in source database |
| 32 | **ms_level** | MS level of the spectrum |
| 33 | **instrument** | Mass spectrometer used |
| 34 | **energy** | Collision/fragmentation energy applied (e.g. 40 eV) |
| 35 | **num_peaks** | Number of peaks recorded in the MS2 spectrum |
| 36 | **MS2** | fortmat: [[mz1, int1], [mz2, int2]...], raw MS2 spectrum, “raw” refers to data taken directly from the source; however, **the MS2 in that source may already have been normalized ** |
| 37 | **MS2_norm** | Normalized MS2 intensities, Intensities are scaled to 0–100; if there are more than 100 peaks, only the 100 highest-intensity peaks are retained. |
| 38 | **novel** | Flag indicating novel lipid entry (y or n); **Only test set have this column**|





