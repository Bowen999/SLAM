## Download
training_set, test_set, test_set2, can be downloaded from: https://huggingface.co/datasets/Bowen999/lipids_ms2/tree/main

## Data Source

- **Training Set**(train_set_no_sn.csv):  
  The training set contains 959,031 entries of lipid MS/MS spectra and corresponding structural information.  
  It is constructed by integrating data from multiple sources, including **MS DIAL Lipidome Atlas**, **GNPS**, **MassBank**, **MoNA**, **PNNL-Lipids**, **HCE**, and **IOBA_NHC**.

- **Test Set**(test_set_no_sn.csv):  
  The test set is sourced from **MassSpecGym**, containing 4,631 MS/MS spectra.  
  Among them, 1,372 spectra are **novel** (not present in the training set).

- **Test Set2**:  
MS2 data of standards from Li Lab and Nova Medical Testing (non-public data)"
*Preprocessing for **comprehensive MS2 databases** (like MassSpecGym) involved retaining only lipids by filtering the **InChI Key main block** using references from the MS DIAL Lipidome Atlas and Swiss Lipids*.


## Dataset Description
| Column  | Column Name | Description |
|:---|:---|:---|
| 1 | **index** | A unique index for each spectrum. |
| 2 | **name** | Full lipid name including class and detailed acyl-chain notation (e.g. “PC 18:1/16:0”). |
| 3 | **simple_name** | Simplified lipid name summarizing total carbons and double bonds (e.g. “PC 34:1”). |
| 4 | **class** | Lipid class abbreviation (e.g. PC, PE, TAG). |
| 5 | **chain** | Acyl-chain composition string (e.g. “18:1/16:0”). |
| 6 | **num_chain** | The total number of acyl chains attached to the lipid backbone. |
| 7 | **num_c** | The total number of carbon atoms in all acyl chains combined. |
| 8 | **num_db** | The total number of carbon-carbon double bonds (C=C) in all acyl chains combined. |
| 9 | **num_c_1** | Number of carbons in chain 1. **Note:** `num_c_1` can be 0, but the `num_c` values for all four chains cannot all be 0 simultaneously. |
| 10 | **num_db_1** | Number of C=C double bonds in chain 1. |
| 11 | **num_c_2** | Number of carbons in chain 2. |
| 12 | **num_db_2** | Number of C=C double bonds in chain 2. |
| 13 | **num_c_3** | Number of carbons in chain 3. |
| 14 | **num_db_3** | Number of C=C double bonds in chain 3. |
| 15 | **num_c_4** | Number of carbons in chain 4. |
| 16 | **num_db_4** | Number of C=C double bonds in chain 4. |
| 17 | **precursor_mz** | Mass-to-charge ratio (m/z) of the precursor ion. |
| 18 | **adduct** | Adduct ion type detected (e.g. [M+H]+, [M+Na]+). |
| 19 | **ion_mode** | Ionization mode (Positive or Negative). |
| 20 | **charge** | Observed charge state of the ion. |
| 21 | **smiles** | SMILES string representation of the molecule. |
| 22 | **inchi** | Full IUPAC InChI identifier. |
| 23 | **inchikey** | InChIKey. |
| 24 | **inchikey_main** | Main layer (first 14 characters) of the InChIKey. |
| 25 | **exact_mass** | Theoretical monoisotopic neutral mass. |
| 26 | **formula** | Chemical formula (e.g. C₃₆H₇₀O₈P). |
| 27 | **synonyms** | Alternative names. |
| 28 | **retention_time** | Liquid chromatographic retention time. |
| 29 | **lib_quality** | Library match quality score or confidence (1 is best). |
| 30 | **source** | Origin of the entry (database name). |
| 31 | **spectrum_id** | Unique identifier for the MS/MS spectrum in the source database. |
| 32 | **ms_level** | MS level of the spectrum. |
| 33 | **instrument** | Mass spectrometer used. |
| 34 | **energy** | Collision/fragmentation energy applied (e.g. 40 eV). |
| 35 | **num_peaks** | Number of peaks recorded in the MS2 spectrum. |
| 36 | **MS2** | Raw MS2 spectrum in the format: `[[mz1, int1], [mz2, int2]...]`. "Raw" refers to data from the source, which may have already been normalized. |
| 37 | **MS2_norm** | Normalized MS2 intensities, scaled from 0–100. If more than 100 peaks, only the 100 highest-intensity peaks are retained. |
| 38 | **MS2_frag_formula** | The chemical formula for each fragment in the MS2 spectrum, corresponding to the peaks in `MS2_norm`. |
| 39 | **ref** | Chain information ignoring Sn position. Format: `[total_carbons, total_double_bonds, c_chain1, db_chain1, c_chain2, db_chain2, ...]`. Example: `[38, 1, 18, 1, 20, 0]` |
=======
