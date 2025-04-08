# How to download
download from [hugging face](https://huggingface.co/datasets/Bowen999/lipids_ms2)   

or use Python to use it directly
```
from datasets import load_dataset

# Load each sub-dataset
ms_dial = load_dataset("Bowen999/lipids_ms2", data_dir="ms-dial")
gym = load_dataset("Bowen999/lipids_ms2", data_dir="mass_spec_gym")
gnps_mb_mona = load_dataset("Bowen999/lipids_ms2", data_dir="gnps_massbank_mona")

# Convert to pandas DataFrames
ms_dial_df = ms_dial['train'].to_pandas()
gym_df = gym['train'].to_pandas()
gnps_df = gnps_mb_mona['train'].to_pandas()
``` 




# Data source
## MS DIAL (only lipidome atlas) 
* Source: [Supplementary information](https://www.nature.com/articles/s41587-020-0531-2#Sec17) of Tsugawa, H., Ikeda, K., Takahashi, M. et al. A lipidome atlas in MS-DIAL 4. Nat Biotechnol 38, 1159–1163 (2020). https://doi.org/10.1038/s41587-020-0531-2
* Intruments: TOF
* Complete MS DIAL DB (not only lipids): https://zenodo.org/records/10953284


## MassSpecGym:
* Source: https://huggingface.co/datasets/roman-bushuiev/MassSpecGym
* Instruments: TOF, Orbitrap

## GNPS
* Source: https://external.gnps2.org/gnpslibrary

## MassBank
* Source: https://external.gnps2.org/gnpslibrary

## PNNL-LIPIDS
* Source: https://external.gnps2.org/gnpslibrary

## MONA
* Source: https://external.gnps2.org/gnpslibrary

## Li Lab Lipid Standrad (not open)
Source: Li Lab


