from rdkit import Chem
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import mgf2csv

gnps = mgf2csv.parse_mgf('GNPS/GNPS-LIBRARY.mgf')
massbank = mgf2csv.parse_mgf('MassBank/MASSBANK.mgf')
mona = mgf2csv.parse_mgf('MONA/MONA.mgf')

post_gnps = mgf2csv.post_process_df(gnps)
post_massbank = mgf2csv.post_process_df(massbank)
post_mona = mgf2csv.post_process_df(mona)

gnps_df = mgf2csv.process_adduct_df(post_gnps)
massbank_df = mgf2csv.process_adduct_df(post_massbank)
mona_df = mgf2csv.process_adduct_df(post_mona)

print("GNPS: {} -> {} -> {}".format(len(gnps), len(post_gnps), len(gnps_df)))
print("MassBank: {} -> {} -> {}".format(len(massbank), len(post_massbank), len(massbank_df)))
print("MONA: {} -> {} -> {}".format(len(mona), len(post_mona), len(mona_df)))

gnps_df.to_csv('GNPS/GNPS.csv')
massbank_df.to_csv('MassBank/MASSBANK.csv')
mona_df.to_csv('MONA/MONA.csv')