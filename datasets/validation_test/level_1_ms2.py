from tqdm import tqdm

def filter_mgf_by_library_quality(input_path, output_path, required_quality=1, max_ions=None):
    # First, count total blocks for tqdm
    with open(input_path, 'r') as infile:
        total_blocks = sum(1 for line in infile if line.strip() == 'BEGIN IONS')

    kept_count = 0
    block_count = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile, tqdm(total=total_blocks, desc="Processing IONS") as pbar:
        keep_block = False
        current_block = []

        for line in infile:
            if line.strip() == 'BEGIN IONS':
                current_block = [line]
                keep_block = False
                block_count += 1
            elif line.strip() == 'END IONS':
                current_block.append(line)
                if keep_block:
                    outfile.writelines(current_block)
                    outfile.write("\n\n")  # Two blank lines between blocks
                    kept_count += 1
                    if max_ions is not None and kept_count >= max_ions:
                        break
                pbar.update(1)
            else:
                current_block.append(line)
                if line.startswith('LIBRARYQUALITY='):
                    try:
                        quality = int(line.strip().split('=')[1])
                        if quality == required_quality:
                            keep_block = True
                    except ValueError:
                        continue

    print(f"\n✅ Number of IONS kept with LIBRARYQUALITY={required_quality}: {kept_count}")




# Example usage
filter_mgf_by_library_quality('/Users/bowen/Desktop/DeepLipid/datasets/GNPS/GNPS-LIBRARY.mgf', '/Users/bowen/Desktop/DeepLipid/datasets/validation_test/GNPS_level_1.mgf', max_ions=100)
