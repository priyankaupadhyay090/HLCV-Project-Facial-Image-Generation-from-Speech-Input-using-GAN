from pathlib import Path
from tqdm import tqdm
import pickle

cwd = Path(__file__).resolve().parent
celeba_dir = cwd / 'mmca' / 'celeba-caption'
output_dir = cwd / 'mmca' / 'captions_pickles'

try:
    output_dir.mkdir(parents=True, exist_ok=False)
except FileExistsError:
    print(f'{output_dir} is already there')
else:
    print(f'{output_dir} created to store caption pickle files')


def process_file(infile):
    captions = []
    with open(infile, 'r') as f:
        for line in f:
            captions.append(line.lower().rstrip())
    return captions


def main():
    idx = 0
    for file in tqdm(celeba_dir.iterdir(), desc="caption file"):
        list_of_texts = process_file(file)
        out_filename = output_dir / f"{str(file.stem)}.pickle"
        pickle.dump(list_of_texts, open(out_filename, 'wb'))
        idx += 1

    print(f'Processed {idx} files to pickle files.')

    """ 
    for file in output_dir.iterdir():
        list_of_texts = pickle.load(open(file, 'rb'))
        print(list_of_texts)
    """


if __name__ == '__main__':
    main()
