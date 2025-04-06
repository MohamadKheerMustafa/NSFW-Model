from pathlib import Path
from app.model import reorganize_dataset

def main():
    dataset_dir = Path(__file__).resolve().parent / 'dataset'
    print(f"Reorganizing dataset in: {dataset_dir}")
    reorganize_dataset(dataset_dir)
    print('Dataset reorganization completed successfully!')

if __name__ == '__main__':
    main()