import urllib.request
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def decompress_gz(gz_path: Path, output_path: Path):
    """Decompress a .gz file."""
    print(f"Decompressing {gz_path.name}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out, length=16*1024*1024)  # 16MB chunks
    print(f"✓ Created {output_path.name}")

def download_pubchem_data(data_dir: str = "./pubchem_data"):
    """Download and prepare PubChem data files."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    files_to_download = [
        {
            "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz",
            "filename": "CID-Synonym-filtered.gz",
            "decompressed": "CID-Synonym-filtered",
            "description": "Name to CID mapping (filtered, high quality)"
        },
        {
            "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
            "filename": "CID-SMILES.gz",
            "decompressed": "CID-SMILES",
            "description": "CID to SMILES mapping"
        },
        {
            "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Parent.gz",
            "filename": "CID-Parent.gz",
            "decompressed": "CID-Parent",
            "description": "CID to Parent CID mapping"
        },
        {
            "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Title.gz",
            "filename": "CID-Title.gz",
            "decompressed": "CID-Title",
            "description": "CID to preferred name"
        },
        {
            "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/Extras/Reference-Collection.tsv.gz",
            "filename": "Reference-Collection.tsv.gz",
            "decompressed": "Reference-Collection.tsv",
            "description": "SID reference collection (preferred names)"
        },
    ]

    print("="*70)
    print("PubChem Data Downloader")
    print("="*70)
    print(f"\nDownload location: {data_path.absolute()}")
    print(f"\nFiles to download: {len(files_to_download)}")
    print("\n" + "="*70)

    for file_info in files_to_download:
        gz_path = data_path / file_info["filename"]
        decompressed_path = data_path / file_info["decompressed"]

        print(f"\n[{file_info['description']}]")

        # Skip if decompressed file already exists
        if decompressed_path.exists():
            print(f"✓ {file_info['decompressed']} already exists, skipping")
            continue

        # Download compressed file if doesn't exist
        if not gz_path.exists():
            print(f"Downloading {file_info['filename']}...")
            try:
                download_with_progress(file_info["url"], gz_path)
                print(f"✓ Downloaded {file_info['filename']}")
            except Exception as e:
                print(f"✗ Error downloading {file_info['filename']}: {e}")
                continue
        else:
            print(f"✓ {file_info['filename']} already exists, skipping download")

        # Decompress
        if gz_path.exists():
            try:
                decompress_gz(gz_path, decompressed_path)
                # Remove .gz file after decompression to save space
                print(f"  Removing {gz_path.name} to save space...")
                gz_path.unlink()
            except Exception as e:
                print(f"✗ Error decompressing {file_info['filename']}: {e}")
                continue

    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)

    # Check which files are present
    present = []
    missing = []
    for file_info in files_to_download:
        decompressed_path = data_path / file_info["decompressed"]
        if decompressed_path.exists():
            present.append(file_info["decompressed"])
        else:
            missing.append(file_info["decompressed"])

    print(f"\n✓ Present: {len(present)}/{len(files_to_download)}")
    for f in present:
        print(f"  - {f}")

    if missing:
        print(f"\n✗ Missing: {len(missing)}")
        for f in missing:
            print(f"  - {f}")

    print(f"\nFiles location: {data_path.absolute()}")
    print(f"\nEstimated disk space used: ~10-12 GB")
    print("\n" + "="*70)
    print("Next Step:")
    print("="*70)
    print("Run create_pubchem_db() to build the database from downloaded files")
    print("="*70)

def check_files_status(data_dir: str = "./pubchem_data"):
    """Check status of downloaded files."""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"Directory {data_dir} does not exist!")
        return

    print("\nFile Status:")
    print("="*70)

    expected_files = [
        "CID-Synonym-filtered",
        "CID-SMILES",
        "CID-Parent",
        "CID-Title",
        "Reference-Collection.tsv"
    ]

    total_size = 0
    for filename in expected_files:
        filepath = data_path / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024**3)  # GB
            total_size += size
            print(f"✓ {filename:<30} ({size:.2f} GB)")
        else:
            print(f"✗ {filename:<30} (missing)")

    print("="*70)
    print(f"Total disk space: {total_size:.2f} GB")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_files_status()
    else:
        download_pubchem_data()