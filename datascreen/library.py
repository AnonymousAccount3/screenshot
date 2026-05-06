"""Drug library management and configuration."""
import os
import json
import math
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
import warnings

import pandas as pd
import numpy as np

from .drug_graph import incremental_merge_drugs, load_graph_from_json
from .drug_lookup import extract_drug_info
from .pubchem_download import download_pubchem_data
from .pubchem_database import PubChemLocalDatabase, create_pubchem_db


def extract_drug_names(
    df: pd.DataFrame,
    drug_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract unique drug names from a DataFrame.

    Args:
        df: DataFrame containing drug columns
        drug_columns: List of column names to extract drugs from.
            If None, auto-detects columns named drug1, drug2, drug3, ...

    Returns:
        Sorted list of unique, non-null drug names
    """
    if drug_columns is None:
        drug_columns = []
        i = 1
        while f"drug{i}" in df.columns:
            drug_columns.append(f"drug{i}")
            i += 1

    if not drug_columns:
        raise ValueError("No drug columns found in DataFrame")

    drug_set = set()
    for col in drug_columns:
        drug_set.update(df[col].dropna().unique())
    drug_set.discard("")
    drug_set.discard("nan")

    return sorted(drug_set)


@dataclass
class DrugLibraryConfig:
    """Configuration for drug library settings (not location)."""

    graph_file: str = "drug_graph.json"
    remap_file: str = "drug_id_remap.csv"
    drugs_file: str = "all_drugs.csv"
    pubchem_cache_dir: Optional[Path] = None
    use_local_pubchem: bool = False
    use_pubchem_api: bool = True

    def __post_init__(self):
        """Ensure paths are Path objects and initialize PubChem if needed."""
        if self.pubchem_cache_dir:
            self.pubchem_cache_dir = Path(self.pubchem_cache_dir)
            self.use_local_pubchem = True

            # Check if PubChem database needs to be set up
            if not self._pubchem_db_exists():
                print(f"\n{'='*70}")
                print("PubChem Database Setup Required")
                print(f"{'='*70}")
                print(f"Local PubChem database not found at: {self.pubchem_cache_dir}")
                print("This is a one-time setup that will:")
                print("  1. Download ~15-20 GB of PubChem data")
                print("  2. Build a local SQLite database for fast lookups")
                print("  3. Take approximately 30-60 minutes depending on connection")
                print(f"{'='*70}\n")

                response = input("Would you like to download and set up PubChem now? (yes/no): ")
                if response.lower() in ['yes', 'y']:
                    self._setup_pubchem_database()
                else:
                    print("\nSkipping PubChem setup. You can set it up later by running:")
                    print(f"  from datascreen import download_pubchem_data, create_pubchem_db")
                    print(f"  download_pubchem_data('{self.pubchem_cache_dir}')")
                    print(f"  create_pubchem_db('{self.pubchem_cache_dir}')")
                    print("\nFalling back to PubChem API for drug lookups.")
                    self.use_local_pubchem = False

    def _pubchem_db_exists(self) -> bool:
        """Check if PubChem database exists."""
        if not self.pubchem_cache_dir:
            return False

        db_path = self.pubchem_cache_dir / "pubchem.db"
        return db_path.exists()

    def _setup_pubchem_database(self):
        """Download PubChem data and create database."""
        print("\nStep 1: Downloading PubChem data files...")
        print(f"{'='*70}\n")

        try:
            download_pubchem_data(data_dir=str(self.pubchem_cache_dir))
        except Exception as e:
            print(f"\n✗ Error downloading PubChem data: {e}")
            print("Falling back to PubChem API for drug lookups.")
            self.use_local_pubchem = False
            return

        print(f"\n{'='*70}")
        print("Step 2: Building PubChem database...")
        print(f"{'='*70}\n")

        try:
            create_pubchem_db(data_dir=str(self.pubchem_cache_dir))
            print(f"\n{'='*70}")
            print("✓ PubChem database setup complete!")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"\n✗ Error creating PubChem database: {e}")
            print("Falling back to PubChem API for drug lookups.")
            self.use_local_pubchem = False

    def get_graph_path(self, library_dir: Path) -> Path:
        """Get full path to drug graph file."""
        return Path(library_dir) / self.graph_file

    def get_remap_path(self, library_dir: Path) -> Path:
        """Get full path to drug ID remap file."""
        return Path(library_dir) / self.remap_file

    def get_drugs_path(self, library_dir: Path) -> Path:
        """Get full path to all drugs file."""
        return Path(library_dir) / self.drugs_file

    def exists(self, library_dir: Path) -> bool:
        """Check if library files exist in directory."""
        library_dir = Path(library_dir)
        return (
            self.get_graph_path(library_dir).exists() and
            self.get_remap_path(library_dir).exists() and
            self.get_drugs_path(library_dir).exists()
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'graph_file': self.graph_file,
            'remap_file': self.remap_file,
            'drugs_file': self.drugs_file,
            'pubchem_cache_dir': str(self.pubchem_cache_dir) if self.pubchem_cache_dir else None,
            'use_local_pubchem': self.use_local_pubchem,
            'use_pubchem_api': self.use_pubchem_api,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DrugLibraryConfig':
        """Create from dictionary."""
        config_dict = config_dict.copy()
        if config_dict.get('pubchem_cache_dir'):
            config_dict['pubchem_cache_dir'] = Path(config_dict['pubchem_cache_dir'])
        return cls(**config_dict)

    def save_config(self, library_dir: Path):
        """Save configuration to JSON file in library directory."""
        library_dir = Path(library_dir)
        library_dir.mkdir(parents=True, exist_ok=True)

        config_path = library_dir / "library_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Library config saved to: {config_path}")

    @classmethod
    def load_config(cls, library_dir: Path) -> 'DrugLibraryConfig':
        """Load configuration from library directory."""
        config_path = Path(library_dir) / "library_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_env(cls, env_var: str = "PUBCHEM_CACHE_DIR") -> 'DrugLibraryConfig':
        """Create configuration from environment variable."""
        pubchem_dir = os.environ.get(env_var)
        if pubchem_dir:
            return cls(pubchem_cache_dir=Path(pubchem_dir))
        else:
            return cls()


class DrugLibrary:
    """
    Drug library - analogous to a tokenizer for NLP models.
    Maps drug names to consensus IDs (vocabulary indices).
    """

    def __init__(self,
                 config: Optional[DrugLibraryConfig] = None,
                 library_dir: Optional[Path] = None,
                 drug_id_remap: Optional[Dict] = None,
                 drug_graph = None,
                 all_drugs: Optional[pd.DataFrame] = None):
        """
        Initialize drug library.

        Args:
            config: Library configuration (settings). If None, uses default config.
            library_dir: Directory where library is/will be stored
            drug_id_remap: Pre-loaded drug ID mapping
            drug_graph: Pre-loaded drug graph
            all_drugs: Pre-loaded drugs dataframe
        """
        self.config = config if config is not None else DrugLibraryConfig()
        self.library_dir = Path(library_dir) if library_dir else None
        self._drug_id_remap = drug_id_remap
        self._drug_graph = drug_graph
        self._all_drugs = all_drugs

        # Lazy load if library_dir provided but data not loaded
        if library_dir and drug_id_remap is None:
            self._loaded = False
        else:
            self._loaded = drug_id_remap is not None

        self._unknown_drug_ids = {}
        self._next_unknown_id = None
        self._n_missing_tokens = 0  # set via set_missing_tokens()

    def _ensure_loaded(self):
        """Lazy load library data if not already loaded."""
        if not self._loaded and self.library_dir:
            self._drug_graph, self._drug_id_remap, self._all_drugs = self._load_from_disk()
            self._loaded = True

    def _load_from_disk(self) -> Tuple:
        """Load drug library files from disk."""
        if not self.library_dir:
            raise ValueError("No library_dir specified")

        if not self.config.exists(self.library_dir):
            raise FileNotFoundError(
                f"Drug library not found at {self.library_dir}. "
                "Initialize with DrugLibrary.initialize() first."
            )

        print(f"Loading drug library from: {self.library_dir}")

        drug_graph = load_graph_from_json(self.config.get_graph_path(self.library_dir))
        drug_id_remap_df = pd.read_csv(self.config.get_remap_path(self.library_dir)).replace(np.nan, None)
        all_drugs = pd.read_csv(self.config.get_drugs_path(self.library_dir))

        drug_id_remap = drug_id_remap_df.set_index("original_id")["consensus_id"].to_dict()

        print(f"  Loaded {len(drug_id_remap)} drug mappings")
        print(f"  Vocabulary size: {len(set(drug_id_remap.values()))}")

        return drug_graph, drug_id_remap, all_drugs

    @property
    def drug_id_remap(self) -> Dict:
        """Get drug ID remap dictionary."""
        self._ensure_loaded()
        return self._drug_id_remap

    @property
    def drug_graph(self):
        """Get drug graph."""
        self._ensure_loaded()
        return self._drug_graph

    @property
    def all_drugs(self) -> pd.DataFrame:
        """Get all drugs dataframe."""
        self._ensure_loaded()
        return self._all_drugs

    @property
    def vocab_size(self) -> int:
        """Minimum embedding size needed for all consensus IDs.

        Returns max(consensus_id) + 1 so that every ID is a valid
        embedding index.  IDs 0, 1, 2 are reserved for padding and
        CLS tokens; real drug IDs start at 3.
        """
        self._ensure_loaded()
        if not self._drug_id_remap:
            return 3  # only the reserved slots
        return max(self._drug_id_remap.values()) + 1

    def set_missing_tokens(self, n_missing_tokens: int):
        """
        Configure the number of missing drug token slots.

        When set, unknown drugs are assigned IDs in the range
        [vocab_size+1, vocab_size+n_missing_tokens] using a deterministic
        hash, so the same unknown drug always gets the same token ID.

        Args:
            n_missing_tokens: Number of reserved missing token IDs (e.g. 100)
        """
        self._n_missing_tokens = n_missing_tokens
        # Clear any previously assigned unknown IDs
        self._unknown_drug_ids = {}
        self._next_unknown_id = None

    def encode(self, drug_name: str) -> int:
        """Encode drug name to consensus ID."""
        self._ensure_loaded()

        if drug_name == "" or pd.isnull(drug_name):
            return 0

        cid = self._drug_id_remap.get(drug_name)
        if cid is not None:
            return cid

        # Assign stable unknown ID per drug name
        if drug_name not in self._unknown_drug_ids:
            if self._n_missing_tokens > 0:
                # Hash into the missing token range [vocab_size, vocab_size+n_missing_tokens)
                # vocab_size = max_id + 1, so it's the first free slot
                token_offset = hash(drug_name) % self._n_missing_tokens
                unknown_id = self.vocab_size + token_offset
            else:
                # Fallback: incrementing IDs above vocab_size
                if self._next_unknown_id is None:
                    self._next_unknown_id = self.vocab_size + 1
                unknown_id = self._next_unknown_id
                self._next_unknown_id += 1

            self._unknown_drug_ids[drug_name] = unknown_id
            warnings.warn(
                f"Unknown drug '{drug_name}' encountered. "
                f"Assigning ID {unknown_id} (> vocab_size).",
                RuntimeWarning,
                stacklevel=2,
            )

        return self._unknown_drug_ids[drug_name]

    def batch_encode(self, drug_names: List[str]) -> List[int]:
        """Encode multiple drug names."""
        return [self.encode(name) for name in drug_names]

    def decode(self, consensus_id: int) -> List[str]:
        """Decode consensus ID to list of possible drug names."""
        self._ensure_loaded()
        return [
            drug_name for drug_name, cid in self._drug_id_remap.items()
            if cid == consensus_id
        ]

    def get_canonical_name(self, drug_name: str) -> Optional[str]:
        """Get canonical (PubChem) name for a drug."""
        self._ensure_loaded()
        consensus_id = self.encode(drug_name)

        # Find any drug with this consensus ID that has a pubchem_name
        drug_names = self.decode(consensus_id)
        if drug_names:
            matching_drugs = self._all_drugs[self._all_drugs['name'].isin(drug_names)]
            pubchem_names = matching_drugs['pubchem_name'].dropna()
            if len(pubchem_names) > 0:
                return pubchem_names.iloc[0]

        return None

    def initialize(self, drug_names: List[str]):
        """
        Initialize library from a list of drug names.

        Looks up each drug via PubChem, builds the drug graph, and assigns
        consensus IDs. Updates library state in-place.
        Call save_pretrained() to persist changes.

        Args:
            drug_names: List of drug names to include in the library.
                Use extract_drug_names() to get these from a DataFrame.
        """
        print(f"Initializing drug library with {len(drug_names)} drugs")

        # Determine PubChem settings from config
        local_pubchem = str(self.config.pubchem_cache_dir) if self.config.use_local_pubchem else None
        use_pubchem_api = self.config.use_pubchem_api

        if local_pubchem:
            print(f"Using local PubChem database at: {local_pubchem}")
        elif use_pubchem_api:
            print("Using PubChem API for drug lookups")
        else:
            print("⚠️  Warning: Neither local PubChem nor API enabled. Drug info may be incomplete.")

        # Extract drug info via PubChem
        df_drug = extract_drug_info(
            drug_names,
            local_pubchem=local_pubchem,
            use_puchem_api=use_pubchem_api
        )
        df_drug = df_drug.loc[~df_drug.name.isna()]

        # Build drug graph and assign consensus IDs
        drug_graph, drug_id_remap, _ = incremental_merge_drugs(
            df_drug, None, None
        )

        # Update internal state
        self._drug_graph = drug_graph
        self._drug_id_remap = drug_id_remap
        self._all_drugs = df_drug.drop_duplicates().reset_index(drop=True)
        self._loaded = True

        print(f"\n✓ Drug library initialized in memory")
        print(f"✓ Vocabulary size: {self.vocab_size}")
        print(f"✓ Total unique drugs: {len(self._all_drugs)}")
        print(f"\nCall save_pretrained(save_path) to persist the library")

    def update(self, drug_names: List[str]):
        """
        Update library with new drug names.

        Only looks up drugs not already in the library. Updates library
        state in-place. Call save_pretrained() to persist changes.

        Args:
            drug_names: List of drug names to add.
                Use extract_drug_names() to get these from a DataFrame.
        """
        print(f"Updating drug library with {len(drug_names)} drug names")

        # Ensure current data is loaded
        self._ensure_loaded()

        # Filter out drugs already in library
        existing_drugs = set(self._drug_id_remap.keys())
        new_drugs = set(drug_names) - existing_drugs

        print(f"  {len(set(drug_names) & existing_drugs)} drugs already in library")
        print(f"  {len(new_drugs)} new drugs to process")

        if len(new_drugs) == 0:
            print(f"\n✓ No new drugs found. Library unchanged.")
            return

        # Determine PubChem settings from config
        local_pubchem = str(self.config.pubchem_cache_dir) if self.config.use_local_pubchem else None
        use_pubchem_api = self.config.use_pubchem_api

        # Extract drug info only for new drugs
        df_drug = extract_drug_info(
            new_drugs,
            local_pubchem=local_pubchem,
            use_puchem_api=use_pubchem_api
        )
        df_drug = df_drug.loc[~df_drug.name.isna()]

        # Merge without updating existing consensus IDs
        drug_graph, drug_id_remap, consensus_id_is_new = incremental_merge_drugs(
            df_drug,
            self._drug_graph,
            self._drug_id_remap,
            update_consensus_ids=False,
        )

        # Count new drugs
        new_drug_count = sum(consensus_id_is_new.values())
        print(f"  Added {new_drug_count} new consensus IDs")

        # Combine with existing drugs
        all_drugs = pd.concat([self._all_drugs, df_drug], ignore_index=True)
        all_drugs = all_drugs.drop_duplicates().reset_index(drop=True)

        # Update internal state
        self._drug_graph = drug_graph
        self._drug_id_remap = drug_id_remap
        self._all_drugs = all_drugs

        print(f"\n✓ Drug library updated in memory")
        print(f"✓ New vocabulary size: {self.vocab_size}")
        print(f"✓ Total unique drugs: {len(self._all_drugs)}")
        print(f"\nCall save_pretrained(save_path) to persist the changes")

    def get_morgan_fingerprints(
        self,
        drug_names: List[str],
        radius: int = 2,
        n_bits: int = 512,
    ) -> np.ndarray:
        """
        Get Morgan fingerprints for a list of drug names using SMILES from the library.

        Args:
            drug_names: List of drug names (must be in the library)
            radius: Morgan fingerprint radius (default: 2, equivalent to ECFP4)
            n_bits: Number of bits in fingerprint (default: 512)

        Returns:
            numpy array of shape (n_drugs, n_bits). Rows for drugs without
            SMILES are filled with NaN.
        """
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        self._ensure_loaded()

        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

        # Build name → SMILES lookup from all_drugs table
        smiles_lookup = {}
        if self._all_drugs is not None and 'smiles' in self._all_drugs.columns:
            for _, row in self._all_drugs.iterrows():
                name = row.get('name')
                smiles = row.get('smiles')
                if name and not pd.isnull(name) and smiles and not pd.isnull(smiles):
                    smiles_lookup[name] = smiles

        fingerprints = []
        for name in drug_names:
            smiles = smiles_lookup.get(name)
            if smiles is None:
                # Try to find via consensus ID (drug may be under a different alias)
                cid = self._drug_id_remap.get(name)
                if cid is not None:
                    aliases = self.decode(cid)
                    for alias in aliases:
                        smiles = smiles_lookup.get(alias)
                        if smiles is not None:
                            break

            if smiles is not None:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = morgan_gen.GetFingerprint(mol)
                    fingerprints.append(np.array(fp))
                else:
                    warnings.warn(f"Could not parse SMILES for '{name}': {smiles}")
                    fingerprints.append(np.full(n_bits, np.nan))
            else:
                fingerprints.append(np.full(n_bits, np.nan))

        return np.array(fingerprints)

    def save_pretrained(self, save_path: Path):
        """
        Save library to disk.

        Args:
            save_path: Directory path where library will be saved
        """
        self._ensure_loaded()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving drug library to: {save_path}")

        # Save graph
        graph_path = self.config.get_graph_path(save_path)
        graph_data = {
            "nodes": [
                self._json_safe({"id": node, **data})
                for node, data in self._drug_graph.nodes(data=True)
            ],
            "edges": [
                self._json_safe({"source": u, "target": v, **data})
                for u, v, data in self._drug_graph.edges(data=True)
            ]
        }
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        print(f"  Saved graph: {graph_path}")

        # Save remap
        remap_df = pd.Series(self._drug_id_remap).reset_index()
        remap_df.columns = ["original_id", "consensus_id"]
        remap_path = self.config.get_remap_path(save_path)
        remap_df.to_csv(remap_path, index=False)
        print(f"  Saved remap: {remap_path}")

        # Save drugs
        drugs_path = self.config.get_drugs_path(save_path)
        self._all_drugs.to_csv(drugs_path, index=False)
        print(f"  Saved drugs: {drugs_path}")

        # Save config
        self.config.save_config(save_path)

        # Update internal library_dir reference
        self.library_dir = save_path

        print(f"\n✓ Drug library saved to: {save_path}")

    @staticmethod
    def _json_safe(obj):
        """Convert object to JSON-safe format."""
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: DrugLibrary._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [DrugLibrary._json_safe(v) for v in obj]
        return obj

    def to_serializable(self) -> Dict:
        """Convert library to a serializable dictionary for model checkpoints."""
        self._ensure_loaded()

        return {
            'drug_id_remap': self._drug_id_remap,
            'vocab_size': self.vocab_size,
            'num_drugs': len(self._all_drugs) if self._all_drugs is not None else 0,
            'library_config': self.config.to_dict(),
        }

    @classmethod
    def from_serializable(cls, data: Dict) -> 'DrugLibrary':
        """Create library from serialized dictionary."""
        config = DrugLibraryConfig.from_dict(data['library_config'])

        return cls(
            config=config,
            library_dir=None,  # No directory for in-memory library
            drug_id_remap=data['drug_id_remap'],
            drug_graph=None,  # Can be loaded later if needed
            all_drugs=None,   # Can be loaded later if needed
        )

    @classmethod
    def from_pretrained(cls, library_dir: Path) -> 'DrugLibrary':
        """
        Load library from directory.

        Args:
            library_dir: Directory containing the saved library

        Returns:
            DrugLibrary instance
        """
        library_dir = Path(library_dir)

        # Try to load config
        try:
            config = DrugLibraryConfig.load_config(library_dir)
        except FileNotFoundError:
            print(f"⚠️  No config found in {library_dir}, using default config")
            config = DrugLibraryConfig()

        return cls(config=config, library_dir=library_dir)

    def get_stats(self) -> Dict:
        """Get library statistics."""
        if not self._loaded:
            if self.library_dir and self.config.exists(self.library_dir):
                self._ensure_loaded()
            else:
                return {"status": "not_initialized"}

        pubchem_info = "not configured"
        if self.config.use_local_pubchem:
            pubchem_info = f"local ({self.config.pubchem_cache_dir})"
        elif self.config.use_pubchem_api:
            pubchem_info = "API"
        else:
            pubchem_info = "disabled"

        return {
            "status": "initialized",
            "library_dir": str(self.library_dir) if self.library_dir else "in-memory",
            "vocab_size": self.vocab_size,
            "num_unique_drugs": len(self._all_drugs) if self._all_drugs is not None else 0,
            "num_edges": len(self._drug_graph.edges()) if self._drug_graph is not None else 0,
            "pubchem_source": pubchem_info,
        }

    def copy(self) -> 'DrugLibrary':
        """
        Create a deep copy of the drug library.

        Returns:
            New DrugLibrary instance with copied data
        """
        import copy

        self._ensure_loaded()

        # Deep copy the data structures
        new_drug_id_remap = copy.deepcopy(self._drug_id_remap)
        new_drug_graph = self._drug_graph.copy() if self._drug_graph is not None else None
        new_all_drugs = self._all_drugs.copy() if self._all_drugs is not None else None
        new_config = DrugLibraryConfig.from_dict(self.config.to_dict())

        # Create new instance
        new_library = DrugLibrary(
            config=new_config,
            library_dir=None,  # Don't link to same directory
            drug_id_remap=new_drug_id_remap,
            drug_graph=new_drug_graph,
            all_drugs=new_all_drugs
        )

        # Copy unknown drug tracking
        new_library._unknown_drug_ids = copy.deepcopy(self._unknown_drug_ids)
        new_library._next_unknown_id = self._next_unknown_id

        return new_library

    def __repr__(self):
        status = f"vocab_size={self.vocab_size}" if self._loaded else "not loaded"
        location = f", dir={self.library_dir}" if self.library_dir else ""
        return f"DrugLibrary({status}{location})"