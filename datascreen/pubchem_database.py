import sqlite3
from pathlib import Path
import gzip
from typing import Optional, List, Tuple, Any


class PubChemLocalDatabase:
    """Local PubChem database built from bulk downloads."""

    TABLES = {
        'compounds': """
            CREATE TABLE IF NOT EXISTS compounds (
                cid INTEGER PRIMARY KEY,
                canonical_smiles TEXT
            )
        """,
        'compound_synonyms': """
            CREATE TABLE IF NOT EXISTS compound_synonyms (
                cid INTEGER,
                synonym TEXT,
                position INTEGER,
                PRIMARY KEY (cid, synonym)
            )
        """,
        'parent_compounds': """
            CREATE TABLE IF NOT EXISTS parent_compounds (
                cid INTEGER PRIMARY KEY,
                parent_cid INTEGER
            )
        """,
        'compound_titles': """
            CREATE TABLE IF NOT EXISTS compound_titles (
                cid INTEGER PRIMARY KEY,
                title TEXT,
                title_length INTEGER
            )
        """,
        'reference_collection': """
            CREATE TABLE IF NOT EXISTS reference_collection (
                sid TEXT PRIMARY KEY,
                preferred_name TEXT,
                preferred_name_length INTEGER
            )
        """,
        'substance_synonyms': """
            CREATE TABLE IF NOT EXISTS substance_synonyms (
                sid TEXT,
                synonym TEXT,
                position INTEGER,
                PRIMARY KEY (sid, synonym)
            )
        """
    }

    INDICES = {
        'idx_synonym_nocase': 'CREATE INDEX IF NOT EXISTS idx_synonym_nocase ON compound_synonyms(synonym COLLATE NOCASE)',
        'idx_cid_position': 'CREATE INDEX IF NOT EXISTS idx_cid_position ON compound_synonyms(cid, position)',
        'idx_title_nocase': 'CREATE INDEX IF NOT EXISTS idx_title_nocase ON compound_titles(title COLLATE NOCASE)',
        'idx_preferred_name': 'CREATE INDEX IF NOT EXISTS idx_preferred_name ON reference_collection(preferred_name)',
        'idx_substance_synonym_nocase': 'CREATE INDEX IF NOT EXISTS idx_substance_synonym_nocase ON substance_synonyms(synonym COLLATE NOCASE)',
        'idx_sid_position': 'CREATE INDEX IF NOT EXISTS idx_sid_position ON substance_synonyms(sid, position)',
    }

    def __init__(self, data_dir: str = "./pubchem_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "pubchem.db"

        if not self.db_path.exists():
            self._create_schema()

    def _create_schema(self):
        """Create database schema - only called once."""
        print("Creating database schema...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for table_sql in self.TABLES.values():
            cursor.execute(table_sql)

        for index_sql in self.INDICES.values():
            cursor.execute(index_sql)

        conn.commit()
        conn.close()
        print("✓ Schema created")

    def _get_connection(self):
        """Get a database connection with optimized settings."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA cache_size = -64000")
        conn.execute("PRAGMA temp_store = MEMORY")
        return conn

    def _bulk_insert_with_indices(self, conn, table: str, columns: Tuple[str, ...], data: list):
        """Bulk insert with proper index management for specific table only."""
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode = OFF")
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA cache_size = -4000000")
        cursor.execute("PRAGMA locking_mode = EXCLUSIVE")
        cursor.execute("PRAGMA temp_store = MEMORY")

        # Only drop indices for THIS table
        table_indices = [idx_name for idx_name, sql in self.INDICES.items()
                         if f" ON {table}" in sql.lower()]

        for idx_name in table_indices:
            cursor.execute(f"DROP INDEX IF EXISTS {idx_name}")

        placeholders = ','.join(['?'] * len(columns))
        cursor.executemany(
            f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
            data
        )

        conn.commit()

        # Recreate only this table's indices
        print(f"Creating indices for {table}...")
        for idx_name in table_indices:
            cursor.execute(self.INDICES[idx_name])

        print(f"Analyzing {table}...")
        cursor.execute("ANALYZE")
        conn.commit()

    # ===== LOADING METHODS =====

    def load_compound_smiles(self, file_path: str, progress_interval: int = 5_000_000):
        """Load CID-SMILES file.

        Args:
            file_path: Path to the CID-SMILES file (can be .gz)
            progress_interval: Print progress every N lines
        """
        print(f"Loading compounds from {file_path}...")

        if file_path.endswith('.gz'):
            file_handle = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')

        data = []
        with file_handle as f:
            for i, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t', 1)
                    if len(parts) >= 2:
                        cid = int(parts[0])
                        smiles = parts[1]
                        data.append((cid, smiles))

                    if i % progress_interval == 0:
                        print(f"  Read {i:,} lines...")
                except Exception:
                    continue

        print(f"Read {len(data):,} records into memory")

        conn = sqlite3.connect(self.db_path)
        self._bulk_insert_with_indices(conn, 'compounds', ('cid', 'canonical_smiles'), data)
        conn.close()
        print(f"✓ Loaded {len(data):,} records into compounds")

    def load_synonyms(self, file_path: str, progress_interval: int = 5_000_000):
        """
        Load CID-Synonym-filtered file.
        Assumes file format is: CID\tSynonym (one synonym per line, ordered by relevance).
        Position is tracked to preserve synonym relevance ordering.

        Args:
            file_path: Path to the CID-Synonym file (can be .gz)
            progress_interval: Print progress every N lines
        """
        print(f"Loading compound_synonyms from {file_path}...")

        if file_path.endswith('.gz'):
            file_handle = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')

        data = []
        position_tracker = {}  # Track position for each CID

        with file_handle as f:
            for i, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t', 1)
                    if len(parts) >= 2:
                        cid = int(parts[0])
                        synonym = parts[1]

                        # Track position for this CID
                        if cid not in position_tracker:
                            position_tracker[cid] = 0
                        position = position_tracker[cid]
                        position_tracker[cid] += 1

                        data.append((cid, synonym, position))

                    if i % progress_interval == 0:
                        print(f"  Read {i:,} lines...")
                except Exception:
                    continue

        print(f"Read {len(data):,} records into memory")

        conn = sqlite3.connect(self.db_path)
        self._bulk_insert_with_indices(conn, 'compound_synonyms',
                                       ('cid', 'synonym', 'position'), data)
        conn.close()
        print(f"✓ Loaded {len(data):,} records into compound_synonyms")

    def load_parent_compounds(self, file_path: str, progress_interval: int = 5_000_000):
        """Load CID-Parent file.

        Args:
            file_path: Path to the CID-Parent file (can be .gz)
            progress_interval: Print progress every N lines
        """
        print(f"Loading parent_compounds from {file_path}...")

        if file_path.endswith('.gz'):
            file_handle = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')

        data = []
        with file_handle as f:
            for i, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t', 1)
                    if len(parts) >= 2:
                        cid = int(parts[0])
                        parent_cid = int(parts[1])
                        data.append((cid, parent_cid))

                    if i % progress_interval == 0:
                        print(f"  Read {i:,} lines...")
                except Exception:
                    continue

        print(f"Read {len(data):,} records into memory")

        conn = sqlite3.connect(self.db_path)
        self._bulk_insert_with_indices(conn, 'parent_compounds', ('cid', 'parent_cid'), data)
        conn.close()
        print(f"✓ Loaded {len(data):,} records into parent_compounds")

    def load_compound_titles(self, file_path: str, progress_interval: int = 5_000_000):
        """
        Load CID-Title file.
        Also stores title length for efficient sorting when doing partial matches.

        Args:
            file_path: Path to the CID-Title file (can be .gz)
            progress_interval: Print progress every N lines
        """
        print(f"Loading compound_titles from {file_path}...")

        if file_path.endswith('.gz'):
            file_handle = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')

        data = []
        with file_handle as f:
            for i, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t', 1)
                    if len(parts) >= 2:
                        cid = int(parts[0])
                        title = parts[1]
                        title_length = len(title)

                        data.append((cid, title, title_length))

                    if i % progress_interval == 0:
                        print(f"  Read {i:,} lines...")
                except Exception:
                    continue

        print(f"Read {len(data):,} records into memory")

        conn = sqlite3.connect(self.db_path)
        self._bulk_insert_with_indices(conn, 'compound_titles',
                                       ('cid', 'title', 'title_length'), data)
        conn.close()
        print(f"✓ Loaded {len(data):,} records into compound_titles")

    def load_reference_collection(self, file_path: str, progress_interval: int = 100_000):
        """
        Load Reference-Collection.tsv file with columns: SID, PreferredName, Synonyms.
        Stores everything in lowercase for efficient case-insensitive matching.
        Synonyms are pipe-separated (syn1|syn2|syn3) and stored in a separate table with position tracking.

        Args:
            file_path: Path to the Reference-Collection.tsv file (can be .gz)
            progress_interval: Print progress every N lines
        """
        print(f"Loading reference_collection from {file_path}...")

        if file_path.endswith('.gz'):
            file_handle = gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            file_handle = open(file_path, 'r', encoding='utf-8', errors='ignore')

        collection_data = []
        synonym_data = []

        with file_handle as f:
            # Skip header line
            next(f, None)

            for i, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:  # At least SID and PreferredName
                        sid = parts[0].strip()
                        preferred_name = parts[1].strip().lower() if len(parts) > 1 else ''
                        preferred_name_length = len(preferred_name)

                        collection_data.append((sid, preferred_name, preferred_name_length))

                        # Process synonyms: split by | and track position
                        synonyms_raw = parts[2].strip() if len(parts) > 2 else ''
                        if synonyms_raw:
                            synonyms_list = [syn.strip().lower() for syn in synonyms_raw.split('|') if syn.strip()]
                            for position, synonym in enumerate(synonyms_list):
                                synonym_data.append((sid, synonym, position))

                    if i % progress_interval == 0:
                        print(f"  Read {i:,} lines...")
                except Exception as e:
                    continue

        print(f"Read {len(collection_data):,} substances and {len(synonym_data):,} synonyms into memory")

        conn = sqlite3.connect(self.db_path)

        # Insert into reference_collection
        self._bulk_insert_with_indices(conn, 'reference_collection',
                                       ('sid', 'preferred_name', 'preferred_name_length'),
                                       collection_data)

        # Insert into substance_synonyms
        self._bulk_insert_with_indices(conn, 'substance_synonyms',
                                       ('sid', 'synonym', 'position'),
                                       synonym_data)

        conn.close()
        print(f"✓ Loaded {len(collection_data):,} records into reference_collection")
        print(f"✓ Loaded {len(synonym_data):,} records into substance_synonyms")

    # ===== QUERY METHODS =====

    def _query_single(self, sql: str, params: tuple) -> Optional[Any]:
        """Execute query and return single value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def _query_multiple(self, sql: str, params: tuple) -> List[Tuple]:
        """Execute query and return multiple rows."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        return results

    def lookup_cid_by_name(self, name: str, fuzzy=True) -> Optional[int]:
        """
        Look up CID by name (case-insensitive).
        Strategy:
        1. Check for exact match in title
        2. Check for partial match in title (shortest title wins)
        3. Check for exact match in synonyms (by relevance/position)
        """
        if not name or not name.strip():
            return None

        name = name.strip().lower()

        # Priority 1: Exact title match
        cid_from_title = self._query_single(
            "SELECT cid FROM compound_titles WHERE title = ? COLLATE NOCASE",
            (name,)
        )
        if cid_from_title:
            return cid_from_title

        # Priority 3: Exact synonym match (ordered by position/relevance)
        cid_from_synonym = self._query_single(
            "SELECT cid FROM compound_synonyms WHERE synonym = ? COLLATE NOCASE ORDER BY position ASC LIMIT 1",
            (name,)
        )
        if cid_from_synonym:
            return cid_from_synonym

        if fuzzy:
            return self.fuzzy_lookup_cid_by_name(name)
        else:
            return None

    def lookup_all_cids_by_name(self, name: str) -> List[Tuple[int, int]]:
        """
        Look up ALL CIDs by exact name (case-insensitive) with their synonym position.
        Returns list of (cid, position) tuples, ordered by position (relevance).
        Lower position = more relevant.
        """
        results = self._query_multiple(
            "SELECT cid, position FROM compound_synonyms WHERE synonym = ? COLLATE NOCASE ORDER BY position ASC",
            (name,)
        )
        return results

    def lookup_sid_by_name(self, name: str, fuzzy=True) -> Optional[str]:
        """
        Look up SID by name using the reference_collection table.

        Priority:
        1. Exact match in PreferredName (case-insensitive)
        2. Partial match in PreferredName (case-insensitive) - return SID with smallest PreferredName
        3. Exact match in Synonyms (case-insensitive, ordered by position/relevance)
        4. Return None if no match found

        Args:
            name: The compound name to search for

        Returns:
            SID (str) or None
        """
        if not name or not name.strip():
            return None

        name_lower = name.strip().lower()

        # Step 1: Check for exact match in PreferredName
        result = self._query_single(
            "SELECT sid FROM reference_collection WHERE preferred_name = ? LIMIT 1",
            (name_lower,)
        )
        if result:
            return result

        # Step 3: Check for exact match in Synonyms (ordered by position/relevance)
        result = self._query_single(
            """SELECT sid FROM substance_synonyms
               WHERE synonym = ? COLLATE NOCASE
               ORDER BY position ASC
               LIMIT 1""",
            (name_lower,)
        )
        if result:
            return result

        if fuzzy:
            return self.fuzzy_lookup_sid_by_name(name_lower)
        else:
            return None

    def _generate_fuzzy_variants(self, name: str) -> List[str]:
        """Generate fuzzy name variants for lookup."""
        name = name.strip().lower()
        variants = []

        # Remove parenthetical content
        if "(" in name:
            variants.append(name.split("(")[0].strip())

        # Hyphen variations
        if "-" in name:
            variants.append(name.replace("-", " "))
            variants.append(name.replace("-", ""))

        return [v for v in variants if v and v != name]

    def fuzzy_lookup_cid_by_name(self, name: str) -> Optional[int]:
        for variant in self._generate_fuzzy_variants(name):
            cid = self.lookup_cid_by_name(variant, fuzzy=False)
            if cid:
                return cid
        return None

    def fuzzy_lookup_sid_by_name(self, name: str) -> Optional[str]:
        for variant in self._generate_fuzzy_variants(name):
            sid = self.lookup_sid_by_name(variant, fuzzy=False)
            if sid:
                return sid
        return None

    def lookup_all_sids_by_name(self, name: str) -> List[Tuple[str, int]]:
        """
        Look up ALL SIDs by exact name (case-insensitive) with their synonym position.
        Returns list of (sid, position) tuples, ordered by position (relevance).
        Lower position = more relevant.
        """
        if not name or not name.strip():
            return []

        name_lower = name.strip().lower()

        results = self._query_multiple(
            "SELECT sid, position FROM substance_synonyms WHERE synonym = ? COLLATE NOCASE ORDER BY position ASC",
            (name_lower,)
        )
        return results

    def get_smiles(self, cid: int) -> Optional[str]:
        """Get SMILES string for a CID."""
        return self._query_single(
            "SELECT canonical_smiles FROM compounds WHERE cid = ?",
            (cid,)
        )

    def get_parent_cid(self, cid: int) -> Optional[int]:
        """Get parent compound CID."""
        return self._query_single(
            "SELECT parent_cid FROM parent_compounds WHERE cid = ?",
            (cid,)
        )

    def get_title(self, cid: int) -> Optional[str]:
        """Get preferred compound name."""
        return self._query_single(
            "SELECT title FROM compound_titles WHERE cid = ?",
            (cid,)
        )

    def get_all_synonyms(self, cid: int) -> List[str]:
        """Get all synonyms for a compound, ordered by relevance (position)."""
        results = self._query_multiple(
            "SELECT synonym FROM compound_synonyms WHERE cid = ? ORDER BY position ASC",
            (cid,)
        )
        return [r[0] for r in results]

    def get_preferred_name(self, sid: str) -> Optional[str]:
        """Get preferred name for a substance."""
        return self._query_single(
            "SELECT preferred_name FROM reference_collection WHERE sid = ?",
            (sid,)
        )

    def get_all_substance_synonyms(self, sid: str) -> List[str]:
        """Get all synonyms for a substance, ordered by relevance (position)."""
        results = self._query_multiple(
            "SELECT synonym FROM substance_synonyms WHERE sid = ? ORDER BY position ASC",
            (sid,)
        )
        return [r[0] for r in results]

    def get_stats(self) -> dict:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}
        for table in self.TABLES.keys():
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        conn.close()
        return stats


def create_pubchem_db(data_dir="./pubchem_data", force_reload=False):
    """Create PubChem database from downloaded files."""
    db = PubChemLocalDatabase(data_dir=data_dir)

    stats = db.get_stats()

    # Load compounds if empty or force_reload
    if force_reload or stats['compounds'] == 0:
        print("Loading compounds...")
        db.load_compound_smiles(f"{data_dir}/CID-SMILES")
    else:
        print(f"✓ compounds already loaded ({stats['compounds']:,} records)")

    # Load compound_synonyms if empty or force_reload
    if force_reload or stats['compound_synonyms'] == 0:
        print("Loading compound_synonyms...")
        db.load_synonyms(f"{data_dir}/CID-Synonym-filtered")
    else:
        print(f"✓ compound_synonyms already loaded ({stats['compound_synonyms']:,} records)")

    # Load parent_compounds if empty or force_reload
    if force_reload or stats['parent_compounds'] == 0:
        print("Loading parent_compounds...")
        db.load_parent_compounds(f"{data_dir}/CID-Parent")
    else:
        print(f"✓ parent_compounds already loaded ({stats['parent_compounds']:,} records)")

    # Load compound_titles if empty or force_reload
    if force_reload or stats['compound_titles'] == 0:
        print("Loading compound_titles...")
        db.load_compound_titles(f"{data_dir}/CID-Title")
    else:
        print(f"✓ compound_titles already loaded ({stats['compound_titles']:,} records)")

    # Load reference_collection if empty or force_reload
    ref_collection_path = Path(data_dir) / "Reference-Collection.tsv"
    if ref_collection_path.exists():
        if force_reload or stats['reference_collection'] == 0:
            print("Loading reference_collection...")
            db.load_reference_collection(str(ref_collection_path))
        else:
            print(f"✓ reference_collection already loaded ({stats['reference_collection']:,} records)")
    else:
        print("⚠️  Reference-Collection.tsv not found, skipping")

    print("\nFinal database stats:")
    final_stats = db.get_stats()
    for table, count in final_stats.items():
        print(f"  {table}: {count:,} records")

    return db