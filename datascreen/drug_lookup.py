"""Drug information lookup from PubChem."""

import json
import os
import re
import urllib.parse
from dataclasses import dataclass
from functools import cache
from typing import Optional
import requests
from tqdm import tqdm
import pandas as pd

from requests_cache import CachedSession
from requests.adapters import HTTPAdapter, Retry

from rdkit import Chem

from .pubchem_database import PubChemLocalDatabase


ETL_REQUEST_CACHE_VAR = "ETL_REQUEST_CACHE"


def add_retry_to_session(
    session,
    retries=5,
    backoff_factor=0.3,
    status_forcelist=(429, 500, 501, 502, 503, 504),
):
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@cache
def get_cached_session():
    if "ETL_REQUEST_CACHE" in os.environ:
        cache_location = os.environ["ETL_REQUEST_CACHE"]
        session = CachedSession(
            cache_location, allowable_codes=(200, 404, 400), backend="sqlite"
        )
    else:
        session = CachedSession(backend="memory")

    return add_retry_to_session(session)


def find_nodes(T, key_string, value_string):
    if isinstance(T, list):
        for x in T:
            for y in find_nodes(x, key_string, value_string):
                yield y
    if isinstance(T, dict):
        if key_string in T and T[key_string] == value_string:
            yield T
        else:
            for k in T:
                for x in find_nodes(T[k], key_string, value_string):
                    yield x


@dataclass
class PUGLookupResponse:
    cid: Optional[str]
    sid: Optional[str]
    canonical_smiles: Optional[str]
    drug_class: Optional[list] = None
    parent_cid: Optional[str] = None
    title: Optional[str] = None


def replace_non_ascii_characters_with_plus(keyword):
    # normalize weird hyphens that aren't an ascii hyphen so we don't lose them
    keyword = re.sub(r"[‐᠆﹣－⁃−]+", "-", keyword)
    return re.sub(r"[^\x00-\x7f]", "+", keyword).strip()


def replace_meaningless_ascii_characters_with_space(keyword):
    return re.sub(r"[^A-Za-z\d()\-+]", " ", keyword).strip()


def pug_rest_fuzzy_name_lookup(keyword) -> PUGLookupResponse:
    """Fuzzy name lookup using PubChem REST API."""
    cleaned_keyword = replace_meaningless_ascii_characters_with_space(
        replace_non_ascii_characters_with_plus(keyword)
    )

    cleaned_keyword_parts = re.split("\s+", cleaned_keyword)

    query = {
        "select": "*",
        "collection": "compound",
        "where": {"ands": [{"*": x} for x in cleaned_keyword_parts]},
        "order": ["relevancescore,desc"],
        "start": 1,
        "limit": 1,
        "width": 1000000,
        "listids": 0,
    }

    url = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi"

    res = get_cached_session().get(
        url, params={"query": json.dumps(query), "infmt": "json", "outfmt": "json"}
    )

    if res.ok:
        try:
            top_search_results = res.json()["SDQOutputSet"][0]["rows"]
        except KeyError:
            top_search_results = None
    else:
        top_search_results = None

    if not top_search_results:
        query["collection"] = "substance"

        res = get_cached_session().get(
            url, params={"query": json.dumps(query), "infmt": "json", "outfmt": "json"}
        )

        if res.ok:
            try:
                top_search_results = res.json()["SDQOutputSet"][0]["rows"]
            except KeyError:
                pass

        if not top_search_results:
            return PUGLookupResponse(cid=None, sid=None, canonical_smiles=None)

    top_search_result = top_search_results[0]

    cid = top_search_result.get("cid")
    sid = top_search_result.get("sid")
    canonical_smiles = top_search_result.get("isosmiles")

    return PUGLookupResponse(cid=cid, sid=sid, canonical_smiles=canonical_smiles)


def get_parent_compound_cid(cid):
    """Get parent compound CID from PubChem API."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{str(int(float(cid)))}/JSON/"
    res = get_cached_session().get(url)
    if not res.ok:
        return None
    record = res.json()
    try:
        parent_compound_text = next(
            find_nodes(record, "TOCHeading", "Parent Compound")
        )["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
    except StopIteration:
        return None

    return parent_compound_text.split(" ")[1]


def pug_rest_name_lookup(keyword) -> PUGLookupResponse:
    """Keyword search a drug name in the PubChem Compound and Substance databases."""

    quoted_keyword = urllib.parse.quote(
        replace_non_ascii_characters_with_plus(keyword), safe="~()*!.'"
    )

    # remove all %20 from quoted_keyword, its not allowed
    quoted_keyword = quoted_keyword.replace("%20", "")

    compound_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quoted_keyword}/property/ConnectivitySMILES,InChI/JSON"
    )

    substance_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/name/"
        f"{quoted_keyword}/JSON"
    )

    compound_response = get_cached_session().get(compound_url)

    try:
        substance_response = get_cached_session().get(substance_url)
    except requests.exceptions.RetryError:
        substance_response = requests.Response()
        substance_response.status_code = 500

    sid = None
    cid = None
    smiles = None

    if substance_response.ok:
        try:
            sid = str(substance_response.json()["PC_Substances"][0]["sid"]["id"])
        except KeyError:
            pass

    if compound_response.ok:
        try:
            cid = str(compound_response.json()["PropertyTable"]["Properties"][0]["CID"])
        except KeyError:
            pass

        try:
            smiles = compound_response.json()["PropertyTable"]["Properties"][0][
                "ConnectivitySMILES"
            ]
        except KeyError:
            pass

    return PUGLookupResponse(cid=cid, sid=sid, canonical_smiles=smiles)


def drug_pubchem_name_lookup(cid=None, sid=None) -> Optional[str]:
    """Look up preferred drug name from PubChem API."""
    if cid:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
        res = get_cached_session().get(url)

        if res.ok:
            return res.json()["Record"]["RecordTitle"]

    elif sid:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/substance/{sid}/JSON/"
        res = get_cached_session().get(url)

        if res.ok:
            return res.json()["Record"]["RecordTitle"]

    return None


@dataclass
class DrugLookupResult:
    """Result from local PubChem database lookup."""
    cid: Optional[int] = None
    sid: Optional[int] = None
    canonical_smiles: Optional[str] = None
    parent_cid: Optional[int] = None
    title: Optional[str] = None


def get_drug_info_from_local_pubchem(
    name: str,
    path: str = "./pubchem_data"
) -> DrugLookupResult:
    """
    Look up drug information from local PubChem database.

    Lookup strategy:
    1. Try to find CID by name (compound synonym)
    2. If found, get SMILES, parent CID, and title from CID
    3. If not found, try SID lookup
    4. If SID found, convert to CID and get compound info

    Args:
        name: Drug name or synonym to search for
        path: Path to PubChem data directory

    Returns:
        DrugLookupResult with available information
    """
    db = PubChemLocalDatabase(data_dir=path)

    # First, try direct CID lookup by name
    cid = db.lookup_cid_by_name(name)

    if cid:
        # Found CID directly - get all compound info
        smiles = db.get_smiles(cid)
        parent_cid = db.get_parent_cid(cid)
        title = db.get_title(cid)
        sid = None

        return DrugLookupResult(
            cid=cid,
            sid=sid,
            canonical_smiles=smiles,
            parent_cid=parent_cid,
            title=title
        )

    # If no CID found, try SID lookup
    sid = db.lookup_sid_by_name(name)

    if sid:
        return DrugLookupResult(sid=sid)

    # Nothing found
    return DrugLookupResult()


@cache
def canonicalize_smiles_string(smiles):
    """Canonicalize SMILES string using RDKit."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return smiles

    return Chem.MolToSmiles(mol)


@cache
def drug_object_from_name(
    drug_description,
    local_pubchem=None,
    use_puchem_api=True,
):
    """
    Get drug information from drug name.

    Args:
        drug_description: Drug name
        local_pubchem: Path to local PubChem database
        use_puchem_api: Whether to use PubChem API as fallback

    Returns:
        Dictionary with drug information
    """
    if local_pubchem:
        result = get_drug_info_from_local_pubchem(
            name=drug_description,
            path=local_pubchem
        )

    if not local_pubchem or ((result.cid is None and result.sid is None) and use_puchem_api):
        print("Don't find in database", drug_description)

        result: PUGLookupResponse = pug_rest_name_lookup(drug_description)

        if result.cid is None and result.sid is None:
            result: PUGLookupResponse = pug_rest_fuzzy_name_lookup(drug_description)

        print(result)

        result.parent_cid = get_parent_compound_cid(result.cid) if result.cid is not None else None
        result.title = drug_pubchem_name_lookup(cid=result.cid, sid=result.sid)

    drug = dict(
        smiles=(
            canonicalize_smiles_string(result.canonical_smiles)
            if result.canonical_smiles is not None
            else None
        ),
        pubchem_compound_id=result.cid,
        pubchem_substance_id=result.sid,
        pubchem_parent_compound_id=result.parent_cid,
        name=drug_description,
        pubchem_name=result.title,
    )

    return drug


def extract_drug_info(drug_list, local_pubchem=None, use_puchem_api=True):
    """Extract drug information for a list of drugs."""
    all_drug_info = []
    for drug in tqdm(drug_list, desc="Extracting drug info..."):
        drug_info = drug_object_from_name(
            drug,
            local_pubchem=local_pubchem,
            use_puchem_api=use_puchem_api
        )
        all_drug_info.append(drug_info)
    return pd.DataFrame(all_drug_info)