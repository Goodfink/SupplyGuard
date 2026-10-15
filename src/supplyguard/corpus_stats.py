from dataclasses import dataclass

@dataclass
class CorpusStats:
    total_json: int = 0
    ghsa: int = 0
    malicious: int = 0
    withdrawn: int = 0
    published_in_date_range: int = 0
    has_package: int = 0
    has_description: int = 0
    valid_candidates: int = 0
    invalid_json: int = 0