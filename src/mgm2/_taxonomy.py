"""Taxonomy matching extracted from the validated dataset builder."""
from typing import Dict, List, Tuple, Optional
import h5py
import numpy as np

class TaxonomyLookup:
    TAXON_RANKS = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species')

    TAXON_PREFIX_TO_RANK = {
        'sk': 'kingdom',
        'k': 'kingdom',
        'd': 'kingdom',
        'p': 'phylum',
        'c': 'class',
        'o': 'order',
        'f': 'family',
        'g': 'genus',
        's': 'species',
    }

    @staticmethod
    def _decode_h5_strings(values) -> List[str]:
        decoded = []
        for value in values:
            if isinstance(value, (bytes, bytearray, np.bytes_)):
                decoded.append(value.decode('utf-8', errors='replace'))
            else:
                decoded.append(str(value))
        return decoded

    @staticmethod
    def _clean_taxon_name(value: str) -> str:
        value = str(value).strip()
        if not value or value.lower() in {'unclassified', 'unknown', 'none', 'nan'}:
            return ''
        return value

    @classmethod
    def _parse_taxonomy_label(cls, label: str) -> Dict[str, str]:
        """Parse plain or rank-prefixed taxonomy labels."""
        if '__' not in str(label):
            parts = [cls._clean_taxon_name(part) for part in str(label).split(';')]
            parts = [part for part in parts if part]
            return {rank: value for rank, value in zip(cls.TAXON_RANKS, parts)}

        rank_to_name: Dict[str, str] = {}
        root_priority = -1
        for token in str(label).split(';'):
            token = token.strip()
            if '__' not in token:
                continue
            prefix, name = token.split('__', 1)
            rank = cls.TAXON_PREFIX_TO_RANK.get(prefix.strip().lower())
            name = cls._clean_taxon_name(name)
            if rank is not None and name:
                if rank == 'kingdom':
                    # The prototype hierarchy has one root slot. MGnify may
                    # contain both superkingdom and kingdom (Eukaryota/Fungi).
                    # Prefer the domain/superkingdom regardless of token order.
                    priority = int(prefix.strip().lower() in {'sk', 'd'})
                    if priority < root_priority:
                        continue
                    root_priority = priority
                rank_to_name[rank] = name
        return rank_to_name

    @classmethod
    def _build_taxon_keys(cls, label: str) -> List[Tuple[str, str]]:
        rank_to_name = cls._parse_taxonomy_label(label)
        keys = []
        path = []
        for rank in cls.TAXON_RANKS:
            name = rank_to_name.get(rank)
            if not name:
                continue
            path.append(name)
            keys.append((rank, f"{rank}|{';'.join(path)}"))
        return keys

    @staticmethod
    def _normalize_taxon_alias(name: str) -> str:
        return ' '.join(str(name).strip().replace('_', ' ').split()).casefold()

    @classmethod
    def _normalize_taxon_path(cls, path: str) -> str:
        return ';'.join(cls._normalize_taxon_alias(name) for name in path.split(';'))

    @classmethod
    def _terminal_alias_candidates(cls, label: str) -> List[Tuple[str, str]]:
        label = str(label).strip()
        if ';' not in label and '__' not in label:
            return [(rank, label) for rank in reversed(cls.TAXON_RANKS)]

        rank_to_name = cls._parse_taxonomy_label(label)
        return [
            (rank, rank_to_name[rank])
            for rank in reversed(cls.TAXON_RANKS)
            if rank in rank_to_name
        ]

    @classmethod
    def _lookup_terminal_alias(
        cls,
        label: str,
        terminal_aliases: Dict[str, Dict[str, str]],
    ) -> Optional[Tuple[str, str]]:
        for rank, name in cls._terminal_alias_candidates(label):
            alias = cls._normalize_taxon_alias(name)
            matched_key = terminal_aliases.get(rank, {}).get(alias)
            if matched_key is not None:
                return rank, matched_key
        return None

    @classmethod
    def _resolve_taxonomy_match(cls, label, fallback_map, terminal_aliases):
        """Try exact path then an unambiguous name at EACH rank before ascending.

        A changed parent lineage must not hide a known genus behind a matching
        kingdom. Name aliases are unique within their rank; when a root is
        supplied, reject aliases belonging to a different kingdom/domain.
        """
        exact_keys = dict(cls._build_taxon_keys(label))
        parsed = cls._parse_taxonomy_label(label)
        bare_name = ';' not in str(label) and '__' not in str(label)
        root = None if bare_name else parsed.get('kingdom')
        for rank, name in cls._terminal_alias_candidates(label):
            key = exact_keys.get(rank)
            if key in fallback_map:
                return rank, key
            # Older prototype tables assigned ranks positionally after dropping
            # empty levels. The identical full lineage still identifies the
            # same taxon even if its stored rank is shifted (e.g. an order
            # labelled as class). Require a unique, complete normalized path.
            if key is not None:
                path_key = terminal_aliases.get('_paths', {}).get(
                    cls._normalize_taxon_path(key.partition('|')[2]))
                if path_key is not None:
                    return f'{rank}_path', path_key
            key = terminal_aliases.get(rank, {}).get(cls._normalize_taxon_alias(name))
            if key is None:
                continue
            candidate_root = key.partition('|')[2].split(';', 1)[0]
            if root and cls._normalize_taxon_alias(root) != cls._normalize_taxon_alias(candidate_root):
                continue
            return f'{rank}_name', key
        return None

    def _load_fallback_embeddings(self):
        if not self.fallback_embedding_path:
            return {}, None, {}
        with h5py.File(self.fallback_embedding_path, 'r') as f:
            taxon_keys = self._decode_h5_strings(f['taxon_keys'][:])
            taxon_embeddings = f['taxon_embeddings'][:].astype(np.float32)
            global_embedding = f['global_embedding'][:].astype(np.float32) if 'global_embedding' in f else None
            taxon_ranks = (
                self._decode_h5_strings(f['taxon_ranks'][:])
                if 'taxon_ranks' in f
                else [key.partition('|')[0] for key in taxon_keys]
            )
            taxon_paths = (
                self._decode_h5_strings(f['taxon_paths'][:])
                if 'taxon_paths' in f
                else [key.partition('|')[2] for key in taxon_keys]
            )

        if len(taxon_embeddings) != len(taxon_keys):
            raise ValueError('Fallback taxon_embeddings and taxon_keys must have the same length')
        if len(taxon_ranks) != len(taxon_keys) or len(taxon_paths) != len(taxon_keys):
            raise ValueError('Fallback taxonomy metadata must align with taxon_keys')

        fallback_map = {key: taxon_embeddings[idx] for idx, key in enumerate(taxon_keys)}
        alias_candidates: Dict[str, Dict[str, set]] = {}
        path_candidates: Dict[str, set] = {}
        for key, rank, path in zip(taxon_keys, taxon_ranks, taxon_paths):
            path_candidates.setdefault(self._normalize_taxon_path(str(path)), set()).add(key)
            rank = str(rank).strip().lower()
            terminal_name = self._clean_taxon_name(str(path).rsplit(';', 1)[-1])
            if rank not in self.TAXON_RANKS or not terminal_name:
                continue
            alias = self._normalize_taxon_alias(terminal_name)
            alias_candidates.setdefault(rank, {}).setdefault(alias, set()).add(key)

        terminal_aliases = {
            rank: {
                alias: next(iter(keys))
                for alias, keys in rank_aliases.items()
                if len(keys) == 1
            }
            for rank, rank_aliases in alias_candidates.items()
        }
        terminal_aliases['_paths'] = {
            path: next(iter(keys)) for path, keys in path_candidates.items() if len(keys) == 1
        }
        return fallback_map, global_embedding, terminal_aliases
