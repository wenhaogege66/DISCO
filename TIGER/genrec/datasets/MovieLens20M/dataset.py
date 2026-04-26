import os
import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class MovieLens20M(AbstractDataset):
    """
    MovieLens-20M dataset.

    Raw files expected at: cache/MovieLens-20M/raw/ratings.csv  movies.csv
    Download from: https://grouplens.org/datasets/movielens/20m/

    Config keys:
        seq_len   (int)  : truncation length; users with fewer interactions are discarded.
                           Also determines the cache subdirectory (len{seq_len}).
        user_core (int)  : K-core threshold for users  (default 5)
        item_core (int)  : K-core threshold for items  (default 5)
        metadata  (str)  : 'sentence' or 'none'
    """

    # Shared raw directory — all seq_len variants read the same source files
    RAW_SUBDIR = "raw"

    def __init__(self, config: dict):
        super().__init__(config)

        self.seq_len = int(config.get("seq_len", 60))
        self.log(f"[DATASET] MovieLens-20M  seq_len={self.seq_len}")

        # Each seq_len gets its own processed cache
        self.cache_dir = os.path.join(
            config["cache_dir"], "MovieLens-20M", f"len{self.seq_len}"
        )
        self.log(f"[DATASET] Cache directory: {self.cache_dir}")
        self._download_and_process_raw()

    # ------------------------------------------------------------------
    # Raw data loading
    # ------------------------------------------------------------------

    def _raw_dir(self) -> str:
        return os.path.join(config_cache_root(self.config), "MovieLens-20M", self.RAW_SUBDIR)

    def _load_ratings(self, raw_dir: str) -> List[Tuple[str, str, int]]:
        """Return list of (userId, movieId, timestamp) sorted by timestamp per user."""
        self.log("[DATASET] Loading ratings.csv ...")
        path = os.path.join(raw_dir, "ratings.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ratings.csv not found at {path}.\n"
                "Download ml-20m.zip from https://grouplens.org/datasets/movielens/20m/ "
                f"and extract ratings.csv + movies.csv into {raw_dir}"
            )
        records = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Reading ratings"):
                records.append((row["userId"], row["movieId"], int(float(row["timestamp"]))))
        self.log(f"[DATASET] Loaded {len(records):,} ratings")
        return records

    def _load_movies(self, raw_dir: str) -> Dict[str, str]:
        """Return {movieId: metadata_sentence}."""
        self.log("[DATASET] Loading movies.csv ...")
        path = os.path.join(raw_dir, "movies.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"movies.csv not found at {path}.")
        movie2meta = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = clean_text(row["title"])
                genres_raw = row.get("genres", "(no genres listed)")
                if genres_raw == "(no genres listed)":
                    genres_str = "Unknown"
                else:
                    genres_str = ", ".join(genres_raw.split("|"))
                movie2meta[row["movieId"]] = f"Title: {title}. Genres: {genres_str}."
        self.log(f"[DATASET] Loaded metadata for {len(movie2meta):,} movies")
        return movie2meta

    # ------------------------------------------------------------------
    # Interaction building
    # ------------------------------------------------------------------

    def _build_user_seqs(self, records: List[Tuple[str, str, int]]) -> Dict[str, List[str]]:
        """Group by user, sort by timestamp, return {userId: [movieId, ...]}."""
        raw: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for uid, mid, ts in records:
            raw[uid].append((mid, ts))
        user_seqs = {}
        for uid, pairs in raw.items():
            pairs.sort(key=lambda x: x[1])
            user_seqs[uid] = [p[0] for p in pairs]
        return user_seqs

    # ------------------------------------------------------------------
    # K-core filtering (same pattern as Yelp / Steam)
    # ------------------------------------------------------------------

    def _check_kcore(self, user_items, user_core, item_core):
        user_count: Dict[str, int] = defaultdict(int)
        item_count: Dict[str, int] = defaultdict(int)
        for uid, items in user_items.items():
            for item in items:
                user_count[uid] += 1
                item_count[item] += 1
        for cnt in user_count.values():
            if cnt < user_core:
                return user_count, item_count, False
        for cnt in item_count.values():
            if cnt < item_core:
                return user_count, item_count, False
        return user_count, item_count, True

    def _filter_kcore(self, user_items, user_core, item_core):
        self.log(f"[DATASET] K-core filtering (user≥{user_core}, item≥{item_core}) ...")
        user_count, item_count, ok = self._check_kcore(user_items, user_core, item_core)
        iteration = 0
        while not ok:
            iteration += 1
            to_delete = [u for u, c in user_count.items() if c < user_core]
            for u in to_delete:
                user_items.pop(u, None)
            for uid in list(user_items):
                user_items[uid] = [i for i in user_items[uid] if item_count[i] >= item_core]
                if not user_items[uid]:
                    user_items.pop(uid)
            user_count, item_count, ok = self._check_kcore(user_items, user_core, item_core)
            self.log(f"[DATASET]   iteration {iteration}: {len(user_items):,} users remain")
        return user_items

    # ------------------------------------------------------------------
    # Truncation
    # ------------------------------------------------------------------

    def _truncate(self, user_items: Dict[str, List[str]], seq_len: int) -> Dict[str, List[str]]:
        """Discard users with < seq_len items; keep last seq_len for the rest."""
        self.log(f"[DATASET] Truncating to seq_len={seq_len} ...")
        before = len(user_items)
        result = {}
        for uid, items in user_items.items():
            if len(items) >= seq_len:
                result[uid] = items[-seq_len:]   # keep most recent
        self.log(
            f"[DATASET] Truncation: {before:,} → {len(result):,} users "
            f"(dropped {before - len(result):,} with <{seq_len} interactions)"
        )
        return result

    # ------------------------------------------------------------------
    # ID mapping
    # ------------------------------------------------------------------

    def _id_map(self, user_items: Dict[str, List[str]]):
        self.log("[DATASET] Mapping IDs to sequential integers ...")
        u2id = self.id_mapping["user2id"]
        i2id = self.id_mapping["item2id"]
        id2u = self.id_mapping["id2user"]
        id2i = self.id_mapping["id2item"]

        final: Dict[str, List[str]] = {}
        for uid, items in user_items.items():
            if uid not in u2id:
                u2id[uid] = len(u2id)
                id2u.append(uid)
            mapped = []
            for item in items:
                if item not in i2id:
                    i2id[item] = len(i2id)
                    id2i.append(item)
                mapped.append(item)          # keep original IDs in all_item_seqs
            final[uid] = mapped

        n_users = len(u2id) - 1
        n_items = len(i2id) - 1
        self.log(f"[DATASET] {n_users:,} users, {n_items:,} items")
        return final

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _process_meta(self, movie2meta: Dict[str, str], output_dir: str) -> Optional[Dict[str, str]]:
        mode = self.config.get("metadata", "sentence")
        meta_file = os.path.join(output_dir, f"metadata.{mode}.json")
        if os.path.exists(meta_file):
            self.log(f"[DATASET] Loading metadata from {meta_file}")
            with open(meta_file) as f:
                return json.load(f)
        if mode == "none":
            return None
        # Filter to items actually in id_mapping
        item_ids = set(self.id_mapping["item2id"].keys()) - {"[PAD]"}
        item2meta = {mid: movie2meta.get(mid, "") for mid in item_ids}
        os.makedirs(output_dir, exist_ok=True)
        with open(meta_file, "w") as f:
            json.dump(item2meta, f)
        self.log(f"[DATASET] Saved metadata for {len(item2meta):,} items → {meta_file}")
        return item2meta

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _download_and_process_raw(self):
        processed_dir = os.path.join(self.cache_dir, "processed")
        id_map_file = os.path.join(processed_dir, "id_mapping.json")
        seq_file = os.path.join(processed_dir, "all_item_seqs.json")
        meta_mode = self.config.get("metadata", "sentence")
        meta_file = os.path.join(processed_dir, f"metadata.{meta_mode}.json")

        files_ready = (
            os.path.exists(id_map_file)
            and os.path.exists(seq_file)
            and (meta_mode == "none" or os.path.exists(meta_file))
        )
        if files_ready:
            self.log(f"[DATASET] Loading processed data from {processed_dir}")
            with open(id_map_file) as f:
                self.id_mapping = json.load(f)
            with open(seq_file) as f:
                self.all_item_seqs = json.load(f)
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    self.item2meta = json.load(f)
            return

        raw_dir = os.path.join(config_cache_root(self.config), "MovieLens-20M", self.RAW_SUBDIR)
        user_core = int(self.config.get("user_core", 5))
        item_core = int(self.config.get("item_core", 5))

        # 1. Load raw data
        records = self._load_ratings(raw_dir)
        movie2meta = self._load_movies(raw_dir)

        # 2. Build sequences
        user_seqs = self._build_user_seqs(records)
        self.log(f"[DATASET] Initial: {len(user_seqs):,} users")

        # 3. K-core filtering
        user_seqs = self._filter_kcore(user_seqs, user_core, item_core)

        # 4. Truncation
        user_seqs = self._truncate(user_seqs, self.seq_len)

        # 5. ID mapping
        self.all_item_seqs = self._id_map(user_seqs)

        # 6. Stats
        lengths = [len(v) for v in self.all_item_seqs.values()]
        self.log(
            f"[DATASET] Final: {len(self.all_item_seqs):,} users, "
            f"{self.n_items - 1:,} items, "
            f"avg_len={np.mean(lengths):.1f}"
        )

        # 7. Save
        os.makedirs(processed_dir, exist_ok=True)
        with open(id_map_file, "w") as f:
            json.dump(self.id_mapping, f)
        with open(seq_file, "w") as f:
            json.dump(self.all_item_seqs, f)

        # 8. Metadata
        self.item2meta = self._process_meta(movie2meta, processed_dir)


# ---------------------------------------------------------------------------
# Helper: resolve the root cache dir from config (strips trailing slash)
# ---------------------------------------------------------------------------

def config_cache_root(config: dict) -> str:
    """Return the absolute root cache directory (e.g. /home/sjj/wenhao/TIGER/cache)."""
    raw = config.get("cache_dir", "cache/")
    # config["cache_dir"] is typically "cache/" (relative to TIGER root)
    # When running from TIGER/, this resolves correctly.
    return os.path.abspath(raw)
