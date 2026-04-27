#!/usr/bin/env bash
# Sync Yelp semantic artifacts across DISCO and LETTER from TIGER semantic IDs.
# Sequence files are untouched (train/valid/test.txt, *.inter.json stay as-is).

set -euo pipefail

TIGER_PROCESSED="/home/sjj/wenhao/TIGER/cache/Yelp/Yelp_2020/processed"
DISCO_YELP="/home/sjj/wenhao/DISCO/datasets/Yelp"
LETTER_YELP="/home/sjj/wenhao/LETTER/data/Yelp"
BACKUP_ROOT="/home/sjj/wenhao/TIGER/backup_yelp_semantic"
TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/${TS}"

mkdir -p "${BACKUP_DIR}" "${DISCO_YELP}" "${LETTER_YELP}"

for f in id_mapping.json sentence-t5-base_256,256,256,256.sem_ids sentence-t5-base.sent_emb metadata.sentence.json; do
  if [ ! -f "${TIGER_PROCESSED}/${f}" ]; then
    echo "ERROR: missing ${TIGER_PROCESSED}/${f}"
    exit 1
  fi
done

for f in clhe_sid.npy clhe_token.json clhe_weight.npy; do
  if [ -f "${DISCO_YELP}/${f}" ]; then
    cp "${DISCO_YELP}/${f}" "${BACKUP_DIR}/${f}.bak"
  fi
done
if [ -f "${LETTER_YELP}/Yelp.index.json" ]; then
  cp "${LETTER_YELP}/Yelp.index.json" "${BACKUP_DIR}/Yelp.index.json.bak"
fi

python3 - <<'PY'
import json
import numpy as np
from pathlib import Path

TIGER_PROCESSED = Path('/home/sjj/wenhao/TIGER/cache/Yelp/Yelp_2020/processed')
DISCO_YELP = Path('/home/sjj/wenhao/DISCO/datasets/Yelp')
LETTER_YELP = Path('/home/sjj/wenhao/LETTER/data/Yelp')

RQ_N_CODEBOOKS = 3
RQ_CODEBOOK_SIZE = 256

with (TIGER_PROCESSED / 'id_mapping.json').open() as f:
    id_mapping = json.load(f)
with (TIGER_PROCESSED / 'sentence-t5-base_256,256,256,256.sem_ids').open() as f:
    tiger_item2sid = json.load(f)

n_items = len(id_mapping['item2id']) - 1

embs = np.fromfile(TIGER_PROCESSED / 'sentence-t5-base.sent_emb', dtype=np.float32).reshape(n_items, 64)

# DISCO sid/token from TIGER
sid = {}
for iid_1based in range(1, n_items + 1):
    orig = id_mapping['id2item'][iid_1based]
    a, b, c, d = tiger_item2sid[orig]
    sid[iid_1based - 1] = (int(a), int(b), int(c), int(d))

# clhe_sid.npy
np.save(DISCO_YELP / 'clhe_sid.npy', np.array(sid, dtype=object))

# clhe_token.json
token = {}
for item_id, (a, b, c, d) in sid.items():
    token[str(item_id)] = [
        a + 1,
        b + RQ_CODEBOOK_SIZE + 1,
        c + RQ_CODEBOOK_SIZE * 2 + 1,
        d + RQ_CODEBOOK_SIZE * 3,
    ]
with (DISCO_YELP / 'clhe_token.json').open('w') as f:
    json.dump(token, f, indent=2)

# clhe_weight.npy (centroids from TIGER sid assignment)
global_mean = embs.mean(axis=0)
weight = np.zeros((RQ_N_CODEBOOKS * RQ_CODEBOOK_SIZE, embs.shape[1]), dtype=np.float32)
for codebook_idx in range(RQ_N_CODEBOOKS):
    assignments = np.array([sid[i][codebook_idx] for i in range(len(sid))], dtype=np.int32)
    for code in range(RQ_CODEBOOK_SIZE):
        mask = assignments == code
        row = codebook_idx * RQ_CODEBOOK_SIZE + code
        if np.any(mask):
            weight[row] = embs[mask].mean(axis=0)
        else:
            weight[row] = global_mean
np.save(DISCO_YELP / 'clhe_weight.npy', weight)

# LETTER Yelp.index.json from TIGER sid
letter_index = {}
for item_id, (a, b, c, d) in sid.items():
    letter_index[str(item_id)] = [f'<a_{a}>', f'<b_{b}>', f'<c_{c}>', f'<d_{d}>']
with (LETTER_YELP / 'Yelp.index.json').open('w') as f:
    json.dump(letter_index, f, indent=2)

print(f'Updated DISCO semantic files in: {DISCO_YELP}')
print(f'Updated LETTER semantic file in: {LETTER_YELP / "Yelp.index.json"}')
print(f'Items: {n_items}')
PY

echo "Backups saved to: ${BACKUP_DIR}"
echo "Done. Sequence files were not changed."
