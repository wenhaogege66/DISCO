#!/usr/bin/env bash
# train_tiger_movielens20m.sh
#
# For each seq_len in [60, 90, 120]:
#   1. TIGER processes raw data  →  cache/MovieLens-20M/len{N}/processed/
#   2. DISCO/LETTER conversion   →  DISCO/datasets/MovieLens-20M/len{N}/ + LETTER/data/MovieLens-20M/len{N}/
#
# Prerequisites:
#   Place ml-20m raw files at:
#     /home/sjj/wenhao/TIGER/cache/MovieLens-20M/raw/ratings.csv
#     /home/sjj/wenhao/TIGER/cache/MovieLens-20M/raw/movies.csv
#   Download: https://grouplens.org/datasets/movielens/20m/
#
# Usage:
#   bash train_tiger_movielens20m.sh              # process all 3 lengths
#   bash train_tiger_movielens20m.sh 60           # process only len60
#   bash train_tiger_movielens20m.sh 60 90        # process len60 and len90

set -euo pipefail

TIGER_DIR="/home/sjj/wenhao/TIGER"
DISCO_CONVERT="/home/sjj/wenhao/DISCO/convert_movielens_to_ddbc.py"
CONDA_ENV="DDBC"
TIGER_GPU="${TIGER_GPU:-1}"                 # default use GPU1 to avoid busy GPU0
SENT_EMB_BATCH_SIZE="${SENT_EMB_BATCH_SIZE:-64}"  # lower batch to avoid OOM

# Determine which seq_lens to process
if [ $# -eq 0 ]; then
    SEQ_LENS=(60 90 120)
else
    SEQ_LENS=("$@")
fi

echo "================================================================"
echo "  MovieLens-20M data pipeline"
echo "  seq_lens: ${SEQ_LENS[*]}"
echo "================================================================"

# Verify raw files exist before starting
RAW_DIR="${TIGER_DIR}/cache/MovieLens-20M/raw"
for f in ratings.csv movies.csv; do
    if [ ! -f "${RAW_DIR}/${f}" ]; then
        echo "ERROR: ${RAW_DIR}/${f} not found."
        echo "Download ml-20m.zip from https://grouplens.org/datasets/movielens/20m/"
        echo "and extract ratings.csv + movies.csv into ${RAW_DIR}/"
        exit 1
    fi
done

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [seq_len=${SEQ_LEN}] Step 1/2: TIGER data processing"
    echo "──────────────────────────────────────────────────────────────"
    cd "${TIGER_DIR}"
    CUDA_VISIBLE_DEVICES="${TIGER_GPU}" conda run -n "${CONDA_ENV}" python main.py \
        --model=TIGER \
        --dataset=MovieLens20M \
        --category=MovieLens20M \
        --seq_len="${SEQ_LEN}" \
        --max_item_seq_len="${SEQ_LEN}" \
        --sent_emb_batch_size="${SENT_EMB_BATCH_SIZE}" \
        --process_only

    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  [seq_len=${SEQ_LEN}] Step 2/2: DISCO conversion"
    echo "──────────────────────────────────────────────────────────────"
    conda run -n "${CONDA_ENV}" python "${DISCO_CONVERT}" --seq_len "${SEQ_LEN}"

    echo "  [seq_len=${SEQ_LEN}] Done."
done

echo ""
echo "================================================================"
echo "  All done. Generated files:"
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    echo "  TIGER: ${TIGER_DIR}/cache/MovieLens-20M/len${SEQ_LEN}/processed/"
    echo "  DISCO: /home/sjj/wenhao/DISCO/datasets/MovieLens-20M/len${SEQ_LEN}/"
done
echo ""
echo "  DISCO training example (len60, 3 codebooks):"
echo "    python DISCO/main.py data=movielens20m_len60 seq_len=60 \\"
echo "      rq_n_codebooks=3 rq_codebook_size=256 model.length=302"
echo "  (len90 → model.length=452, len120 → model.length=602)"
echo "================================================================"
