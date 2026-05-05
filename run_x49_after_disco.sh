#!/bin/bash
# Wait for DISCO x49 process (PID 3191543) to finish, then run TIGER & LETTER x49
DISCO_PID=3191543
LOG_DIR="/home/sjj/wenhao/logs/ml60/canditate49"

echo "[$(date)] Waiting for DISCO x49 (PID $DISCO_PID) to complete..."

while kill -0 $DISCO_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date)] DISCO x49 finished! Running TIGER and LETTER x49..."

bash /home/sjj/wenhao/run_x49_tiger_letter.sh

echo "[$(date)] All x49 evaluations complete."
