#!/bin/bash
# fetch_model.sh: pull the Hy4-FP8 files listed in hy4_files.txt from the dev node's file server into /mnt/nvme
# (a file already at its listed size is skipped: reruns resume)
S=http://172.31.47.244:8090
cd /mnt/nvme && curl -s $S/work/dist/hy4_files.txt -o /mnt/nvme/hy4_files.txt
export S
cat /mnt/nvme/hy4_files.txt | xargs -P 32 -L 1 bash -c 'sz=$0; f=$1; [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$sz" ] && exit 0; mkdir -p "$(dirname "$f")"; curl -s --retry 5 -o "$f" "$S/$f" || echo "FAIL $f"'
echo FETCH_DONE $(du -sb /mnt/nvme/models/Hy4-FP8 | cut -f1)
