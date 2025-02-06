#!/bin/bash

# 사용법: ./script.sh DATASET_DIR
DATASET_DIR="$1"

# 인자 체크
if [[ -z "$DATASET_DIR" ]]; then
    echo "Usage: $0 DATASET_DIR"
    exit 1
fi

# DATASET_DIR의 바로 하위 폴더들을 순회
for subfolder in "$DATASET_DIR"/*/; do
    if [[ -d "$subfolder" ]]; then
        echo "Processing folder: $subfolder"
        # 해당 폴더로 이동한 후 명령어 실행
        (
            cd "$subfolder" || exit
            obj2mjcf --obj-dir textured_objs --save-mjcf
        )
    fi
done
