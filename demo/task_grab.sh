#!/bin/bash

# for demo purposes; extract participant/task data from kinematic_data.csv

TASK=$1
ID_NUM=$2
CSV="/Users/vickiwong/Desktop/cs4100/AI-Handwriting-Alzheimer-Diagnosis/datasets/kinematic_data.csv"

ID="id_${ID_NUM}"
OUTPUT="task_${TASK}_${ID}.csv"

awk -F',' -v id="$ID" -v task="$TASK" '
BEGIN {
    OFS=","
}

NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i == "ID" || $i == "class") {
            keep[++n] = i
        }
        if ($i == "ID") {
            id_col = i
        }
        if (match($i, /[0-9]+$/)) {
            suffix = substr($i, RSTART, RLENGTH)
            if (suffix == task) {
                keep[++n] = i
            }
        }
    }
    for (j=1; j<=n; j++) {
        printf "%s", $(keep[j])
        if (j < n) printf OFS
    }
    printf "\n"
    next
}

$(id_col) == id {
    for (j=1; j<=n; j++) {
        printf "%s", $(keep[j])
        if (j < n) printf OFS
    }
    printf "\n"
    exit
}
' "$CSV" > "$OUTPUT"

echo "Saved $OUTPUT"