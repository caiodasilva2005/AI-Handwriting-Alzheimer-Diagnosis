#!/bin/bash

# usage: ./extract_patient.sh [ID]
# ID is global; Alzheimer's patients are 1-89, healthy controls 90-174

ID_NUM=$1
CSV="/Users/vickiwong/Desktop/cs4100/AI-Handwriting-Alzheimer-Diagnosis/datasets/kinematic_data.csv"

ID="id_${ID_NUM}"
OUTPUT="${ID}.csv"

awk -F',' -v id="$ID" '
BEGIN {
    OFS=","
}

NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i == "ID") {
            id_col = i
            break
        }
    }

    print
    next
}

$(id_col) == id {
    print
    exit
}
' "$CSV" > "$OUTPUT"

echo "Saved $OUTPUT"