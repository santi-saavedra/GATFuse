#!/usr/bin/env bash
#
# Align an RNA-seq sample with STAR and produce the three files GATFuse needs:
#   <prefix>Aligned.sortedByCoord.out.bam
#   <prefix>Chimeric.out.junction
#   <prefix>ReadsPerGene.out.tab
#
# The chimeric-detection parameters below are the ones the published model was
# trained with. Changing them changes the candidate set GATFuse sees, so keep
# them unless you retrain.
#
# Usage:
#   scripts/run_star.sh -i STAR_INDEX -1 READS_1.fastq [-2 READS_2.fastq] [-o OUTDIR] [-p PREFIX] [-t THREADS]

set -euo pipefail

THREADS=8
OUTDIR="."
PREFIX="sample_"
READS_2=""

usage() {
    cat >&2 <<'USAGE'
Usage: run_star.sh -i STAR_INDEX -1 READS_1 [-2 READS_2] [-o OUTDIR] [-p PREFIX] [-t THREADS]

  -i  STAR genome index directory (built with the same GTF passed to GATFuse)
  -1  FASTQ of read 1 (gzipped input is detected automatically)
  -2  FASTQ of read 2; omit for single-end data
  -o  output directory                    (default: .)
  -p  output file prefix                  (default: sample_)
  -t  threads                             (default: 8)
USAGE
    exit 2
}

while getopts ":i:1:2:o:p:t:h" opt; do
    case "$opt" in
        i) INDEX="$OPTARG" ;;
        1) READS_1="$OPTARG" ;;
        2) READS_2="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        p) PREFIX="$OPTARG" ;;
        t) THREADS="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "${INDEX:-}" ] || [ -z "${READS_1:-}" ]; then
    usage
fi
command -v STAR >/dev/null 2>&1 || { echo "STAR is not on PATH." >&2; exit 3; }
[ -d "$INDEX" ] || { echo "STAR index directory not found: $INDEX" >&2; exit 3; }
[ -f "$READS_1" ] || { echo "FASTQ not found: $READS_1" >&2; exit 3; }
if [ -n "$READS_2" ] && [ ! -f "$READS_2" ]; then
    echo "FASTQ not found: $READS_2" >&2
    exit 3
fi

mkdir -p "$OUTDIR"

READ_COMMAND=()
case "$READS_1" in
    *.gz) READ_COMMAND=(--readFilesCommand zcat) ;;
esac

STAR \
    --runThreadN "$THREADS" \
    --genomeDir "$INDEX" \
    --readFilesIn "$READS_1" ${READS_2:+"$READS_2"} \
    "${READ_COMMAND[@]}" \
    --outFileNamePrefix "${OUTDIR%/}/${PREFIX}" \
    --outSAMtype BAM SortedByCoordinate \
    --quantMode GeneCounts \
    --outFilterMultimapNmax 20 \
    --outSAMmapqUnique 60 \
    --chimSegmentMin 10 \
    --chimJunctionOverhangMin 10 \
    --chimScoreMin 1 \
    --chimScoreDropMax 30 \
    --chimOutType Junctions WithinBAM SoftClip \
    --chimOutJunctionFormat 1

echo "STAR finished. GATFuse inputs:" >&2
echo "  --bam                ${OUTDIR%/}/${PREFIX}Aligned.sortedByCoord.out.bam" >&2
echo "  --chimeric-junctions ${OUTDIR%/}/${PREFIX}Chimeric.out.junction" >&2
echo "  --gene-counts        ${OUTDIR%/}/${PREFIX}ReadsPerGene.out.tab" >&2
