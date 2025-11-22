#!/bin/bash

# Download common SpGEMM benchmark matrices from SuiteSparse
# Based on typical matrices used in GPU SpGEMM papers

MATRIX_DIR="benchmark_matrices"
mkdir -p $MATRIX_DIR

echo "==================================================="
echo "Downloading SpGEMM Benchmark Matrices"
echo "==================================================="

# Base URL for SuiteSparse Matrix Collection
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# Common SpGEMM benchmark matrices (small to medium for testing)
declare -A MATRICES=(
    # Social Networks
    ["soc-Epinions1"]="SNAP/soc-Epinions1"
    ["ca-GrQc"]="SNAP/ca-GrQc"

    # Web Graphs
    ["web-Google"]="SNAP/web-Google"
    ["web-Stanford"]="SNAP/web-Stanford"

    # Citation Networks
    ["cit-Patents"]="SNAP/cit-Patents"

    # Collaboration Networks
    ["ca-CondMat"]="SNAP/ca-CondMat"

    # Circuits (smaller ones)
    ["rajat15"]="Rajat/rajat15"

    # Biology
    ["PROTEINS_OSM"]="Gleich/PROTEINS"
)

download_count=0
failed_count=0

for name in "${!MATRICES[@]}"; do
    path="${MATRICES[$name]}"
    url="$BASE_URL/$path.tar.gz"

    echo ""
    echo "---------------------------------------------------"
    echo "Downloading: $name"
    echo "URL: $url"

    wget -q --timeout=30 --tries=2 -O "$MATRIX_DIR/${name}.tar.gz" "$url"

    if [ $? -eq 0 ] && [ -f "$MATRIX_DIR/${name}.tar.gz" ]; then
        echo "✓ Downloaded successfully"

        # Extract
        cd $MATRIX_DIR
        tar -xzf "${name}.tar.gz" 2>/dev/null

        # Find and move .mtx file
        find . -name "*.mtx" -exec mv {} "${name}.mtx" \; 2>/dev/null

        # Cleanup
        rm -rf */
        rm "${name}.tar.gz"

        if [ -f "${name}.mtx" ]; then
            SIZE=$(du -h "${name}.mtx" | cut -f1)
            LINES=$(wc -l < "${name}.mtx" 2>/dev/null || echo "?")
            echo "  → ${name}.mtx (${SIZE}, ${LINES} lines)"
            ((download_count++))
        else
            echo "  ✗ Failed to extract .mtx file"
            ((failed_count++))
        fi

        cd ..
    else
        echo "✗ Download failed"
        ((failed_count++))
    fi
done

echo ""
echo "==================================================="
echo "Download Summary"
echo "==================================================="
echo "Successfully downloaded: $download_count matrices"
echo "Failed: $failed_count"
echo ""

if [ $download_count -gt 0 ]; then
    echo "Available benchmark matrices:"
    ls -lh $MATRIX_DIR/*.mtx 2>/dev/null || echo "  (none)"
    echo ""
    echo "To test SpMM:"
    echo "  ./build/main benchmark_matrices/<matrix>.mtx"
    echo ""
    echo "To test SpGEMM (A × A):"
    echo "  ./build/main benchmark_matrices/<matrix>.mtx benchmark_matrices/<matrix>.mtx"
else
    echo "No matrices downloaded successfully."
    echo "Creating fallback synthetic matrices..."

    # Create synthetic matrices as fallback
    python3 <<'PYTHON'
import random
import sys

def create_random_sparse_matrix(filename, rows, cols, nnz):
    """Create a random sparse matrix in MatrixMarket format"""
    entries = set()
    while len(entries) < nnz:
        r = random.randint(1, rows)
        c = random.randint(1, cols)
        entries.add((r, c))

    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{rows} {cols} {nnz}\n")
        for r, c in sorted(entries):
            val = random.uniform(0.1, 10.0)
            f.write(f"{r} {c} {val:.6f}\n")
    print(f"  Created {filename}: {rows}×{cols} with {nnz} non-zeros")

create_random_sparse_matrix("benchmark_matrices/synthetic_1000.mtx", 1000, 1000, 10000)
create_random_sparse_matrix("benchmark_matrices/synthetic_5000.mtx", 5000, 5000, 50000)
print("\nSynthetic matrices created successfully!")
PYTHON
fi

echo ""
echo "==================================================="
echo "Ready for benchmarking!"
echo "==================================================="
