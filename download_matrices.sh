#!/bin/bash

# Script to download test matrices from SuiteSparse Matrix Collection
# https://sparse.tamu.edu/

MATRIX_DIR="test_matrices"
mkdir -p $MATRIX_DIR

echo "Downloading SuiteSparse test matrices..."

# Small matrices for quick testing
SMALL_MATRICES=(
    "https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/chesapeake.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Bai/dwt_234.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/Bai/qc324.tar.gz"
)

# Medium matrices
MEDIUM_MATRICES=(
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/ca-GrQc.tar.gz"
    "https://suitesparse-collection-website.herokuapp.com/MM/SNAP/soc-Epinions1.tar.gz"
)

# Alternative: Direct download links (more reliable)
echo "Note: SuiteSparse website may have changed. Using alternative method..."

# Using wget to download specific matrices
wget -P $MATRIX_DIR https://suitesparse-collection-website.herokuapp.com/MM/Bai/dwt_234.tar.gz 2>/dev/null
wget -P $MATRIX_DIR https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/chesapeake.tar.gz 2>/dev/null

# Extract if downloads succeeded
cd $MATRIX_DIR
for file in *.tar.gz; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xzf "$file"
        # Find and move .mtx file to root of test_matrices
        find . -name "*.mtx" -exec mv {} . \;
        rm -rf $(tar -tzf "$file" | head -1 | cut -f1 -d"/")
        rm "$file"
    fi
done

cd ..

# If downloads failed, create simple test matrices
if [ ! -f "$MATRIX_DIR/*.mtx" ]; then
    echo "Download failed or no .mtx files found. Creating simple test matrices..."

    # Create a simple 10x10 sparse matrix
    cat > $MATRIX_DIR/test_10x10.mtx <<EOF
%%MatrixMarket matrix coordinate real general
10 10 20
1 1 1.0
1 5 2.0
2 2 3.0
2 7 4.0
3 3 5.0
3 8 6.0
4 4 7.0
4 9 8.0
5 5 9.0
5 10 10.0
6 1 11.0
6 6 12.0
7 2 13.0
7 7 14.0
8 3 15.0
8 8 16.0
9 4 17.0
9 9 18.0
10 5 19.0
10 10 20.0
EOF

    # Create a simple 100x100 sparse matrix
    cat > $MATRIX_DIR/test_100x100.mtx <<EOF
%%MatrixMarket matrix coordinate real general
100 100 300
EOF

    # Generate random sparse entries
    for i in $(seq 1 300); do
        row=$((RANDOM % 100 + 1))
        col=$((RANDOM % 100 + 1))
        val=$(awk -v seed=$RANDOM 'BEGIN{srand(seed); print rand()}')
        echo "$row $col $val" >> $MATRIX_DIR/test_100x100.mtx
    done

    echo "Created synthetic test matrices."
fi

echo "Test matrices ready in $MATRIX_DIR/"
ls -lh $MATRIX_DIR/*.mtx

echo ""
echo "To test SpMM, run:"
echo "  ./main test_matrices/<matrix>.mtx"
echo ""
echo "To test SpGEMM, run:"
echo "  ./main test_matrices/<matrix1>.mtx test_matrices/<matrix2>.mtx"
