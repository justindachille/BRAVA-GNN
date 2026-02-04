#!/bin/bash

TARGET_DIR="datasets/real_graph"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit

echo "--- Starting Download & Processing in $TARGET_DIR ---"

get_snap_gz() {
    url=$1
    name=$2
    
    if [ -f "${name}.txt" ] && [ -s "${name}.txt" ]; then
        echo "Skipping $name (already exists)."
        return
    fi

    filename=$(basename "$url")
    echo "Processing $name..."
    wget -qN "$url"
    
    # Decompress
    gzip -d -f "$filename"
    unzipped_name="${filename%.gz}"
    
    # Find the file (handle potential renaming or wget behavior)
    if [ -f "$unzipped_name" ]; then
        mv "$unzipped_name" "$name.txt"
        # Remove comments (lines starting with #)
        sed -i '/^#/d' "$name.txt"
    else
        # Fallback: look for the most recent txt file
        found=$(ls -t *.txt 2>/dev/null | head -n 1)
        if [ -f "$found" ] && [ "$found" != "$name.txt" ]; then
            mv "$found" "$name.txt"
            sed -i '/^#/d' "$name.txt"
        fi
    fi
}

get_nrvis_zip() {
    url=$1
    name=$2
    
    if [ -f "${name}.txt" ] && [ -s "${name}.txt" ]; then
        echo "Skipping $name (already exists)."
        return
    fi

    filename=$(basename "$url")
    echo "Processing $name..."
    wget -qN "$url"
    unzip -o -q "$filename"
    
    # Find largest data file in the unzipped contents
    data_file=$(find . -maxdepth 2 -type f -not -name "*.zip" -not -name "*.txt" | xargs ls -S 2>/dev/null | head -n 1)
    
    if [ -f "$data_file" ]; then
        mv "$data_file" "$name.txt"
        # Remove header lines (lines starting with %) often found in NRVis data
        sed -i '/^%/d' "$name.txt"
    else
        echo "Error: Failed to process $name"
    fi
    # Cleanup artifacts
    rm -f "$filename" readme* *.html
}

echo "=== Road Networks ==="
get_nrvis_zip "https://nrvis.com/download/data/road/road-luxembourg-osm.zip" "road-luxembourg-osm"
get_snap_gz "https://snap.stanford.edu/data/roadNet-PA.txt.gz" "road-roadNet-PA"
get_nrvis_zip "https://nrvis.com/download/data/road/road-belgium-osm.zip" "road-belgium-osm"
get_snap_gz "https://snap.stanford.edu/data/roadNet-CA.txt.gz" "road-roadNet-CA"
get_nrvis_zip "https://nrvis.com/download/data/road/road-netherlands-osm.zip" "road-netherlands-osm"

echo "=== Social and Web Networks ==="
get_snap_gz "https://snap.stanford.edu/data/p2p-Gnutella31.txt.gz" "p2p-Gnutella31"
get_snap_gz "https://snap.stanford.edu/data/soc-Epinions1.txt.gz" "soc-Epinions1"
get_snap_gz "https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz" "soc-Slashdot0902"
get_snap_gz "https://snap.stanford.edu/data/email-EuAll.txt.gz" "email-EuAll"
get_snap_gz "https://snap.stanford.edu/data/web-Google.txt.gz" "web-Google"

get_snap_gz "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz" "soc-Pokec"
get_snap_gz "https://snap.stanford.edu/data/wiki-TopCats.txt.gz" "wiki-topcats"
get_snap_gz "https://snap.stanford.edu/data/wiki-Talk.txt.gz" "wiki-Talk"
get_snap_gz "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz" "soc-LiveJournal1"

# NOTE: The following datasets are sourced from the ABCDE repository release 
# and are handled by baselines/convert_abcde.py:
# - com-youtube
# - amazon
# - cit-Patents
# - com-lj
# - dblp

echo "--- Done. Files in $TARGET_DIR: ---"
ls -lh *.txt