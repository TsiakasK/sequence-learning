#!/usr/bin/env bash

./mds.py
echo "MDS clustering done"

./extract_engagement_signal.py
echo "engagement extraction done"

./normalize_engagement_signal.py
echo "engagement normalization done"

./statistical_analysis.py

