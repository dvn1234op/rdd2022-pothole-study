# Data Leakage & Duplicate Report

- Total images scanned: 38385
- Exact perceptual duplicates found: 73

## Impact on Splitting
These duplicate or near-duplicate frames MUST be kept in the same data split (train or test) to prevent data leakage. When we establish fixed splits, we will group images by their pHash.

