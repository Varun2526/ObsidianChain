"""Dataset loading and normalisation.

Reads Elliptic++ CSVs from the mounted data directory and normalises them
into a stable internal schema. Loaders are pure: path in, dataframe out,
no downloads, no caching to anywhere outside data/processed.

Note: this subpackage is named `io` but does not shadow the standard
library for absolute imports elsewhere in the codebase.
"""
