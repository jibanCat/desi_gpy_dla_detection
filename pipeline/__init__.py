"""HBI results-store pipeline — DAG driver that CHAINS existing producers into a
structured, reproducible store (implements ``pipeline/IMPLEMENTATION_PLAN.md``
tasks 3-5). This package does NOT rewrite any science; every stage is a thin
wrapper around an existing producer entry point, writing into a write-once
``ResultStore`` leaf with a provenance stamp.
"""
