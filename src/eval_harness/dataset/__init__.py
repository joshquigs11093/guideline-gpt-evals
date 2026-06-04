"""Eval dataset construction, loading, and validation.

The generator (used once, then retired) produces candidate Q&A pairs; the loader
reads and schema-validates the curated set; the validator enforces corpus
consistency in CI.
"""
