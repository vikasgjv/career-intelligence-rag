"""
Configuration constants for evidence classification thresholds and scoring weights.
These initial heuristics can be tuned through evaluation datasets.
"""

# Similarity score thresholds
STRONG_EVIDENCE_THRESHOLD = 0.40
PARTIAL_EVIDENCE_THRESHOLD = 0.30

# Evidence weights for Overall Match Coverage calculation
WEIGHT_STRONG_EVIDENCE = 1.0
WEIGHT_PARTIAL_EVIDENCE = 0.5
WEIGHT_MISSING_EVIDENCE = 0.0

# Ollama local LLM configuration
OLLAMA_MODEL = "llama3.2:3b"
