# LexBrief: Report Outline

## 1. Introduction
- Problem: long, implicitly structured Indian judgments, and the cost of manual case briefs
- Idea: rhetorical roles as structure for extractive briefing
- Contributions (bullet list)

## 2. Related Work
- Rhetorical role labelling (BUILD / InRhetoricalRoles, LegalEval 2023)
- Legal-domain language models (InLegalBERT, LegalBERT)
- Legal summarisation (IN-Ext, IN-Abs; extractive vs. abstractive)
- Hierarchical sequence labelling (BiLSTM-CRF, HSLN)

## 3. Data
- BUILD rhetorical-role corpus: sources, 13 labels, statistics
- IN-Ext expert summaries: segment labels, statistics
- Preprocessing: sentence splitting, alignment, splits
- Fine-to-coarse label mapping (table)

## 4. Method
- M0 TF-IDF baseline, M1 sentence encoders, M2 context window
- M3a cached embeddings + BiLSTM-CRF; M3b end-to-end
- Brief generation: learned per-role word budgets, sentence scoring, selection
- Training details (6 GB GPU, bf16, hyperparameters)

## 5. Experiments
- Research questions
- Classifier evaluation protocol (macro-F1, per-label F1)
- Brief evaluation protocol (ROUGE, per-role ROUGE, role coverage)
- Gold-vs-predicted roles study and label-noise analysis
- Baselines (lead, TextRank/LexRank, role-agnostic, oracle)

## 6. Results
- Classifier results table and confusion matrix
- Brief results table with significance tests
- Per-role analysis and qualitative example

## 7. Demo
- Streamlit app: upload, role-coloured judgment, brief, export
- Screenshots

## 8. Limitations
- Domain and court coverage, label noise, extractive-only output, compute constraints
- Ethical considerations and future work
