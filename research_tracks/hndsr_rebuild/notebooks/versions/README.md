# Versioned Kaggle Notebooks

- `vR.x_HNDSR.ipynb` is reserved for scratch-trained notebook versions.
- `vR.P.x_HNDSR.ipynb` is reserved for externally pretrained notebook versions.
- Do not overwrite a reviewed notebook version.
- Pair every notebook with:
  - `research_tracks/hndsr_rebuild/docs/notebooks/<stem>.md`
  - `research_tracks/hndsr_rebuild/reports/reviews/<stem>.review.md`
- Run `research_tracks/hndsr_rebuild/scripts/validate_notebook_version.py` before handing a notebook to Kaggle.
