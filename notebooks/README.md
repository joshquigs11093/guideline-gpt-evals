# Notebooks

`analysis.ipynb` is the long-form, narrative complement to the dashboard. Where
the dashboard is browseable, the notebook walks through all six experiments as a
story: hypothesis → method → result → takeaway (spec §7), plus cross-experiment
insights, limitations, and future work.

## Viewing

- **Rendered HTML:** open `analysis.html` in a browser (charts load via the
  Plotly CDN). This is the zero-setup way to read it.
- **On nbviewer:** paste the GitHub URL of `analysis.ipynb` into
  [nbviewer](https://nbviewer.org) for interactive charts. (GitHub's own
  notebook preview strips the chart JavaScript, so charts look blank there.)
- **Locally:** `jupyter lab notebooks/analysis.ipynb`.

## Regenerating

The notebook is generated and executed from the committed results, then exported:

```bash
python scripts/build_notebook.py    # build + execute -> analysis.ipynb
python scripts/export_notebook.py   # convert -> analysis.html
```

> The numbers are **synthetic** (see `results/SYNTHETIC_DATA.md`) until real
> experiment runs exist.
