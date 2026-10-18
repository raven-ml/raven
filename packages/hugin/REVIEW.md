# Reviewing hugin

## Design

- A fix in one mark for clipping, layout or scaling: does it belong in the
  scale, the panel or the layout, where every mark would get it?
- A per-backend branch (PNG, SVG, PDF) for what the scene should state.

## Semantics

References: the PNG, SVG and PDF specifications, OpenType for the font,
ISO 8601 and RFC 3339 for time axes.

Failure here is a figure that draws something other than its data: a mark
outside its panel, a scale applied twice or not at all, a bin of zero
width, a line reduced where the reduction changes what is drawn. Then
output that differs between systems.

- Empty data, one point, NaN and infinities, a zero-width domain.
- Clipping in facets, grids and zoomed panels.
- Ticks at domain edges and for very small or large ranges.
- A golden image change: read the diff and name why it changed.

## Skip

`test/golden/`, generated example images, `bench/*.thumper`.
