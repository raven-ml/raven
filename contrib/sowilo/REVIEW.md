# Reviewing sowilo

## Design

- An operation written with its own loop where nx operations compose it;
  rune can't differentiate a loop sowilo hides.

## Semantics

References: the standard each color conversion names (ITU-R BT.601 for
luma), and the paper of each named filter.

Failure here is an image changed at its borders or in its range: a border
mode applied wrong, values clipped or wrapped in an integer dtype, a color
conversion with the wrong weights.

- Borders: one-pixel images, kernels larger than the image.
- Integer dtypes at their extremes.
- Gradients through each operation where the `.mli` promises them.
