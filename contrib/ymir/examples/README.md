# Ymir Examples

Each example teaches one part of ymir on small synthetic data and runs in
well under a second. Start with `01-units` and work through them in order.

| Example | Concept |
|---------|---------|
| [`01-units`](./01-units/) | Units and quantities |
| [`02-constants-and-names`](./02-constants-and-names/) | Constants and names |
| [`03-frames-and-directions`](./03-frames-and-directions/) | Frames and directions |
| [`04-transforms`](./04-transforms/) | Transforms |
| [`05-fits-images`](./05-fits-images/) | FITS images and headers |
| [`06-fits-tables`](./06-fits-tables/) | FITS tables |
| [`07-fits-wcs`](./07-fits-wcs/) | World coordinates from a header |
| [`08-grids-and-regions`](./08-grids-and-regions/) | Grids and regions |
| [`09-observations`](./09-observations/) | Observations and aperture sums |
| [`10-cosmology`](./10-cosmology/) | Background cosmology |
| [`11-derivatives`](./11-derivatives/) | Derivatives and compilation |

Run one from the repository root:

```bash
dune exec contrib/ymir/examples/01-units/main.exe
```
