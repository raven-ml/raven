# nx-io

Input/Output library for Nx

nx-io provides functions to load and save Nx tensors in image formats,
NumPy formats (`.npy`, `.npz`), and more.

## Features

- Image I/O: `load_image`, `save_image` (PNG, JPEG, BMP, TGA, GIF)
- NumPy `.npy` format: `load_npy`, `save_npy`
- NumPy `.npz` archives: `load_npz`, `load_npz_member`, `save_npz`
- `packed_nx` for runtime-detected dtypes
- Conversion utilities: `to_float32`, `to_uint8`, etc.

## Quick Start

```ocaml
open Nx
open Nx_io

(* Image I/O *)
let () =
  (* uint8 nx [|H;W;C|] or [|H;W|] *)
  let img = load_image "photo.png" in
  save_image img "copy.png";

  (* NumPy I/O *)
  let arr = load_npy "array.npy" |> to_float32 in
  save_npy arr "out.npy";

  (* NPZ archive *)
  let archive = load_npz "bundle.npz"
  match Hashtbl.find_opt archive "data" with
  | Some packed_arr -> 
    let arr = to_float32 packed_arr in
    save_npz ("data", P arr) "out.npz"
  | None -> failwith "data not found"
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
