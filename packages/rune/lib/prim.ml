(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx's operations as functions, each evaluated in the current interpretation:
   what the transformations' rules issue. *)

open Nx.Op

let unary k x = eval (Unary (k, x))
let binary k x y = eval (Binary (k, x, y))
let cmp k x y = eval (Compare (k, x, y))
let where c x y = eval (Where (c, x, y))
let reduce k ~axes x = eval (Reduce (k, axes, x))
let scan k ~axis x = eval (Scan (k, axis, x))
let arg_reduce k ~axis x = eval (Arg_reduce (k, axis, x))
let sort ~descending ~axis x = eval (Sort { descending; axis; x })
let argsort ~descending ~axis x = eval (Argsort { descending; axis; x })
let pad padding v x = eval (Pad (padding, v, x))
let cat ~axis xs = eval (Cat (axis, xs))
let cast dtype x = eval (Convert (Cast, dtype, x))
let bitcast dtype x = eval (Convert (Bitcast, dtype, x))
let threefry key ctr = eval (Threefry (key, ctr))
let gather ~axis indices x = eval (Gather (axis, indices, x))

let scatter ~mode ~unique ~axis ~indices ~updates into =
  eval (Scatter { mode; unique; axis; indices; updates; into })

let update x ~starts v = eval (Update (x, starts, v))

let unfold ~kernel_size ~stride ~dilation ~padding x =
  eval (Unfold { kernel_size; stride; dilation; padding; x })

let fold ~output_size ~kernel_size ~stride ~dilation ~padding x =
  eval (Fold { output_size; kernel_size; stride; dilation; padding; x })

let matmul x y = eval (Matmul (x, y))
let fft ~inverse ~axes x = eval (Fft { inverse; axes; x })
let rfft dtype ~axes x = eval (Rfft { dtype; axes; x })
let irfft ?s dtype ~axes x = eval (Irfft { dtype; axes; s; x })
let copy x = eval (Contiguous x)
let cholesky ~upper x = eval (Cholesky { upper; x })
let qr ~reduced x = eval (Qr { reduced; x })
let lu x = eval (Lu x)

let solve_triangular ~upper ~transpose ~unit_diag a b =
  eval (Solve_triangular { upper; transpose; unit_diag; a; b })

let move x m = eval (Move (x, m))
let reshape x shape = move x (Reshape shape)
let expand x shape = move x (Expand shape)
let permute x axes = move x (Permute axes)
let shrink x limits = move x (Shrink limits)
let flip x dims = move x (Flip dims)

let sliding_window x ~axis ~window ~step =
  move x (Window { axis; size = window; step })

let place p x = eval (Place (p, x))
