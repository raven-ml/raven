(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Calls whose reports the cram test reads: a consumed argument lent, then a
   retrace, then a consumed argument whose memory is borrowed. *)

let step =
  Rune_next.Rune.jit
    Nx.Ptree.(consumes (pair tensor tensor) @@ returns (pair tensor tensor))
    (fun (keys, values) -> (Nx.add_s keys 1., Nx.mul_s values 2.))

let () =
  let pair n = (Nx.zeros Nx.float32 [| n |], Nx.ones Nx.float32 [| n |]) in
  ignore (step (pair 4));
  ignore (step (pair 8));
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 8 in
  ignore
    (step
       ( Nx.of_bigarray (Bigarray.genarray_of_array1 ba),
         Nx.ones Nx.float32 [| 8 |] ))

(* Retraces for a view's strides and for where its run starts within 16 bytes of
   memory. *)
let () =
  let neg = Rune_next.Rune.jit' Nx.neg in
  let grid = Nx.reshape [| 2; 3 |] (Nx.arange_f Nx.float32 0. 6. 1.) in
  ignore (neg grid);
  ignore
    (neg
       (Nx.transpose (Nx.reshape [| 3; 2 |] (Nx.arange_f Nx.float32 0. 6. 1.))));
  let a = Nx.arange_f Nx.float32 0. 12. 1. in
  ignore (neg (Nx.slice [ R (0, 4) ] a));
  ignore (neg (Nx.slice [ R (1, 5) ] a))
