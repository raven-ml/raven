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
