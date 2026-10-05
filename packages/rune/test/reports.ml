(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Calls whose reports the cram test reads: a consumed argument lent, then a
   retrace, then a consumed argument whose memory is borrowed. *)

let step =
  Rune.jit
    Nx.Ptree.(consumes (pair tensor tensor) @@ returns (pair tensor tensor))
    (fun (keys, values) -> (Nx.add_s keys 1., Nx.mul_s values 2.))

let () =
  let pair n =
    (Nx.copy (Nx.zeros Nx.float32 [| n |]), Nx.copy (Nx.ones Nx.float32 [| n |]))
  in
  ignore (step (pair 4));
  ignore (step (pair 8));
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 8 in
  ignore
    (step
       ( Nx.of_bigarray (Bigarray.genarray_of_array1 ba),
         Nx.copy (Nx.ones Nx.float32 [| 8 |]) ))

(* Retraces for a view's strides and for where its run starts within 16 bytes of
   memory. *)
let () =
  let neg = Rune.jit' Nx.neg in
  let grid = Nx.reshape [| 2; 3 |] (Nx.arange_f Nx.float32 0. 6. 1.) in
  ignore (neg grid);
  ignore
    (neg
       (Nx.transpose (Nx.reshape [| 3; 2 |] (Nx.arange_f Nx.float32 0. 6. 1.))));
  let a = Nx.arange_f Nx.float32 0. 12. 1. in
  ignore (neg (Nx.slice [ R (0, 4) ] a));
  ignore (neg (Nx.slice [ R (1, 5) ] a))

(* A consumed leaf no result can take: another dtype. *)
let () =
  ignore
    (Rune.jit
       Nx.Ptree.(consumes tensor @@ returns tensor)
       (fun a -> Nx.cast Nx.float64 a)
       (Nx.ones Nx.float32 [| 4 |]))

(* A retrace for a setting a caller changed around the call. *)
let () =
  let neg = Rune.jit' Nx.neg in
  let a = Nx.ones Nx.float32 [| 4 |] in
  ignore (neg a);
  Tolk.Setting.context
    [ B (Tolk.Setting.noopt, true) ]
    (fun () -> ignore (neg a))

(* A retrace for the counters of the profile being taken: one counter whose name
   holds "; " against two. *)
let () =
  let neg = Rune.jit' Nx.neg in
  let a = Nx.ones Nx.float32 [| 4 |] in
  let under counters =
    let p = Nx_device.Profile.start ~counters () in
    ignore (neg a);
    ignore (Nx_device.Profile.stop p)
  in
  under [ "a; b" ];
  under [ "a"; "b" ]

(* A function compiled with a search reports each kernel it searches; another
   compiled without one, in the same process, reports none. *)
let () =
  let f a = Nx.add_s (Nx.mul a a) 0.625 in
  ignore (Rune.jit' ~beam:1 f (Nx.ones Nx.float32 [| 4 |]));
  let g a = Nx.add_s (Nx.mul a a) 0.875 in
  ignore (Rune.jit' g (Nx.ones Nx.float32 [| 4 |]));
  (* An explicit width overrides BEAM, [0] searching nothing; another width
     searches again. *)
  Tolk.Setting.context
    [ B (Tolk.Setting.beam, 1) ]
    (fun () -> ignore (Rune.jit' ~beam:0 g (Nx.ones Nx.float32 [| 4 |])));
  ignore (Rune.jit' ~beam:2 f (Nx.ones Nx.float32 [| 4 |]))
