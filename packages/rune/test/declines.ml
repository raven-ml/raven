(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Calls over two devices whose declines and moves the cram test reads: one
   that donates none of its arguments, then one that donates its argument, then
   one that places its cut argument whole on each device, then an iterate whose
   stop lies on the first device alone, where the program computes on the
   second, its first argument's. *)

let () =
  let ds =
    List.init 2 (fun k ->
        Result.get_ok (Rig.software (Printf.sprintf "declines-%d" k)))
  in
  let module S = (val Nx.devices ds) in
  let x () = Nx.place (S.split ~axis:0) (Nx.ones Nx.float32 [| 4 |]) in
  let f = Rune.jit' (fun a -> Nx.mul a a) in
  ignore (Nx.to_array (f (x ())));
  ignore (Nx.to_array (f (Nx.donate (x ()))));
  let g = Rune.jit' (fun a -> Nx.mul_s (Nx.place S.on a) 2.) in
  ignore (Nx.to_array (g (x ())));
  let on k = Nx.Placement.device S.v (List.nth ds k) in
  let s = Nx.place (on 1) (Nx.ones Nx.float32 [||]) in
  let x = Nx.place (on 0) (Nx.ones Nx.float32 [| 4 |]) in
  let halve x = Nx.mul_s x 0.5 in
  let small x = Nx.less_s (Nx.max x) 0.01 in
  let h =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
      (fun s x -> (Nx.neg s, Rune.iterate' ~max:10 ~until:small halve x))
  in
  ignore (Nx.to_array (snd (h s x)))
