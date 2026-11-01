(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Calls over two devices whose declines the cram test reads: one that donates
   none of its arguments, then one that donates its argument, then one that
   places its cut argument whole on each device. *)

let () =
  let ds =
    List.init 2 (fun k ->
        Result.get_ok (Rig.memory_device (Printf.sprintf "declines-%d" k)))
  in
  let module S = (val Nx.devices ds) in
  let x () = Nx.place (S.split ~axis:0) (Nx.ones Nx.float32 [| 4 |]) in
  let f = Rune.jit' (fun a -> Nx.mul a a) in
  ignore (Nx.to_array (f (x ())));
  ignore (Nx.to_array (f (Nx.donate (x ()))));
  let g = Rune.jit' (fun a -> Nx.mul_s (Nx.place S.on a) 2.) in
  ignore (Nx.to_array (g (x ())))
