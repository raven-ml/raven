(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Past 62 bits the range of codes, [2^bits], no longer fits an int64. *)
let max_bits = 62

let invalid fmt = Printf.ksprintf invalid_arg fmt

(* Bit [bits - 1 - c] for number [c], so number 0 is the most significant, as
   upstream's LFQ masks with 2^(arange(bits - 1, -1, -1)). *)
let weights at bits =
  Nx.place at
    (Nx.create Nx.int64 [| bits |]
       (Array.init bits (fun c -> Int64.shift_left 1L (bits - 1 - c))))

(* A number of exactly 0 gives a clear bit, as upstream's [x > 0]. *)
let encode z =
  let s = Nx.shape z in
  let r = Array.length s in
  if r = 0 then invalid "Aion.Lfq.encode: a 0-d array, a last axis expected";
  if s.(r - 1) > max_bits then
    invalid "Aion.Lfq.encode: last axis of %d entries, at most %d expected"
      s.(r - 1) max_bits;
  let w = weights (Nx.beside z) s.(r - 1) in
  Nx.sum ~axes:[ -1 ] (Nx.mul (Nx.cast Nx.int64 (Nx.greater_s z 0.)) w)

let decode dt ~bits k =
  if bits < 0 || bits > max_bits then
    invalid "Aion.Lfq.decode: %d bits, [0, %d] expected" bits max_bits;
  let n = Int64.shift_left 1L bits in
  let inside = Nx.logical_and (Nx.greater_equal_s k 0L) (Nx.less_s k n) in
  Nx.check Nx.Ptree.tensor inside k (fun i k ->
      Invalid_argument
        (Format.asprintf "Aion.Lfq.decode: code %Ld at %a outside [0, %Ld)"
           (Nx.item [] k) Nx.pp_shape i n));
  let k = Nx.unsqueeze ~axes:[ -1 ] k in
  let set = Nx.not_equal_s (Nx.bitwise_and k (weights (Nx.beside k) bits)) 0L in
  let one = Nx.ones_at (Nx.beside k) dt [||] in
  Nx.where set one (Nx.neg one)
