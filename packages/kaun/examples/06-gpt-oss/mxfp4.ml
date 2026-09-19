(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type blocks = (int, Nx.uint8_elt) Nx.t
type scales = (int, Nx.uint8_elt) Nx.t

let group_bytes = 16

(* Values are computed at float32, where every product of a code and a scale is
   exact, and cast once: bfloat16 arithmetic is emulated on the CPU device. *)

(* nx has no ldexp: the 256 powers of two are a table. *)
let scale_values =
  Array.init 256 (fun e ->
      if e = 255 then Float.nan else Float.ldexp 1.0 (e - 127))

let group_scales scales =
  let table = Nx.create Nx.float32 [| 256 |] scale_values in
  let indices = Nx.reshape [| -1 |] (Nx.cast Nx.int32 scales) in
  Nx.reshape (Nx.shape scales) (Nx.take ~indices table)

(* The magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 are m / 2 up to 4, m - 2 for 5
   and 6, then 6. Arithmetic, where a 16-entry table would be a gather: the
   scheduler materialises a gather's indices, four bytes for each weight. *)
let code_values codes =
  let m = Nx.cast Nx.float32 (Nx.bitwise_and codes (Nx.scalar Nx.uint8 7)) in
  let magnitude =
    Nx.where (Nx.less_s m 5.0) (Nx.mul_s m 0.5)
      (Nx.where (Nx.less_s m 7.0) (Nx.sub_s m 2.0) (Nx.full_like m 6.0))
  in
  let sign = Nx.cast Nx.float32 (Nx.rshift codes 3) in
  Nx.mul magnitude (Nx.rsub_s 1.0 (Nx.mul_s sign 2.0))

let dequant blocks scales dt =
  let shape = Nx.shape blocks in
  let rank = Array.length shape in
  if rank < 2 || shape.(rank - 1) <> group_bytes then
    invalid_arg "Mxfp4.dequant: blocks must have shape [...; groups; 16]";
  if Nx.shape scales <> Array.sub shape 0 (rank - 1) then
    invalid_arg
      "Mxfp4.dequant: scales must have the shape of blocks without its last \
       axis";
  let low = Nx.bitwise_and blocks (Nx.scalar Nx.uint8 15) in
  let high = Nx.rshift blocks 4 in
  let values = code_values (Nx.stack ~axis:(-1) [ low; high ]) in
  let scale =
    Nx.reshape (Array.append (Nx.shape scales) [| 1; 1 |]) (group_scales scales)
  in
  let out = Array.sub shape 0 (rank - 1) in
  out.(rank - 2) <- shape.(rank - 2) * group_bytes * 2;
  Nx.reshape out (Nx.cast dt (Nx.mul values scale))

let take_rows ids t =
  let rows = Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] ids) t in
  let rest = Array.sub (Nx.shape t) 1 (Nx.ndim t - 1) in
  Nx.reshape (Array.append (Nx.shape ids) rest) rows

let dequant_rows blocks scales ids dt =
  dequant (take_rows ids blocks) (take_rows ids scales) dt
