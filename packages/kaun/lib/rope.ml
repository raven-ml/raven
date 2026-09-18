(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = float array

let base ~fn ~theta ~head_dim =
  if head_dim <= 0 || head_dim mod 2 <> 0 then
    Printf.ksprintf invalid_arg
      "Rope.%s: head_dim must be positive and even, got %d" fn head_dim;
  if theta <= 0.0 then
    Printf.ksprintf invalid_arg "Rope.%s: theta must be positive, got %g" fn
      theta;
  Array.init (head_dim / 2) (fun i ->
      theta ** (-2.0 *. float_of_int i /. float_of_int head_dim))

let make ?(theta = 10000.0) ~head_dim () = base ~fn:"make" ~theta ~head_dim

let llama3 ~theta ~head_dim ~factor ~low_freq_factor ~high_freq_factor
    ~original_context =
  if factor <= 0.0 || low_freq_factor <= 0.0 || original_context <= 0 then
    invalid_arg
      "Rope.llama3: factor, low_freq_factor and original_context must be \
       positive";
  if high_freq_factor <= low_freq_factor then
    invalid_arg "Rope.llama3: high_freq_factor must exceed low_freq_factor";
  let context = float_of_int original_context in
  let low_wavelen = context /. low_freq_factor in
  let high_wavelen = context /. high_freq_factor in
  Array.map
    (fun f ->
      let wavelen = 2.0 *. Float.pi /. f in
      if wavelen < high_wavelen then f
      else if wavelen > low_wavelen then f /. factor
      else
        let smooth =
          ((context /. wavelen) -. low_freq_factor)
          /. (high_freq_factor -. low_freq_factor)
        in
        ((1.0 -. smooth) *. f /. factor) +. (smooth *. f))
    (base ~fn:"llama3" ~theta ~head_dim)

let frequencies t = Array.copy t

let apply t ~pos x =
  let shape = Nx.shape x in
  if Array.length shape <> 4 then
    invalid_arg "Rope.apply: x must have shape [batch; heads; seq; head_dim]";
  let batch = shape.(0) and seq = shape.(2) and head_dim = shape.(3) in
  let half = Array.length t in
  if head_dim <> 2 * half then
    Printf.ksprintf invalid_arg
      "Rope.apply: x has head_dim %d but the frequencies are for %d" head_dim
      (2 * half);
  (match Nx.shape pos with
  | [| b; s |] when (b = batch || b = 1) && s = seq -> ()
  | _ ->
      Printf.ksprintf invalid_arg
        "Rope.apply: pos must have shape [%d; %d] or [1; %d]" batch seq seq);
  (* Angles at float32 whatever [x]'s dtype: a half precision product of a long
     position and a small frequency has neither the range nor the mantissa. *)
  let freqs = Nx.create Nx.float32 [| 1; 1; 1; half |] t in
  let angles =
    Nx.mul
      (Nx.reshape [| Nx.dim 0 pos; 1; seq; 1 |] (Nx.cast Nx.float32 pos))
      freqs
  in
  let dt = Nx.dtype x in
  let cos = Nx.cast dt (Nx.cos angles) and sin = Nx.cast dt (Nx.sin angles) in
  let x1 = Nx.slice [ A; A; A; R (0, half) ] x in
  let x2 = Nx.slice [ A; A; A; R (half, head_dim) ] x in
  Nx.concatenate ~axis:3
    [
      Nx.sub (Nx.mul x1 cos) (Nx.mul x2 sin);
      Nx.add (Nx.mul x2 cos) (Nx.mul x1 sin);
    ]
