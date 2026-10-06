(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  frequencies : float array;
  cos : Nx.float32_t; (* [| context; pairs |] *)
  sin : Nx.float32_t;
}

(* The angles in float64, rounded once to float32: a float32 product of a long
   position and a small frequency loses the angle's low bits. *)
let table ~context frequencies f =
  let pairs = Array.length frequencies in
  let values = Array.make (context * pairs) 0.0 in
  for p = 0 to context - 1 do
    for i = 0 to pairs - 1 do
      values.((p * pairs) + i) <- f (float_of_int p *. frequencies.(i))
    done
  done;
  Nx.create Nx.float32 [| context; pairs |] values

let of_frequencies ~context frequencies =
  if Array.length frequencies = 0 then
    invalid_arg "Rope.of_frequencies: no frequency";
  if not (Array.for_all Float.is_finite frequencies) then
    invalid_arg "Rope.of_frequencies: a frequency is not finite";
  if context <= 0 then
    Printf.ksprintf invalid_arg
      "Rope.of_frequencies: context must be positive, got %d" context;
  let frequencies = Array.copy frequencies in
  {
    frequencies;
    cos = table ~context frequencies Float.cos;
    sin = table ~context frequencies Float.sin;
  }

let base ~fn ~theta ~head_dim =
  if head_dim <= 0 || head_dim mod 2 <> 0 then
    Printf.ksprintf invalid_arg
      "Rope.%s: head_dim must be positive and even, got %d" fn head_dim;
  if theta <= 0.0 then
    Printf.ksprintf invalid_arg "Rope.%s: theta must be positive, got %g" fn
      theta;
  Array.init (head_dim / 2) (fun i ->
      theta ** (-2.0 *. float_of_int i /. float_of_int head_dim))

let make ?(theta = 10000.0) ~head_dim ~context () =
  of_frequencies ~context (base ~fn:"make" ~theta ~head_dim)

let llama3 ~theta ~head_dim ~factor ~low_freq_factor ~high_freq_factor
    ~original_context ~context =
  if factor <= 0.0 || low_freq_factor <= 0.0 || original_context <= 0 then
    invalid_arg
      "Rope.llama3: factor, low_freq_factor and original_context must be \
       positive";
  if high_freq_factor <= low_freq_factor then
    invalid_arg "Rope.llama3: high_freq_factor must exceed low_freq_factor";
  let original = float_of_int original_context in
  let low_wavelen = original /. low_freq_factor in
  let high_wavelen = original /. high_freq_factor in
  Array.map
    (fun f ->
      let wavelen = 2.0 *. Float.pi /. f in
      if wavelen < high_wavelen then f
      else if wavelen > low_wavelen then f /. factor
      else
        let smooth =
          ((original /. wavelen) -. low_freq_factor)
          /. (high_freq_factor -. low_freq_factor)
        in
        ((1.0 -. smooth) *. f /. factor) +. (smooth *. f))
    (base ~fn:"llama3" ~theta ~head_dim)
  |> of_frequencies ~context

let yarn ~theta ~head_dim ~factor ~beta_fast ~beta_slow ~original_context
    ~context =
  if factor < 1.0 || beta_slow <= 0.0 || original_context <= 0 then
    invalid_arg
      "Rope.yarn: factor must be at least 1 and beta_slow and original_context \
       positive";
  if beta_fast <= beta_slow then
    invalid_arg "Rope.yarn: beta_fast must exceed beta_slow";
  let dim = float_of_int head_dim in
  (* The pair that makes [turns] full turns over the original context. *)
  let pair turns =
    dim
    *. log (float_of_int original_context /. (turns *. 2.0 *. Float.pi))
    /. (2.0 *. log theta)
  in
  let low = Float.max (pair beta_fast) 0.0 in
  let high = Float.min (pair beta_slow) (dim -. 1.0) in
  let high = if high = low then high +. 0.001 else high in
  Array.mapi
    (fun i f ->
      let ramp = (float_of_int i -. low) /. (high -. low) in
      let ramp = Float.min 1.0 (Float.max 0.0 ramp) in
      (ramp *. f /. factor) +. ((1.0 -. ramp) *. f))
    (base ~fn:"yarn" ~theta ~head_dim)
  |> of_frequencies ~context

let frequencies t = Array.copy t.frequencies
let context t = Nx.dim 0 t.cos

let apply t ~pos x =
  let shape = Nx.shape x in
  if Array.length shape <> 4 then
    invalid_arg "Rope.apply: x must have shape [batch; heads; seq; head_dim]";
  let batch = shape.(0) and seq = shape.(2) and head_dim = shape.(3) in
  let half = Array.length t.frequencies in
  if head_dim <> 2 * half then
    Printf.ksprintf invalid_arg
      "Rope.apply: x has head_dim %d but the frequencies are for %d" head_dim
      (2 * half);
  (match Nx.shape pos with
  | [| b; s |] when (b = batch || b = 1) && s = seq -> ()
  | _ ->
      Printf.ksprintf invalid_arg
        "Rope.apply: pos must have shape [%d; %d] or [1; %d]" batch seq seq);
  let dt = Nx.dtype x in
  let rows table =
    Nx.cast dt (Nx.unsqueeze ~axes:[ 1 ] (Nx.take ~axis:0 ~indices:pos table))
  in
  let cos = rows t.cos and sin = rows t.sin in
  let x1 = Nx.slice [ A; A; A; R (0, half) ] x in
  let x2 = Nx.slice [ A; A; A; R (half, head_dim) ] x in
  Nx.concatenate ~axis:3
    [
      Nx.sub (Nx.mul x1 cos) (Nx.mul x2 sin);
      Nx.add (Nx.mul x2 cos) (Nx.mul x1 sin);
    ]
