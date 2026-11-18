(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, _) t = { w : 'a; b : 'a option }

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let pp_ints a =
  "[|" ^ String.concat "; " (Array.to_list (Array.map string_of_int a)) ^ "|]"

let walk c { w; b } =
  let open Nx.Ptree.Walk in
  let w = field c "w" leaf w in
  let b = field c "b" (option leaf) b in
  { w; b }

let make ?(w_init = Init.glorot_uniform) ?(bias_init = Init.zeros)
    ?(bias = true) ~key ~in_channels ~out_channels ~kernel_size dtype =
  if Array.length kernel_size = 0 then
    invalid_arg "Conv_transpose.make: kernel_size has no extent";
  if in_channels <= 0 || out_channels <= 0 || Array.exists (( >= ) 0) kernel_size
  then
    invalid_argf
      "Conv_transpose.make: channels and kernel extents must be positive, got \
       in_channels=%d out_channels=%d kernel_size=%s"
      in_channels out_channels (pp_ints kernel_size);
  (* The fans of the transposed map at unit stride: one output reads
     [in_channels] channels at each tap, one input feeds [out_channels]. *)
  let taps = Array.fold_left ( * ) 1 kernel_size in
  let fan_in = in_channels * taps and fan_out = out_channels * taps in
  let keys = Nx.Rng.split key in
  let shape = Array.concat [ [| in_channels |]; kernel_size; [| out_channels |] ] in
  let w = w_init keys.(0) ~fan_in ~fan_out dtype shape in
  let b =
    if bias then
      Some (bias_init keys.(1) ~fan_in ~fan_out dtype [| out_channels |])
    else None
  in
  { w; b }

let init ~key ~in_channels ~out_channels ~kernel_size =
  make ~key ~in_channels ~out_channels ~kernel_size Nx.float32

let apply ?stride ?padding ?dilation p x =
  Nx.conv_transpose ?init:p.b ?stride ?padding ?dilation x p.w
