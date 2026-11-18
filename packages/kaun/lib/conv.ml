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
    ?(bias = true) ?(groups = 1) ~key ~in_channels ~out_channels ~kernel_size
    dtype =
  if Array.length kernel_size = 0 then
    invalid_arg "Conv.make: kernel_size has no extent";
  if in_channels <= 0 || out_channels <= 0 || Array.exists (( >= ) 0) kernel_size
  then
    invalid_argf
      "Conv.make: channels and kernel extents must be positive, got \
       in_channels=%d out_channels=%d kernel_size=%s"
      in_channels out_channels (pp_ints kernel_size);
  if groups <= 0 || in_channels mod groups <> 0 || out_channels mod groups <> 0
  then
    invalid_argf
      "Conv.make: %d groups do not divide in_channels=%d and out_channels=%d"
      groups in_channels out_channels;
  (* The inputs one output reads and the outputs one input feeds, at unit
     stride (He et al., 2015, §2.2): a group's channels times the taps. *)
  let taps = Array.fold_left ( * ) 1 kernel_size in
  let fan_in = in_channels / groups * taps in
  let fan_out = out_channels / groups * taps in
  let keys = Nx.Rng.split key in
  let shape =
    Array.concat [ [| out_channels |]; kernel_size; [| in_channels / groups |] ]
  in
  let w = w_init keys.(0) ~fan_in ~fan_out dtype shape in
  let b =
    if bias then
      Some (bias_init keys.(1) ~fan_in ~fan_out dtype [| out_channels |])
    else None
  in
  { w; b }

let init ~key ~in_channels ~out_channels ~kernel_size =
  make ~key ~in_channels ~out_channels ~kernel_size Nx.float32

let apply ?stride ?padding ?dilation ?groups p x =
  Nx.conv ?init:p.b ?stride ?padding ?dilation ?groups x p.w
