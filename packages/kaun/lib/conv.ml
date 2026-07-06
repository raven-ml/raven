(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'b params = { w : (float, 'b) Nx.t; b : (float, 'b) Nx.t option }
type t = Nx.float32_elt params

let map (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t) { w; b } =
  { w = f w; b = (match b with None -> None | Some b -> Some (f b)) }

let map2 (f : 'a 'c. ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t) p q =
  let b =
    match (p.b, q.b) with
    | Some pb, Some qb -> Some (f pb qb)
    | None, None -> None
    | Some _, None | None, Some _ -> invalid_arg "Conv.map2: bias mismatch"
  in
  { w = f p.w q.w; b }

let iter (f : 'a 'c. ('a, 'c) Nx.t -> unit) { w; b } =
  f w;
  match b with None -> () | Some b -> f b

let names p = match p.b with None -> [ "w" ] | Some _ -> [ "w"; "b" ]

let make ?(w_init = Init.glorot_uniform) ?(bias_init = Init.zeros)
    ?(bias = true) ~in_channels ~out_channels ~kernel_size dtype =
  let kh, kw = kernel_size in
  if in_channels <= 0 || out_channels <= 0 || kh <= 0 || kw <= 0 then
    Printf.ksprintf invalid_arg
      "Conv.make: channels and kernel size must be positive, got \
       in_channels=%d out_channels=%d kernel_size=(%d, %d)"
      in_channels out_channels kh kw;
  let fan_in = in_channels * kh * kw in
  let fan_out = out_channels * kh * kw in
  let w =
    w_init ~fan_in ~fan_out dtype [| out_channels; in_channels; kh; kw |]
  in
  let b =
    if bias then Some (bias_init ~fan_in ~fan_out dtype [| out_channels |])
    else None
  in
  { w; b }

let init ~in_channels ~out_channels ~kernel_size =
  make ~in_channels ~out_channels ~kernel_size Nx.float32

(* TF-style padding: [`Valid] pads nothing; [`Same] pads so that the output size
   is [ceil (size / stride)], splitting any odd total towards the end. *)
let pad_pair padding size kernel stride =
  match padding with
  | `Valid -> (0, 0)
  | `Same ->
      let out = (size + stride - 1) / stride in
      let total = Stdlib.max 0 (((out - 1) * stride) + kernel - size) in
      (total / 2, total - (total / 2))

let apply ?(stride = (1, 1)) ?(padding = `Valid) p x =
  let sh, sw = stride in
  if sh <= 0 || sw <= 0 then
    Printf.ksprintf invalid_arg
      "Conv.apply: stride must be positive, got (%d, %d)" sh sw;
  let ws = Nx.shape p.w in
  let co = ws.(0) and ci = ws.(1) and kh = ws.(2) and kw = ws.(3) in
  let xs = Nx.shape x in
  if Array.length xs <> 4 then
    Printf.ksprintf invalid_arg
      "Conv.apply: input must be [batch; channels; height; width], got rank %d"
      (Array.length xs);
  if xs.(1) <> ci then
    Printf.ksprintf invalid_arg
      "Conv.apply: input has %d channels but the layer expects %d" xs.(1) ci;
  let n = xs.(0) and h = xs.(2) and w = xs.(3) in
  let ph0, ph1 = pad_pair padding h kh sh in
  let pw0, pw1 = pad_pair padding w kw sw in
  let oh = ((h + ph0 + ph1 - kh) / sh) + 1 in
  let ow = ((w + pw0 + pw1 - kw) / sw) + 1 in
  if oh <= 0 || ow <= 0 then
    Printf.ksprintf invalid_arg
      "Conv.apply: kernel (%d, %d) does not fit input (%d, %d)" kh kw h w;
  (* im2col: patches has shape [n; ci; kh*kw; oh*ow]; contract the filter axes
     with a batched matmul. Differentiable end to end (unfold, reshape and
     matmul all have autodiff rules). *)
  let patches =
    Nx.extract_patches ~kernel_size:[| kh; kw |] ~stride:[| sh; sw |]
      ~dilation:[| 1; 1 |]
      ~padding:[| (ph0, ph1); (pw0, pw1) |]
      x
  in
  let patches = Nx.reshape [| n; ci * kh * kw; oh * ow |] patches in
  let w_mat = Nx.reshape [| co; ci * kh * kw |] p.w in
  let y = Nx.reshape [| n; co; oh; ow |] (Nx.matmul w_mat patches) in
  match p.b with None -> y | Some b -> Nx.add y (Nx.reshape [| co; 1; 1 |] b)
