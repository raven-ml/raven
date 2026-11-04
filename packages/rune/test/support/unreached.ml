(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

type t = {
  dtype : Nx.Dtype.any;
  combine : Nx.combine;
  n : int;
  transposed : bool;
}

let combine_name : Nx.combine -> string = function
  | Set -> "set"
  | Add -> "add"
  | Max -> "max"
  | Min -> "min"

let pp ppf c =
  let (Nx.Dtype.Any dt) = c.dtype in
  Format.fprintf ppf "%s %a n %d%s" (combine_name c.combine) Nx.pp_dtype dt
    c.n
    (if c.transposed then " transposed" else "")

let cases =
  let dtypes = Nx.Dtype.[ Any Nx.float16; Any Nx.bfloat16; Any Nx.float32 ] in
  let combines = Nx.[ Set; Add; Max; Min ] in
  List.concat_map
    (fun dtype ->
      List.concat_map
        (fun combine ->
          List.concat_map
            (fun n ->
              List.map
                (fun transposed -> { dtype; combine; n; transposed })
                [ false; true ])
            [ 4; 40 ])
        combines)
    dtypes

(* Elements are their bits, unsigned in an [int]: an OCaml float would quiet a
   signalling NaN. *)

(* A format's signalling NaNs of both signs, a quiet NaN, both zeros,
   subnormals of both signs and [1], and the update [0.25]. *)
let target (type s) (dt : (float, s) Nx.dtype) =
  match dt with
  | Nx.Dtype.Float16 ->
      [| 0x7C01; 0xFD55; 0x7E01; 0x8000; 0x0000; 0x0001; 0x83FF; 0x3C00 |]
  | Nx.Dtype.Bfloat16 ->
      [| 0x7F81; 0xFFA5; 0x7FC1; 0x8000; 0x0000; 0x0001; 0x807F; 0x3F80 |]
  | Nx.Dtype.Float32 ->
      [|
        0x7F800001;
        0xFFA00005;
        0x7FC00001;
        0x80000000;
        0x00000000;
        0x00000001;
        0x807FFFFF;
        0x3F800000;
      |]
  | _ -> invalid_arg "Unreached.law: not float16, bfloat16 or float32"

let update (type s) (dt : (float, s) Nx.dtype) =
  match dt with
  | Nx.Dtype.Float16 -> 0x3400
  | Nx.Dtype.Bfloat16 -> 0x3E80
  | _ -> 0x3E800000

let of_bits (type s) (dt : (float, s) Nx.dtype) shape bs :
    (float, s, Nx.host) Nx.t =
  match dt with
  | Nx.Dtype.Float32 ->
      Nx.bitcast dt (Nx.create Nx.int32 shape (Array.map Int32.of_int bs))
  | _ -> Nx.bitcast dt (Nx.create Nx.uint16 shape bs)

let to_bits (type s) (dt : (float, s) Nx.dtype) (x : (float, s, Nx.host) Nx.t)
    =
  match dt with
  | Nx.Dtype.Float32 ->
      Array.map
        (fun b -> Int32.to_int b land 0xFFFFFFFF)
        (Nx.to_array (Nx.bitcast Nx.int32 x))
  | _ -> Nx.to_array (Nx.bitcast Nx.uint16 x)

(* The extent of the target's other two axes. *)
let width = 4

let run (type s) (module S : Nx.Devices) (dt : (float, s) Nx.dtype) c =
  let pool = target dt in
  let n = c.n and w = width in
  let row = w * w in
  (* The target's base, [[w; w; n]] where the function transposes it. *)
  let base = if c.transposed then [| w; w; n |] else [| n; w; w |] in
  let raws = Array.init (n * row) (fun j -> pool.(j mod Array.length pool)) in
  let at r a b =
    if c.transposed then raws.((((b * w) + a) * n) + r)
    else raws.((r * row) + (a * w) + b)
  in
  let x = of_bits dt base raws in
  let t = [| n; w; w |] in
  let u = of_bits dt t (Array.make (n * row) (update dt)) in
  let i = Nx.create Nx.int64 t (Array.make (n * row) 0L) in
  let f i u x =
    let x = if c.transposed then Nx.transpose x else x in
    Nx.scatter ~combine:c.combine ~axis:0 i u x
  in
  let g =
    Rune.jit Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor) f
  in
  let put y = Nx.place S.on y in
  let before = Rune.evaluations () in
  let y = to_bits dt (Nx.place Nx.Host.on (g (put i) (put u) (put x))) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  (* Rows past the first, which no update reaches. *)
  let shown f =
    List.init ((n - 1) * row) (fun j ->
        let r = 1 + (j / row) and a = j / w mod w and b = j mod w in
        Printf.sprintf "%d,%d,%d: %x" r a b (f r a b))
  in
  let got r a b = y.((r * row) + (a * w) + b) in
  equal (list string) (shown at) (shown got)

let law s c =
  let (Nx.Dtype.Any dt) = c.dtype in
  match dt with
  | Nx.Dtype.Float16 -> run s dt c
  | Nx.Dtype.Bfloat16 -> run s dt c
  | Nx.Dtype.Float32 -> run s dt c
  | _ -> invalid_arg "Unreached.law: not float16, bfloat16 or float32"
