(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/renderer/tc.py. *)

open Tolk_uop

let strf = Printf.sprintf
let pow2 n = 1 lsl n
let take n xs = List.filteri (fun i _ -> i < n) xs

type fragment = string list * string list

type t = {
  dtype_in : Dtype.t;
  dtype_out : Dtype.t;
  frag_a : fragment;
  frag_b : fragment;
  frag_c : fragment;
  dims : int * int * int;
  threads : int;
}

let bit_index c =
  if String.length c < 2 || not (String.contains "nmk" c.[0]) then
    invalid_arg ("Tc.create: invalid tile coordinate " ^ c);
  match int_of_string_opt (String.sub c 1 (String.length c - 1)) with
  | Some n when n >= 0 && n < Sys.int_size - 2 && c = strf "%c%d" c.[0] n -> n
  | _ -> invalid_arg ("Tc.create: invalid tile coordinate " ^ c)

let coords frag_a frag_c =
  let al, ae = frag_a and cl, ce = frag_c in
  let used = al @ ae @ cl @ ce in
  List.concat_map (fun dim ->
      let last = List.fold_left (fun last c ->
          if c.[0] = dim then max last (bit_index c) else last) (-1) used in
      List.init (last + 1) (fun i -> strf "%c%d" dim i)) [ 'n'; 'm'; 'k' ]

let axis_coords t = coords t.frag_a t.frag_c

let base_upcast_axes t =
  List.rev (List.filter (fun c -> c.[0] = 'k') (axis_coords t) @ snd t.frag_c)

let relabel t =
  List.map (fun (lanes, elements) ->
      let slots = fst t.frag_c @ List.rev (take (List.length elements) (base_upcast_axes t)) in
      List.combine (lanes @ elements) slots) [ t.frag_a; t.frag_b ]

let create ~dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c =
  let check condition message = if not condition then invalid_arg ("Tc.create: " ^ message) in
  List.iter (fun (lanes, elements) -> List.iter (fun c -> ignore (bit_index c)) (lanes @ elements))
    [ frag_a; frag_b; frag_c ];
  check (List.length (fst frag_c) < Sys.int_size - 1) "lane count exceeds host integer range";
  let coordinates = coords frag_a frag_c in
  List.iter (fun ((lanes, elements), dimensions) ->
      let all = lanes @ elements in
      let own = List.filter (fun c -> String.contains dimensions c.[0]) coordinates in
      check (List.length lanes = List.length (fst frag_c)) "fragment has the wrong lane count";
      check (List.length (List.sort_uniq String.compare all) = List.length all
             && List.for_all (fun c -> List.mem c own) elements
             && List.for_all (fun c -> List.mem c all) own
             && List.for_all (fun c -> List.mem c own
                  || (List.mem c coordinates && String.contains "mn" c.[0])) all)
        "fragment must cover its tile bits exactly, with broadcasts only in lanes")
    [ frag_a, "mk"; frag_b, "kn"; frag_c, "mn" ];
  let k_bits (lanes, elements) = List.filter (fun c -> c.[0] = 'k') (elements @ lanes) in
  check (k_bits frag_a = k_bits frag_b) "input fragments must relabel K identically";
  let extent dim = pow2 (List.length (List.filter (fun c -> c.[0] = dim) coordinates)) in
  { dtype_in; dtype_out; frag_a; frag_b; frag_c;
    dims = extent 'n', extent 'm', extent 'k'; threads = pow2 (List.length (fst frag_c)) }

(* Hardware fragment descriptions. Bits are ordered least significant first. *)

let labels dim count = List.init count (fun i -> strf "%c%d" dim i)
let drop count xs = List.filteri (fun i _ -> i >= count) xs
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n lsr 1)

let mk ~frag_a ~frag_b ~frag_c dtypes =
  List.map (fun (dtype_in, dtype_out) ->
      create ~dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c) dtypes

let mma k dtype_in dtype_out =
  let k = labels 'k' (log2 k) and g = log2 (4 / Dtype.itemsize dtype_in) in
  let lanes = take 2 (drop g k) in
  let elements = take g k @ drop (g + 2) k in
  create ~dtype_in ~dtype_out
    ~frag_a:(lanes @ [ "m0"; "m1"; "m2" ], take g elements @ [ "m3" ] @ drop g elements)
    ~frag_b:(lanes @ [ "n0"; "n1"; "n2" ], elements)
    ~frag_c:([ "n1"; "n2"; "m0"; "m1"; "m2" ], [ "n0"; "m3" ])

let cuda_81616 = List.map (fun (di, do_) -> mma 16 di do_)
    Dtype.[ Float16, Float32; Bfloat16, Float32; Float16, Float16 ]
let cuda_81632_f8 = List.map (fun di -> mma 32 di Dtype.float32) Dtype.[ Fp8e4m3; Fp8e5m2 ]
let cuda_8168_f16 = List.map (fun do_ -> mma 8 Dtype.float16 do_) Dtype.[ Float32; Float16 ]
let cuda_8168_tf32 = [ mma 8 Dtype.float32 Dtype.float32 ]
let cuda_sm75 = cuda_8168_f16
let cuda_sm80 = cuda_81616 @ cuda_8168_f16 @ cuda_8168_tf32
let cuda_sm89 = cuda_sm80 @ cuda_81632_f8

let amd_rdna3 = mk
    ~frag_a:([ "m0"; "m1"; "m2"; "m3"; "n0" ], [ "k0"; "k1"; "k2"; "k3" ])
    ~frag_b:([ "n0"; "n1"; "n2"; "n3"; "m0" ], [ "k0"; "k1"; "k2"; "k3" ])
    ~frag_c:([ "n0"; "n1"; "n2"; "n3"; "m0" ], [ "m1"; "m2"; "m3" ])
    Dtype.[ Float16, Float32; Float16, Float16; Bfloat16, Float32; Int8, Int32 ]

let amd_rdna4 = mk
    ~frag_a:([ "m0"; "m1"; "m2"; "m3"; "k2" ], [ "k0"; "k1"; "k3" ])
    ~frag_b:([ "n0"; "n1"; "n2"; "n3"; "k2" ], [ "k0"; "k1"; "k3" ])
    ~frag_c:([ "n0"; "n1"; "n2"; "n3"; "m3" ], [ "m0"; "m1"; "m2" ])
    Dtype.[ Float16, Float32; Float16, Float16; Bfloat16, Float32; Bfloat16, Bfloat16 ]

let mfma k dtype_in dtype_out =
  let kl = log2 (min k 64 / 4) in
  let k = labels 'k' (log2 k) in
  let lanes = take 2 (drop kl k) and elements = take kl k @ drop (kl + 2) k in
  create ~dtype_in ~dtype_out
    ~frag_a:([ "m0"; "m1"; "m2"; "m3" ] @ lanes, elements)
    ~frag_b:([ "n0"; "n1"; "n2"; "n3" ] @ lanes, elements)
    ~frag_c:([ "n0"; "n1"; "n2"; "n3"; "m2"; "m3" ], [ "m0"; "m1" ])

let amd_cdna_161616 = List.map (fun di -> mfma 16 di Dtype.float32) Dtype.[ Float16; Bfloat16 ]
let amd_cdna_161632 = List.map (fun di -> mfma 32 di Dtype.float32)
    Dtype.[ Fp8e5m2; Fp8e4m3; Float16; Bfloat16 ]
let amd_cdna_1616128 = List.map (fun di -> mfma 128 di Dtype.float32) Dtype.[ Fp8e5m2; Fp8e4m3 ]
let amd_cdna3 = List.map (fun di -> mfma 32 di Dtype.float32)
    Dtype.[ Fp8e5m2fnuz; Fp8e4m3fnuz ] @ amd_cdna_161616
let amd_cdna4 = amd_cdna_1616128 @ amd_cdna_161632 @ amd_cdna_161616

let metal = mk
    ~frag_a:([ "k1"; "m0"; "m1"; "k2"; "m2" ], [ "k0" ])
    ~frag_b:([ "n1"; "k0"; "k1"; "n2"; "k2" ], [ "n0" ])
    ~frag_c:([ "n1"; "m0"; "m1"; "n2"; "m2" ], [ "n0" ])
    Dtype.[ Float32, Float32; Float16, Float32; Float16, Float16;
            Bfloat16, Float32; Bfloat16, Bfloat16 ]

(* Operand type names go verbatim into the emitted tensor-core function
   name, so they are the target's spelling of the type rather than tolk's
   short dtype tag. A dtype with no spelling here has no place in a
   tensor-core table. *)
let dtype_name = function
  | Dtype.Float16 -> "half"
  | Dtype.Bfloat16 -> "__bf16"
  | Dtype.Float32 -> "float"
  | Dtype.Float64 -> "double"
  | Dtype.Int8 -> "char"
  | Dtype.Int32 -> "int"
  | Dtype.Fp8e4m3 -> "float8_e4m3"
  | Dtype.Fp8e5m2 -> "float8_e5m2"
  | Dtype.Fp8e4m3fnuz -> "float8_e4m3fnuz"
  | Dtype.Fp8e5m2fnuz -> "float8_e5m2fnuz"
  | dt ->
      invalid_arg
        (strf "Tc.dtype_name: no tensor-core name for %s" (Dtype.to_string dt))

let to_string (tc : t) =
  let n, m, k = tc.dims in
  strf "WMMA_%d_%d_%d_%s_%s" n m k (dtype_name tc.dtype_in)
    (dtype_name tc.dtype_out)

