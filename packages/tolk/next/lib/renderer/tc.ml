(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type bit = N of int | M of int | K of int

let equal_bit (b0 : bit) b1 = b0 = b1
let dim = function N _ -> 'n' | M _ -> 'm' | K _ -> 'k'
let index = function N i | M i | K i -> i
let of_dim d i = match d with 'n' -> N i | 'm' -> M i | _ -> K i
let pp_bit ppf b = Format.fprintf ppf "%c%d" (dim b) (index b)

type fragment = { lanes : bit list; elements : bit list }

let frag lanes elements = { lanes; elements }

let pp_tuple pp_elt ppf = function
  | [ x ] -> Format.fprintf ppf "(%a,)" pp_elt x
  | xs ->
      let sep ppf () = Format.pp_print_string ppf ", " in
      Format.fprintf ppf "(%a)" (Format.pp_print_list ~pp_sep:sep pp_elt) xs

let pp_fragment ppf f =
  let pp_quoted ppf b = Format.fprintf ppf "'%a'" pp_bit b in
  pp_tuple (pp_tuple pp_quoted) ppf [ f.lanes; f.elements ]

type t = {
  dtype_in : Dtype.t;
  dtype_out : Dtype.t;
  frag_a : fragment;
  frag_b : fragment;
  frag_c : fragment;
}

let bits f = f.lanes @ f.elements
let of_dims ds = List.filter (fun b -> String.contains ds (dim b))

let axis_coords tc =
  let used = bits tc.frag_a @ bits tc.frag_c in
  let top d =
    List.fold_left
      (fun m b -> if dim b = d then max m (index b) else m)
      (-1) used
  in
  List.concat_map (fun d -> List.init (top d + 1) (of_dim d)) [ 'n'; 'm'; 'k' ]

let base_upcast_axes tc =
  List.rev (of_dims "k" (axis_coords tc) @ tc.frag_c.elements)

let relabel tc =
  let pairs f =
    let slots = List.take (List.length f.elements) (base_upcast_axes tc) in
    let rec zip ys cs =
      match (ys, cs) with y :: ys, c :: cs -> (c, y) :: zip ys cs | _ -> []
    in
    zip (tc.frag_c.lanes @ List.rev slots) (bits f)
  in
  (pairs tc.frag_a, pairs tc.frag_b)

let frag_coords tc =
  let coord f ax lane elem =
    let part d =
      let sum bs v =
        let bit j b = if dim b = d then ((v lsr j) land 1) lsl index b else 0 in
        List.fold_left ( + ) 0 (List.mapi bit bs)
      in
      sum f.lanes lane + sum f.elements elem
    in
    (part ax.[0], part ax.[1])
  in
  let coords f ax =
    Array.init
      (1 lsl List.length f.lanes)
      (fun lane -> Array.init (1 lsl List.length f.elements) (coord f ax lane))
  in
  (coords tc.frag_a "mk", coords tc.frag_b "kn", coords tc.frag_c "mn")

let dims tc =
  let size d = 1 lsl List.length (of_dims d (axis_coords tc)) in
  (size "n", size "m", size "k")

let threads tc = 1 lsl List.length tc.frag_c.lanes

let check tc =
  let coords = axis_coords tc in
  let subset bs0 bs1 = List.for_all (fun b -> List.mem b bs1) bs0 in
  let check_fragment f ds =
    let own = of_dims ds coords and all = bits f in
    let distinct = List.length (List.sort_uniq compare all) = List.length all in
    if List.length f.lanes <> List.length tc.frag_c.lanes then
      invalid_arg
        (Format.asprintf "fragment %a has the wrong lane count" pp_fragment f);
    if
      not
        (distinct && subset f.elements own && subset own all
        && subset all (own @ of_dims "mn" coords))
    then
      invalid_arg
        (Format.asprintf "fragment %a isn't distinct bits covering %s"
           pp_fragment f ds)
  in
  check_fragment tc.frag_a "mk";
  check_fragment tc.frag_b "kn";
  check_fragment tc.frag_c "mn";
  let ks f = of_dims "k" (f.elements @ f.lanes) in
  if ks tc.frag_a <> ks tc.frag_b then
    invalid_arg
      (Format.asprintf "A holds its k bits as %a and B as %a" (pp_tuple pp_bit)
         (ks tc.frag_a) (pp_tuple pp_bit) (ks tc.frag_b))

let v ~dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c =
  let tc = { dtype_in; dtype_out; frag_a; frag_b; frag_c } in
  check tc;
  tc

let equal (tc0 : t) tc1 = tc0 = tc1

let pp ppf tc =
  Format.fprintf ppf
    "TensorCore(dtype_in=%a, dtype_out=%a, frag_a=%a, frag_b=%a, frag_c=%a)"
    Dtype.pp tc.dtype_in Dtype.pp tc.dtype_out pp_fragment tc.frag_a pp_fragment
    tc.frag_b pp_fragment tc.frag_c

let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)
let ks n = List.init (log2 n) (fun i -> K i)

(* The k bits from i to j, excluded, are the lane's; the others the
   elements'. *)
let split_k k i j =
  (List.take (j - i) (List.drop i k), List.take i k @ List.drop j k)

(* NVIDIA *)

(* mma.m16n8kK: a lane is a thread in its group (two k bits), then its group (m0
   to m2); the elements, least significant first, are the 2^g k bits packed in a
   32-bit register, m3 (A's row + 8), then the other k bits. *)
let mma k_size dtype_in dtype_out =
  let g = log2 (4 / Dtype.itemsize dtype_in) in
  let lane, elem = split_k (ks k_size) g (g + 2) in
  v ~dtype_in ~dtype_out
    ~frag_a:
      (frag
         (lane @ [ M 0; M 1; M 2 ])
         (List.take g elem @ (M 3 :: List.drop g elem)))
    ~frag_b:(frag (lane @ [ N 0; N 1; N 2 ]) elem)
    ~frag_c:(frag [ N 1; N 2; M 0; M 1; M 2 ] [ N 0; M 3 ])

let cuda_81616 =
  List.map
    (fun (di, dout) -> mma 16 di dout)
    Dtype.[ (Float16, Float32); (Bfloat16, Float32); (Float16, Float16) ]

let cuda_81632_f8 =
  List.map (fun di -> mma 32 di Float32) Dtype.[ Fp8e4m3; Fp8e5m2 ]

let cuda_8168_f16 =
  List.map (fun dout -> mma 8 Float16 dout) Dtype.[ Float32; Float16 ]

let cuda_8168_tf32 = [ mma 8 Float32 Float32 ]
let cuda_sm75 = cuda_8168_f16
let cuda_sm80 = cuda_81616 @ cuda_8168_f16 @ cuda_8168_tf32
let cuda_sm89 = cuda_sm80 @ cuda_81632_f8

let cuda arch =
  let n = String.length arch - 3 in
  let is_digit c = c >= '0' && c <= '9' in
  if n <= 0 || not (String.for_all is_digit (String.sub arch 3 n)) then
    invalid_arg (Printf.sprintf "%S has no compute capability" arch);
  let ver = int_of_string (String.sub arch 3 n) in
  if ver >= 89 then cuda_sm89
  else if ver >= 80 then cuda_sm80
  else if ver >= 75 then cuda_sm75
  else []

(* AMD *)

let cores ~frag_a ~frag_b ~frag_c =
  List.map (fun (dtype_in, dtype_out) ->
      v ~dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c)

let amd_rdna3 =
  cores
    ~frag_a:(frag [ M 0; M 1; M 2; M 3; N 0 ] [ K 0; K 1; K 2; K 3 ])
    ~frag_b:(frag [ N 0; N 1; N 2; N 3; M 0 ] [ K 0; K 1; K 2; K 3 ])
    ~frag_c:(frag [ N 0; N 1; N 2; N 3; M 0 ] [ M 1; M 2; M 3 ])
    Dtype.
      [
        (Float16, Float32);
        (Float16, Float16);
        (Bfloat16, Float32);
        (Int8, Int32);
      ]

let amd_rdna4 =
  cores
    ~frag_a:(frag [ M 0; M 1; M 2; M 3; K 2 ] [ K 0; K 1; K 3 ])
    ~frag_b:(frag [ N 0; N 1; N 2; N 3; K 2 ] [ K 0; K 1; K 3 ])
    ~frag_c:(frag [ N 0; N 1; N 2; N 3; M 3 ] [ M 0; M 1; M 2 ])
    Dtype.
      [
        (Float16, Float32);
        (Float16, Float16);
        (Bfloat16, Float32);
        (Bfloat16, Bfloat16);
      ]

(* 16x16xK: A[i,k] is element k mod K_L of lane i + 16 * (k / K_L), with K_L =
   K/4; the 8-bit floats' K = 128 is two halves of K = 64, k / 64 the high
   element. *)
let mfma k_size dtype_in dtype_out =
  let kl = log2 (min k_size 64 / 4) in
  let lane, elem = split_k (ks k_size) kl (kl + 2) in
  v ~dtype_in ~dtype_out
    ~frag_a:(frag ([ M 0; M 1; M 2; M 3 ] @ lane) elem)
    ~frag_b:(frag ([ N 0; N 1; N 2; N 3 ] @ lane) elem)
    ~frag_c:(frag [ N 0; N 1; N 2; N 3; M 2; M 3 ] [ M 0; M 1 ])

let amd_cdna_161616 =
  List.map (fun di -> mfma 16 di Float32) Dtype.[ Float16; Bfloat16 ]

let amd_cdna_161632 =
  List.map
    (fun di -> mfma 32 di Float32)
    Dtype.[ Fp8e5m2; Fp8e4m3; Float16; Bfloat16 ]

let amd_cdna_1616128 =
  List.map (fun di -> mfma 128 di Float32) Dtype.[ Fp8e5m2; Fp8e4m3 ]

let amd_cdna3_161632 =
  List.map (fun di -> mfma 32 di Float32) Dtype.[ Fp8e5m2fnuz; Fp8e4m3fnuz ]

let amd_cdna3 = amd_cdna3_161632 @ amd_cdna_161616
let amd_cdna4 = amd_cdna_1616128 @ amd_cdna_161632 @ amd_cdna_161616

let amd = function
  | "gfx942" -> amd_cdna3
  | "gfx950" -> amd_cdna4
  | "gfx1200" | "gfx1201" -> amd_rdna4
  | _ -> amd_rdna3

(* Apple Metal *)

let metal =
  cores
    ~frag_a:(frag [ K 1; M 0; M 1; K 2; M 2 ] [ K 0 ])
    ~frag_b:(frag [ N 1; K 0; K 1; N 2; K 2 ] [ N 0 ])
    ~frag_c:(frag [ N 1; M 0; M 1; N 2; M 2 ] [ N 0 ])
    Dtype.
      [
        (Float32, Float32);
        (Float16, Float32);
        (Float16, Float16);
        (Bfloat16, Float32);
        (Bfloat16, Bfloat16);
      ]
