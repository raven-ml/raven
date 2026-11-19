(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type transform = Identity | Log of float | Asinh_asinh

(* A compiled codec's programs ({!jit}): encode, at float32 and at float64,
   and decode without its check. *)
type 'd programs = {
  p_encode32 : 'd Nx.float32_t -> 'd Nx.int64_t;
  p_encode64 : 'd Nx.float64_t -> 'd Nx.int64_t;
  p_decode : 'd Nx.int64_t -> 'd Nx.float32_t;
}

(* [edges] is [c] codebooks of [n] edges, [c; n]. *)
type 'd t = {
  transform : transform;
  edges : 'd Nx.float32_t;
  programs : 'd programs option;
}

let v transform edges =
  match Nx.shape edges with
  | [| c; n |] when c > 0 && n > 0 -> { transform; edges; programs = None }
  | s ->
      Refuse.invalid "Scalar.v" "edges of shape %a, [c; n] expected" Nx.pp_shape
        s

let transform t = t.transform
let edges t = t.edges

(* The codecs of upstream's aion/codecs/config.py (MODALITY_CODEC_MAPPING) with
   the parameters of their config.json on the Hub: every log codec has
   min_log_value -3, and the Gaia XP codecs compress by arcsinh twice. *)
let transform_of : Modality.scalar -> transform = function
  | Flux_g | Flux_r | Flux_i | Flux_z | Flux_w1 | Flux_w2 | Flux_w3 | Flux_w4
  | Shape_r | Flux_g_gaia | Flux_bp_gaia | Flux_rp_gaia | Parallax ->
      Log (exp (-3.))
  | Xp_bp | Xp_rp -> Asinh_asinh
  | Shape_e1 | Shape_e2 | Ebv | Z | A_g | A_r | A_i | A_z | A_y | Mag_g | Mag_r
  | Mag_i | Mag_z | Mag_y | Shape11 | Shape22 | Shape12 | Ra | Dec ->
      Identity

let directory : Modality.scalar -> string = function
  | Flux_g -> "FLUX_G"
  | Flux_r -> "FLUX_R"
  | Flux_i -> "FLUX_I"
  | Flux_z -> "FLUX_Z"
  | Flux_w1 -> "FLUX_W1"
  | Flux_w2 -> "FLUX_W2"
  | Flux_w3 -> "FLUX_W3"
  | Flux_w4 -> "FLUX_W4"
  | Shape_r -> "SHAPE_R"
  | Shape_e1 -> "SHAPE_E1"
  | Shape_e2 -> "SHAPE_E2"
  | Ebv -> "EBV"
  | Z -> "Z"
  | A_g -> "a_g"
  | A_r -> "a_r"
  | A_i -> "a_i"
  | A_z -> "a_z"
  | A_y -> "a_y"
  | Mag_g -> "g_cmodel_mag"
  | Mag_r -> "r_cmodel_mag"
  | Mag_i -> "i_cmodel_mag"
  | Mag_z -> "z_cmodel_mag"
  | Mag_y -> "y_cmodel_mag"
  | Shape11 -> "i_sdssshape_shape11"
  | Shape22 -> "i_sdssshape_shape22"
  | Shape12 -> "i_sdssshape_shape12"
  | Flux_g_gaia -> "phot_g_mean_flux"
  | Flux_bp_gaia -> "phot_bp_mean_flux"
  | Flux_rp_gaia -> "phot_rp_mean_flux"
  | Parallax -> "parallax"
  | Ra -> "ra"
  | Dec -> "dec"
  | Xp_bp -> "bp_coefficients"
  | Xp_rp -> "rp_coefficients"

(* The [c] codebooks of a codec's file. A reservoir codec stores its one
   codebook as [_quantizer._index_to_val], the redshift codec as
   [_quantizer.buckets], and the Gaia XP codecs one per channel as
   [_quantizer.quantizers.<i>._index_to_val], [i] from 0. The redshift codec's
   stored grid spans [0, 6] where AION-1's constructor builds [0, 1]: the stored
   buffer is what AION-1's code reads, so it is read here too. *)
let codebooks name c archive =
  let fail fmt = Refuse.failure "Scalar.load" ("%s: " ^^ fmt) name in
  let row key =
    Option.map
      (fun p ->
        let e = Nx.unpack Nx.float32 p in
        if Nx.ndim e <> 1 then fail "%s is not 1-d" key;
        e)
      (Nx_io.Archive.find key archive)
  in
  let channel i = Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i in
  let rows =
    match (row "_quantizer._index_to_val", row "_quantizer.buckets") with
    | Some e, _ | None, Some e -> [ e ]
    | None, None ->
        if Option.is_some (row (channel c)) then fail "more than %d codebooks" c;
        List.init c (fun i ->
            match row (channel i) with
            | Some e -> e
            | None -> fail "no codebook %s" (channel i))
  in
  if List.length rows <> c then
    fail "%d codebook, %d expected" (List.length rows) c;
  let n = Nx.dim 0 (List.hd rows) in
  if List.exists (fun e -> Nx.dim 0 e <> n) rows then
    fail "codebooks of several lengths";
  Nx.stack rows

let load ?cache_dir ?offline m =
  let name = directory m in
  let c = Modality.tokens (Scalar m) in
  v (transform_of m) (codebooks name c (Hub.codec ?cache_dir ?offline name))

(* The clamp's logarithm is an edge most of a codebook's mass can sit on: 36 %
   of FLUX_W3's and 48 % of FLUX_W4's training sample lie on log(float32 e⁻³) =
   -3, so a [log] one ulp off at that one input would move half their tokens.
   test_scalar pins it. *)
let forward tr x =
  match tr with
  | Identity -> x
  | Log m -> Nx.log (Nx.clamp ~min:m x)
  | Asinh_asinh -> Nx.asinh (Nx.asinh x)

let inverse tr y =
  match tr with
  | Identity -> y
  | Log _ -> Nx.exp y
  | Asinh_asinh -> Nx.sinh (Nx.sinh y)

(* A codec of [c] channels takes values whose last axis has [c] entries. *)
let channels (type v s d) fn t (x : (v, s, d) Nx.t) =
  let c = Nx.dim 0 t.edges and s = Nx.shape x in
  let r = Array.length s in
  if c > 1 && (r = 0 || s.(r - 1) <> c) then
    Refuse.invalid fn "last axis of %d entries, %d expected"
      (if r = 0 then 0 else s.(r - 1))
      c

(* The position of each value's codebook in the flattened edges: [i * n] for
   channel [i], along the values' last axis. *)
let offsets (type v s d) fn t (x : (v, s, d) Nx.t) : d Nx.int64_t =
  channels fn t x;
  let c = Nx.dim 0 t.edges and n = Nx.dim 1 t.edges in
  let s = Nx.shape x and at = Nx.beside x in
  if c = 1 then Nx.zeros_at at Nx.int64 s
  else Nx.broadcast_to s (Nx.arange_at at Nx.int64 0 (c * n) n)

(* The bisection's rounds: each one at least halves the [hi - lo] of every
   search, which starts at [n - 1]. *)
let rounds n =
  let rec bits k = if k = 0 then 0 else 1 + bits (k lsr 1) in
  bits (n - 1)

(* torch.bucketize(y, e[:-1], right=False), the search AION's tokens were made
   with, run on every value at once: [flat] holds the edges, [base] the position
   of each value's codebook in it. Its step is a comparison [e(mid) >= y], which
   is false at a NaN edge and for a NaN [y]; both go right. Upstream's redshift
   codec searches all [n] edges and clamps the result to [n - 1], which is this
   search wherever the edges are in order and finite, as the stored grid is. *)
let search ~n ~base flat y =
  let lo = ref base and hi = ref (Nx.add_s base (Int64.of_int (n - 1))) in
  for _ = 1 to rounds n do
    let mid = Nx.add !lo (Nx.div_s (Nx.sub !hi !lo) 2L) in
    let open_ = Nx.less !lo !hi in
    let left = Nx.greater_equal (Nx.take mid flat) y in
    lo :=
      Nx.where
        (Nx.logical_and open_ (Nx.logical_not left))
        (Nx.add_s mid 1L) !lo;
    hi := Nx.where (Nx.logical_and open_ left) mid !hi
  done;
  Nx.sub !lo base

(* torch.bucketize compares in the promoted type of its input and its float32
   edges: float64 for a float64 input, which a float32 comparison would round (1
   + 1e-12 lies above the edge 1, its float32 rounding on it), and float32 for
   any narrower one. *)
let encode_raw (type s d) t (x : (float, s, d) Nx.t) : d Nx.int64_t =
  let n = Nx.dim 1 t.edges in
  let base = offsets "Scalar.encode" t x in
  let flat = Nx.flatten t.edges in
  let y = forward t.transform x in
  match Nx.dtype x with
  | Nx.Dtype.Float64 -> search ~n ~base (Nx.cast Nx.float64 flat) y
  | _ -> search ~n ~base flat (Nx.cast Nx.float32 y)

(* A compiled codec keeps a program for float32 and one for float64, and
   refuses other floats, which a set without eager kernels could not encode. *)
let encode (type s d) t (x : (float, s, d) Nx.t) : d Nx.int64_t =
  channels "Scalar.encode" t x;
  match (t.programs, Nx.dtype x) with
  | None, _ -> encode_raw t x
  | Some p, Nx.Dtype.Float32 -> p.p_encode32 x
  | Some p, Nx.Dtype.Float64 -> p.p_encode64 x
  | Some _, dt ->
      Refuse.invalid "Scalar.encode"
        "%s values, a compiled codec takes float32 or float64"
        (Nx.Dtype.name dt)

(* The token [n] is the class past the last edge, which a transformer may
   predict (redshift's vocabulary holds it), so the refusal names it. *)
let check_decode t k =
  let n = Nx.dim 1 t.edges in
  channels "Scalar.decode" t k;
  let inside =
    Nx.logical_and (Nx.greater_equal_s k 0L) (Nx.less_s k (Int64.of_int n))
  in
  Nx.check Nx.Ptree.tensor inside k (fun i k ->
      let k = Nx.item [] k in
      Refuse.exn "Scalar.decode" "token %Ld at %a outside [0, %d)%s" k
        Nx.pp_shape i n
        (if k = Int64.of_int n then
           ", the class past the last edge, whose value is unknown"
         else ""))

let decode_raw (type d) t (k : d Nx.int64_t) : d Nx.float32_t =
  let base = offsets "Scalar.decode" t k in
  inverse t.transform (Nx.take (Nx.add k base) (Nx.flatten t.edges))

(* A compiled codec checks a copy on the host, so that a refusal comes before
   its program runs. *)
let decode t k =
  match t.programs with
  | None ->
      check_decode t k;
      decode_raw t k
  | Some p ->
      check_decode t (Nx.place Nx.Host.on k);
      p.p_decode k

(* Compiling *)

let jit t =
  let t = { t with programs = None } in
  let open Nx.Ptree in
  let programs =
    {
      p_encode32 = Rune.jit (tensor @-> returns tensor) (encode_raw t);
      p_encode64 = Rune.jit (tensor @-> returns tensor) (encode_raw t);
      p_decode = Rune.jit (tensor @-> returns tensor) (decode_raw t);
    }
  in
  { t with programs = Some programs }

let place p t =
  let t' = { t with edges = Nx.place p t.edges; programs = None } in
  if Option.is_some t.programs then jit t' else t'
