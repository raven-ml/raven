(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A floor is reported by its bits, so that each floor has programs of its
   own. *)
let walk_floor c m =
  Nx.Ptree.Walk.case c (Printf.sprintf "%h" m);
  m

type transform = Identity | Log of float [@ptree.walk walk_floor] | Asinh_asinh
[@@deriving ptree]

(* [edges] is [c] codebooks of [n] edges, [c; n]. *)
type 'd t = { transform : transform; edges : 'd Nx.float32_t }
[@@deriving ptree]

let v transform edges =
  match Nx.shape edges with
  | [| c; n |] when c > 0 && n > 0 -> { transform; edges }
  | s ->
      invalid_arg
        (Format.asprintf "Aion.Scalar.v: edges of shape %a, [c; n] expected"
           Nx.pp_shape s)

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

(* The [c] codebooks of a codec's file. A reservoir codec stores its one
   codebook as [_quantizer._index_to_val], the redshift codec as
   [_quantizer.buckets], and the Gaia XP codecs one per channel as
   [_quantizer.quantizers.<i>._index_to_val], [i] from 0. The redshift codec's
   stored grid spans [0, 6] where AION-1's constructor builds [0, 1]: the stored
   buffer is what AION-1's code reads, so it is read here too. *)
let codebooks m archive =
  let key = Modality.key (Scalar m) in
  let fail fmt =
    Printf.ksprintf failwith ("Aion.Scalar.of_archive: %s: " ^^ fmt) key
  in
  let c = Modality.tokens (Scalar m) in
  (* Probes through [names], which [import] does not note as reads. *)
  let has name = List.mem name (Nx_io.Archive.names archive) in
  let book name =
    let e = Nx_io.Archive.tensor Nx.float32 name archive in
    if Nx.ndim e <> 1 then fail "%s is not 1-d" name;
    e
  in
  let channel i = Printf.sprintf "_quantizer.quantizers.%d._index_to_val" i in
  (* Only the layout read is asked for, so an entry of another is unread. *)
  let rows =
    if has "_quantizer._index_to_val" then [ book "_quantizer._index_to_val" ]
    else if has "_quantizer.buckets" then [ book "_quantizer.buckets" ]
    else begin
      if has (channel c) then fail "more than %d codebooks" c;
      List.init c (fun i -> book (channel i))
    end
  in
  if List.length rows <> c then
    fail "%d codebook, %d expected" (List.length rows) c;
  let n = Nx.dim 0 (List.hd rows) in
  if List.exists (fun e -> Nx.dim 0 e <> n) rows then
    fail "codebooks of several lengths";
  Nx.stack rows

(* Each codebook's [_reservoir] is the training sample its quantiles were
   computed from, which encoding never reads: [_quantizer._reservoir] for a
   codec of one channel, [_quantizer.quantizers.<i>._reservoir] for channel [i]
   of the others. *)
let reservoir c name =
  if c = 1 then name = "_quantizer._reservoir"
  else
    List.exists
      (fun i -> name = Printf.sprintf "_quantizer.quantizers.%d._reservoir" i)
      (List.init c Fun.id)

let of_archive m a =
  let unread = reservoir (Modality.tokens (Scalar m)) in
  v (transform_of m) (Nx_io.Archive.import ~unread (codebooks m) a)

(* AION-1's codebooks hold 1024 edges, redshift's grid included. *)
let codebook = 1024

let init ~key m =
  let c = Modality.tokens (Scalar m) in
  let at = Nx.Rng.placement key in
  let e = Nx.Rng.normal_at at ~key Nx.float32 [| c; codebook |] in
  v (transform_of m) (fst (Nx.sort ~axis:1 e))

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

let values t = inverse t.transform t.edges

(* A codec of [c] channels takes values whose last axis has [c] entries. *)
let channels (type v s d) fn t (x : (v, s, d) Nx.t) =
  let c = Nx.dim 0 t.edges and s = Nx.shape x in
  let r = Array.length s in
  if c > 1 && (r = 0 || s.(r - 1) <> c) then
    invalid_arg
      (Printf.sprintf "Aion.%s: last axis of %d entries, %d expected" fn
         (if r = 0 then 0 else s.(r - 1))
         c)

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
let encode (type s d) t (x : (float, s, d) Nx.t) : d Nx.int64_t =
  let n = Nx.dim 1 t.edges in
  let base = offsets "Scalar.encode" t x in
  let flat = Nx.flatten t.edges in
  let y = forward t.transform x in
  match Nx.dtype x with
  | Nx.Dtype.Float64 -> search ~n ~base (Nx.cast Nx.float64 flat) y
  | _ -> search ~n ~base flat (Nx.cast Nx.float32 y)

(* The token [n] is the class past the last edge, which a transformer may
   predict (redshift's vocabulary holds it), so the refusal names it. *)
let decode (type d) t (k : d Nx.int64_t) : d Nx.float32_t =
  let n = Nx.dim 1 t.edges in
  let base = offsets "Scalar.decode" t k in
  let inside =
    Nx.logical_and (Nx.greater_equal_s k 0L) (Nx.less_s k (Int64.of_int n))
  in
  Nx.check Nx.Ptree.tensor inside k (fun i k ->
      let k = Nx.item [] k in
      Invalid_argument
        (Format.asprintf "Aion.Scalar.decode: token %Ld at %a outside [0, %d)%s"
           k Nx.pp_shape i n
           (if k = Int64.of_int n then
              ", the class past the last edge, whose value is unknown"
            else "")));
  inverse t.transform (Nx.take (Nx.add k base) (Nx.flatten t.edges))
