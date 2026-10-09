(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's records and fill on the Mac's GPU, through rig, and the GPU's
   float arithmetic under the build's options. Every test skips on a machine
   with no Metal GPU. *)

open Windtrap
module S = Nx_metal_support
module Dt = Nx_array.Dtype

let dev =
  let t = lazy (S.open_ ()) in
  fun () ->
    match Lazy.force t with
    | Some t -> t
    | None -> skip ~reason:"no Metal GPU" ()

let floats o = S.view Bigarray.float32 o
let words o = S.view Bigarray.int32 o
let bytes o = String.init (Bigarray.Array1.dim o) (fun i -> o.{i})

(* Bits, printed as a float: -0 differs from +0. *)
let bits =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun x y -> Int64.bits_of_float x = Int64.bits_of_float y)

(* Runs *)

(* A run's launches run in order, each reading what the one before wrote:
   [generate] fills [a], then [move] copies [a] into [b] and [b] into [c]. *)
let ordered () =
  let t = dev () in
  let n = 3 * 1024 in
  let a = S.operand t (4 * n)
  and b = S.operand t (4 * n)
  and c = S.operand t (4 * n) in
  let move ~dst ~src =
    S.launch "move"
      ~groups:(S.groups (n / 4))
      ~addrs:[ S.address dst; S.address src; 0; 0 ]
      ~words:[ 1; n / 4 ]
  in
  let generate =
    S.launch "generate" ~groups:(S.groups n)
      ~addrs:[ S.address a ]
      ~words:[ n; Dt.code Dt.Uint32; 11; 0 ]
  in
  ignore (S.run t (S.seq [ generate; move ~dst:b ~src:a; move ~dst:c ~src:b ]));
  let view o = bytes (S.view Bigarray.char o) in
  equal string (view a) (view c)

let runs =
  group ~timeout:60. "runs" [ test "a run's launches run in order" ordered ]

(* Arithmetic *)

let n = 1 lsl 20

(* [a·b + c] rounds the product, then the sum: no contraction. Exponents in
   [-20, 20] keep every operand and result normal. *)
let rounded_apart () =
  let t = dev () in
  let abc = S.operand t (4 * 3 * n) in
  S.generate ~spread:20 t abc Dt.Float32 (3 * n) ~seed:5;
  let one = S.probe t "probe_contract" abc ~which:0 n in
  let fused = S.probe t "probe_contract" abc ~which:1 n in
  let apart, _, fma_wrong =
    S.probe_contract (floats abc) (floats one) (floats fused)
  in
  equal ~msg:"a*b + c differs from the product and sum rounded apart" int 0
    apart;
  equal ~msg:"fma differs from the fused result" int 0 fma_wrong

(* The GPU flushes float32 subnormals: an operand reads as zero and a result is
   written as a zero of its sign. *)
let tiny = Int32.float_of_bits 1l (* 2^-149 *)
let least_normal = Int32.float_of_bits 0x00800000l (* 2^-126 *)

let flushes =
  let pp ppf (a, b, c, _) = Format.fprintf ppf "%h * %h + %h" a b c in
  cases ~name:(Format.asprintf "%a" pp)
    "float32 subnormals flush, keeping their sign"
    [
      (tiny, 1., 0., 0.);
      (-.tiny, 1., -0., -0.);
      (least_normal, 0.5, 0., 0.);
      (-.least_normal, 0.5, -0., -0.);
      (tiny, 0x1p126, 0., 0.);
      (least_normal, 1., tiny, least_normal);
    ]
    (fun (a, b, c, want) ->
      let t = dev () in
      let abc = S.operand t 12 in
      let v = floats abc in
      v.{0} <- a;
      v.{1} <- b;
      v.{2} <- c;
      equal bits want (floats (S.probe t "probe_contract" abc ~which:0 1)).{0})

(* Division and square roots are correctly rounded, subnormals flushed.
   Exponents in [-126, 126] give quotients down to 2^-252, so many subnormal
   ones. *)
let div_sqrt () =
  let t = dev () in
  let xy = S.operand t (4 * 2 * n) in
  S.generate ~spread:126 t xy Dt.Float32 (2 * n) ~seed:6;
  let div = S.probe t "probe_div_sqrt" xy ~which:0 n in
  let sqrt = S.probe t "probe_div_sqrt" xy ~which:1 n in
  let div_wrong, sqrt_wrong, subnormal =
    S.probe_div_sqrt (floats xy) (floats div) (floats sqrt)
  in
  greater ~msg:"subnormal quotients drawn" int ~than:0 subnormal;
  equal ~msg:"quotients" int 0 div_wrong;
  equal ~msg:"roots" int 0 sqrt_wrong

(* Half keeps its subnormals: conversions from and to float32 give nx_dtype.h's
   codes, and h + h is the exact sum rounded to half. *)
let half () =
  let t = dev () in
  let xs = S.operand t (4 * n) in
  S.generate t xs Dt.Uint32 n ~seed:9;
  let probe which = words (S.probe t "probe_half" xs ~which n) in
  let to_wrong, from_wrong, sum_wrong =
    S.probe_half (words xs) (probe 0) (probe 1) (probe 2)
  in
  equal ~msg:"half(x)" int 0 to_wrong;
  equal ~msg:"float(h)" int 0 from_wrong;
  equal ~msg:"h + h" int 0 sum_wrong

(* nx_dtype.h's codecs give the same codes on the GPU as on the host: every code
   decodes, and drawn floats of a wide spread, with the edges, encode. *)
let codecs =
  let formats =
    Dt.
      [
        (Any Float16, 1 lsl 16);
        (Any Bfloat16, 1 lsl 16);
        (Any Float8_e4m3fn, 256);
        (Any Float8_e5m2, 256);
        (Any Float4_e2m1fn, 16);
      ]
  in
  let edges =
    [ 0.; -0.; infinity; neg_infinity; nan; tiny; -.tiny; least_normal; 1e-40 ]
  in
  cases
    ~name:(fun (Dt.Any dt, _) -> Dt.name dt)
    "nx_dtype.h's codecs agree with the host's" formats
    (fun (Dt.Any dt, codes) ->
      let t = dev () in
      let c = S.operand t (4 * codes) in
      let v = words c in
      for i = 0 to codes - 1 do
        v.{i} <- Int32.of_int i
      done;
      let f = S.operand t (4 * n) in
      S.generate ~spread:40 t f Dt.Float32 n ~seed:7;
      List.iteri (fun i x -> (floats f).{i} <- x) edges;
      let code = Dt.code dt in
      let decoded = S.probe ~dtype:code t "probe_codec" c ~which:0 codes in
      let encoded = S.probe ~dtype:code t "probe_codec" f ~which:1 n in
      let decode_wrong, encode_wrong, _ =
        S.probe_codec dt (words c) (words decoded) (words f) (words encoded)
      in
      equal ~msg:"decodings" int 0 decode_wrong;
      equal ~msg:"encodings" int 0 encode_wrong)

let arithmetic =
  group ~timeout:60. "arithmetic"
    [
      test "a*b + c rounds twice" rounded_apart;
      flushes;
      test "division and square roots round correctly" div_sqrt;
      test "half keeps its subnormals" half;
      codecs;
    ]

let () =
  S.hold_gpu ();
  exit (run "nx_metal" [ runs; arithmetic ])
