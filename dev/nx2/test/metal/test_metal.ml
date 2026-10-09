(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx.metal's records and fill on the Mac's GPU, through rig, the GPU's float
   arithmetic under the build's options, and nx.metal's contraction. Every test
   skips on a machine with no Metal GPU. *)

open Windtrap
module S = Nx_metal_support
module Dt = Nx_array.Dtype

let strf = Printf.sprintf

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

(* Contract *)

type init = No_init | Full | Bias

type case = {
  dt : Dt.any;
  out : Dt.any;
  a_t : bool; (* a stored [k][m] *)
  b_t : bool; (* b stored [n][k] *)
  batch : int;
  m : int;
  n : int;
  k : int;
  init : init;
  pad : int; (* extra elements per row of a and b *)
  bpad : int; (* extra elements per batch of a and b *)
  spread : int;
}

let pp_case ppf c =
  let name (Dt.Any dt) = Dt.name dt in
  Format.fprintf ppf
    "%s -> %s, a%s b%s, %d x (%d x %d x %d), init %s, pad %d, bpad %d, spread \
     %d"
    (name c.dt) (name c.out)
    (if c.a_t then "^T" else "")
    (if c.b_t then "^T" else "")
    c.batch c.m c.n c.k
    (match c.init with No_init -> "none" | Full -> "full" | Bias -> "bias")
    c.pad c.bpad c.spread

let floats_dt = [ Dt.Any Dt.Float32; Dt.Any Dt.Float16; Dt.Any Dt.Bfloat16 ]

(* Extents at, around and between the dense kernels' 32 and 64 wide tiles and 16
   and 32 long steps, or whole tiles, [tile] each. *)
let extent_gen tile =
  Gen.one_of
    [
      Gen.of_list
        [ 0; 1; 2; 7; 8; 9; 15; 16; 17; 31; 32; 33; 63; 64; 65; 100; 129 ];
      Gen.map (fun x -> tile * x) (Gen.int_range 1 3);
    ]

let case =
  let open Gen in
  let+ dt = of_list floats_dt
  and+ out = of_list floats_dt
  and+ a_t = bool
  and+ b_t = bool
  and+ batch = int_range 1 3
  and+ m = frequency [ (1, of_list [ 1 ]); (4, extent_gen 64) ]
  and+ n = extent_gen 64
  and+ k = Gen.one_of [ extent_gen 16; Gen.of_list [ 300; 1000; 2880 ] ]
  and+ init = of_list [ No_init; Full; Bias ]
  and+ pad = of_list [ 0; 3 ]
  and+ bpad = of_list [ 0; 1; 4 ]
  and+ spread = of_list [ 0; 8 ] in
  { dt; out; a_t; b_t; batch; m; n; k; init; pad; bpad; spread }

let case = Gen.with_pp pp_case case

(* An operand of [rows] x [cols] per batch, stored [cols][rows] if [trans], each
   stored row [pad] elements longer and each batch [bpad], with values from
   [seed]. *)
let matrix t (Dt.Any dt) ~trans ~batch ~rows ~cols ~pad ~bpad ~seed ~spread =
  let ld = (if trans then rows else cols) + pad in
  let per = ((if trans then cols else rows) * ld) + bpad in
  let o = S.operand t (max 1 (Dt.bytes dt (batch * per))) in
  if batch * per > 0 then S.generate ~spread t o dt (batch * per) ~seed;
  S.arg o dt (if trans then (per, 1, ld) else (per, ld, 1))

let init_arg t c (Dt.Any dt) =
  match c.init with
  | No_init -> None
  | Full ->
      Some
        (matrix t (Dt.Any dt) ~trans:false ~batch:c.batch ~rows:c.m ~cols:c.n
           ~pad:0 ~bpad:0 ~seed:3 ~spread:c.spread)
  | Bias ->
      let o = S.operand t (max 1 (Dt.bytes dt c.n)) in
      if c.n > 0 then S.generate ~spread:c.spread t o dt c.n ~seed:4;
      Some (S.arg o dt (0, 0, 1))

(* The call's operands and its run. *)
let call ?acc t c =
  let a =
    matrix t c.dt ~trans:c.a_t ~batch:c.batch ~rows:c.m ~cols:c.k ~pad:c.pad
      ~bpad:c.bpad ~seed:1 ~spread:c.spread
  in
  let b =
    matrix t c.dt ~trans:c.b_t ~batch:c.batch ~rows:c.k ~cols:c.n ~pad:c.pad
      ~bpad:c.bpad ~seed:2 ~spread:c.spread
  in
  let (Dt.Any out_dt) = c.out in
  let mn = c.batch * c.m * c.n in
  let out =
    S.arg (S.operand t (max 1 (Dt.bytes out_dt mn))) out_dt (c.m * c.n, c.n, 1)
  in
  let init = init_arg t c c.out in
  let dims = (c.batch, c.m, c.n, c.k) in
  let run =
    require_some ~msg:"the planner declined"
      (S.plan_contract ?init ?acc t dims ~a ~b ~out)
  in
  (dims, a, b, out, init, run)

(* Every output is within the contraction's bound of the exact result. *)
let contract_bound =
  let empty =
    {
      dt = Dt.Any Dt.Float32;
      out = Dt.Any Dt.Bfloat16;
      a_t = false;
      b_t = false;
      batch = 2;
      m = 65;
      n = 3;
      k = 0;
      init = Bias;
      pad = 0;
      bpad = 0;
      spread = 8;
    }
  in
  (* 128 or more 64 x 64 tiles run on them, whole or reaching past m, n, k. *)
  let large dt ~a_t ~b_t ~m ~n ~k ~init =
    { empty with dt; out = dt; a_t; b_t; m; n; k; init; spread = 0 }
  in
  let examples =
    [
      empty;
      large (Dt.Any Dt.Float32) ~a_t:false ~b_t:false ~m:512 ~n:512 ~k:64
        ~init:No_init;
      large (Dt.Any Dt.Float16) ~a_t:true ~b_t:false ~m:577 ~n:520 ~k:45
        ~init:Full;
      large (Dt.Any Dt.Bfloat16) ~a_t:false ~b_t:true ~m:512 ~n:576 ~k:64
        ~init:Bias;
      large (Dt.Any Dt.Float32) ~a_t:true ~b_t:true ~m:520 ~n:513 ~k:33
        ~init:Bias;
      (* Few rows of a split along k, on wide tiles. *)
      {
        (large (Dt.Any Dt.Float16) ~a_t:false ~b_t:true ~m:9 ~n:200 ~k:2880
           ~init:Full)
        with
        batch = 1;
      };
    ]
  in
  prop ~count:60 ~examples "a float contraction is within its bound" case
    (fun c ->
      cover "an empty sum" (c.k = 0);
      cover "a transposed operand" (c.a_t || c.b_t);
      cover "a bias" (c.init = Bias);
      cover "narrow out" (c.out <> Dt.Any Dt.Float32);
      cover "a batch pad" (c.batch > 1 && c.bpad > 0);
      let t = dev () in
      let dims, a, b, out, init, run = call t c in
      let kernels = S.entries run in
      let launches_one prefix = String.starts_with ~prefix in
      let launches prefix = List.exists (launches_one prefix) kernels in
      let dense =
        List.filter
          (fun k -> launches_one "contract_f" k || launches_one "contract_bf" k)
          kernels
      in
      (* A dense kernel's tile, by its name's suffix past "_edge". *)
      let tile k =
        let k =
          if String.ends_with ~suffix:"_edge" k then
            String.sub k 0 (String.length k - 5)
          else k
        in
        if String.ends_with ~suffix:"_s" k then `Small
        else if String.ends_with ~suffix:"_w" k then `Wide
        else `Large
      in
      let tiles x = List.exists (fun k -> tile k = x) dense in
      cover "64 x 64 tiles" (tiles `Large);
      cover "32 x 32 tiles" (tiles `Small);
      cover "16 x 64 tiles" (tiles `Wide);
      let edge = List.exists (String.ends_with ~suffix:"_edge") dense in
      cover "whole tiles" (dense <> [] && not edge);
      cover "edge tiles" edge;
      cover "a skinny product, b stored [k][n]" (launches "skinny_" && not c.b_t);
      cover "a skinny product, b stored [n][k]" (launches "skinny_" && c.b_t);
      cover "a split along k" (launches "contract_combine");
      ignore (S.run t run);
      let worst, at = S.contract_error ?init dims ~a ~b ~out in
      at_most
        ~msg:(strf "output %d's error over its bound" at)
        float_exact ~than:1. worst)

(* Integer contractions wrap in the accumulator and reach out as a cast from it
   does. *)

type int_case = { case : case; acc : Dt.any }

let int_dt =
  Dt.
    [
      Any Int8;
      Any Uint8;
      Any Int16;
      Any Uint16;
      Any Int32;
      Any Uint32;
      Any Int64;
    ]

let int_case =
  let open Gen in
  let+ c = case
  and+ dt = of_list int_dt
  and+ out = of_list Dt.[ Any Int8; Any Int32; Any Int64 ]
  and+ acc = of_list Dt.[ Any Int32; Any Uint32; Any Int64 ] in
  { case = { c with dt; out; spread = 0 }; acc }

let int_case =
  Gen.with_pp
    (fun ppf c ->
      let (Dt.Any acc) = c.acc in
      Format.fprintf ppf "%a, acc %s" pp_case c.case (Dt.name acc))
    int_case

let int_example ?(out = Dt.Any Dt.Int32) ?(batch = 1) ?(bpad = 0) dt ~m ~k =
  {
    case =
      {
        dt;
        out;
        a_t = false;
        b_t = true;
        batch;
        m;
        n = m;
        k;
        init = Full;
        pad = 0;
        bpad;
        spread = 0;
      };
    acc = Dt.Any Dt.Int32;
  }

let int_examples =
  [
    (* Bytes into 32 bits sum on the matrix units in float32 chunks: sums longer
       than a chunk, 1,024 terms of int8 or 258 of uint8. *)
    int_example (Dt.Any Dt.Int8) ~m:64 ~k:2112;
    int_example (Dt.Any Dt.Int8) ~m:65 ~k:2101;
    int_example (Dt.Any Dt.Uint8) ~m:64 ~k:608;
    int_example (Dt.Any Dt.Uint8) ~m:65 ~k:601;
    (* Negative sums in 32 bits into a 64-bit out, on the matrix units and on
       the SIMD units. *)
    int_example ~out:(Dt.Any Dt.Int64) (Dt.Any Dt.Int8) ~m:64 ~k:64;
    int_example ~out:(Dt.Any Dt.Int64) (Dt.Any Dt.Int16) ~m:64 ~k:64;
    (* Batches whose bytes start off 16-byte alignment. *)
    int_example ~batch:2 ~bpad:1 (Dt.Any Dt.Int8) ~m:64 ~k:64;
    int_example ~batch:2 ~bpad:2 (Dt.Any Dt.Int8) ~m:64 ~k:64;
  ]

let contract_wraps =
  prop ~count:40 ~examples:int_examples "an integer contraction wraps" int_case
    (fun { case = c; acc } ->
      cover "64-bit accumulator" (acc = Dt.Any Dt.Int64);
      cover "narrow operands" (c.dt = Dt.Any Dt.Int8 || c.dt = Dt.Any Dt.Uint8);
      cover "a batch pad" (c.batch > 1 && c.bpad > 0);
      let t = dev () in
      let dims, a, b, out, init, run = call ~acc t c in
      ignore (S.run t run);
      let wrong, first = S.contract_wrong ?init ~acc dims ~a ~b ~out in
      equal ~msg:(strf "outputs wrong, the first %d" first) int 0 wrong)

(* float64 has no arithmetic on Apple GPUs: the library declines it. *)
let declines_float64 () =
  let t = dev () in
  let m =
    matrix t (Dt.Any Dt.Float64) ~trans:false ~batch:1 ~rows:4 ~cols:4 ~pad:0
      ~bpad:0 ~seed:1 ~spread:0
  in
  let plan =
    S.plan_contract ~acc:(Dt.Any Dt.Float64) t (1, 4, 4, 4) ~a:m ~b:m ~out:m
  in
  is_none ~pp:(fun ppf _ -> Format.pp_print_string ppf "a run") plan

(* Determinism: a call's results are the same bits each time it runs. *)

let bytes_of (arg : S.arg) n =
  let v = S.view Bigarray.char (S.arg_operand arg) in
  String.init n (fun i -> v.{i})

let deterministic c () =
  let t = dev () in
  let _, _, _, out, _, run = call t c in
  let (Dt.Any out_dt) = c.out in
  let n = Dt.bytes out_dt (c.batch * c.m * c.n) in
  let go = S.prepare t run in
  ignore (go ());
  let first = bytes_of out n in
  ignore (go ());
  equal ~msg:"run twice" string first (bytes_of out n)

let determinism =
  let case dt ~batch ~m ~n ~k ~b_t =
    {
      dt;
      out = dt;
      a_t = false;
      b_t;
      batch;
      m;
      n;
      k;
      init = Full;
      pad = 0;
      bpad = 0;
      spread = 8;
    }
  in
  cases
    ~name:(Format.asprintf "%a" pp_case)
    "determinism"
    [
      case (Dt.Any Dt.Float32) ~batch:2 ~m:200 ~n:300 ~k:1000 ~b_t:false;
      case (Dt.Any Dt.Bfloat16) ~batch:2 ~m:130 ~n:257 ~k:2000 ~b_t:true;
      case (Dt.Any Dt.Float16) ~batch:1 ~m:768 ~n:776 ~k:300 ~b_t:false;
      case (Dt.Any Dt.Float16) ~batch:2 ~m:1 ~n:1000 ~k:3000 ~b_t:true;
      case (Dt.Any Dt.Bfloat16) ~batch:2 ~m:1 ~n:1000 ~k:3000 ~b_t:false;
      case (Dt.Any Dt.Float32) ~batch:2 ~m:9 ~n:1000 ~k:3000 ~b_t:false;
      case (Dt.Any Dt.Float32) ~batch:1 ~m:64 ~n:100 ~k:2048 ~b_t:false;
    ]
    (fun c -> deterministic c ())

let contract =
  group ~timeout:120. "contract"
    [
      contract_bound;
      contract_wraps;
      test "float64 declines" declines_float64;
      determinism;
    ]

let () =
  S.hold_gpu ();
  exit (run "nx_metal" [ runs; arithmetic; contract ])
