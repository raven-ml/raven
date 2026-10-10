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

(* The operands' values past the drawn ones. *)
type values =
  | Drawn
  | Cancelling (* a's second half along k negates its first; b's repeats *)
  | Special of int (* NaN, ±inf, -0 and a subnormal at places from a seed *)
  | Subnormal of int (* subnormals at a tenth of the places, from a seed *)
  | Extreme (* integers: every element its dtype's extreme *)
  | Underflow (* float32: every product negative and below the least normal *)

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
  values : values;
}

let pp_values ppf = function
  | Drawn -> Format.pp_print_string ppf "drawn"
  | Cancelling -> Format.pp_print_string ppf "cancelling"
  | Special s -> Format.fprintf ppf "special %d" s
  | Subnormal s -> Format.fprintf ppf "subnormal %d" s
  | Extreme -> Format.pp_print_string ppf "extreme"
  | Underflow -> Format.pp_print_string ppf "underflow"

let pp_case ppf c =
  let name (Dt.Any dt) = Dt.name dt in
  Format.fprintf ppf
    "%s -> %s, a%s b%s, %d x (%d x %d x %d), init %s, pad %d, bpad %d, spread \
     %d, %a"
    (name c.dt) (name c.out)
    (if c.a_t then "^T" else "")
    (if c.b_t then "^T" else "")
    c.batch c.m c.n c.k
    (match c.init with No_init -> "none" | Full -> "full" | Bias -> "bias")
    c.pad c.bpad c.spread pp_values c.values

let floats_dt = [ Dt.Any Dt.Float32; Dt.Any Dt.Float16; Dt.Any Dt.Bfloat16 ]

(* Extents at, around and between the dense kernels' 16, 32 and 64 wide tiles
   and 16 and 32 long steps, or whole tiles, [tile] each. *)
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
  and+ spread = of_list [ 0; 8; 40 ]
  and+ values =
    let* seed = int_range 0 1_000_000 in
    frequency
      [
        (4, constant Drawn);
        (1, constant Cancelling);
        (1, constant (Special seed));
        (1, constant (Subnormal seed));
      ]
  in
  { dt; out; a_t; b_t; batch; m; n; k; init; pad; bpad; spread; values }

let case = Gen.with_pp pp_case case

(* An operand's memory as an argument, its length in elements, and the index of
   its element (p, r, c). *)
type matrix = { arg : S.arg; len : int; at : int -> int -> int -> int }

(* An operand of [rows] x [cols] per batch, stored [cols][rows] if [trans], each
   stored row [pad] elements longer and each batch [bpad], with values from
   [seed]. *)
let matrix t (Dt.Any dt) ~trans ~batch ~rows ~cols ~pad ~bpad ~seed ~spread =
  let ld = (if trans then rows else cols) + pad in
  let per = ((if trans then cols else rows) * ld) + bpad in
  let len = batch * per in
  let o = S.operand t (max 1 (Dt.bytes dt len)) in
  if len > 0 then S.generate ~spread t o dt len ~seed;
  let s1, s2 = if trans then (1, ld) else (ld, 1) in
  {
    arg = S.arg o dt (per, s1, s2);
    len;
    at = (fun p r c -> (p * per) + (r * s1) + (c * s2));
  }

(* A matrix's elements as unsigned codes of [size] bytes: get and set. *)
let codes x size =
  let o = S.arg_operand x.arg in
  match size with
  | 1 ->
      let v = S.view Bigarray.int8_unsigned o in
      ((fun i -> v.{i}), fun i c -> v.{i} <- c land 0xff)
  | 2 ->
      let v = S.view Bigarray.int16_unsigned o in
      ((fun i -> v.{i}), fun i c -> v.{i} <- c land 0xffff)
  | _ ->
      let v = S.view Bigarray.int32 o in
      ( (fun i -> Int32.to_int v.{i} land 0xffffffff),
        fun i c -> v.{i} <- Int32.of_int c )

(* NaN, +inf, -inf, -0 and a subnormal, as codes of the float [dt]. *)
let specials = function
  | Dt.Any Dt.Float32 ->
      [ 0x7fc00000; 0x7f800000; 0xff800000; 0x80000000; 0x00080000 ]
  | Dt.Any Dt.Float16 -> [ 0x7e00; 0x7c00; 0xfc00; 0x8000; 0x0200 ]
  | _ -> [ 0x7fc0; 0x7f80; 0xff80; 0x8000; 0x0008 ]

(* The integer [dt]'s extreme, as a code: its least if signed, its greatest if
   not. *)
let extreme (Dt.Any dt) =
  let bits = Dt.bits dt in
  match Dt.Any dt with
  | Dt.Any (Dt.Uint8 | Dt.Uint16 | Dt.Uint32) -> (1 lsl bits) - 1
  | _ -> 1 lsl (bits - 1)

(* Writes [c.values] into a, which is m x k, and b, which is k x n. *)
let write_values c a b =
  let (Dt.Any dt) = c.dt in
  let size = Dt.bytes dt 1 in
  let sign = 1 lsl ((8 * size) - 1) in
  let at_random seed x f =
    if x.len > 0 then begin
      let r = Random.State.make [| seed |] in
      let get, set = codes x size in
      f (fun () -> Random.State.int r x.len) get set
    end
  in
  match c.values with
  | Drawn -> ()
  | Cancelling ->
      let ga, sa = codes a size and gb, sb = codes b size in
      let h = c.k / 2 in
      for p = 0 to c.batch - 1 do
        for l = 0 to h - 1 do
          for i = 0 to c.m - 1 do
            sa (a.at p i (l + h)) (ga (a.at p i l) lxor sign)
          done;
          for j = 0 to c.n - 1 do
            sb (b.at p (l + h) j) (gb (b.at p l j))
          done
        done
      done
  | Special seed ->
      List.iteri
        (fun i x ->
          at_random (seed + i) x (fun place _ set ->
              List.iter (fun code -> set (place ()) code) (specials c.dt)))
        [ a; b ]
  | Subnormal seed ->
      let sub = List.nth (specials c.dt) 4 in
      List.iteri
        (fun i x ->
          at_random (seed + i) x (fun place _ set ->
              for e = 0 to x.len / 10 do
                set (place ()) (if e land 1 = 0 then sub else sub lor sign)
              done))
        [ a; b ]
  | Extreme ->
      List.iter
        (fun x ->
          let _, set = codes x size in
          for i = 0 to x.len - 1 do
            set i (extreme c.dt)
          done)
        [ a; b ]
  | Underflow ->
      (* 2^-70 and -2^-70: each sum is a subnormal, which the GPU writes as
         -0. *)
      List.iter
        (fun (x, code) ->
          let _, set = codes x size in
          for i = 0 to x.len - 1 do
            set i code
          done)
        [ (a, 0x1c800000); (b, 0x9c800000) ]

let init_arg t c (Dt.Any dt) =
  match c.init with
  | No_init -> None
  | Full ->
      Some
        (matrix t (Dt.Any dt) ~trans:false ~batch:c.batch ~rows:c.m ~cols:c.n
           ~pad:0 ~bpad:0 ~seed:3 ~spread:c.spread)
          .arg
  | Bias ->
      let o = S.operand t (max 1 (Dt.bytes dt c.n)) in
      if c.n > 0 then S.generate ~spread:c.spread t o dt c.n ~seed:4;
      Some (S.arg o dt (0, 0, 1))

(* The call's operands and its run. *)
let call ?acc t c =
  let operand ~trans ~rows ~cols ~seed =
    matrix t c.dt ~trans ~batch:c.batch ~rows ~cols ~pad:c.pad ~bpad:c.bpad
      ~seed ~spread:c.spread
  in
  let a = operand ~trans:c.a_t ~rows:c.m ~cols:c.k ~seed:1 in
  let b = operand ~trans:c.b_t ~rows:c.k ~cols:c.n ~seed:2 in
  write_values c a b;
  let (Dt.Any out_dt) = c.out in
  let mn = c.batch * c.m * c.n in
  let out =
    S.arg (S.operand t (max 1 (Dt.bytes out_dt mn))) out_dt (c.m * c.n, c.n, 1)
  in
  let init = init_arg t c c.out in
  let dims = (c.batch, c.m, c.n, c.k) in
  let a = a.arg and b = b.arg in
  let run =
    require_some ~msg:"the planner declined"
      (S.plan_contract ?init ?acc t dims ~a ~b ~out)
  in
  (dims, a, b, out, init, run)

(* Every output is within the contraction's bound of the exact result, or is the
   NaN or infinity its terms give. *)
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
      values = Drawn;
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
      (* A stored [k][m] of 16 MiB: read where it lies, never copied. *)
      large (Dt.Any Dt.Float32) ~a_t:true ~b_t:false ~m:1024 ~n:32 ~k:4096
        ~init:No_init;
      (* Few rows of a split along k, on wide tiles. *)
      {
        (large (Dt.Any Dt.Float16) ~a_t:false ~b_t:true ~m:9 ~n:200 ~k:2880
           ~init:Full)
        with
        batch = 1;
      };
      (* Subnormal operands times 2^40: a flushed operand loses up to 2^-126
         times the other. *)
      {
        empty with
        dt = Dt.Any Dt.Bfloat16;
        out = Dt.Any Dt.Float32;
        m = 40;
        n = 40;
        k = 64;
        spread = 40;
        values = Subnormal 7;
      };
    ]
  in
  prop ~count:80 ~examples "a float contraction is within its bound" case
    (fun c ->
      cover "an empty sum" (c.k = 0);
      cover "a transposed operand" (c.a_t || c.b_t);
      cover "a bias" (c.init = Bias);
      cover "narrow out" (c.out <> Dt.Any Dt.Float32);
      cover "a batch pad" (c.batch > 1 && c.bpad > 0);
      cover "wide spread" (c.spread = 40);
      cover "cancellation" (c.values = Cancelling && c.k > 1);
      cover "NaN, infinities, -0"
        (match c.values with Special _ -> c.m * c.n * c.k > 0 | _ -> false);
      cover "subnormal operands"
        (match c.values with Subnormal _ -> c.m * c.n * c.k > 0 | _ -> false);
      let t = dev () in
      let dims, a, b, out, init, run = call t c in
      (* A call's scratch is its split's parts alone, whatever its operands: at
         most 2 x 256 tiles of 64 x 64 float32 outputs. *)
      at_most ~msg:"scratch bytes" int ~than:(8 * 1024 * 1024) (S.scratch run);
      let kernels = S.entries run in
      let launches_one prefix = String.starts_with ~prefix in
      let launches prefix = List.exists (launches_one prefix) kernels in
      let dense =
        List.filter
          (fun k -> launches_one "contract_f" k || launches_one "contract_bf" k)
          kernels
      in
      (* A dense kernel's tile, by its name's last part: s small, w and the
         orders wide, the orders alone large. Large tiles read whole tiles;
         small and wide ones, tiles reaching past the matrix. *)
      let tile k =
        match List.rev (String.split_on_char '_' k) with
        | "s" :: _ -> `Small
        | last :: _ when last.[0] = 'w' -> `Wide
        | _ -> `Large
      in
      let tiles x = List.exists (fun k -> tile k = x) dense in
      cover "64 x 64 tiles" (tiles `Large);
      cover "32 x 32 tiles" (tiles `Small);
      cover "16 x 64 tiles" (tiles `Wide);
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
  and+ acc =
    of_list Dt.[ Any Int32; Any Uint32; Any Int64; Any Int16; Any Uint8 ]
  in
  { case = { c with dt; out; spread = 0; values = Drawn }; acc }

let int_case =
  Gen.with_pp
    (fun ppf c ->
      let (Dt.Any acc) = c.acc in
      Format.fprintf ppf "%a, acc %s" pp_case c.case (Dt.name acc))
    int_case

let int_example ?(out = Dt.Any Dt.Int32) ?(acc = Dt.Any Dt.Int32) ?(batch = 1)
    ?(bpad = 0) ?(values = Drawn) dt ~m ~k =
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
        values;
      };
    acc;
  }

let int_examples =
  [
    (* int8 into 32 bits sums on the matrix units in float32 chunks of 1,024
       terms, in whole tiles: sums longer than a chunk, and the extremes'
       longest; past whole tiles, and uint8, on the SIMD units. *)
    int_example (Dt.Any Dt.Int8) ~m:64 ~k:2112;
    int_example (Dt.Any Dt.Int8) ~m:65 ~k:2101;
    int_example (Dt.Any Dt.Uint8) ~m:64 ~k:608;
    int_example (Dt.Any Dt.Uint8) ~m:65 ~k:601;
    int_example ~values:Extreme (Dt.Any Dt.Int8) ~m:64 ~k:2112;
    int_example ~values:Extreme (Dt.Any Dt.Uint8) ~m:64 ~k:608;
    int_example ~values:Extreme (Dt.Any Dt.Int16) ~m:20 ~k:100;
    (* Negative sums in 32 bits into a 64-bit out, on the matrix units and on
       the SIMD units. *)
    int_example ~out:(Dt.Any Dt.Int64) (Dt.Any Dt.Int8) ~m:64 ~k:64;
    int_example ~out:(Dt.Any Dt.Int64) (Dt.Any Dt.Int16) ~m:64 ~k:64;
    (* Unsigned sums of 2^31 or more into a 64-bit out widen by no sign, on the
       SIMD units, the matrix units and one row. *)
    int_example ~acc:(Dt.Any Dt.Uint32) ~out:(Dt.Any Dt.Uint64)
      (Dt.Any Dt.Uint16) ~m:17 ~k:300;
    int_example ~acc:(Dt.Any Dt.Uint32) ~out:(Dt.Any Dt.Int64) ~values:Extreme
      (Dt.Any Dt.Uint8) ~m:64 ~k:608;
    int_example ~acc:(Dt.Any Dt.Uint32) ~out:(Dt.Any Dt.Uint64)
      (Dt.Any Dt.Int8) ~m:64 ~k:64;
    int_example ~acc:(Dt.Any Dt.Uint32) ~out:(Dt.Any Dt.Uint64)
      (Dt.Any Dt.Uint16) ~m:1 ~k:300;
    (* Accumulators narrower than 32 bits wrap to their width, then reach out
       widened by their sign. *)
    int_example ~acc:(Dt.Any Dt.Int16) ~out:(Dt.Any Dt.Int64) (Dt.Any Dt.Int16)
      ~m:17 ~k:300;
    int_example ~acc:(Dt.Any Dt.Uint8) ~out:(Dt.Any Dt.Int32) (Dt.Any Dt.Int8)
      ~m:17 ~k:300;
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

(* Every kernel of the library runs in some call, and its results are right:
   each instance is named by a call that launches it. *)
let every_kernel () =
  let t = dev () in
  let seen = Hashtbl.create 128 in
  let launch ?acc c =
    let dims, a, b, out, init, run = call ?acc t c in
    List.iter (fun k -> Hashtbl.replace seen k ()) (S.entries run);
    ignore (S.run t run);
    (dims, a, b, out, init)
  in
  let base =
    {
      dt = Dt.Any Dt.Float32;
      out = Dt.Any Dt.Float32;
      a_t = false;
      b_t = false;
      batch = 1;
      m = 1;
      n = 1;
      k = 1;
      init = No_init;
      pad = 0;
      bpad = 0;
      spread = 0;
      values = Drawn;
    }
  in
  let orders = [ (false, false); (false, true); (true, false); (true, true) ] in
  (* Large tiles take 128 or more whole ones; small ones fewer, or tiles past
     the matrix; wide ones few rows. *)
  let shapes =
    [
      (128, 64, 64, 32);
      (32, 65, 65, 3);
      (1, 64, 64, 32);
      (1, 33, 33, 33);
      (1, 9, 70, 40);
      (1, 1, 70, 40);
      (1, 40, 40, 1024);
    ]
  in
  List.iter
    (fun dt ->
      List.iter
        (fun (a_t, b_t) ->
          List.iter
            (fun (batch, m, n, k) ->
              let c = { base with dt; out = dt; a_t; b_t; batch; m; n; k } in
              let dims, a, b, out, init = launch c in
              let worst, at = S.contract_error ?init dims ~a ~b ~out in
              at_most
                ~msg:
                  (strf "%s: output %d's error over its bound"
                     (Format.asprintf "%a" pp_case c)
                     at)
                float_exact ~than:1. worst)
            shapes)
        orders)
    floats_dt;
  let ints =
    List.concat_map
      (fun dt ->
        List.concat_map
          (fun (a_t, b_t) ->
            [
              (dt, a_t, b_t, 64, 64, Dt.Any Dt.Int32);
              (dt, a_t, b_t, 65, 65, Dt.Any Dt.Int32);
            ])
          orders)
      [ Dt.Any Dt.Int8; Dt.Any Dt.Uint8 ]
    @ [
        (Dt.Any Dt.Int16, false, false, 20, 20, Dt.Any Dt.Int32);
        (Dt.Any Dt.Int16, false, false, 20, 20, Dt.Any Dt.Int64);
      ]
  in
  List.iter
    (fun (dt, a_t, b_t, m, k, acc) ->
      let c = { base with dt; out = Dt.Any Dt.Int32; a_t; b_t; m; n = m; k } in
      let dims, a, b, out, init = launch ~acc c in
      let wrong, first = S.contract_wrong ?init ~acc dims ~a ~b ~out in
      equal
        ~msg:
          (strf "%s: outputs wrong, the first %d"
             (Format.asprintf "%a" pp_case c)
             first)
        int 0 wrong)
    ints;
  let library k =
    String.starts_with ~prefix:"contract_" k
    || String.starts_with ~prefix:"skinny_" k
  in
  let missing =
    List.filter
      (fun k -> library k && not (Hashtbl.mem seen k))
      (Array.to_list S.kernels)
  in
  equal ~msg:"kernels no call launched" (list string) [] missing

(* The plan's acceptance: over every (a, b, acc, out) of the dtypes, at a shape
   of each kernel class, the plan accepts exactly the stated set, and an
   accepted call's results are within the bound (floats) or exact (integers).
   Out is filled with 0xAA bytes before each run, so a path that writes nothing
   fails. The reference refuses a dtype it cannot read. *)
let every_quadruple () =
  let t = dev () in
  let floats = function
    | Dt.Any (Dt.Float32 | Dt.Float16 | Dt.Bfloat16) -> true
    | _ -> false
  in
  let integers = function
    | Dt.Any
        ( Dt.Int8 | Dt.Uint8 | Dt.Int16 | Dt.Uint16 | Dt.Int32 | Dt.Uint32
        | Dt.Int64 | Dt.Uint64 ) ->
        true
    | _ -> false
  in
  (* Accepted: a = b, both float32, float16 or bfloat16, acc float32, out any of
     the three; or a = b, integers of 8 to 64 bits, acc and out integers of 8 to
     64 bits. *)
  let expected da db acc dout =
    da = db
    && ((floats da && acc = Dt.Any Dt.Float32 && floats dout)
       || (integers da && integers acc && integers dout))
  in
  (* Small and wide tiles, one row, and a shape of the SIMD integer tile. *)
  let shapes = [ (64, 64, 64); (3, 70, 300); (1, 70, 300); (17, 129, 30) ] in
  List.iter
    (fun (m, n, k) ->
      let mem len = S.operand t (8 * len) in
      let am = mem (m * k) and bm = mem (k * n) and om = mem (m * n) in
      let arg o (Dt.Any dt) strides = S.arg o dt strides in
      List.iter
        (fun da ->
          List.iter
            (fun db ->
              List.iter
                (fun acc ->
                  List.iter
                    (fun dout ->
                      let a = arg am da (m * k, k, 1)
                      and b = arg bm db (k * n, n, 1)
                      and out = arg om dout (m * n, n, 1) in
                      let dims = (1, m, n, k) in
                      let plan = S.plan_contract ~acc t dims ~a ~b ~out in
                      let name (Dt.Any dt) = Dt.name dt in
                      let call () =
                        strf "%s x %s, acc %s -> %s, %d x %d x %d" (name da)
                          (name db) (name acc) (name dout) m n k
                      in
                      let want = expected da db acc dout in
                      if want <> Option.is_some plan then
                        equal
                          ~msg:(strf "%s: accepted" (call ()))
                          bool want (Option.is_some plan);
                      match plan with
                      | None -> ()
                      | Some run ->
                          let draw o (Dt.Any dt) len seed =
                            S.generate ~spread:8 t o dt len ~seed
                          in
                          draw am da (m * k) 1;
                          draw bm db (k * n) 2;
                          Bigarray.Array1.fill (S.view Bigarray.char om) '\xaa';
                          ignore (S.run t run);
                          if floats acc then
                            let worst, at = S.contract_error dims ~a ~b ~out in
                            at_most
                              ~msg:
                                (strf "%s: output %d's error over its bound"
                                   (call ()) at)
                              float_exact ~than:1. worst
                          else
                            let wrong, first =
                              S.contract_wrong ~acc dims ~a ~b ~out
                            in
                            equal
                              ~msg:
                                (strf "%s: outputs wrong, the first %d"
                                   (call ()) first)
                              int 0 wrong)
                    Dt.all)
                Dt.all)
            Dt.all)
        Dt.all)
    shapes

(* The bound's check sees a NaN output of a finite sum, wherever it lies
   among the outputs. *)
let nan_output =
  cases ~name:(strf "output %d")
    "a NaN output of a finite sum is over its bound" [ 0; 31; 63 ]
    (fun at ->
      let t = dev () in
      let c =
        {
          dt = Dt.Any Dt.Float32;
          out = Dt.Any Dt.Float32;
          a_t = false;
          b_t = false;
          batch = 1;
          m = 8;
          n = 8;
          k = 4;
          init = No_init;
          pad = 0;
          bpad = 0;
          spread = 0;
          values = Drawn;
        }
      in
      let dims, a, b, out, init, run = call t c in
      ignore (S.run t run);
      (floats (S.arg_operand out)).{at} <- Float.nan;
      let worst, where = S.contract_error ?init dims ~a ~b ~out in
      equal ~msg:"the output over its bound" int at where;
      greater ~msg:"its error over its bound" float_exact ~than:1. worst)

(* float64 has no arithmetic on Apple GPUs: the library declines it. *)
let declines_float64 () =
  let t = dev () in
  let m =
    (matrix t (Dt.Any Dt.Float64) ~trans:false ~batch:1 ~rows:4 ~cols:4 ~pad:0
       ~bpad:0 ~seed:1 ~spread:0)
      .arg
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
      values = Drawn;
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

(* Layouts: a contraction's bits are a function of its operands' values and
   shapes, never of the order a and b are stored in. *)

(* [x], an operand of [rows] x [cols] per batch, copied element for element into
   a fresh one stored [cols][rows] if [trans]. *)
let relaid t c x ~trans ~rows ~cols =
  let y =
    matrix t c.dt ~trans ~batch:c.batch ~rows ~cols ~pad:c.pad ~bpad:c.bpad
      ~seed:0 ~spread:0
  in
  let (Dt.Any dt) = c.dt in
  let get, _ = codes x (Dt.bytes dt 1) and _, set = codes y (Dt.bytes dt 1) in
  for p = 0 to c.batch - 1 do
    for r = 0 to rows - 1 do
      for q = 0 to cols - 1 do
        set (y.at p r q) (get (x.at p r q))
      done
    done
  done;
  y

(* Whether the code [x] of the float [dt] is a NaN: a NaN result is some NaN. *)
let nan_code dt x =
  match dt with
  | Dt.Any Dt.Float32 -> x land 0x7f800000 = 0x7f800000 && x land 0x7fffff <> 0
  | Dt.Any Dt.Float16 -> x land 0x7c00 = 0x7c00 && x land 0x3ff <> 0
  | _ -> x land 0x7f80 = 0x7f80 && x land 0x7f <> 0

let orders_agree c =
  let t = dev () in
  let a =
    matrix t c.dt ~trans:false ~batch:c.batch ~rows:c.m ~cols:c.k ~pad:c.pad
      ~bpad:c.bpad ~seed:1 ~spread:c.spread
  and b =
    matrix t c.dt ~trans:false ~batch:c.batch ~rows:c.k ~cols:c.n ~pad:c.pad
      ~bpad:c.bpad ~seed:2 ~spread:c.spread
  in
  write_values c a b;
  let a_t = relaid t c a ~trans:true ~rows:c.m ~cols:c.k
  and b_t = relaid t c b ~trans:true ~rows:c.k ~cols:c.n in
  let init = init_arg t c c.out in
  let (Dt.Any out_dt) = c.out in
  let size = Dt.bytes out_dt 1 in
  let mn = c.batch * c.m * c.n in
  (* The outputs' codes of the call with a and b. *)
  let outputs a b =
    let out =
      matrix t c.out ~trans:false ~batch:c.batch ~rows:c.m ~cols:c.n ~pad:0
        ~bpad:0 ~seed:5 ~spread:0
    in
    let run =
      require_some ~msg:"the planner declined"
        (S.plan_contract ?init t (c.batch, c.m, c.n, c.k) ~a:a.arg ~b:b.arg
           ~out:out.arg)
    in
    ignore (S.run t run);
    let get, _ = codes out size in
    (S.entries run, Array.init mn get)
  in
  let entries, nn = outputs a b in
  cover "a skinny product"
    (List.exists (String.starts_with ~prefix:"skinny_") entries);
  cover "a split along k" (List.mem "contract_combine" entries);
  List.iter
    (fun (name, a, b) ->
      let _, y = outputs a b in
      let differs i =
        y.(i) <> nn.(i) && not (nan_code c.out y.(i) && nan_code c.out nn.(i))
      in
      let wrong = List.filter differs (List.init mn Fun.id) in
      let show i =
        strf "output %d: %#x, a and b C-contiguous %#x" i y.(i) nn.(i)
      in
      equal ~msg:name (list string) []
        (List.filteri (fun j _ -> j < 4) (List.map show wrong)))
    [
      ("b stored [n][k]", a, b_t);
      ("a stored [k][m]", a_t, b);
      ("both transposed", a_t, b_t);
    ]

let orders =
  let example dt ~m ~n ~k =
    {
      dt;
      out = Dt.Any Dt.Float32;
      a_t = false;
      b_t = false;
      batch = 1;
      m;
      n;
      k;
      init = No_init;
      pad = 0;
      bpad = 0;
      spread = 8;
      values = Drawn;
    }
  in
  let examples =
    [
      (* One row: the skinny kernel, b read along either axis. *)
      example (Dt.Any Dt.Bfloat16) ~m:1 ~n:70 ~k:2880;
      example (Dt.Any Dt.Float32) ~m:1 ~n:70 ~k:300;
      (* 40 rows past whole tiles: 32 x 32 tiles count 158 and split k in 2, 16
         x 64 tiles count 120 and would split it in 4. *)
      example (Dt.Any Dt.Float32) ~m:40 ~n:2500 ~k:4096;
      (* Sums of -0 over 48 terms: 16-step tiles end on k, 32-step tiles pass
         it. *)
      {
        (example (Dt.Any Dt.Float32) ~m:17 ~n:17 ~k:48) with
        values = Underflow;
      };
    ]
  in
  prop ~count:30 ~examples "a contraction's bits do not depend on its orders"
    case orders_agree

let contract =
  group ~timeout:120. "contract"
    [
      contract_bound;
      contract_wraps;
      nan_output;
      test "float64 declines" declines_float64;
      test "every kernel runs" every_kernel;
      test "the plan declines or is right" every_quadruple;
      determinism;
      orders;
    ]

let () =
  S.hold_gpu ();
  exit (run "nx_metal" [ runs; arithmetic; contract ])
