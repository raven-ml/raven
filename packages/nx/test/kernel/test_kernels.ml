(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx_kernel.S's calls and maps, run through every kernel library the host runs,
   in nx_kernel.mli's order: what a call that does not fit, a refusal, a
   destination, a shared buffer, a box and a NaN result give; check_dst, padded
   loads, computes_on and map; then each kind of Nx_kernel.Prog a map computes,
   against a reference built element by element, the exact ones exactly and the
   others within Prog's bounds. *)

open Windtrap
open Nx_array_gen
open Kernels_cases
module A = Nx_array
module D = Nx_array.Dtype
module B = Rig.Buffer
module Support = Nx_kernels_support

(* [op] of [a] by [b]'s kernels, into a fresh C-contiguous array of [dt] on
   [b]'s device, read back where the host reads it; [None] if the kernels
   decline a kind they do not state they compute. *)
let run_op (b : Support.backend) op dt a =
  let module K = (val b.kernels) in
  let a = Support.on b a in
  let dst = A.create b.device dt (L.shape (A.layout a)) in
  match Support.map1 (module K) op ~dst a with
  | A.Done -> Some (Support.host dst)
  | A.Declined ->
      check_declined ~into:(D.Any dt) b (K1 op) (D.Any (A.dtype a));
      None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

let declines r =
  cover "computed" (Option.is_some r);
  classify "declined" (Option.is_none r)

(* [x]'s bytes on the host, as {!bits_of} gives them. *)
let bytes_of x = bits_of (Support.host x)

(* Bits *)

(* The first eight elements of [want] and [got], arrays of one dtype and shape,
   that [same i w g] tells apart, [w] and [g] the bytes of element [i], each
   beside the element of [src] it came from where given. *)
let mismatches ?src same (want : _ A.t) (got : _ A.t) =
  let d = D.Any (A.dtype want) in
  let per = max 1 (D.bits (A.dtype want) / 8) in
  let w = bits_of want and g = bits_of got in
  let hex bits per i =
    String.concat ""
      (List.init per (fun j -> Printf.sprintf "%02x" bits.((i * per) + j)))
  in
  let source i =
    match src with
    | None -> ""
    | Some (A.Any s) ->
        let sp = max 1 (D.bits (A.dtype s) / 8) in
        " from " ^ hex (bits_of s) sp i
  in
  let element bits i = (elements_of d (Array.sub bits (i * per) per)).(0) in
  let bad = ref [] in
  if w <> g then
    for i = (Array.length w / per) - 1 downto 0 do
      if not (same i (element w i) (element g i)) then
        bad :=
          Printf.sprintf "element %d: %s, expected %s%s" i (hex g per i)
            (hex w per i) (source i)
          :: !bad
    done;
  List.filteri (fun i _ -> i < 8) !bad

(* The elements whose bits differ. *)
let bits_differ ?src want got = mismatches ?src (fun _ e f -> e = f) want got

(* The elements whose bits differ, any NaN for a NaN: a kind that computes a NaN
   fixes no NaN's bits (nx_kernel.mli). *)
let values_differ want got =
  let same = Testable.equal (Elements.same (D.Any (A.dtype want))) in
  mismatches (fun _ e f -> same e f) want got

let no_mismatch = equal (list string) []

(* Copies *)

let law_copy b (Case a) =
  let r = run_op b Copy (A.dtype a) a in
  declines r;
  Option.iter (fun dst -> no_mismatch (bits_differ a dst)) r

let test_copy_nan_payloads b () =
  let bits = [| 0x7fc00001l; 0xffa00002l; 0x80000000l; 0x7f800001l |] in
  let f = Option.get (A.bitcast D.Float32 (A.of_array D.Uint32 [| 4 |] bits)) in
  Option.iter
    (fun dst ->
      equal (array int32) bits
        (A.to_array (Option.get (A.bitcast D.Uint32 dst))))
    (run_op b Copy D.Float32 f)

(* A drawn case, of a few hundred thousand elements one time in five. *)
let any_size = Gen.frequency [ (4, case); (1, large) ]

(* Casts *)

(* Whether a cast from [s] to [d] keeps a float's bits: into its own dtype, and
   a float32 or float64 into or out of a complex number's real part. *)
let keeps_bits (D.Any s as from) (D.Any d as into) =
  match (from, into) with
  | D.Any D.Float32, D.Any D.Complex64
  | D.Any D.Complex64, D.Any D.Float32
  | D.Any D.Float64, D.Any D.Complex128
  | D.Any D.Complex128, D.Any D.Float64 ->
      true
  | _ -> D.equal s d

(* The elements where [a] cast to [d] is not [dst]: the reference's bits, or
   where the cast does not keep bits and the reference is a NaN, a NaN,
   signalling only where [a]'s element is: a cast's NaN keeps neither its sign
   nor its payload (Dtype's stores). *)
let miscast a d dst =
  let want = reference a d in
  let dd = D.Any d and sd = D.Any (A.dtype a) in
  let nan_rule = not (keeps_bits sd dd) in
  let src = elements_of sd (bits_of a) in
  let same i e f =
    if nan_rule && Elements.nan_of dd e then
      Elements.nan_of dd f
      && ((not (Elements.signalling dd f)) || Elements.signalling sd src.(i))
    else e = f
  in
  mismatches ~src:(A.Any a) same want dst

let law_cast b (Case a, D.Any d) =
  let r = run_op b Cast d a in
  declines r;
  Option.iter (fun dst -> no_mismatch (miscast a d dst)) r

(* A case and a destination dtype, the case's own one time in five. *)
let cast_pair c =
  let open Gen in
  let* (Case a as c) = c in
  let+ d =
    frequency [ (4, dtypes); (1, constant ~pp:pp_dtype (D.Any (A.dtype a))) ]
  in
  (c, d)

(* [a] cast to every dtype, or to [dts], is the reference. *)
let casts_are_reference ?(dts = D.all) b a =
  List.iter
    (fun (D.Any d) ->
      Option.iter
        (fun dst -> equal ~msg:(D.name d) (list string) [] (miscast a d dst))
        (run_op b Cast d a))
    dts

(* The float and complex dtypes, whose casts round. *)
let rounding =
  List.filter (fun (D.Any d) -> D.is D.Float d || D.is D.Complex d) D.all

(* The floats narrower than float32: the dtypes a float32 rounds into. A float32
   into float32 is a copy, into float64 and complex128 exact, and into complex64
   its bits (Prog's Cast). *)
let narrow_floats =
  List.filter (fun (D.Any d) -> D.is D.Float d && D.bits d < 32) D.all

(* Every code of a format of at most 16 bits, cast to every dtype. *)
let test_every_code b (D.Any s) =
  let w = D.bits s in
  let n = 1 lsl w in
  let codes = Array.init n Fun.id in
  let a =
    match w with
    | 1 -> A.Any (A.of_array D.Bit [| 2 |] [| false; true |])
    | 4 -> A.Any (A.of_array D.Uint4 [| n |] codes)
    | 8 -> A.Any (A.of_array D.Uint8 [| n |] codes)
    | _ -> A.Any (A.of_array D.Uint16 [| n |] codes)
  in
  let (A.Any p) = a in
  let a =
    match D.bits s with
    | 1 -> A.expect s (A.Any p)
    | _ -> Option.get (A.bitcast s p)
  in
  casts_are_reference b a

let narrow = List.filter (fun (D.Any dt) -> D.bits dt <= 16) D.all

(* Floats about every rounding point of the narrow float formats, as float32 and
   as float64. *)

(* The low halves of {!sweep32}: about a float16 or bfloat16 rounding bit, all
   ones, and every value of the top 4 bits with the rest clear. The last hold
   float16's ties at every exponent with its kept bit odd and even, the rounding
   bits of exponents -15 and -16 and the tie at 65520, past which a float rounds
   to infinity. *)
let sweep32_lows =
  Array.of_list
    (List.sort_uniq compare
       ([ 0; 1; 0x0FFF; 0x1000; 0x1001; 0x7FFF; 0x8000; 0x8001; 0xFFFF ]
       @ List.init 16 (fun k -> k lsl 12)))

(* Element [i] of {!sweep32}: the float32 of high half [i / l] and low half
   {!sweep32_lows}[.(i mod l)], [l] the lows' count. *)
let sweep32_at i =
  let l = Array.length sweep32_lows in
  Int32.float_of_bits
    (Int32.of_int (((i / l) lsl 16) lor sweep32_lows.(i mod l)))

(* float32 of every high half and each of {!sweep32_lows}, the largest float32
   below a power of two among them; every float8 and float4 rounding bit lies in
   the high half. *)
let sweep32 = lazy (Array.init (0x10000 * Array.length sweep32_lows) sweep32_at)

(* The float64 neighbours of float32 values with their low 12 bits clear, which
   hold every tie of every narrow format: rounding through float32 to nearest
   first would land on a tie. *)
let sweep64 =
  lazy
    (Array.concat
       (List.init 0x10000 (fun hi ->
            let x =
              Int32.float_of_bits (Int32.of_int ((hi lsl 16) lor 0x1000))
            in
            if Float.is_finite x then [| Float.pred x; x; Float.succ x |]
            else [| x |])))

let test_sweep ~dts (type s) (s : (float, s) D.t) xs b () =
  let xs = Lazy.force xs in
  casts_are_reference ~dts b (A.of_array s [| Array.length xs |] xs)

(* Every length from 0 to 64, cast to every dtype, so that each run's vector
   loops and its scalar tail meet every cast: float32 about 1.0 from {!sweep32},
   and bfloat16 codes from 0x7F7C, which cross into NaN. *)
let test_lengths b () =
  let at = 0x3F80 * Array.length sweep32_lows in
  for n = 0 to 64 do
    let xs = Array.init n (fun i -> sweep32_at (at + i)) in
    casts_are_reference ~dts:rounding b (A.of_array D.Float32 [| n |] xs);
    let codes =
      A.of_array D.Uint16 [| n |] (Array.init n (fun i -> 0x7F7C + i))
    in
    casts_are_reference b (Option.get (A.bitcast D.Bfloat16 codes))
  done

(* Integers about every rounding point of the float formats: m·2^e and the
   points half a unit about it, for m of 2 to 54 significant bits, as int64 and
   uint64, cast to every dtype. *)
let ties =
  let open Int64 in
  List.concat_map
    (fun p ->
      List.concat_map
        (fun e ->
          let m = logor (shift_left 1L (p - 1)) 1L in
          let x = shift_left m e and half = shift_left 1L (e - 1) in
          List.concat_map
            (fun y -> [ y; pred y; succ y; neg y ])
            [ x; add x half; sub x half ])
        (List.init (64 - p) (fun e -> e + 1)))
    [ 2; 3; 4; 8; 11; 24; 25; 53; 54 ]

let test_integer_ties b () =
  let n = List.length ties in
  let srcs =
    [
      A.Any (A.of_array D.Int64 [| n |] (Array.of_list ties));
      A.Any (A.of_array D.Uint64 [| n |] (Array.of_list ties));
    ]
  in
  List.iter
    (fun (A.Any a) ->
      List.iter
        (fun (D.Any d) ->
          Option.iter
            (fun dst ->
              equal
                ~msg:(Printf.sprintf "%s to %s" (D.name (A.dtype a)) (D.name d))
                (list string) [] (miscast a d dst))
            (run_op b Cast d a))
        D.all)
    srcs

(* Every special of a dtype, cast to every dtype. *)
let test_specials b (D.Any s) =
  let sp = specials s in
  let a =
    A.v s
      (L.contiguous [| List.length sp |])
      (B.of_string (String.concat "" sp))
  in
  casts_are_reference b a

let with_specials =
  List.filter (fun (D.Any s) -> D.bits s >= 8 && specials s <> []) D.all

(* Bitcasts *)

(* Every kind of one operand. *)
let op1s =
  Nx_kernel.Prog.[ Copy; Cast; Bitcast ]
  @ List.map (fun (u, _) -> Nx_kernel.Prog.Unary u) unaries

let law_bitcast (b : Support.backend) (Case x, D.Any d) =
  let module K = (val b.kernels) in
  let dt = A.dtype x in
  let x = Support.on b x in
  cover "computed" (b.computes (K1 Bitcast) (D.Any dt));
  cover "into another dtype" (not (D.equal dt d));
  answers b (K1 Bitcast) (D.Any dt) (D.Any d)
    (L.shape (A.layout x))
    ~accepted:true
    ~want:(fun () -> bytes_of x)
    (fun dst -> Support.map1 (module K) Bitcast ~dst x)

(* A case and a dtype Bitcast takes it into: one of its width. *)
let same_width c =
  let open Gen in
  let* (Case a as c) = c in
  let dt = A.dtype a in
  let takes = List.filter (fun (D.Any d) -> P.accepts1 Bitcast dt d) D.all in
  let+ d = of_list ~pp:pp_dtype takes in
  (c, d)

(* Kinds of three operands *)

(* Pairs of the dtypes [dts]. *)
let pairs_of dts =
  Gen.with_pp pp_pair
    (let open Gen in
     let* (Case x) = case_of dts in
     let+ y = alike x in
     Pair (x, y))

let law_where (b : Support.backend) (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let x = Support.on b x and y = Support.on b y in
  let w = max 1 (D.bits dt / 8) in
  let where (type s) (c : (bool, s) A.t) =
    let c = Support.on b c in
    answers b (K3 Where) (D.Any dt) (D.Any dt) shape ~accepted:true
      ~want:(fun () ->
        let cs = A.to_array (Support.host c)
        and xs = bytes_of x
        and ys = bytes_of y in
        Array.concat
          (List.mapi
             (fun i c -> Array.sub (if c then xs else ys) (i * w) w)
             (Array.to_list cs)))
      (fun dst ->
        Support.map3 (module K) Where ~dst:(A.expect dt (A.Any dst)) c x y)
  in
  cover "a bool condition" (seed mod 2 = 0);
  cover "a bit condition" (seed mod 2 = 1);
  if seed mod 2 = 0 then where (seeded D.Bool shape seed)
  else where (seeded D.Bit shape seed)

let fma_pairs =
  pairs_of
    (Gen.of_list ~pp:pp_dtype
       (List.filter (fun (D.Any d) -> P.accepts3 Fma d d) D.all))

let law_fma (b : Support.backend) (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let dt = A.dtype x and shape = L.shape (A.layout x) in
  let z = seeded dt shape (seed + 1) in
  let x' = Support.on b x and y' = Support.on b y and z' = Support.on b z in
  cover "computed" (b.computes (K3 Fma) (D.Any dt));
  cover "a narrow float" (D.is D.Float dt && D.bits dt < 32);
  cover "an integer" (D.is D.Signed dt || D.is D.Unsigned dt);
  answers b (K3 Fma) (D.Any dt) (D.Any dt) shape ~accepted:true
    ~want:(fun () -> expected_on b "fma" dt ~compare:false [| x; y; z |])
    (fun dst ->
      Support.map3 (module K) Fma ~dst:(A.expect dt (A.Any dst)) x' y' z')

(* Kinds of no operand *)

let law_const (b : Support.backend) (D.Any dt, shape, seed) =
  let module K = (val b.kernels) in
  let w = max 1 (D.bits dt / 8) in
  let n = Array.fold_left ( * ) 1 shape in
  let e = bits_of (seeded dt [| 1 |] seed) in
  let fill = String.init (Array.length e) (fun i -> Char.chr e.(i)) in
  cover "sub-byte" (D.bits dt < 8);
  cover "no element" (n = 0);
  answers b (K0 (Fill fill)) (D.Any dt) (D.Any dt) (Iarray.of_array shape)
    ~accepted:true
    ~want:(fun () -> Array.concat (List.init n (fun _ -> Array.sub e 0 w)))
    (fun dst -> Support.map0 (module K) (Fill fill) ~dst)

(* A sub-byte destination, which the draws reach one case in six. *)
let sub_byte_const = [ (D.Any D.Int4, [| 3; 3 |], 1) ]

let law_coord (b : Support.backend) (shape, seed) =
  let module K = (val b.kernels) in
  let axis = seed mod Array.length shape in
  cover "an axis past the first" (axis > 0);
  answers b (K0 (Iota axis)) (D.Any D.Int64) (D.Any D.Int64)
    (Iarray.of_array shape) ~accepted:true
    ~want:(fun () ->
      let i = List.map (fun idx -> Int64.of_int idx.(axis)) (indices shape) in
      bits_of (A.of_array D.Int64 (Iarray.of_array shape) (Array.of_list i)))
    (fun dst -> Support.map0 (module K) (Iota axis) ~dst)

(* A Const of a drawn dtype, or a Coord along a drawn axis into an int64, into a
   C-contiguous destination of shape [shape] at an offset of [lead] elements
   into a buffer of [trail] more past it. *)
type inside = {
  op : Support.op0;
  dt : D.any;
  shape : int array;
  lead : int;
  trail : int;
}

let pp_inside ppf c =
  Format.fprintf ppf "%s %a %a after %d, before %d"
    (match c.op with
    | Fill _ -> "Const"
    | Iota a -> Printf.sprintf "Coord %d" a)
    pp_dtype c.dt pp_ints c.shape c.lead c.trail

(* Const, and Coord along an axis, into a C-contiguous destination at an offset
   into a larger buffer, mid-byte for a sub-byte dtype, store what they store
   into a fresh array and leave the elements around it; a library that declines
   one there declines it inside too, writing nothing. *)
let law_inside (b : Support.backend) c =
  let module K = (val b.kernels) in
  let (D.Any dt) = c.dt in
  let n = Array.fold_left ( * ) 1 c.shape in
  let s = Iarray.of_array c.shape in
  let strides = L.strides (L.contiguous s) in
  let lead = c.lead in
  cover "a Const" (match c.op with Fill _ -> true | Iota _ -> false);
  cover "a Coord" (match c.op with Iota _ -> true | Fill _ -> false);
  cover "sub-byte, mid-byte" (D.bits dt < 8 && lead mod (8 / D.bits dt) <> 0);
  if Rig.equal b.device Rig.host then
    cover "on the job's threads" (n >= 1_000_000);
  let w = max 1 (D.bits dt / 8) in
  let span = lead + n + c.trail in
  let whole = Support.on b (seeded dt [| span |] span) in
  let dst = A.v dt (L.v ~offset:lead ~strides s) (A.buffer whole) in
  let fresh = A.create b.device dt s in
  let before = bytes_of whole in
  match Support.map0 (module K) c.op ~dst:fresh with
  | A.Declined ->
      check_declined b (K0 c.op) c.dt;
      equal ~msg:"inside" answer A.Declined (Support.map0 (module K) c.op ~dst);
      equal ~msg:"declined writes nothing" (array int) before (bytes_of whole)
  | r ->
      equal ~msg:"fresh" answer A.Done r;
      let e = bytes_of fresh in
      equal ~msg:"inside" answer A.Done (Support.map0 (module K) c.op ~dst);
      let want =
        Array.mapi
          (fun i x ->
            if i / w >= lead && i / w < lead + n then e.(i - (lead * w)) else x)
          before
      in
      equal (array int) want (bytes_of whole)

(* A Const's bits: an element of [dt] from [seed]. *)
let const_of (D.Any dt) seed =
  let e = bits_of (seeded dt [| 1 |] seed) in
  Support.Fill (String.init (Array.length e) (fun i -> Char.chr e.(i)))

let insides =
  Gen.with_pp pp_inside
    (let open Gen in
     let* around = pair (int_range 0 9) (int_range 0 9) in
     let lead, trail = around in
     let const =
       let* dt = dtypes in
       let* shape = shape in
       let+ seed = nat in
       { op = const_of dt seed; dt; shape; lead; trail }
     in
     let coord =
       let* shape = array ~size:(int_range 1 4) extent in
       let+ axis = nat in
       {
         op = Iota (axis mod Array.length shape);
         dt = D.Any D.Int64;
         shape;
         lead;
         trail;
       }
     in
     frequency [ (1, const); (1, coord) ])

(* Sub-byte destinations of a million elements, starting and ending mid-byte,
   and a Coord of as many, on nx.cpu's job threads: its tables alone run
   them. *)
let large_insides =
  List.map
    (fun dt ->
      { op = const_of dt 7; dt; shape = [| 1_000_001 |]; lead = 3; trail = 5 })
    D.[ Any Int4; Any Bit; Any Float4_e2m1fn ]
  @ [
      {
        op = Iota 0;
        dt = D.Any D.Int64;
        shape = [| 1_000_001 |];
        lead = 3;
        trail = 5;
      };
    ]

(* Sub-byte destinations starting mid-byte, which the draws reach one case in
   18. *)
let mid_insides =
  List.map
    (fun (dt, (lead, trail)) ->
      { op = const_of dt 3; dt; shape = [| 2; 5 |]; lead; trail })
    D.[ (Any Int4, (3, 2)); (Any Bit, (5, 1)) ]

(* Calls that do not fit and refusals *)

(* A map that does not fit: by a destination's or a load's dtype, a coordinate
   past the loop's rank, an empty loop's included, a count of destinations or
   loads, or their shapes. [refusal] is the refusal map_refusal names it by,
   where it reads the misfit: it reads no shape but the rank. *)
type misfit = {
  name : string;
  what : string;
  refusal : A.answer option;
  program : P.t;
  dsts : (D.any * int array) array;
  loads : (D.any * int array) array;
}

let map_misfits =
  let f32 = D.Any D.Float32 and i64 = D.Any D.Int64 in
  let double =
    P.v ~ins:[| f32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 1 |]
  in
  let two =
    P.v ~ins:[| f32 |] [| P.In 0; Op2 (Binary Add, 0, 0) |] ~outs:[| 0; 1 |]
  in
  let coord axis = P.v ~ins:[||] [| P.Coord axis |] ~outs:[| 0 |] in
  let cast = P.v ~ins:[||] [| P.Coord 1; Op1 (Cast, f32, 0) |] ~outs:[| 1 |] in
  let row name what refusal program dsts loads =
    { name; what; refusal; program; dsts; loads }
  in
  let wrong = Some A.Wrong_dtype and shape = Some A.Shape_mismatch in
  let v3 = [| (f32, [| 3 |]) |] in
  [
    row "a destination of another dtype" "a dtype" wrong double
      [| (D.Any D.Float64, [| 3 |]) |]
      v3;
    row "a load of another dtype" "a dtype" wrong double v3
      [| (D.Any D.Float64, [| 3 |]) |];
    row "a coordinate into a destination of another dtype" "a dtype" wrong
      (coord 0) v3 [||];
    row "a coordinate past the loop's rank" "an attribute" shape (coord 1)
      [| (i64, [| 3 |]) |]
      [||];
    row "a coordinate past the rank of a loop of no element" "an attribute"
      shape (coord 1)
      [| (i64, [| 0 |]) |]
      [||];
    row "a cast coordinate past the rank of a loop of no element" "an attribute"
      shape cast
      [| (f32, [| 0 |]) |]
      [||];
    row "no destination" "a count" shape double [||] v3;
    row "two destinations" "a count" shape double (Array.append v3 v3) v3;
    row "no load" "a count" shape double v3 [||];
    row "two loads" "a count" shape double v3 (Array.append v3 v3);
    row "a load of another shape" "a shape" None double v3 [| (f32, [| 4 |]) |];
    row "a load of another rank" "a shape" None double v3
      [| (f32, [| 3; 1 |]) |];
    row "destinations of two shapes" "a shape" None two
      [| (f32, [| 3 |]); (f32, [| 4 |]) |]
      v3;
  ]

(* The dtypes of destinations and loads as their dtypes and shapes give them, as
   Nx_kernel's rules read them. *)
let specs = { Nx_kernel.arrays = Array.length; dtype = (fun a i -> fst a.(i)) }

let check_map_misfit (b : Support.backend) r =
  let module K = (val b.kernels) in
  let loads =
    Array.map
      (fun (D.Any d, s) ->
        Nx_kernel.Plain (A.create b.device d (Iarray.of_array s)))
      r.loads
  in
  Misfit.check ~msg:r.name ~what:r.what b r.dsts (fun g ds ->
      K.map r.program ~dsts:(Iarray.of_array ds)
        (Iarray.map (Misfit.load g) (Iarray.of_array loads)))

let check_misfit_named r =
  let rank = if r.dsts = [||] then 0 else Array.length (snd r.dsts.(0)) in
  match
    Nx_kernel.map_refusal r.program ~rank ~dsts:specs r.dsts ~loads:specs
      r.loads
  with
  | Some (got, _) -> equal (option answer) r.refusal (Some got)
  | None -> failf "the rule finds it fits"

(* A program the draws make, of one or two loads of [dt] on a loop of rank
   [rank], from [seed], up to 400 nodes past its loads where [long]. *)
let law_map_fits (D.Any dt, rank, seed, long) =
  let st = Random.State.make [| seed |] in
  let ins =
    if Random.State.bool st then [| D.Any dt; D.Any dt |] else [| D.Any dt |]
  in
  let length = if long then Some 400 else None in
  let p, types = program ?length st ins ~room:16 ~rank in
  let dsts = Array.map (fun o -> (types.(o), [||])) (Iarray.to_array p.outs) in
  let loads = Array.map (fun d -> (d, [||])) ins in
  equal ~msg:(describe p)
    (option (pair answer string))
    None
    (Nx_kernel.map_refusal p ~rank ~dsts:specs dsts ~loads:specs loads)

let fitting_programs =
  Gen.quad dtypes (Gen.int_range 0 4) Gen.int
    (Gen.frequency [ (4, Gen.constant false); (1, Gen.constant true) ])

(* A float32 map of a copy and a negation of one load, into two destinations. *)
let copy_and_neg =
  let f32 = D.Any D.Float32 in
  P.v ~ins:[| f32 |] [| P.In 0; Op1 (Unary Neg, f32, 0) |] ~outs:[| 0; 1 |]

(* Refusals of {!copy_and_neg} over float32 arrays of 5 elements: for its second
   destination's memory admitting only Read, for a dead load, and for its second
   destination reaching one element five times. *)
let refusals =
  [
    ("a destination admitting only Read", A.Read_only);
    ("a dead load", A.Dead_buffer);
    ("a destination reaching an element twice", A.Repeated_elements);
  ]

(* [x] on [b]'s device over memory that admits only [Read]: a borrow of host
   pages holding a copy of its bytes. *)
let read_only (b : Support.backend) x =
  let y = Nx_array_support.mapped_read_only (Support.host x) in
  if Rig.equal b.device Rig.host then y
  else
    match B.borrow b.device (A.buffer y) with
    | Some buf -> A.v (A.dtype y) (A.layout y) buf
    | None -> skip ~reason:"the device maps no host pages" ()

let check_refusal (b : Support.backend) (name, want) =
  let module K = (val b.kernels) in
  let n = 5 in
  let x = Support.on b (seeded D.Float32 [| n |] 1) in
  let first = Support.on b (seeded D.Float32 [| n |] 2) in
  let before = bytes_of first in
  let second = Support.on b (seeded D.Float32 [| n |] 3) in
  let x, second =
    match want with
    | A.Read_only -> (x, read_only b second)
    | A.Dead_buffer ->
        let dead = Support.on b (A.copy x) in
        ignore (B.free (A.buffer dead));
        (dead, second)
    | _ ->
        ( x,
          Option.get
            (A.move (M.Broadcast [| n |]) (A.create b.device D.Float32 [| 1 |]))
        )
  in
  equal ~msg:name answer want
    (K.map copy_and_neg ~dsts:[| A.Any first; A.Any second |] [| Plain x |]);
  equal ~msg:"the first destination" (array int) before (bytes_of first)

(* While another domain collects and compacts, copies of arrays no other value
   holds run with the runtime released: their buffers stay alive until the call
   ends. *)
let test_collect_during_call b () =
  let stop = Atomic.make false in
  let gc =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          Gc.compact ()
        done)
  in
  let n = 1 lsl 18 in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set stop true;
      Domain.join gc)
    (fun () ->
      for round = 1 to 4 do
        let src = A.of_array D.Float32 [| n |] (Array.make n 1.5) in
        Option.iter
          (fun dst ->
            equal
              ~msg:(Printf.sprintf "round %d" round)
              float_exact 1.5
              (A.get dst [| n - 1 |]))
          (run_op b Cast D.Float64 src)
      done)

(* Destinations *)

(* Destinations of [n] elements that are not C-contiguous, in the middle of a
   buffer of [4 n] elements: every other element from [n], and elements [n] to
   [2 n - 1] reversed. *)
let scattered n =
  [
    ("stepped", L.v ~offset:n ~strides:[| 2 |] [| n |]);
    ("reversed", L.v ~offset:((2 * n) - 1) ~strides:[| -1 |] [| n |]);
  ]

(* The loops nx_kernel.mli lets write a strided destination. *)
type loop = Copy_loop | Neg_loop | Sum_loop | Scan_loop

let loop_name = function
  | Copy_loop -> "a copy"
  | Neg_loop -> "a negation"
  | Sum_loop -> "a sum"
  | Scan_loop -> "a scan"

let scattered_rows =
  List.concat_map
    (fun call -> List.map (fun (what, _) -> (call, what)) (scattered 7))
    [ Copy_loop; Neg_loop; Sum_loop; Scan_loop ]

(* [call] into a stepped or a reversed destination gives the bits it gives into
   a C-contiguous one, and changes no other element of the destination's
   buffer. *)
let test_scattered_loop (b : Support.backend) (call, what) =
  let module K = (val b.kernels) in
  let n = 7 and f32 = D.Any D.Float32 in
  let xs = Array.init (2 * n) (fun i -> Float.of_int (i + 1)) in
  let x = Support.on b (A.of_array D.Float32 [| n |] (Array.sub xs 0 n)) in
  let rows = Support.on b (A.of_array D.Float32 [| n; 2 |] xs) in
  let copy = P.v ~ins:[| f32 |] [| P.In 0 |] ~outs:[| 0 |] in
  let neg =
    P.v ~ins:[| f32 |] [| P.In 0; Op1 (Unary Neg, f32, 0) |] ~outs:[| 1 |]
  in
  let f y =
    match call with
    | Copy_loop -> K.map copy ~dsts:[| A.Any y |] [| Plain x |]
    | Neg_loop -> K.map neg ~dsts:[| A.Any y |] [| Plain x |]
    | Sum_loop ->
        K.reduce copy ~shape:[| n; 2 |] ~axes:[| 1 |]
          [| (Monoid Sum, 0) |]
          ~dsts:[| A.Any y |] [| Plain rows |]
    | Scan_loop ->
        K.scan copy ~axis:0 (Monoid Sum, 0) ~dsts:[| A.Any y |] [| Plain x |]
  in
  let l = List.assoc what (scattered n) in
  let fresh = A.create b.device D.Float32 [| n |] in
  equal ~msg:"C-contiguous" answer A.Done (f fresh);
  let want = A.to_array (Support.host fresh) in
  let guard = Array.init (4 * n) (fun i -> Float.of_int (-i - 1)) in
  let base = Support.on b (A.of_array D.Float32 [| 4 * n |] guard) in
  let y = A.v D.Float32 l (A.buffer base) in
  equal answer A.Done (f y);
  equal (array float_exact) want (A.to_array (Support.host y));
  let reached = Array.make (4 * n) false in
  for i = 0 to n - 1 do
    reached.(L.offset l + (i * L.stride l 0)) <- true
  done;
  let kept =
    Array.mapi
      (fun i v -> if reached.(i) then guard.(i) else v)
      (A.to_array (Support.host base))
  in
  equal ~msg:"the other elements of its buffer" (array float_exact) guard kept

(* A polymorphic maker of destinations. *)
type maker = { mk : 'v 's. ('v, 's) D.t -> ('v, 's) A.t }

(* Each kernel whose destinations must be C-contiguous, the destination [what],
   and its call with [what] by [s] and its operands and every other destination
   by [fresh]. A contraction's, which nx.cpu's, nx.cuda's and nx.amd's docs let
   be strided, is not among them. *)
let contiguous_calls :
    (string
    * string
    * ((module Nx_kernel.S) -> fresh:maker -> maker -> A.answer))
    list =
  let pick what name s fresh = if what = name then s else fresh in
  let ops fresh =
    (fresh.mk D.Int64, fresh.mk D.Float32, fresh.mk D.Complex64)
  in
  let pad =
    { Nx_kernel.lo = [| 0 |]; hi = [| 0 |]; interior = [| 0 |]; windows = [||] }
  in
  let one kernel f = [ (kernel, "dst", f) ] in
  let each kernel whats f =
    List.map (fun what -> (kernel, what, f what)) whats
  in
  List.concat
    [
      one "gather" (fun (module K : Nx_kernel.S) ~fresh s ->
          let i64, f32, _ = ops fresh in
          K.gather ~axis:0 ~dst:(s.mk D.Float32) i64 f32);
      one "gather_ranges" (fun (module K : Nx_kernel.S) ~fresh s ->
          let i64, f32, _ = ops fresh in
          K.gather_ranges ~dst:(s.mk D.Float32) ~starts:i64 ~ends:i64 f32);
      one "scatter" (fun (module K : Nx_kernel.S) ~fresh s ->
          let i64, f32, _ = ops fresh in
          K.scatter Set ~unique:false ~axis:0 ~dst:(s.mk D.Float32) ~into:f32
            i64 f32);
      each "sort" [ "values"; "positions" ]
        (fun what (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          let a name = pick what name s fresh in
          K.sort ~axis:0 ~descending:false
            ~values:((a "values").mk D.Float32)
            ~positions:((a "positions").mk D.Int64)
            f32);
      each "unique" [ "ids"; "first"; "counts"; "count" ]
        (fun what (module K : Nx_kernel.S) ~fresh s ->
          let i64, _, _ = ops fresh in
          let a name = (pick what name s fresh).mk D.Int64 in
          K.unique ~ids:(a "ids") ~first:(a "first") ~counts:(a "counts")
            ~count:(a "count") i64);
      one "assemble" (fun (module K : Nx_kernel.S) ~fresh:_ s ->
          K.assemble ~fill:0. ~dst:(s.mk D.Float32) [||]);
      one "fold" (fun (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          K.fold pad ~dst:(s.mk D.Float32) f32);
      one "c2c" (fun (module K : Nx_kernel.S) ~fresh s ->
          let _, _, c64 = ops fresh in
          K.c2c Forward ~axes:[| 0 |] ~dst:(s.mk D.Complex64) c64);
      one "r2c" (fun (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          K.r2c ~axes:[| 0 |] ~dst:(s.mk D.Complex64) f32);
      one "c2r" (fun (module K : Nx_kernel.S) ~fresh s ->
          let _, _, c64 = ops fresh in
          K.c2r ~axes:[| 0 |] ~dst:(s.mk D.Float32) c64);
      one "cholesky" (fun (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          K.cholesky Lower ~dst:(s.mk D.Float32) f32);
      each "lu" [ "lu"; "pivots"; "perm" ]
        (fun what (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          let a name = pick what name s fresh in
          K.lu ~lu:((a "lu").mk D.Float32) ~pivots:((a "pivots").mk D.Int64)
            ~perm:((a "perm").mk D.Int64) f32);
      each "qr" [ "q"; "r" ] (fun what (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          let a name = (pick what name s fresh).mk D.Float32 in
          K.qr ~q:(a "q") ~r:(a "r") f32);
      each "svd" [ "s"; "u"; "vh" ]
        (fun what (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          let a name = (pick what name s fresh).mk D.Float32 in
          K.svd ~s:(a "s") ~vectors:(~u:(a "u"), ~vh:(a "vh")) f32);
      each "eigh" [ "w"; "v" ] (fun what (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          let a name = (pick what name s fresh).mk D.Float32 in
          K.eigh ~w:(a "w") ~v:(a "v") f32);
      each "eig" [ "w"; "v" ] (fun what (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          let a name = (pick what name s fresh).mk D.Complex64 in
          K.eig ~w:(a "w") ~v:(a "v") f32);
      one "solve_triangular" (fun (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          K.solve_triangular Lower ~adjoint:false ~unit_diagonal:false
            ~dst:(s.mk D.Float32) f32 f32);
      each "attention" [ "o"; "lse" ]
        (fun what (module K : Nx_kernel.S) ~fresh s ->
          let _, f32, _ = ops fresh in
          let a name = (pick what name s fresh).mk D.Float32 in
          K.attention
            (P.v ~ins:[| D.Any D.Float32 |] [| P.In 0 |] ~outs:[| 0; 0 |])
            ~scale:1. ~o:(a "o") ~lse:(a "lse") f32
            (Nx_kernel.Dense { k = f32; v = f32 })
            [||]);
    ]

(* nx_kernel.mli: every kernel's destination but a loop's must be C-contiguous,
   and a library raises for one that is not, before any write, as
   Nx_kernel.check_dst does. The destination [what], stepped or reversed, the
   others fresh, raises Invalid_argument naming the kernel, the argument and its
   layout, and its buffer keeps its bytes. *)
let test_contiguous_dst (b : Support.backend) (kernel, what, call) =
  let n = 4 in
  let fresh = { mk = (fun dt -> A.create b.device dt [| n |]) } in
  List.iter
    (fun (layout, l) ->
      let msg = layout in
      let bytes = ref [] in
      let s =
        {
          mk =
            (fun dt ->
              let size = 4 * n * D.bits dt / 8 in
              let codes = Array.init size (fun i -> i land 0xff) in
              let base = Support.on b (A.of_array D.Uint8 [| size |] codes) in
              bytes := (base, codes) :: !bytes;
              A.v dt l (A.buffer base));
        }
      in
      let module K = (val b.kernels) in
      let want =
        Format.asprintf "%s %s: %s's layout %a is not C-contiguous" K.name
          kernel what L.pp l
      in
      raises ~msg (Invalid_argument want) (fun () -> call b.kernels ~fresh s);
      List.iter
        (fun (base, codes) ->
          equal ~msg:(msg ^ ": its buffer") (array int) codes
            (A.to_array (Support.host base)))
        !bytes)
    (scattered n)

(* Offsets *)

(* The alignment Nx_array promises an array's first element: its dtype's width
   for dtypes of a byte or more, one byte for narrower ones, one component's for
   complex ones. *)
let alignment (type v s) (dt : (v, s) D.t) =
  let bits = D.bits dt in
  if bits < 8 then 1 else if D.is D.Complex dt then bits / 16 else bits / 8

type offsets = {
  dt : D.any;
  shape : int array;
  pad : int;
  rev : bool;
  src : int;
  dst : int;
  seed : int;
}

let pp_offsets ppf c =
  let (D.Any dt) = c.dt in
  Format.fprintf ppf "%a %a%s from byte %d into byte %d%s, seed %d" D.pp dt
    pp_ints c.shape
    (if c.pad = 0 then "" else Printf.sprintf ", rows %d apart" c.pad)
    c.src c.dst
    (if c.rev then ", reversed" else "")
    c.seed

(* A dtype, complex half the time; a shape of a few axes, or of no element or of
   one, or rows whose extent is a multiple of 4 that lie [pad] elements apart
   past their last: the rows of 8 of rows of 9, or rows 1 to 4 apart, a stride a
   multiple of 4 or not; and the bytes a source and a destination start at past
   a buffer's start: each a multiple of the dtype's alignment below 16, the
   widest access a kernel makes, 2, 4, 8 or 12 most often; and destinations
   whose rows run reversed, half the time. *)
let offset_cases =
  let open Gen in
  let complexes = of_list ~pp:pp_dtype D.[ Any Complex64; Any Complex128 ] in
  let* (D.Any d as dt) = frequency [ (1, dtypes); (1, complexes) ] in
  let a = alignment d in
  let ints = of_list ~pp:Format.pp_print_int in
  let at =
    frequency
      [
        (1, ints (List.init (16 / a) (( * ) a)));
        (2, ints (List.filter (fun o -> o mod a = 0) [ 2; 4; 8; 12 ]));
      ]
  in
  let rows =
    let* r = int_range 2 3 in
    let+ outer = array ~size:(constant (r - 1)) extent
    and+ last = int_range 1 4 in
    Array.append outer [| 4 * last |]
  in
  let plain =
    let+ s = array ~size:(int_range 1 3) extent in
    (0, s)
  in
  let eight_of_nine =
    let+ r = extent in
    (1, [| r; 8 |])
  in
  let apart =
    let* pad = int_range 1 4 in
    let+ s = rows in
    (pad, s)
  in
  let edge =
    let+ s =
      of_list ~pp:pp_ints
        [ [| 0 |]; [| 1 |]; [| 1; 1 |]; [| 1; 4 |]; [| 0; 4 |]; [| 4; 0 |] ]
    in
    (0, s)
  in
  let* pad, shape =
    frequency [ (2, plain); (1, edge); (1, eight_of_nine); (1, apart) ]
  in
  let* src = at in
  let* dst = at in
  let* rev = bool in
  let+ seed = int in
  { dt; shape; pad; rev; src; dst; seed }

(* Rows 4 elements apart past their last, which the draws reach one case in 20;
   a float16 array from byte 2, which they reach one case in 12; and complex64
   arrays from 4 bytes past 8 whose copies of a transpose move blocks and zip
   pairs, and whose scatter sorts its updates. *)
let offset_examples =
  {
    dt = D.Any D.Float32;
    shape = [| 3; 8 |];
    pad = 4;
    rev = true;
    src = 4;
    dst = 8;
    seed = 1;
  }
  :: {
       dt = D.Any D.Float16;
       shape = [| 3; 5 |];
       pad = 0;
       rev = false;
       src = 2;
       dst = 2;
       seed = 3;
     }
  :: List.map
       (fun shape ->
         {
           dt = D.Any D.Complex64;
           shape;
           pad = 0;
           rev = false;
           src = 4;
           dst = 4;
           seed = 7;
         })
       [ [| 8; 8 |]; [| 2; 9 |]; [| 1 lsl 16 |] ]

(* The first [s.(i)] elements of each axis [i] of [x]. *)
let rows_of x s =
  let all e = { Nx_array.Move.start = 0; count = e; step = 1 } in
  Option.get
    (A.move (Nx_array.Move.Slice (Iarray.of_array (Array.map all s))) x)

(* [x] with its last axis reversed, where it has elements. *)
let reversed x =
  let s = Iarray.to_array (L.shape (A.layout x)) in
  let r = Array.length s in
  if r = 0 || Array.exists (( = ) 0) s then x
  else
    let range i e =
      if i < r - 1 then { Nx_array.Move.start = 0; count = e; step = 1 }
      else { Nx_array.Move.start = e - 1; count = e; step = -1 }
    in
    Option.get
      (A.move (Nx_array.Move.Slice (Iarray.of_array (Array.mapi range s))) x)

(* [x] over shape [s] from its first row, broadcast along every other axis, or
   where [last], from its first column, broadcast along the last axis: a stride
   of 0 before the last, or as the last. *)
let spread (type v s) ~last (x : (v, s) A.t) s : (v, s) A.t =
  let r = Array.length s in
  let one i e = if i = r - 1 = last then Int.min 1 e else e in
  let first = rows_of x (Array.mapi one s) in
  Option.get (A.move (Nx_array.Move.Broadcast (Iarray.of_array s)) first)

(* The maximum of four loads of [dt] as a tape of three maximums, which no
   kernel library runs as one kind. *)
let maximums dt =
  P.v ~ins:[| dt; dt; dt; dt |]
    [|
      P.In 0;
      In 1;
      In 2;
      In 3;
      Op2 (Binary Maximum, 0, 1);
      Op2 (Binary Maximum, 2, 3);
      Op2 (Binary Maximum, 4, 5);
    |]
    ~outs:[| 6 |]

(* [x]'s bytes on [b]'s device in a fresh buffer from its byte [at]. *)
let placed (type v s) (b : Support.backend) at (x : (v, s) A.t) : (v, s) A.t =
  let src = A.buffer x in
  let n = B.length src in
  let dst = B.view (B.create b.device (at + n)) ~first:at ~length:n in
  B.copy ~src ~dst;
  A.v (A.dtype x) (A.layout x) dst

(* An array of [dt] and shape [s] on [b]'s device in a fresh buffer from its
   byte [at], C-contiguous, or the rows of one whose last extent is [pad]
   more. *)
let blank (type v s) ?(pad = 0) ?(rev = false) (b : Support.backend) at
    (dt : (v, s) D.t) s : (v, s) A.t =
  let r = Array.length s in
  let whole = Array.mapi (fun i e -> if i = r - 1 then e + pad else e) s in
  let n = max 1 (D.bytes dt (Array.fold_left ( * ) 1 whole)) in
  let buffer = B.view (B.create b.device (at + n)) ~first:at ~length:n in
  let x = A.v dt (L.contiguous (Iarray.of_array whole)) buffer in
  let x = if pad = 0 then x else rows_of x s in
  if rev then reversed x else x

(* [c]'s source: its elements from [c]'s seed, C-contiguous or rows. *)
let offset_source (type v s) c (dt : (v, s) D.t) : (v, s) A.t =
  let r = Array.length c.shape in
  let whole =
    Array.mapi (fun i e -> if i = r - 1 then e + c.pad else e) c.shape
  in
  let x = seeded dt (Iarray.of_array whole) c.seed in
  if c.pad = 0 then x else rows_of x c.shape

(* A copy, of [x] and of its last two axes swapped, a padded load's map, a map
   of the source and itself at the destination's byte, the same with the
   source's broadcast rows and columns as a tape of maximums, a where, a gather
   and a scatter by Set along the first axis, a sum along the last axis and a
   sum's scan along the first from a carry, each with its sources at byte [src]
   and its destination, scattered-into array and carry at byte [dst], or the
   multiple of the carrier's alignment below it, the two maximums, the sum and
   the scan into rows [pad] apart, reversed where [rev]: each answer, whether it
   moves bits or computes them, and each result where the host reads it. The
   positions are drawn from one before the first axis to one past it, so a
   gather reads zeros and a scatter drops updates, and several updates share a
   target. *)
let offset_runs (type v s) (b : Support.backend) c (x : (v, s) A.t) ~src ~dst
    ~pad ~rev =
  let module K = (val b.kernels) in
  let dt = A.dtype x and s = c.shape in
  let x' = placed b dst x in
  let x = placed b src x in
  let into s = blank b dst dt s in
  let view s = blank ~pad ~rev b dst dt s in
  let ran name f y = (name, `Moves, f y, y) in
  let computed name f y = (name, `Computes, f y, y) in
  let p = Reduce_cases.identity c.dt in
  let r = Array.length s - 1 in
  let swapped =
    if r < 1 then []
    else
      let perm = Array.init (r + 1) Fun.id in
      perm.(r - 1) <- r;
      perm.(r) <- r - 1;
      let t = Option.get (A.move (M.Permute (Iarray.of_array perm)) x) in
      let ts = Array.map (fun a -> s.(a)) perm in
      [
        ran "copy of the last axes swapped"
          (fun y -> Support.map1 (module K) Copy ~dst:y t)
          (into ts);
      ]
  in
  let n = Array.fold_left ( * ) 1 s in
  let st = Random.State.make [| c.seed + 2 |] in
  let at () = Int64.of_int (Random.State.int st (s.(0) + 2) - 1) in
  let positions =
    Support.on b
      (A.of_array D.Int64 (Iarray.of_array s) (Array.init n (fun _ -> at ())))
  in
  let cond = placed b src (seeded D.Bool (Iarray.of_array s) (c.seed + 3)) in
  let other = placed b src (seeded dt (Iarray.of_array s) (c.seed + 4)) in
  let target = placed b dst (seeded dt (Iarray.of_array s) (c.seed + 5)) in
  let maps =
    [ ran "copy" (fun y -> Support.map1 (module K) Copy ~dst:y x) (into s) ]
    @ swapped
    @ [
        ran "padded load"
          (fun y -> K.map p ~dsts:[| A.Any y |] [| unpadded x |])
          (into s);
        computed "maximum of two loads"
          (fun y -> Support.map2 (module K) (Binary Maximum) ~dst:y x x')
          (view s);
        computed "a tape of maximums over broadcast loads"
          (fun y ->
            K.map (maximums c.dt) ~dsts:[| A.Any y |]
              [|
                Plain x;
                Plain x';
                Plain (spread ~last:false x s);
                Plain (spread ~last:true x s);
              |])
          (view s);
        ran "where"
          (fun y -> Support.map3 (module K) Where ~dst:y cond x other)
          (into s);
        ran "gather" (fun y -> K.gather ~axis:0 ~dst:y positions x) (into s);
        ran "scatter by Set"
          (fun y ->
            K.scatter Set ~unique:false ~axis:0 ~dst:y ~into:target positions x)
          (into s);
      ]
  in
  if not (Reduce_cases.accepts Sum c.dt) then maps
  else
    let (D.Any ct) = Nx_kernel.carrier c.dt in
    let carry_shape = Array.mapi (fun i e -> if i = 0 then 1 else e) s in
    let carry =
      placed b
        (dst / alignment ct * alignment ct)
        (seeded ct (Iarray.of_array carry_shape) (c.seed + 1))
    in
    maps
    @ [
        computed "sum"
          (fun y ->
            K.reduce p ~shape:(Iarray.of_array s) ~axes:(Iarray.of_list [ r ])
              [| (Monoid Sum, 0) |]
              ~dsts:[| A.Any y |] [| Plain x |])
          (view (Array.sub s 0 r));
        computed "scan from a carry"
          (fun y ->
            K.scan ~carry:(A.Any carry) p ~axis:0 (Monoid Sum, 0)
              ~dsts:[| A.Any y |] [| Plain x |])
          (view s);
      ]

(* A map, a reduction, a scan, a gather and a scatter over arrays at any offset
   their dtype's alignment admits give the answers and the bits they give over
   arrays at a buffer's start, through views whose rows lie apart by a multiple
   of 4 elements or not, and broadcast ones; and into destinations whose rows
   lie apart or run reversed (nx_kernel.mli: a loop's destination may be
   strided), the bits they give into C-contiguous ones. A NaN that a kind
   computes is compared as a class: its bits may change with the layouts and the
   addresses (nx_cpu.mli). *)
let law_offsets (b : Support.backend) c =
  let (D.Any dt) = c.dt in
  let complex bytes = D.is D.Complex dt && 8 * bytes = D.bits dt in
  let off o = o mod (D.bits dt / 8) <> 0 in
  let r = Array.length c.shape - 1 in
  let last k = if r < 1 then 0 else c.shape.(r - k) in
  cover "complex64 from 4 bytes past 8" (complex 8 && off c.src);
  cover "complex64 into 4 bytes past 8" (complex 8 && off c.dst);
  cover "complex128 from 8 bytes past 16" (complex 16 && off c.src);
  cover "complex128 into 8 bytes past 16" (complex 16 && off c.dst);
  List.iter
    (fun at ->
      cover
        (Printf.sprintf "a base %d bytes past 16" at)
        (c.src = at || c.dst = at))
    [ 2; 4; 8; 12 ];
  cover "rows apart by no multiple of 4" (c.pad mod 4 <> 0);
  cover "rows apart by a multiple of 4" (c.pad = 4);
  cover "no element" (Array.exists (( = ) 0) c.shape);
  cover "one element" (Array.for_all (( = ) 1) c.shape);
  cover "rows of 8 of rows of 9"
    (c.pad = 1 && Array.length c.shape = 2 && c.shape.(1) = 8);
  cover "a reversed destination" (c.rev && Array.exists (( < ) 1) c.shape);
  (* nx.cpu's copies of a transpose: square blocks of 4 x 4 8-byte elements, and
     two columns zipped. *)
  cover "complex64 swapped in blocks from 4 bytes past 8"
    (complex 8 && off c.src && last 1 >= 4 && last 0 >= 4);
  cover "complex64 pairs swapped from 4 bytes past 8"
    (complex 8 && off c.src && last 1 = 2 && last 0 >= 2);
  (* nx.cpu sorts the updates of a scatter of 64Ki or more by target on a host
     of more than one thread. *)
  cover "a scatter of 64Ki complex64 updates from 4 bytes past 8"
    (complex 8 && off c.src && c.shape.(0) >= 1 lsl 16);
  let x = offset_source c dt in
  let want = offset_runs b c x ~src:0 ~dst:0 ~pad:0 ~rev:false in
  let got = offset_runs b c x ~src:c.src ~dst:c.dst ~pad:c.pad ~rev:c.rev in
  List.iter2
    (fun (msg, how, a, w) (_, _, a', g) ->
      equal ~msg answer a a';
      if a = A.Done then
        let w = A.copy (Support.host w) and g = A.copy (Support.host g) in
        equal ~msg (list string) []
          (if how = `Computes then values_differ w g else bits_differ w g))
    want got

(* Sharing *)

(* [k dst a] into a fresh destination, then into [a] itself as a C-contiguous
   copy of [a0], [a]'s elements on the host, [a] being an operand [k] reads only
   at the result's own index; then the check that both give one answer, and on
   [Done] the same bits, any NaN for a NaN where [values]. A law makes all its
   calls before it checks any, so that the device runs them without a round trip
   between. *)
let in_place ?(values = false) ~msg (b : Support.backend) (a, a0) k =
  let fresh = A.create b.device (A.dtype a) (L.shape (A.layout a)) in
  let want = k fresh a in
  let alias = Support.on b (A.copy a0) in
  let got = k alias alias in
  fun () ->
    equal ~msg answer want got;
    cover "computed" (want = A.Done);
    if want = A.Done then
      let w = Support.host fresh and g = Support.host alias in
      equal ~msg (list string) []
        (if values then values_differ w g else bits_differ w g)

let binaries = List.filter (function P.Binary _ -> true | _ -> false) op2s

(* A [Plain] load of [map], of one node or more, may be the destination itself:
   the kernels read each index before they write it. *)
let law_in_place (b : Support.backend) (Pair (x0, y0), seed) =
  let module K = (val b.kernels) in
  let dt = A.dtype x0 and shape = L.shape (A.layout x0) in
  let x = Support.on b x0 and y = Support.on b y0 in
  let z0 = seeded dt shape (seed + 1) in
  let z = Support.on b z0 in
  let c = Support.on b (seeded D.Bool shape seed) in
  let map2 k ~dst x y = Support.map2 (module K) k ~dst x y in
  let ones =
    List.map
      (fun op ->
        in_place ~values:(any_nan (K1 op)) ~msg:"one operand into itself" b
          (x, x0) (fun dst x -> Support.map1 (module K) op ~dst x))
      (List.filter (fun op -> P.accepts1 op dt dt) op1s)
  in
  let twos =
    List.concat_map
      (fun k ->
        let msg into = Printf.sprintf "%s into %s" (name2 k) into in
        let values = any_nan (K2 k) in
        [
          in_place ~values ~msg:(msg "the first") b (x, x0) (fun dst x ->
              map2 k ~dst x y);
          in_place ~values ~msg:(msg "the second") b (y, y0) (fun dst y ->
              map2 k ~dst x y);
          in_place ~values ~msg:(msg "both") b (x, x0) (fun dst x ->
              map2 k ~dst x x);
        ])
      (List.filter (fun k -> P.accepts2 k dt) binaries)
  in
  let wheres =
    [
      in_place ~msg:"where into the second" b (x, x0) (fun dst x ->
          Support.map3 (module K) Where ~dst c x y);
      in_place ~msg:"where into the third" b (y, y0) (fun dst y ->
          Support.map3 (module K) Where ~dst c x y);
    ]
  in
  let fma =
    if not (P.accepts3 Fma dt dt) then []
    else
      [
        in_place ~values:true ~msg:"fma into the addend" b (z, z0) (fun dst z ->
            Support.map3 (module K) Fma ~dst x y z);
      ]
  in
  let load =
    if not (P.accepts2 (Binary Add) dt) then []
    else
      let p =
        P.v ~ins:[| D.Any dt; D.Any dt |]
          [| P.In 0; In 1; Op2 (Binary Add, 0, 1) |]
          ~outs:[| 2 |]
      in
      [
        in_place ~values:true ~msg:"map into a load" b (x, x0) (fun dst x ->
            K.map p ~dsts:[| A.Any dst |] Nx_kernel.[| Plain x; Plain y |]);
      ]
  in
  List.iter (fun check -> check ()) (ones @ twos @ wheres @ fma @ load)

(* A drawn program of several nodes whose first output, its last node cast to
   its loads' dtype, is written over one of its loads, as a donated operand's
   is: the bits it gives into fresh destinations, any NaN for a NaN. A second
   output may follow: a kind of that load, read after the first output is
   written, or the load itself as an output, as it is, as a Copy, a Cast into
   its own dtype or a Bitcast. *)
let law_map_over_load (b : Support.backend) (Pair (x, y), seed) =
  let module K = (val b.kernels) in
  let st = Random.State.make [| seed |] in
  let shape = L.shape (A.layout x) in
  let (D.Any d as dt) = D.Any (A.dtype x) in
  let ops =
    if Random.State.bool st then
      [| A.Any (Support.on b x); A.Any (Support.on b y) |]
    else [| A.Any (Support.on b x) |]
  in
  let ins = Array.map (fun _ -> dt) ops in
  let p, _ = program st ins ~room:16 ~rank:(Iarray.length shape) in
  let n = Iarray.length p.nodes in
  let k = Random.State.int st (Array.length ops) in
  let pick l = List.nth l (Random.State.int st (List.length l)) in
  let later =
    List.filter (fun u -> P.accepts1 (Unary u) d d) (List.map fst unaries)
  in
  let same = List.filter (fun (D.Any e) -> D.bits e = D.bits d) D.all in
  let second =
    match Random.State.int st 6 with
    | (0 | 1) when later <> [] -> `Reread [ P.Op1 (Unary (pick later), dt, k) ]
    | 0 | 1 | 2 -> `None
    | 3 -> `Output []
    | 4 ->
        let move = if Random.State.bool st then P.Copy else Cast in
        `Output [ P.Op1 (move, dt, k) ]
    | _ -> `Output [ P.Op1 (Bitcast, pick same, k) ]
  in
  let extra, outs =
    match second with
    | `None -> ([], [| n |])
    | `Reread e -> (e, [| n; n + 1 |])
    | `Output [] -> ([], [| n; k |])
    | `Output e -> (e, [| n; n + 1 |])
  in
  let q =
    P.v ~ins:(Iarray.of_array ins)
      (Iarray.of_list
         (Iarray.to_list p.nodes @ (P.Op1 (Cast, dt, n - 1) :: extra)))
      ~outs:(Iarray.of_array outs)
  in
  let fresh () =
    Array.map
      (fun o ->
        let (D.Any e) = P.dtype q o in
        A.Any (A.create b.device e shape))
      outs
  in
  let loads ops =
    Iarray.map (fun (A.Any x) -> Nx_kernel.Plain x) (Iarray.of_array ops)
  in
  let want = fresh () in
  let answered = K.map q ~dsts:(Iarray.of_array want) (loads ops) in
  let over =
    A.Any (Support.on b (A.copy (Support.host (A.expect d ops.(k)))))
  in
  let got = fresh () in
  got.(0) <- over;
  let ops' = Array.copy ops in
  ops'.(k) <- over;
  let msg = describe q in
  equal ~msg answer answered (K.map q ~dsts:(Iarray.of_array got) (loads ops'));
  let computed = answered = A.Done in
  cover "computed" computed;
  cover "the load read after the output is written"
    (computed && match second with `Reread _ -> true | _ -> false);
  cover "the load an output"
    (computed && match second with `Output _ -> true | _ -> false);
  cover "several nodes" (computed && n > Array.length ops + 1);
  if computed then
    Array.iteri
      (fun i (A.Any w) ->
        let (A.Any g) = got.(i) in
        equal
          ~msg:(Printf.sprintf "output %d of %s" i msg)
          (same_values (P.dtype q outs.(i)))
          (bytes_of w) (bytes_of g))
      want

(* A one-element pair, which the draws reach one case in five. *)
let one_element_pair =
  Pair (seeded D.Float32 [| 1 |] 3, seeded D.Float32 [| 1 |] 4)

(* A load identical to a destination through another view of its buffer is read
   before the destination is written: float32 offset 1 over a buffer and offset
   0 over the view of the buffer from its fourth byte are one array. *)
let test_map_identical_view (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 5 in
  let whole =
    Support.on b (A.of_array D.Float32 [| n + 1 |] [| 0.; 1.; 2.; 3.; 4.; 5. |])
  in
  let buffer = A.buffer whole in
  let x = A.v D.Float32 (L.v ~offset:1 ~strides:[| 1 |] [| n |]) buffer in
  let neg =
    A.v D.Float32 (L.contiguous [| n |])
      (B.view buffer ~first:4 ~length:(4 * n))
  in
  let sum = A.create b.device D.Float32 [| n |] in
  let p =
    P.v ~ins:[| D.Any D.Float32 |]
      [| P.In 0; Op1 (Unary Neg, D.Any D.Float32, 0); Op2 (Binary Add, 0, 0) |]
      ~outs:[| 1; 2 |]
  in
  equal answer A.Done
    (K.map p ~dsts:[| A.Any neg; A.Any sum |] [| Nx_kernel.Plain x |]);
  equal ~msg:"neg" (array float_exact)
    [| -1.; -2.; -3.; -4.; -5. |]
    (A.to_array (Support.host neg));
  equal ~msg:"add" (array float_exact) [| 2.; 4.; 6.; 8.; 10. |]
    (A.to_array (Support.host sum))

(* A map of 100 outputs, each into the load it reads: too wide for one of
   nx.cuda's launches, as is the copy, into the destinations, of the temporaries
   that nx_cuda.mli's loop section says such outputs go through. Output [k] is
   load [k] plus [k]. *)
let test_wide_in_place (b : Support.backend) () =
  let module K = (val b.kernels) in
  let n = 100 and m = 5 and i64 = D.Any D.Int64 in
  let xs =
    Array.init n (fun k ->
        Support.on b
          (A.of_array D.Int64 [| m |]
             (Array.init m (fun i -> Int64.of_int (k * i)))))
  in
  let p =
    P.v
      ~ins:(Iarray.init n (fun _ -> i64))
      (Iarray.of_array
         (Array.concat
            [
              Array.init n (fun k -> P.In k);
              Array.init n (fun k ->
                  P.Const (i64, P.bits D.Int64 (Int64.of_int k)));
              Array.init n (fun k -> P.Op2 (Binary Add, k, n + k));
            ]))
      ~outs:(Iarray.init n (fun k -> (2 * n) + k))
  in
  equal answer A.Done
    (K.map p
       ~dsts:(Iarray.of_array (Array.map (fun x -> A.Any x) xs))
       (Iarray.of_array (Array.map (fun x -> Nx_kernel.Plain x) xs)));
  Array.iteri
    (fun k x ->
      equal
        ~msg:(Printf.sprintf "output %d" k)
        (array int64)
        (Array.init m (fun i -> Int64.of_int ((k * i) + k)))
        (A.to_array (Support.host x)))
    xs

(* Writes that share bytes with a load they are not identical to, or with
   another destination: a float32 destination over a load one element further
   into its buffer, over a load broadcast from its first element, an int4
   destination over a load one nibble further, and a second destination one
   element further. Each the arrays of 5 elements over a buffer of 6, the
   destination from its start, the other array from [offset] at [stride]. *)
let overlaps =
  [
    ("a load one element further", D.Any D.Float32, 1, 1, `Load);
    ("a load broadcast from its first element", D.Any D.Float32, 0, 0, `Load);
    ("an int4 load one nibble further", D.Any D.Int4, 1, 1, `Load);
    ("a second destination one element further", D.Any D.Float32, 1, 1, `Dst);
  ]

(* The copy, refused as Overlapping, and every byte of the buffer as it was. *)
let check_overlap (b : Support.backend) (_, D.Any dt, offset, stride, other) =
  let module K = (val b.kernels) in
  let n = 5 in
  let whole = Support.on b (seeded dt [| n + 1 |] 5) in
  let before = bytes_of whole in
  let at offset strides =
    A.v dt (L.v ~offset ~strides:[| strides |] [| n |]) (A.buffer whole)
  in
  let dst = at 0 1 and y = at offset stride in
  let d = D.Any dt in
  let answered =
    match other with
    | `Load ->
        let p = P.v ~ins:[| d |] [| P.In 0 |] ~outs:[| 0 |] in
        K.map p ~dsts:[| A.Any dst |] [| Plain y |]
    | `Dst ->
        let p = P.v ~ins:[| d |] [| P.In 0 |] ~outs:[| 0; 0 |] in
        let x = Support.on b (seeded dt [| n |] 6) in
        K.map p ~dsts:[| A.Any dst; A.Any y |] [| Plain x |]
  in
  equal answer A.Overlapping answered;
  equal ~msg:"the buffer's bytes" (array int) before (bytes_of whole)

(* Boxes *)

(* A box of a shape [s]: along each axis a start and a count within the
   extent. *)
let box_of s =
  let open Gen in
  let rec axes = function
    | [] -> constant []
    | e :: rest ->
        let* start = int_range 0 e in
        let* count = int_range 0 (e - start) in
        let+ r = axes rest in
        { M.start; count; step = 1 } :: r
  in
  map Array.of_list (axes (Array.to_list s))

let pp_box ppf (Pair (x, _), box) =
  Format.fprintf ppf "%a %a box %a" D.pp (A.dtype x) L.pp (A.layout x)
    (Format.pp_print_list ~pp_sep:Format.pp_print_space
       (fun ppf (r : M.range) -> Format.fprintf ppf "%d+%d" r.start r.count))
    (Array.to_list box)

let box_cases =
  Gen.with_pp pp_box
    (let open Gen in
     let* (D.Any dt) =
       of_list ~pp:pp_dtype
         D.
           [
             Any Int32;
             Any Float32;
             Any Uint8;
             Any Int4;
             Any Float64;
             Any Int16;
             Any Bfloat16;
           ]
     in
     let* s = array ~size:(int_range 1 3) (int_range 1 6) in
     let* x = drawn dt (Iarray.of_array s)
     and+ y = drawn dt (Iarray.of_array s) in
     let whole = Array.map (fun e -> { M.start = 0; count = e; step = 1 }) s in
     let+ box = frequency [ (3, box_of s); (1, constant whole) ] in
     (Pair (x, y), box))

(* A result whose bits the contract fixes depends on its own index's operand
   elements alone: over a box of the indices, the results computed from the
   operands' box alone have the whole result's bits there, for an exact kind of
   two operands, a copy of a view and an integer sum over the last axis. *)
let law_box (b : Support.backend) (Pair (x, y), box) =
  let module K = (val b.kernels) in
  let dt = A.dtype x in
  let shape = Iarray.to_array (L.shape (A.layout x)) in
  let r = Array.length box in
  let slice ranges a =
    if Array.length ranges = 0 then a
    else Option.get (A.move (M.Slice (Iarray.of_array ranges)) a)
  in
  let fresh a = A.create b.device dt (L.shape (A.layout a)) in
  let fresh2 (a, _) = fresh a in
  cover "an empty box" (Array.exists (fun (g : M.range) -> g.count = 0) box);
  cover "the whole as the box"
    (Array.for_all2
       (fun (g : M.range) e -> g.start = 0 && g.count = e)
       box shape);
  (* [call] into a destination, [fresh] for the operands, over the whole
     operands [a] and over their box [a']; [sub] is the result's box. Whether
     both computed. *)
  let both ~msg ~sub fresh call a a' =
    let d = fresh a and d' = fresh a' in
    match (call ~dst:d a, call ~dst:d' a') with
    | A.Done, A.Done ->
        equal ~msg (list string) []
          (values_differ
             (A.copy (slice sub (Support.host d)))
             (Support.host d'));
        true
    | A.Declined, A.Declined ->
        collect "declined";
        false
    | p, q ->
        failf "%s answered %a over the whole and %a over the box" msg
          Nx_array_support.pp_answer p Nx_array_support.pp_answer q
  in
  let x = Support.on b x and y = Support.on b y in
  let exact =
    P.accepts2 (Binary Add) dt
    && both ~msg:"an exact kind of two operands" ~sub:box fresh2
         (fun ~dst (a, c) -> Support.map2 (module K) (Binary Add) ~dst a c)
         (x, y)
         (slice box x, slice box y)
  in
  cover "an exact kind of two operands" exact;
  let flipped =
    slice
      (Array.mapi
         (fun i e ->
           if i = r - 1 then { M.start = e - 1; count = e; step = -1 }
           else { M.start = 0; count = e; step = 1 })
         shape)
      x
  in
  cover "a copy of a view"
    (both ~msg:"a copy of a view" ~sub:box fresh
       (fun ~dst a -> Support.map1 (module K) Copy ~dst a)
       flipped (slice box flipped));
  let integer =
    (D.is D.Signed dt || D.is D.Unsigned dt)
    && Nx_kernel.accepts (Monoid Sum) dt
  in
  let summed =
    integer
    &&
    let id = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |] in
    let whole = { M.start = 0; count = shape.(r - 1); step = 1 } in
    let obox = Array.mapi (fun i g -> if i = r - 1 then whole else g) box in
    let sum ~dst a =
      let dst = A.Any dst in
      K.reduce id
        ~shape:(L.shape (A.layout a))
        ~axes:[| r - 1 |]
        [| (Monoid Sum, 0) |]
        ~dsts:[| dst |] [| Nx_kernel.Plain a |]
    in
    let d =
      A.create b.device dt (Iarray.of_array (Array.sub shape 0 (r - 1)))
    in
    let d' =
      A.create b.device dt
        (Iarray.of_array
           (Array.map (fun (g : M.range) -> g.count) (Array.sub box 0 (r - 1))))
    in
    match (sum ~dst:d x, sum ~dst:d' (slice obox x)) with
    | A.Done, A.Done ->
        equal ~msg:"sum" (list string) []
          (bits_differ
             (A.copy (slice (Array.sub box 0 (r - 1)) (Support.host d)))
             (Support.host d'));
        true
    | A.Declined, A.Declined ->
        collect "declined";
        false
    | p, q ->
        failf "sum answered %a over the whole and %a over the box"
          Nx_array_support.pp_answer p Nx_array_support.pp_answer q
  in
  cover "an integer sum" summed

(* NaN results *)

(* The dtypes whose NaNs have a quiet bit, each with operands of every special
   for a dtype of 32 bits or more, of every code for a narrower one, beside the
   same reversed. *)
let quiet_bits =
  let every (type v s) (dt : (v, s) D.t) =
    let n = 1 lsl D.bits dt in
    let codes = A.of_array D.Uint16 [| n |] (Array.init n Fun.id) in
    A.Any
      (match dt with
      | D.Float8_e5m2 ->
          Option.get
            (A.bitcast dt (A.of_array D.Uint8 [| n |] (Array.init n Fun.id)))
      | _ -> Option.get (A.bitcast dt codes))
  in
  [
    (D.Any D.Float32, lazy (A.Any (specials_only D.Float32 300 1)));
    (D.Any D.Float64, lazy (A.Any (specials_only D.Float64 300 2)));
    (D.Any D.Complex64, lazy (A.Any (specials_only D.Complex64 300 3)));
    (D.Any D.Complex128, lazy (A.Any (specials_only D.Complex128 300 4)));
    (D.Any D.Float16, lazy (every D.Float16));
    (D.Any D.Bfloat16, lazy (every D.Bfloat16));
    (D.Any D.Float8_e5m2, lazy (every D.Float8_e5m2));
  ]

(* nx_kernel.mli: a NaN result is signalling only where an operand is. Each kind
   of one and two operands the dtype takes, over [x] and over [x] and [x]
   reversed, Neg and Abs included, which keep a NaN's bits. *)
let check_quiet (b : Support.backend) (D.Any dt, x) =
  let module K = (val b.kernels) in
  let (A.Any x) = Lazy.force x in
  let x = A.expect dt (A.Any x) in
  let n = L.numel (A.layout x) in
  let r =
    Option.get
      (A.move (M.Slice [| { M.start = n - 1; count = n; step = -1 } |]) x)
  in
  let d = D.Any dt in
  let els a = elements_of d (bits_of a) in
  let xs = els x and rs = els r in
  let check msg answered dst operands =
    if answered = A.Done then
      Array.iteri
        (fun i e ->
          if
            Elements.signalling d e
            && not (List.exists (fun o -> Elements.signalling d o.(i)) operands)
          then
            failf "%s: element %d is a signalling NaN from quiet operands" msg i)
        (els (Support.host dst))
  in
  let x' = Support.on b x and r' = Support.on b r in
  List.iter
    (fun (u, name) ->
      let k = P.Unary u in
      if P.accepts1 k dt dt then
        let dst = A.create b.device dt [| n |] in
        check name (Support.map1 (module K) k ~dst x') dst [ xs ])
    unaries;
  List.iter
    (fun k ->
      if P.accepts2 k dt then
        let dst = A.create b.device dt [| n |] in
        check (name2 k) (Support.map2 (module K) k ~dst x' r') dst [ xs; rs ])
    binaries

(* check_dst *)

(* Destinations the door refuses that are not C-contiguous either: one element
   repeated, a stepped view of a freed buffer and of memory admitting only
   Read. *)
let door_rows =
  [
    ("repeated", A.Repeated_elements);
    ("dead", A.Dead_buffer);
    ("read-only", A.Read_only);
  ]

let check_door_first (b : Support.backend) (_, want) =
  let module K = (val b.kernels) in
  let n = 4 in
  let stepped = L.v ~strides:[| 2 |] [| n |] in
  let dst =
    match want with
    | A.Repeated_elements ->
        Option.get
          (A.move (M.Broadcast [| n |]) (A.create b.device D.Float32 [| 1 |]))
    | A.Dead_buffer ->
        let base = A.create b.device D.Float32 [| 2 * n |] in
        let dst = A.v D.Float32 stepped (A.buffer base) in
        ignore (B.free (A.buffer base));
        dst
    | _ ->
        let base =
          read_only b (A.of_array D.Float32 [| 2 * n |] (Array.make (2 * n) 0.))
        in
        A.v D.Float32 stepped (A.buffer base)
  in
  let idx = Support.on b (A.of_array D.Int64 [| n |] (Array.make n 0L)) in
  let x = Support.on b (A.of_array D.Float32 [| n |] (Array.make n 1.)) in
  equal answer want (K.gather ~axis:0 ~dst idx x)

(* Loads *)

type padding = { x : case; fill : string; pad : Nx_kernel.pad }

let pp_padding ppf c =
  let ints a = Array.to_list (Iarray.to_array a) in
  let pp_list =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ",")
      Format.pp_print_int
  in
  Format.fprintf ppf "%a, lo %a, hi %a, interior %a, %d windows" pp_case c.x
    pp_list (ints c.pad.lo) pp_list (ints c.pad.hi) pp_list
    (ints c.pad.interior)
    (Iarray.length c.pad.windows)

(* An operand from {!case}, padded along each axis by [lo] and [hi] in [-2, 2],
   cropping where negative but to no negative extent, and [interior] in [0, 2];
   one case in four, windows of size, step and dilation 1 or 2 on increasing
   axes; and a fill of the operand's dtype, one of its specials, NaNs with
   payloads among them, one time in two. *)
let paddings =
  Gen.with_pp pp_padding
    (let open Gen in
     let* (Case x as c) = case in
     let dt = A.dtype x in
     let s = Iarray.to_array (L.shape (A.layout x)) in
     let r = Array.length s in
     let* interior = array ~size:(constant r) (int_range 0 2) in
     let* lo = array ~size:(constant r) (int_range (-2) 2) in
     let* hi = array ~size:(constant r) (int_range (-2) 2) in
     let grown i =
       s.(i) + lo.(i) + if s.(i) > 0 then interior.(i) * (s.(i) - 1) else 0
     in
     let hi = Array.mapi (fun i h -> max h (-grown i)) hi in
     let window axis =
       let+ size = int_range 1 2
       and+ step = int_range 1 2
       and+ dilation = int_range 1 2 in
       { M.axis; size; step; dilation }
     in
     let rec windows = function
       | [] -> constant []
       | a :: rest ->
           let* w = window a in
           let+ ws = windows rest in
           w :: ws
     in
     let* axes = subsequence (List.init r Fun.id) in
     let* ws = frequency [ (3, constant []); (1, windows axes) ] in
     let w = max 1 (D.bits dt / 8) in
     let+ fill =
       let bytes = string_of ~size:(constant w) char in
       match specials dt with
       | _ when D.bits dt < 8 ->
           let+ code = int_range 0 ((1 lsl D.bits dt) - 1) in
           String.make 1 (Char.chr code)
       | [] -> bytes
       | sp -> frequency [ (1, bytes); (1, of_list sp) ]
     in
     let pad =
       {
         Nx_kernel.lo = Iarray.of_array lo;
         hi = Iarray.of_array hi;
         interior = Iarray.of_array interior;
         windows = Iarray.of_list ws;
       }
     in
     { x = c; fill; pad })

(* [fill]'s value of [dt], and the bits it stores as: a signalling NaN of a
   float read as an OCaml value is quiet. *)
let fill_value (type v s) (dt : (v, s) D.t) fill =
  let v = A.get (A.v dt (L.contiguous [||]) (B.of_string fill)) [||] in
  let one = A.host dt [||] in
  A.set one [||] v;
  let e = bits_of one in
  (v, String.init (Array.length e) (fun i -> Char.chr e.(i)))

(* nx_kernel.mli: a [Padded] load reads its operand padded with [fill] as [pad]
   says, its loop the padded shape: a copy of it is {!Elements.pad_array}'s
   array, bit for bit. *)
let law_padded (b : Support.backend) { x = Case x; fill; pad } =
  let module K = (val b.kernels) in
  let dt = A.dtype x in
  let v, stored = fill_value dt fill in
  cover "a negative lo" (Iarray.exists (fun l -> l < 0) pad.lo);
  cover "an interior" (Iarray.exists (fun i -> i > 0) pad.interior);
  cover "windows" (Iarray.length pad.windows > 0);
  cover "sub-byte" (D.bits dt < 8);
  match Nx_kernel.Shape.padded pad (Iarray.to_array (L.shape (A.layout x))) with
  | Error e -> failf "the draws made a padding Shape.padded refuses: %s" e
  | Ok shape -> (
      let want = A.expect dt (Elements.pad_array (A.Any x) stored pad) in
      let p = P.v ~ins:[| D.Any dt |] [| P.In 0 |] ~outs:[| 0 |] in
      let dst = A.create b.device dt (Iarray.of_array shape) in
      let load = Nx_kernel.Padded { x = Support.on b x; fill = v; pad } in
      match K.map p ~dsts:[| A.Any dst |] [| load |] with
      | A.Done -> no_mismatch (bits_differ (A.copy want) (Support.host dst))
      | A.Declined -> check_declined b (K1 Copy) (D.Any dt)
      | r -> failf "map answered %a" Nx_array_support.pp_answer r)

(* A padded load and a coordinate: the map holds its loads' padded copies and,
   after them, the coordinate's index. On a domain of its own, whose thread has
   kept no scratch, each buffer the map takes is of its exact size. Each output
   is its load's element plus its column. *)
let test_padded_coordinate (b : Support.backend) () =
  let module K = (val b.kernels) in
  let rows = 4 and cols = 5 in
  let x =
    Support.on b
      (A.of_array D.Int64 [| rows; cols |]
         (Array.init (rows * cols) Int64.of_int))
  in
  let p =
    P.v ~ins:[| D.Any D.Int64 |]
      [| P.In 0; P.Coord 1; Op2 (Binary Add, 0, 1) |]
      ~outs:[| 2 |]
  in
  let dst = A.create b.device D.Int64 [| rows; cols |] in
  let run () = K.map p ~dsts:[| A.Any dst |] [| unpadded x |] in
  equal answer A.Done (Domain.join (Domain.spawn run));
  equal (array int)
    (Array.init (rows * cols) (fun i -> i + (i mod cols)))
    (Array.map Int64.to_int (A.to_array (Support.host dst)))

(* computes_on *)

let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal

let test_computes_on (b : Support.backend) () =
  let module K = (val b.kernels) in
  satisfies ~claim:"a device the kernels compute on" device K.computes_on
    b.device

(* Maps of programs *)

(* Programs of 400 nodes over float32 rows that hold more than eight values at
   once, of seeds whose nodes a library that declines float64 computes: the
   draws reach such a program in under 3% of the cases. *)
let long_programs =
  List.map
    (fun s ->
      ( Pair (seeded D.Float32 [| 3; 5 |] s, seeded D.Float32 [| 3; 5 |] (s + 1)),
        s,
        Long ))
    [ 54; 57; 59 ]

(* Programs whose map holds what the draws reach in under one case in five: a
   where among more than four loop operands, read plain and padded; a sub-byte
   bitcast; a load of one repeated element; and one element. *)
let rare_programs =
  let f32 s = seeded D.Float32 [| 3; 5 |] s
  and int4 s = seeded D.Int4 [| 3; 5 |] s in
  [
    (Pair (f32 1, f32 2), 74, Plain);
    (Pair (f32 1, f32 2), 74, Padded);
    (Pair (int4 1, int4 2), 0, Plain);
    (Pair (repeated D.Float32 5 1.5, seeded D.Float32 [| 5 |] 2), 0, Plain);
    (Pair (seeded D.Float32 [| 1 |] 1, seeded D.Float32 [| 1 |] 2), 12, Plain);
  ]

(* A map too wide for one of nx.cuda's launches gives each output the bits of
   its nodes run one by one, any NaN for a NaN, and a destination identical to a
   [Plain] load receives what a fresh one would. Where a node run alone is
   declined, the map is. *)
let check_wide_map (b : Support.backend) c =
  let module K = (val b.kernels) in
  let p, n = wide_program c in
  let shape = Iarray.of_array (wide_shape c) in
  let loads, xs = wide_loads b c n in
  let types = Array.init (Iarray.length p.nodes) (P.dtype p) in
  let aliased = c.aliased && not c.padded in
  let dsts () =
    Array.mapi
      (fun k o ->
        if aliased && k = 0 then xs.(0)
        else
          let (D.Any d) = types.(o) in
          A.Any (A.create b.device d shape))
      (Iarray.to_array p.outs)
  in
  let map dsts = K.map p ~dsts:(Iarray.of_array dsts) (Iarray.of_array loads) in
  match split b p types shape xs with
  | exception Skip -> equal answer A.Declined (map (dsts ()))
  | values ->
      let dsts = dsts () in
      equal answer A.Done (map dsts);
      Array.iteri
        (fun k (A.Any y) ->
          let (A.Any v) = values.(Iarray.get p.outs k) in
          equal
            ~msg:(Printf.sprintf "output %d" k)
            (same_values (D.Any (A.dtype v)))
            (bytes_of v) (bytes_of y))
        dsts

(* The contract's kinds

   Every kernel library gives a kind its doc, IEEE 754 or modular arithmetic
   defines exactly that result, and holds the others to the bounds of
   Nx_kernel.Prog's Results section. nx_kinds.h's functions give the exact
   results' bits, which test_kinds checks against their meaning. *)

(* The kinds of one operand the contract fixes at [dt] that [dt] takes: every
   one of an integer and a boolean, a float's but the transcendentals, a complex
   number's Neg. *)
let exact1 (D.Any dt) =
  let float = D.is D.Float dt and complex = D.is D.Complex dt in
  List.filter
    (fun ((u : P.unary), _) ->
      P.accepts1 (Unary u) dt dt
      &&
      match u with
      | Neg -> true
      | Recip | Abs | Sign | Sqrt | Floor | Ceil | Round | Trunc -> not complex
      | _ -> not (float || complex))
    unaries

(* The kinds of two operands it fixes that [dt] takes: but Pow, Atan2 and a
   complex number's Mul and Fdiv. *)
let exact2 (D.Any dt) =
  let complex = D.is D.Complex dt in
  List.filter
    (fun k ->
      P.accepts2 k dt
      &&
      match k with
      | P.Binary (Pow | Atan2) -> false
      | Binary (Mul | Fdiv) -> not complex
      | Binary _ | Compare _ -> true)
    op2s

(* A sub-byte operand and pair, which the draws reach one case in six. *)
let sub_byte_case = Case (seeded D.Int4 [| 5 |] 1)
let sub_byte_pair = Pair (seeded D.Int4 [| 5 |] 1, seeded D.Int4 [| 5 |] 2)

(* Prog's Results section's table: each float kind's bound in ulps at float32
   and at float64, with its name in nx_kinds.h. *)
let float_bounds =
  let u k = Support.K1 (P.Unary k) and b k = Support.K2 (P.Binary k) in
  P.
    [
      (u Exp, "exp", 2, 2);
      (u Exp2, "exp2", 2, 2);
      (u Sin, "sin", 2, 2);
      (u Cos, "cos", 2, 2);
      (u Asin, "asin", 2, 2);
      (u Acos, "acos", 2, 2);
      (u Atan, "atan", 2, 2);
      (u Cosh, "cosh", 2, 2);
      (u Tanh, "tanh", 2, 2);
      (u Erf, "erf", 2, 2);
      (u Expm1, "expm1", 1, 2);
      (u Log, "log", 1, 2);
      (u Log2, "log2", 1, 2);
      (u Log1p, "log1p", 1, 2);
      (b Atan2, "atan2", 3, 2);
      (u Sinh, "sinh", 3, 2);
      (u Tan, "tan", 4, 2);
      (b Pow, "pow", 4, 2);
    ]

(* [k] of [xs], and of [ys] for a kind of two operands, by [b]'s kernels into a
   fresh array of [dt]; [None] where they decline it. *)
let run_kind (type v s) (b : Support.backend) k (dt : (v, s) D.t) xs ys =
  let module K = (val b.kernels) in
  let n = Array.length xs in
  let x = Support.on b (A.of_array dt [| n |] xs) in
  let dst = A.create b.device dt [| n |] in
  let r =
    match k with
    | Support.K1 op -> Support.map1 (module K) op ~dst x
    | K2 op ->
        Support.map2
          (module K)
          op ~dst x
          (Support.on b (A.of_array dt [| n |] ys))
    | K0 _ | K3 _ -> invalid_arg "run_kind"
  in
  match r with
  | A.Done -> Some (A.to_array (Support.host dst))
  | A.Declined ->
      check_declined ~into:(D.Any dt) b k (D.Any dt);
      None
  | r -> failf "the kernels answered %a" Nx_array_support.pp_answer r

(* Ordered-bit ranks of binary32 values, -0 and +0 one apart. *)
let rank32 x =
  let b = Int32.to_int (Int32.bits_of_float x) in
  if b < 0 then -(b land 0x7FFF_FFFF) - 1 else b

let of_rank32 r =
  Int32.float_of_bits
    (Int32.of_int (if r < 0 then -(r + 1) lor 0x8000_0000 else r))

let ulps32 a c =
  if Float.is_nan a || Float.is_nan c then
    if Float.is_nan a && Float.is_nan c then 0 else max_int
  else abs (rank32 a - rank32 c)

(* The binary32 values [r], a float64 within 2 ulps of a kind's value, may stand
   for: [r] rounded once, and its neighbour too where [r] is too close to a tie
   to tell. *)
let nearest32 r =
  let a = Int32.float_of_bits (Int32.bits_of_float r) in
  if Float.is_nan r || a = r || not (Float.is_finite a) then [ a ]
  else
    let n = of_rank32 (rank32 a + if r > a then 1 else -1) in
    let mid = (a +. n) /. 2. in
    if Float.abs (r -. mid) <= 0x1p-50 *. Float.abs r then [ a; n ] else [ a ]

(* Fails at the first of [got] past [bound] ulps from [want], as [dist] measures
   them, or a zero of the wrong sign where [want] is one. *)
let within_ulps ~name ~bound ~dist ~show got want =
  Array.iteri
    (fun i g ->
      let w = want.(i) in
      let zero = List.exists (fun w -> w = 0.) w in
      let wrong_zero =
        g = 0. && zero
        && List.for_all (fun w -> Float.sign_bit g <> Float.sign_bit w) w
      in
      let e = if wrong_zero then max_int else dist g w in
      if e > bound then
        failf "%s %s: %h, %d ulps from %s, past %d" name (show i) g
          (if e = max_int then -1 else e)
          (String.concat " or " (List.map (Printf.sprintf "%h") w))
          bound)
    got

(* Each float kind at float64 within its bound of the goldens' correctly rounded
   values. *)
let test_f64_kinds (b : Support.backend) () =
  List.iter
    (fun (k, name, _, bound) ->
      let rows = Array.of_list (Exact.golden name) in
      let arity = Array.length rows.(0) - 2 in
      let col j = Array.map (fun r -> r.(j)) rows in
      let xs = col 0 and ys = if arity = 2 then col 1 else col 0 in
      match run_kind b k D.Float64 xs ys with
      | None -> ()
      | Some got ->
          let show i =
            if arity = 2 then Printf.sprintf "%h %h" xs.(i) ys.(i)
            else Printf.sprintf "%h" xs.(i)
          in
          within_ulps ~name ~bound
            ~dist:(fun g w -> Exact.ulps64 g (List.hd w))
            ~show got
            (Array.map (fun r -> [ r.(arity) ]) rows))
    float_bounds

(* The kind nx_kinds.h names [name]. *)
let kind_named name =
  match List.find_opt (fun (_, n) -> n = name) unaries with
  | Some (u, _) -> Support.K1 (P.Unary u)
  | None -> Support.K2 (List.find (fun k -> name2 k = name) op2s)

(* An Annex F row's kind at its widths: [check ~dtype ~single args want got] of
   each result [got]. At float32, where [b] flushes, a row with a subnormal
   operand or, as [tiny want] says, a nonzero exact result below the least
   normal is outside what prog.mli states and is skipped. *)
let annex_f_at (b : Support.backend) ~tiny ~check (w, name, args, want) =
  let k = kind_named name in
  let at (type s) (dt : (float, s) D.t) ~single round =
    let args = Array.map round args and want = round want in
    let least = 0x1p-126 in
    let subnormal a = a <> 0. && Float.abs a < least in
    let outside =
      single && b.underflow = Flush
      && (Array.exists subnormal args || tiny ~least want)
    in
    if not outside then
      Option.iter
        (fun got -> check ~dtype:(D.name dt) ~single args want got.(0))
        (run_kind b k dt [| args.(0) |] [| args.(Array.length args - 1) |])
  in
  match (w : Annex_f.width) with
  | F64 -> at D.Float64 ~single:false Fun.id
  | F32 -> at D.Float32 ~single:true Fun.id
  | Both ->
      at D.Float64 ~single:false Fun.id;
      at D.Float32 ~single:true Exact.round_f32

(* prog.mli: Annex F's special values and zero signs are exact. A NaN as any
   NaN. *)
let check_annex_f b row =
  annex_f_at b row
    ~tiny:(fun ~least want -> want <> 0. && Float.abs want < least)
    ~check:(fun ~dtype ~single:_ _ want got ->
      equal ~msg:dtype float_exact want got)

(* Annex F's rows of no special value, each the correctly rounded result: within
   the kind's bound in ranks, a zero of the result's sign. Every exact value is
   nonzero, so a result below the least normal is a tiny one. *)
let check_annex_f_rounded b ((_, name, _, _) as row) =
  let _, _, b32, b64 = List.find (fun (_, n, _, _) -> n = name) float_bounds in
  annex_f_at b row
    ~tiny:(fun ~least want -> Float.abs want < least)
    ~check:(fun ~dtype ~single args want got ->
      let bound, dist = if single then (b32, ulps32) else (b64, Exact.ulps64) in
      let show _ =
        String.concat " " (Array.to_list (Array.map (Printf.sprintf "%h") args))
      in
      within_ulps
        ~name:(name ^ " " ^ dtype)
        ~bound
        ~dist:(fun g w -> dist g (List.hd w))
        ~show [| got |] [| [ want ] |])

(* Operands where C99 Annex F gives a kind a special value. *)
let specials32 =
  [| 0.; -0.; infinity; neg_infinity; nan; 1.; -1.; 0.5; 2.; 0x1p-149 |]

let special_pairs =
  let s = [ 0.; -0.; infinity; neg_infinity; nan; 1.; -1.; 0.5; 2.; -3. ] in
  Array.of_list (List.concat_map (fun x -> List.map (fun y -> (x, y)) s) s)

(* [v] and its [k] float32 neighbours on either side, of either sign. *)
let around k v =
  let b = Int32.to_int (Int32.bits_of_float (Float.abs v)) land 0x7FFF_FFFF in
  List.concat_map
    (fun d ->
      let p = b + d in
      if p < 0 || p >= 0x8000_0000 then []
      else
        [
          Int32.float_of_bits (Int32.of_int p);
          Int32.float_of_bits (Int32.of_int (p lor 0x8000_0000));
        ])
    (List.init ((2 * k) + 1) (fun i -> i - k))

(* The float32 values at which nx_kinds.h's f32 kinds switch formulas, saturate
   or clamp, each with eight neighbours of either sign: every (k + 1/2) ln 2 and
   k + 1/2 below the exponentials' overflow and every sqrt 2 2^k, where a
   reduction switches, and the kinds' constants. *)
let switch_points =
  lazy
    (let ln2 = 0.69314718055994531 in
     let halves =
       List.init 153 (fun k -> (Float.of_int k +. 0.5) *. ln2)
       @ List.init 151 (fun k -> Float.of_int k +. 0.5)
     in
     let sqrt2 =
       List.init 277 (fun k -> Float.ldexp 1.4142135623730951 (k - 149))
     in
     let constants =
       [
         88.72283935546875;
         89.;
         -104.;
         103.97208;
         129.;
         151.;
         128.;
         149.;
         126.;
         17.328680;
         26.;
         709.782712893384;
         708.3964185322641;
         745.1332191019411;
         710.;
         746.;
         1025.;
         1076.;
         1024.;
         1074.;
         40.;
         37.42994775023705;
         86.;
         708.;
         0.70710678118654757;
         1.4142135623730951;
         0.29289321881345243;
         0.41421356237309515;
         32768.;
         268435456.;
         0.5;
         1.;
         0.4375;
         0.6875;
         1.1875;
         2.4375;
         0.875;
         3.92;
         1.75;
         2.75;
         4.;
         5.95;
         5.921587195794507;
         90.;
         89.41598629223294;
         712.;
         710.4758600739439;
         9.010913339;
         19.06154746539849;
         13.;
         22.18070977791825;
         44.3614195558365;
         88.722839111672999;
         0x1p-12;
         0x1p-24;
         0x1p-25;
         0x1p-27;
         0x1p-53;
         0x1p-54;
         0x1p-100;
         0x1p24;
         0x1p23;
         0x1p52;
         0x1p53;
         2.;
         3.;
       ]
     in
     Array.of_list (List.concat_map (around 8) (halves @ sqrt2 @ constants)))

(* Every multiple of pi/4 below the trigonometric switch at 2^15, with four
   neighbours of either sign: where sin, cos and tan alone switch. *)
let quarter_turns =
  lazy
    (let pio4 = 0.78539816339744831 in
     let rec multiples k =
       let v = Float.of_int k *. pio4 in
       if v >= 32768. then [] else v :: multiples (k + 1)
     in
     Array.of_list (List.concat_map (around 4) (multiples 1)))

(* Each float kind of [kinds] at float32 within its bound of the kind at
   float64, which the goldens bound at 2 ulps, rounded once, at the operands
   [xs], and for a kind of two operands at the pairs [ps]. *)
let check_f32_kinds ?(kinds = float_bounds) (b : Support.backend) xs ps =
  List.iter
    (fun (k, name, bound, _) ->
      let two = match k with Support.K2 _ -> true | _ -> false in
      let xs, ys =
        if two then (Array.map fst ps, Array.map snd ps) else (xs, xs)
      in
      match (run_kind b k D.Float64 xs ys, run_kind b k D.Float32 xs ys) with
      | None, _ | _, None -> ()
      | Some exact, Some got ->
          let want = Array.map nearest32 exact in
          let show i =
            if two then Printf.sprintf "%h %h" xs.(i) ys.(i)
            else Printf.sprintf "%h" xs.(i)
          in
          within_ulps ~name ~bound
            ~dist:(fun g w ->
              List.fold_left (fun m w -> min m (ulps32 g w)) max_int w)
            ~show got want)
    kinds

let trigonometric =
  List.filter
    (fun (_, name, _, _) -> List.mem name [ "sin"; "cos"; "tan" ])
    float_bounds

(* {!check_f32_kinds} at Annex F's operands and pairs and at the switch points,
   the quarter turns through sin, cos and tan alone. *)
let test_f32_points (b : Support.backend) () =
  check_f32_kinds b
    (Array.append specials32 (Lazy.force switch_points))
    special_pairs;
  check_f32_kinds ~kinds:trigonometric b (Lazy.force quarter_turns) [||]

(* {!check_f32_kinds} at every 2^20th bit pattern from [offset], every binade of
   both signs reached, and at pairs from [seed], half of any patterns and half
   where [pow] and [atan2] neither overflow nor underflow. *)
let f32_step = 1 lsl 20

let law_f32_kinds (b : Support.backend) (offset, seed) =
  let step = f32_step in
  let xs =
    Array.init
      ((1 lsl 32) / step)
      (fun i -> Int32.float_of_bits (Int32.of_int ((i * step) + offset)))
  in
  let r = Random.State.make [| seed |] in
  let pair i =
    if i land 1 = 0 then
      let any () = Int32.float_of_bits (Random.State.bits32 r) in
      (any (), any ())
    else
      let x =
        Float.ldexp (Random.State.float r 1.) (Random.State.int r 9 - 4)
      in
      (Exact.round_f32 x, Exact.round_f32 (Random.State.float r 40. -. 20.))
  in
  check_f32_kinds b xs (Array.init (1 lsl 12) pair)

(* nx_kinds.h's complex section: each transcendental kind's normwise bound in
   units of its parts' u, with its golden's name. pow's is [3 + 3 |w log z|]. *)
let complex_bounds =
  P.
    [
      (Sqrt, "sqrt", 3.);
      (Sign, "sign", 3.);
      (Exp, "exp", 4.);
      (Exp2, "exp2", 4.);
      (Log, "log", 4.);
      (Log2, "log2", 4.);
      (Asin, "asin", 5.);
      (Acos, "acos", 5.);
      (Atan, "atan", 5.);
      (Asinh, "asinh", 5.);
      (Acosh, "acosh", 5.);
      (Atanh, "atanh", 5.);
      (Sin, "sin", 6.);
      (Cos, "cos", 6.);
      (Tan, "tan", 6.);
      (Sinh, "sinh", 6.);
      (Cosh, "cosh", 6.);
      (Tanh, "tanh", 6.);
    ]

let complex_of (re, im) = { Complex.re; im }

(* Each complex transcendental kind, and pow, within its bound of the goldens'
   correctly rounded values, at complex128 and, where the operands are
   binary32's and the value's modulus a normal one, at complex64; on a backend
   that flushes, where no operand part is subnormal at binary32. *)
let test_complex_kinds (b : Support.backend) () =
  let check name k ~bound rows =
    let rows = Array.of_list rows in
    let n = if rows = [||] then 0 else Array.length rows.(0) - 4 in
    let op j r = { Complex.re = r.(2 * j); im = r.((2 * j) + 1) } in
    let at u (type s) (dt : (Complex.t, s) D.t) rows =
      let xs = Array.map (op 0) rows in
      let ys = if n = 4 then Array.map (op 1) rows else xs in
      match run_kind b k dt xs ys with
      | None -> ()
      | Some got ->
          Array.iteri
            (fun i (g : Complex.t) ->
              let row = rows.(i) in
              let e =
                Exact.normwise u (g.re, g.im) (Exact.golden_value row n)
              in
              let bound = bound row in
              if not (e <= bound) then
                failf "%s %s: %h %h, %gu past %gu" name
                  (String.concat " "
                     (List.map (Printf.sprintf "%h")
                        (Array.to_list (Array.sub row 0 n))))
                  g.re g.im e bound)
            got
    in
    at 0x1p-53 D.Complex128 rows;
    let normal a = a = 0. || Float.abs a >= 0x1p-126 in
    let kept r =
      let args = Array.sub r 0 n in
      Exact.single_row args (Exact.golden_value r n)
      && (b.underflow = Gradual || Array.for_all normal args)
    in
    at 0x1p-24 D.Complex64
      (Array.of_list (List.filter kept (Array.to_list rows)))
  in
  List.iter
    (fun (u, name, bound) ->
      check name (Support.K1 (P.Unary u))
        ~bound:(fun _ -> bound)
        (Exact.complex_golden name))
    complex_bounds;
  let pow_bound r =
    let m (re, im) = Float.hypot re im in
    let z = (r.(0), r.(1)) and w = (r.(2), r.(3)) in
    3. +. (3. *. m w *. m (Float.log (m z), Float.atan2 r.(1) r.(0)))
  in
  check "pow" (Support.K2 (P.Binary Pow)) ~bound:pow_bound
    (Exact.complex_golden "pow")

(* Mul, Fdiv and Recip at complex128 and complex64 of operands from [seed] whose
   parts lie in [2^-40, 2^40], one part in 32 of each operand a NaN: [f ~name
   ~bound ~error ~nan x y g] for each result [g] of operands [x] and [y],
   [bound] the kind's in u, [error x y g] [g]'s normwise error in u, and [nan]
   whether an operand the kind reads has a NaN part. *)
let complex_arith (b : Support.backend) seed f =
  let r = Random.State.make [| seed |] in
  let part () =
    let m = 1. +. Random.State.float r 1. in
    let m = if Random.State.bool r then -.m else m in
    Float.ldexp m (Random.State.int r 81 - 40)
  in
  let n = 4096 in
  let draw () =
    Array.init n (fun i ->
        let re = part () and im = part () in
        match i mod 32 with 0 -> (nan, im) | 1 -> (re, nan) | _ -> (re, im))
  in
  let xs = draw () and ys = draw () in
  let has_nan (a, c) = Float.is_nan a || Float.is_nan c in
  let at (type s) u round (dt : (Complex.t, s) D.t) =
    let xs = Array.map round xs and ys = Array.map round ys in
    let zs a = Array.map complex_of a in
    let check name bound exact nan_in = function
      | None -> ()
      | Some got ->
          Array.iteri
            (fun i (g : Complex.t) ->
              let x = xs.(i) and y = ys.(i) in
              let error () = Exact.normwise u (g.re, g.im) (exact x y) in
              f ~name ~bound ~error ~nan:(nan_in x y) x y g)
            got
    in
    let two k = run_kind b (Support.K2 (P.Binary k)) dt (zs xs) (zs ys) in
    let either x y = has_nan x || has_nan y in
    check "mul" 2. Exact.exact_mul either (two Mul);
    check "fdiv" 4. Exact.exact_div either (two Fdiv);
    check "recip" 4.
      (fun _ y -> Exact.exact_div (1., 0.) y)
      (fun _ y -> has_nan y)
      (run_kind b (Support.K1 (P.Unary Recip)) dt (zs ys) (zs ys))
  in
  let f32 (a, c) = (Exact.round_f32 a, Exact.round_f32 c) in
  at 0x1p-53 Fun.id D.Complex128;
  at 0x1p-24 f32 D.Complex64

let pp_arith name (xr, xi) (yr, yi) (g : Complex.t) =
  Printf.sprintf "%s %h %h, %h %h: %h %h" name xr xi yr yi g.re g.im

(* Mul within 2u, Fdiv and Recip within 4u of the exact values, where no operand
   part is a NaN. *)
let law_complex_bounds b seed =
  complex_arith b seed (fun ~name ~bound ~error ~nan x y g ->
      if not nan then
        let e = error () in
        if not (e <= bound) then
          failf "%s, %gu past %gu" (pp_arith name x y g) e bound)

(* An operand with a NaN part makes both parts of Mul, Fdiv and Recip NaN. *)
let law_complex_nans b seed =
  complex_arith b seed (fun ~name ~bound:_ ~error:_ ~nan x y g ->
      if nan && not (Float.is_nan g.re && Float.is_nan g.im) then
        failf "%s, both parts not NaN" (pp_arith name x y g))

(* Real transforms *)

type real = Real : (float, 'r) D.t -> real
type complex = Complex : (Complex.t, 'c) D.t -> complex

(* A real transform takes a real and a complex dtype of one precision, float32
   with complex64 and float64 with complex128, and computes or declines those
   two. Every other pair does not fit: not Done, and nothing written outside its
   destination. *)
let test_real_transforms (b : Support.backend) () =
  let module K = (val b.kernels) in
  let reals =
    D.
      [
        Real Float64;
        Real Float32;
        Real Float16;
        Real Bfloat16;
        Real Float8_e4m3fn;
        Real Float8_e5m2;
        Real Float4_e2m1fn;
      ]
  in
  let pairs = [ ("float32", "complex64"); ("float64", "complex128") ] in
  List.iter
    (fun (Real r) ->
      List.iter
        (fun (Complex c) ->
          let takes = List.mem (D.name r, D.name c) pairs in
          let real s = A.create b.device r s in
          let complex s = A.create b.device c s in
          let check (type v s) name (d : (v, s) D.t) sh call =
            let msg =
              Printf.sprintf "%s of %s and %s" name (D.name r) (D.name c)
            in
            if not takes then
              Misfit.check ~msg ~what:"a dtype" b
                [| (D.Any d, sh) |]
                (fun g ds -> call g (A.expect d ds.(0)))
            else
              let dst = A.create b.device d (Iarray.of_array sh) in
              if call { Misfit.g = Fun.id } dst = A.Wrong_dtype then
                failf "%s answered Wrong_dtype" msg
          in
          check "r2c" c [| 3 |] (fun g dst ->
              K.r2c ~axes:[| 0 |] ~dst (g.g (real [| 4 |])));
          check "c2r" r [| 4 |] (fun g dst ->
              K.c2r ~axes:[| 0 |] ~dst (g.g (complex [| 3 |]))))
        D.[ Complex Complex64; Complex Complex128 ])
    reals

(* The suite *)

let laws (d : Support.declared) =
  let run f x =
    let b = Support.opened d in
    b.around (fun () -> f b x)
  in
  let unit f () =
    let b = Support.opened d in
    b.around (fun () -> f b ())
  in
  let on_cpu = d.label = Support.cpu.name in
  group d.label
    [
      group "calls"
        [
          cases
            ~id:(fun r -> r.name)
            "a map that does not fit answers no Done and writes nothing \
             outside its arrays"
            map_misfits (run check_map_misfit);
          cases ~id:fst "a refusal of a map that fits comes before any write"
            refusals (run check_refusal);
          test "a call keeps its operands alive while another domain collects"
            (unit test_collect_during_call);
        ];
      group "destinations"
        [
          cases
            ~id:(fun (call, what) ->
              loop_name call ^ " into a " ^ what ^ " one")
            "a map, a reduction and a scan into a stepped or reversed \
             destination give the bits of a C-contiguous one"
            scattered_rows (run test_scattered_loop);
          cases
            ~id:(fun (kernel, what, _) -> kernel ^ "'s " ^ what)
            "a kernel but a loop raises Invalid_argument for a destination \
             that is not C-contiguous, before any write"
            contiguous_calls (run test_contiguous_dst);
          prop ~count:200 ~examples:offset_examples
            "kernels at any offset their dtypes admit give the bits they give \
             at a buffer's start"
            (Gen.with_pp pp_offsets offset_cases)
            (run law_offsets);
          prop ~count:200
            ~examples:(mid_insides @ if on_cpu then large_insides else [])
            "Const and Coord store into an offset destination and nothing \
             around it"
            insides (run law_inside);
        ];
      group "sharing"
        [
          prop
            ~examples:
              (List.map
                 (fun p -> (p, 1))
                 ((one_element_pair :: erf_pair :: nan_pairs) @ large_view_pairs))
            "an operand read at the result's own index may be the destination"
            (Gen.pair pairs Gen.nat)
            (run (fun b ((Pair (x, _), _) as c) ->
                 covers (Case x);
                 law_in_place b c));
          prop
            ~examples:
              (List.map
                 (fun p -> (p, 1))
                 (one_element_pair :: large_view_pairs))
            "a map of several nodes may write over a load it reads"
            (Gen.pair pairs Gen.int)
            (run (fun b ((Pair (x, _), _) as c) ->
                 covers (Case x);
                 law_map_over_load b c));
          test
            "a load identical to a destination through another view is read \
             before it is written"
            (unit test_map_identical_view);
          test "a map of 100 outputs may write each into the load it reads"
            (unit test_wide_in_place);
          cases
            ~id:(fun (name, _, _, _, _) -> name)
            "a destination sharing bytes with a load it is not identical to, \
             or with another destination, is refused as Overlapping, before \
             any write"
            overlaps (run check_overlap);
        ];
      group "boxes"
        [
          prop
            "a result the contract fixes has the bits of its box computed alone"
            box_cases (run law_box);
        ];
      group "NaN results"
        [
          cases
            ~id:(fun (D.Any dt, _) -> D.name dt)
            "a computed NaN is signalling only where an operand is" quiet_bits
            (run check_quiet);
        ];
      group "check_dst"
        [
          cases ~id:fst
            "the door's refusal of a destination comes before check_dst's raise"
            door_rows (run check_door_first);
        ];
      group "loads"
        [
          prop ~count:150
            "a Padded load reads its operand padded with fill as pad says"
            paddings (run law_padded);
          test "a Padded load's map reads the coordinates of the padded shape"
            (unit test_padded_coordinate);
        ];
      group "computes_on"
        [
          test "is true of the device the kernels compute on"
            (unit test_computes_on);
        ];
      group "map"
        [
          prop ~count:100
            ~examples:
              (rare_programs
              @ List.map (fun p -> (p, 1, Plain)) large_view_pairs
              @ long_programs)
            "gives the bits of its nodes run one by one" split_cases
            (run (law_split ~declines:(fun _ _ -> false)));
          cases
            ~id:(Format.asprintf "%a" pp_wide)
            "too wide for a launch gives the bits of its nodes run one by one"
            (wide_calls 7) (run check_wide_map);
        ];
      group "Const"
        [
          prop ~examples:sub_byte_const "stores its bits at every index"
            (Gen.triple dtypes shape Gen.nat)
            (run law_const);
        ];
      group "Coord"
        [
          prop "stores each index along its axis as an int64"
            (Gen.pair (Gen.array ~size:(Gen.int_range 1 4) extent) Gen.nat)
            (run law_coord);
        ];
      group "Copy"
        [
          prop ~count:160 ~examples:large_views "stores each element's bits"
            any_size
            (run (fun b c ->
                 covers c;
                 covers_large c;
                 law_copy b c));
          test "keeps a NaN's sign, payload and signalling bit"
            (unit test_copy_nan_payloads);
        ];
      group "Cast"
        [
          prop ~count:160
            ~examples:(List.map (fun c -> (c, D.Any D.Float64)) large_views)
            "is the reference at each element" (cast_pair any_size)
            (run (fun b ((Case a as c), D.Any d) ->
                 covers c;
                 covers_large c;
                 cover "to the source's dtype" (D.equal (A.dtype a) d);
                 law_cast b (c, D.Any d)));
          cases
            ~id:(fun (D.Any d) -> D.name d)
            "of every code of a dtype of at most 16 bits is the reference"
            narrow (run test_every_code);
          test "of float32s about every rounding point is the reference"
            (unit (test_sweep ~dts:narrow_floats D.Float32 sweep32));
          test "of float64s about every tie is the reference"
            (unit (test_sweep ~dts:rounding D.Float64 sweep64));
          test "of every length to 64 is the reference" (unit test_lengths);
          test "of integers about every rounding point is the reference"
            (unit test_integer_ties);
          cases
            ~id:(fun (D.Any d) -> D.name d)
            "of every special is the reference" with_specials
            (run test_specials);
        ];
      group "Bitcast"
        [
          prop ~count:200 "reads each element's bits in a dtype of its width"
            (same_width case)
            (run (fun b ((c, _) as x) ->
                 covers c;
                 law_bitcast b x));
        ];
      group "Where"
        [
          prop ~count:150
            "selects the second operand's element where the condition is true, \
             the third's elsewhere"
            (Gen.pair pairs Gen.nat) (run law_where);
        ];
      group "Fma"
        [
          prop ~count:150
            "is the product plus the third rounded once, wrapping on integers"
            (Gen.pair fma_pairs Gen.nat)
            (run law_fma);
        ];
      group "exact kinds"
        [
          prop ~examples:[ sub_byte_case ]
            "a kind of one operand Prog defines exactly gives its result \
             rounded once"
            case
            (run (fun b (Case x as c) ->
                 covers c;
                 law_map1 ~kinds:(exact1 (D.Any (A.dtype x))) b c));
          prop
            ~examples:((sub_byte_pair :: divisor_pairs) @ threefry_pairs)
            "a kind of two operands Prog defines exactly gives its result \
             rounded once"
            pairs
            (run (fun b (Pair (x, _) as p) ->
                 covers (Case x);
                 law_map2 ~kinds:(exact2 (D.Any (A.dtype x))) b p));
        ];
      group "float bounds"
        [
          test "float64 kinds are within Prog's bounds of the goldens"
            (unit test_f64_kinds);
          cases ~id:Annex_f.id "Annex F's special values are exact" Annex_f.rows
            (run check_annex_f);
          cases ~id:Annex_f.id
            "results of no special value of Annex F's are within Prog's \
             bounds, a zero of its sign"
            Annex_f.rounded
            (run check_annex_f_rounded);
          test
            "float32 kinds are within Prog's bounds at Annex F's operands and \
             the switch points"
            (unit test_f32_points);
          prop ~count:4
            "float32 kinds are within Prog's bounds at every 2^20th bit pattern"
            Gen.(pair (int_range 0 (f32_step - 1)) int)
            (run law_f32_kinds);
        ];
      group "complex bounds"
        [
          test "complex transcendentals are within Prog's bounds of the goldens"
            (unit test_complex_kinds);
          prop ~count:4 "complex Mul is within 2u, Fdiv and Recip within 4u"
            Gen.int (run law_complex_bounds);
          prop ~count:4 "a NaN part makes both parts of Mul, Fdiv and Recip NaN"
            Gen.int (run law_complex_nans);
        ];
      group "r2c and c2r"
        [
          test
            "real transforms take one precision, and another pair is not Done \
             and writes nothing outside its destination"
            (unit test_real_transforms);
        ];
    ]

(* map_refusal reads no device: its laws run once. *)
let map_refusal =
  group "map_refusal"
    [
      cases
        ~id:(fun r -> r.name)
        "names a misfit of dtype, count or coordinate by its refusal"
        (List.filter (fun r -> r.refusal <> None) map_misfits)
        check_misfit_named;
      prop "is None for every map a loop computes" fitting_programs law_map_fits;
    ]

(* The memory of a map too wide for a launch

   nx_cuda.mli: such a map takes, until its work is done, at most a temporary
   for each node of its program, an element of the node's dtype at each index of
   the loop's axes along which its leaves vary. *)

let held_rows = 64
let held_cols = 4096
let held_keys = 130

(* [held_keys] values [key + i] held at once, summed from the last, plus the
   row: [n key + n (n - 1) / 2 + row] at [row, key]. The values and their sums
   vary along the keys alone. *)
let held_program =
  let n = held_keys in
  let i64 = D.Any D.Int64 in
  let value i = 3 + (2 * i) in
  let nodes =
    Array.concat
      [
        [| P.Coord 1; P.Coord 0 |];
        Array.concat
          (List.init n (fun i ->
               [|
                 P.Const (i64, P.bits D.Int64 (Int64.of_int i));
                 P.Op2 (Binary Add, 0, value i - 1);
               |]));
        [| P.Op2 (Binary Add, value (n - 1), value (n - 2)) |];
        Array.init (n - 2) (fun j ->
            P.Op2 (Binary Add, (2 * n) + 2 + j, value (n - 3 - j)));
      ]
  in
  let sum = Array.length nodes - 1 in
  P.v ~ins:[||]
    (Iarray.of_array (Array.append nodes [| P.Op2 (Binary Add, sum, 1) |]))
    ~outs:[| sum + 1 |]

(* The bytes nx_cuda.mli states the map of [held_program] takes at most: its [2n
   - 1] values and sums along the keys, its output along both axes, each from
   16-byte aligned bytes. They lie in a buffer of the call's own, which its
   device counts in its driver's units ({!Rig.allocation}). *)
let held_bytes =
  let aligned n = (n + 15) / 16 * 16 in
  (((2 * held_keys) - 1) * aligned (8 * held_cols))
  + aligned (8 * held_rows * held_cols)

(* [held_program]'s map on [b] into a fresh destination, and the map, called
   once to load the library's code on the device. *)
let held_map (b : Support.backend) =
  let module K = (val b.kernels) in
  let dst = A.create b.device D.Int64 [| held_rows; held_cols |] in
  let map () =
    equal answer A.Done (K.map held_program ~dsts:[| A.Any dst |] [||]);
    Rig.Buffer.wait (A.buffer dst) Read_write
  in
  map ();
  (dst, map)

let test_held_memory (b : Support.backend) () =
  let n = held_keys and d = b.device in
  let dst, map = held_map b in
  let budget = Rig.budget d in
  Rig.set_budget d (Support.settled d + Rig.allocation d held_bytes);
  Fun.protect ~finally:(fun () -> Rig.set_budget d budget) map;
  let want =
    Array.init (held_rows * held_cols) (fun i ->
        let row = i / held_cols and key = i mod held_cols in
        Int64.of_int ((n * key) + (n * (n - 1) / 2) + row))
  in
  equal (array int64) want (A.to_array (Support.host dst))

let test_held_loop (b : Support.backend) () =
  let _, map = held_map b in
  loop_held b ~calls:32 ~bound:(Rig.allocation b.device held_bytes) map

(* A map with a destination identical to its load, and a scan in nx.cuda's own
   order, of [wide_program]'s 130 int64 values held at once: each call is too
   wide for a launch whole and after each part it plans, and is tried again
   after each. nx_cuda.mli bounds its bytes by a temporary for each node and an
   array of the loop's shape, for the destination or the scan's terms; each node
   varies along both axes. They lie in a buffer of the call's own, counted in
   its driver's units ({!Rig.allocation}). *)

let retried =
  { mode = Held; dt = D.Any D.Int64; padded = false; aliased = true; seed = 7 }

let retried_shape = Iarray.of_list [ 16; 1024 ]

let retried_bytes =
  let p, _ = wide_program retried in
  (Iarray.length p.nodes + 1) * 8 * Iarray.fold_left ( * ) 1 retried_shape

(* [k p x y] on [b], [p] [wide_program]'s, [x] a fresh load of [retried_shape]
   and [y] a fresh destination, run once: the call, which waits for its work. *)
let retried_call (b : Support.backend) k =
  let p, _ = wide_program retried in
  let x = Support.on b (seeded D.Int64 retried_shape 7) in
  let y = A.create b.device D.Int64 retried_shape in
  let call () =
    equal answer A.Done (k p x y);
    Rig.Buffer.wait (A.buffer x) Read_write;
    Rig.Buffer.wait (A.buffer y) Read_write
  in
  call ();
  call

let test_retried_map (b : Support.backend) () =
  let module K = (val b.kernels) in
  loop_held b
    ~bound:(Rig.allocation b.device retried_bytes)
    (retried_call b (fun p x y ->
         let p = P.v ~ins:p.ins p.nodes ~outs:[| Iarray.get p.outs 0; 0 |] in
         K.map p ~dsts:[| A.Any x; A.Any y |] [| Nx_kernel.Plain x |]))

let test_retried_scan (b : Support.backend) () =
  let module K = (val b.kernels) in
  loop_held b
    ~bound:(Rig.allocation b.device retried_bytes)
    (retried_call b (fun p x y ->
         K.scan p ~axis:1 (Monoid Sum, 0) ~dsts:[| A.Any y |]
           [| Nx_kernel.Plain x |]))

(* nx_cuda.mli's and nx_amd.mli's memory of a call too wide for a launch. *)
let memory_laws (d : Support.declared) =
  let unit f () = f (Support.opened d) () in
  group (d.label ^ " memory")
    [
      test
        "a map too wide for a launch takes temporaries of the axes its values \
         vary along"
        (unit test_held_memory);
      test "maps too wide for a launch in a loop hold one call's temporaries"
        (unit test_held_loop);
      test
        "maps over their load tried again after each part, in a loop, hold one \
         call's temporaries"
        (unit test_retried_map);
      test
        "scans tried again after each part, in a loop, hold one call's \
         temporaries"
        (unit test_retried_scan);
    ]

(* rig.nv *)

(* A map of [2^31 + 512] elements into a uint16 destination of 4 GiB and 1 KiB,
   each element its coordinate modulo 2^16, as [Cast] stores it: 1024 elements
   read back from its start, across its byte 2^32 and from its end. *)
let past_4_gib (b : Support.backend) =
  let module K = (val b.kernels) in
  let n = (1 lsl 31) + 512 and m = 1024 in
  let y = A.create b.device D.Uint16 (Iarray.of_list [ n ]) in
  let p =
    P.v ~ins:(Iarray.of_list [])
      (Iarray.of_list [ P.Coord 0; P.Op1 (Cast, D.Any D.Uint16, 0) ])
      ~outs:(Iarray.of_list [ 1 ])
  in
  equal answer A.Done
    (K.map p ~dsts:(Iarray.of_list [ A.Any y ]) (Iarray.of_list []));
  let read at =
    let l =
      L.v ~offset:at ~strides:(Iarray.of_list [ 1 ]) (Iarray.of_list [ m ])
    in
    let x = Support.host (A.v D.Uint16 l (A.buffer y)) in
    equal
      ~msg:(Printf.sprintf "from %d" at)
      (array int)
      (Array.init m (fun i -> (at + i) land 0xffff))
      (A.to_array x);
    A.Any x
  in
  [| read 0; read ((1 lsl 31) - (m / 2)); read (n - m) |]

(* rig.nv loads the cubin CUDA loads, so a map through it has CUDA's bits: what
   it can get wrong, its launches, memory and order, shows in them. Each case's
   map runs through CUDA, then through rig.nv. A NaN agrees with any NaN:
   nx_cuda.mli leaves a NaN result's sign and payload unspecified. *)
let nv_laws (first : Support.declared) (nv : Support.declared) =
  let agree ?msg call =
    Elements.agree ?msg (Support.opened first) (Support.opened nv) call
  in
  group nv.label
    [
      group "bits are cuda's, any NaN for a NaN"
        [
          prop ~count:200 "of maps of programs, over padded loads too"
            split_cases (fun ((Pair (x, _) as pair), seed, mode) ->
              covers (Case x);
              cover "padded loads" (mode = Padded);
              cover "a program of hundreds of nodes" (mode = Long);
              agree (fun b -> split_map b (pair, seed, mode)));
          cases ~id:(Format.asprintf "%a" pp_wide)
            "of maps too wide for a launch" (wide_calls 7) (fun c ->
              agree ~msg:(Format.asprintf "%a" pp_wide c) (fun b ->
                  wide_map b c));
          test "of a map into a buffer past 4 GiB" (fun () -> agree past_4_gib);
        ];
    ]

let () =
  let declared =
    Support.declare Support.cpu
    :: List.filter_map Fun.id
         [ Support.declared_cuda; Support.declared_metal; Support.declared_amd ]
  in
  let memory =
    List.filter_map Fun.id
      [ Support.declared_cuda; Support.declared_nv; Support.declared_amd ]
  in
  let nv =
    match (Support.declared_cuda, Support.declared_nv) with
    | Some first, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (run "nx_kernel.kernels"
       ((map_refusal :: List.map laws declared)
       @ List.map memory_laws memory
       @ nv))
