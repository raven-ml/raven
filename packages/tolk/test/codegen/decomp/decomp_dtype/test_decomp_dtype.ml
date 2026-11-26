(* Tests of Tolk.Decomp_dtype: its conversions of the narrow floats are IEEE
   conversions that give the codes Dtype folds, the pass rewrites each kernel as
   its golden states, and the kernels it emulates compute what native ones do:
   narrow floats in float32, rounded once, and 64-bit integers exactly. *)

open Windtrap
open Tolk
open Dtypes
module PM = Ops.Pattern_matcher

let target = Result.get_ok (Helpers.Target.of_string "")
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* Targets and the pass *)

(* [lacking dts] is a target that has every data type but [dts]; [told names f]
   is [f ()] with the setting that names the data types to emulate. *)
let lacking dts = Renderer.v ~native:(fun dt -> not (List.mem dt dts)) target
let everything = lacking []
let told names f = Setting.context [ B (Setting.emulated_dtypes, names) ] f

(* [emulate on kernel] is [kernel] rewritten for the target [on] by the pass as
   code generation runs it, with the weak constants of its rules committed. *)
let emulate on kernel =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Decomp_dtype.ctx on)
    kernel
    (After_sources
       (PM.append Decomp_dtype.pm_dtype_decomps
          (PM.with_ctx Uop_weak.pm_commit_weak)))

let narrows =
  Dtype.[ Float16; Bfloat16; Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz ]

let longs = Dtype.[ Int64; Uint64 ]
let storage dt = if Dtype.bitsize dt = 8 then Dtype.Uint8 else Dtype.Uint16
let size dt = 1 lsl Dtype.bitsize dt
let eight_bit dt = Dtype.bitsize dt = 8

(* Codes and values *)

let as_int v =
  match (v :> Dtype.const) with
  | `Int z -> Bigint.to_int z
  | v -> failf "%a is not an integer" (Testable.pp const) v

let as_z v =
  match (v :> Dtype.const) with
  | `Int z -> z
  | v -> failf "%a is not an integer" (Testable.pp const) v

let as_float v =
  match (v :> Dtype.const) with
  | `Float x -> x
  | v -> failf "%a is not a float" (Testable.pp const) v

(* [encode dt v] is the code of [v] as a [dt] value, rounded as Dtype folds it;
   [decode dt c] is the value of the code [c]. *)
let encode dt v = as_int (Dtype.bitcast dt (storage dt) (Dtype.truncate dt v))

let decode dt c =
  as_float (Dtype.bitcast (storage dt) dt (`Int (Bigint.of_int c)))

(* [bits x] is the code of the float32 [x]; [f32 b] is the float32 of the code
   [b]. Both move a NaN's bits by hand ({!Dtype.bitcast}). *)
let bits x = as_int (Dtype.bitcast Float32 Uint32 (`Float x))
let f32 b = as_float (Dtype.bitcast Uint32 Float32 (`Int (Bigint.of_int b)))
let round32 x = as_float (Dtype.truncate Float32 (`Float x))
let is_nan_code dt c = Float.is_nan (decode dt c)
let is_finite_code dt c = Float.is_finite (decode dt c)

(* [greatest dt] is the code of [dt]'s greatest finite value, [top dt] that
   value, and [edge dt] that value plus half an ulp. *)
let greatest dt =
  let rec down c = if is_finite_code dt c then c else down (c - 1) in
  down ((size dt / 2) - 1)

let top dt = decode dt (greatest dt)
let edge dt = top dt +. ((top dt -. decode dt (greatest dt - 1)) /. 2.)

(* [below x] is the float32 before the positive float32 [x], and [above x] the
   one after it. *)
let below x = f32 (bits x - 1)
let above x = f32 (bits x + 1)

let hex =
  Testable.make ~pp:(fun ppf c -> Format.fprintf ppf "0x%x" c) ~equal:Int.equal

(* [same_nan dt want got] is [true] iff the codes [want] and [got] of [dt] are
   equal, or are both NaNs: a computed NaN's sign and payload are
   unspecified. *)
let same_nan dt want got =
  want = got || (is_nan_code dt want && is_nan_code dt got)

(* [code_of dt] compares codes of [dt] with {!same_nan}. *)
let code_of dt =
  Testable.make
    ~pp:(fun ppf c -> Format.fprintf ppf "0x%x (%h)" c (decode dt c))
    ~equal:(same_nan dt)

let code c = `Int (Bigint.of_int c)
let every_code dt = List.init (size dt) Fun.id
let nan_codes dt = List.filter (is_nan_code dt) (every_code dt)

(* Boundaries

   f2f is straight-line bit arithmetic: its result depends on a float32 through
   its exponent, which fixes how many mantissa bits the target keeps (fewer
   below its least normal, none past half its least subnormal), the bits it
   drops (none, below half, half, above half), the last bit it keeps (a tie
   rounds to even) and whether the kept bits are all ones (the carry into the
   exponent). Two float32s of one exponent alike in those run the same
   operations on other mantissas. So [float32_boundaries dt], every float32
   exponent with the kept bits 0, 1 or all ones and the dropped bits 0, 1, half
   - 1, half, half + 1 or all ones, of both signs, and the infinities, reach
   every case of every exponent: each subnormal shift and its clamp, ties of
   both parities, the carry, the overflow edge and saturation. *)

let mantissa dt = snd (Dtype.finfo dt)
let least_normal_exponent dt = 1 - Transcendental.exponent_bias dt

(* [dropped dt e] is how many of a float32's mantissa bits, its implicit one
   counted as the 24th, the narrowing to [dt] drops at the exponent field [e]:
   past 24 the value is below half [dt]'s least subnormal. *)
let dropped dt e =
  let unbiased = if e = 0 then -126 else e - 127 in
  min 25 (23 - mantissa dt + max 0 (least_normal_exponent dt - unbiased))

(* [patterns width] is 0, 1, half - 1, half, half + 1 and all ones over [width]
   bits. *)
let patterns width =
  if width <= 0 then [ 0 ]
  else
    let half = 1 lsl (width - 1) and ones = (1 lsl width) - 1 in
    List.filter
      (fun p -> p >= 0 && p <= ones)
      [ 0; 1; half - 1; half; half + 1; ones ]
    |> List.sort_uniq compare

let float32_boundaries ?(exponents = List.init 255 Fun.id) dt =
  List.concat_map
    (fun e ->
      let d = min 23 (dropped dt e) in
      let ones = (1 lsl (23 - d)) - 1 in
      List.sort_uniq compare [ 0; min 1 ones; ones ]
      |> List.concat_map (fun k ->
          List.concat_map
            (fun p ->
              let b = (e lsl 23) lor (k lsl d) lor p in
              [ b; b lor 0x80000000 ])
            (patterns d)))
    exponents
  @ [ 0x7f800000; 0xff800000 ]

(* [in_range dt] is the float32 exponent fields from two below [dt]'s least
   subnormal to two past its greatest value: where its narrowing changes. *)
let in_range dt =
  let lo = max 0 (least_normal_exponent dt - mantissa dt - 2 + 127) in
  let hi = min 254 (snd (Float.frexp (top dt)) + 1 + 127) in
  List.init (hi - lo + 1) (fun k -> lo + k)

(* [boundary_codes dt] is every finite code of an 8-bit float, and for a 16-bit
   one the codes of each exponent whose mantissa is 0, 1 or all ones, of both
   signs, the infinities included. *)
let boundary_codes dt =
  if eight_bit dt then
    List.filter (fun c -> not (is_nan_code dt c)) (every_code dt)
  else
    let m = mantissa dt and half = size dt / 2 in
    List.init (half lsr m) (fun e ->
        List.map (fun k -> (e lsl m) lor k) [ 0; 1; (1 lsl m) - 1 ])
    |> List.concat
    |> List.concat_map (fun c -> [ c; c + half ])
    |> List.filter (fun c -> not (is_nan_code dt c))

(* [ties dt] is the midpoints between the positive finite values of
   [boundary_codes dt] and the next codes, the greatest value plus half an ulp
   among them, of both signs. *)
let ties dt =
  List.filter_map
    (fun c ->
      let v = decode dt c in
      if c >= size dt / 2 || not (Float.is_finite v) then None
      else if c = greatest dt then Some (edge dt)
      else Some ((v +. decode dt (c + 1)) /. 2.))
    (boundary_codes dt)
  |> List.concat_map (fun m -> [ m; -.m ])

(* [neighbourhood dt] is each finite code's value one and two float32 ulps each
   side, and each tie one ulp each side, as float32 codes of both signs: with
   [float32_boundaries], an 8-bit float's narrowing at every code. *)
let neighbourhood dt =
  let around x ds = List.map (fun d -> bits (Float.abs x) + d) ds in
  List.concat_map
    (fun c ->
      let v = decode dt c in
      if Float.is_finite v then around v [ -2; -1; 1; 2 ] else [])
    (boundary_codes dt)
  @ List.concat_map (fun m -> around m [ -1; 0; 1 ]) (ties dt)
  |> List.filter (fun b -> b >= 0)
  |> List.concat_map (fun b -> [ b; b lor 0x80000000 ])

(* What the narrowing of the float32 code [b] to [dt] meets: a law's covers. *)
let regimes dt b =
  let e = (b lsr 23) land 0xff and m = b land 0x7fffff in
  let d = min 23 (dropped dt e) in
  let half = if d >= 1 then 1 lsl (d - 1) else -1 in
  let low = m land ((1 lsl d) - 1) and kept = m lsr d in
  let tie = low = half && dropped dt e <= 23 in
  let up = tie && kept land 1 = 1 in
  let all_ones = d < 23 && kept = (1 lsl (23 - d)) - 1 in
  let x = Float.abs (f32 b) in
  [
    ("a tie", tie);
    ("a tie rounded up, to the even code", up);
    ("a carry into the exponent", all_ones && (low > half || up));
    ( "a subnormal result",
      x > 0. && x < Float.ldexp 1. (least_normal_exponent dt) );
    ("past the greatest value", Float.is_finite x && x > top dt);
    ("an infinity", x = Float.infinity);
    ("below half the least subnormal", dropped dt e > 24 && x > 0.);
  ]

(* [float32_codes] draws a narrow float and a float32 of any finite exponent, as
   its bits. *)
let float32_codes =
  Gen.(
    with_pp
      (fun ppf (dt, b) ->
        Format.fprintf ppf "%s 0x%08x (%h)" (alias dt) b (f32 b))
      (let+ dt = of_list narrows
       and+ e = int_range 0 254
       and+ m = int_range 0 0x7fffff
       and+ negative = bool in
       (dt, (e lsl 23) lor m lor if negative then 0x80000000 else 0)))

(* f2f *)

let at_code u c = Interpreter.eval ~params:[ (0, code c) ] u

(* [widening dt] is the float32 code that f2f widens a [dt] code to; [narrowing
   ~sat dt] is the [dt] code that f2f narrows a float32 code to. *)
let widening dt =
  Ops.bitcast (Decomp_dtype.f2f (Shape.param 0 (storage dt)) dt Float32) Uint32

let narrowing ?sat dt =
  Decomp_dtype.f2f ?sat (Shape.param 0 Dtype.Uint32) Float32 dt

let narrowings = lazy (List.map (fun dt -> (dt, narrowing dt)) narrows)

let narrowed_code dt b =
  as_int (at_code (List.assoc dt (Lazy.force narrowings)) b)

(* [narrowed ?sat dt x] is the [dt] code that f2f narrows the float32 [x] to. *)
let narrowed ?sat dt x = as_int (at_code (narrowing ?sat dt) (bits x))
let infinity_of s = Float.copy_sign Float.infinity s

(* [overflows_at_the_edge ?sat dt] checks that narrowing keeps [dt]'s greatest
   value and the float32 below the edge, and gives the code of an infinity, a
   NaN in a format without one, above the edge. The edge is a tie, which rounds
   to the even code: the greatest one if it is even, as e4m3's is. *)
let overflows_at_the_edge ?sat dt =
  List.iter
    (fun s ->
      let x = narrowed ?sat dt
      and greatest = encode dt (`Float (s *. top dt)) in
      let infinity = encode dt (`Float (infinity_of s)) in
      let tie = if greatest land 1 = 0 then greatest else infinity in
      equal ~msg:"the greatest value" hex greatest (x (s *. top dt));
      equal ~msg:"below the edge" hex greatest (x (s *. below (edge dt)));
      equal ~msg:"the edge" (code_of dt) tie (x (s *. edge dt));
      equal ~msg:"above the edge" (code_of dt) infinity
        (x (s *. above (edge dt)));
      equal ~msg:"an infinity" (code_of dt) infinity (x (infinity_of s)))
    [ 1.; -1. ]

(* The quiet float32 NaNs, and the signalling ones, of both signs. *)
let quiet_nans = [ 0x7fc00000; 0x7fc12345; 0x7fe00001; 0x7fffffff ]
let signalling_nans = [ 0x7f800001; 0x7fa00000; 0x7fbfffff ]
let both_signs = List.concat_map (fun b -> [ b; b lor 0x80000000 ])

(* [is_quiet dt c] is [true] iff [c] is a quiet NaN code of [dt]: its mantissa's
   top bit is set, or it is the one NaN of an fnuz format. [is_quiet_bits b] is
   the same for the float32 code [b]. *)
let is_quiet dt c =
  let quiet = 1 lsl (mantissa dt - 1) in
  is_nan_code dt c && (List.mem dt Dtype.fp8_fnuz || c land quiet <> 0)

let is_nan_bits b = b land 0x7f800000 = 0x7f800000 && b land 0x7fffff <> 0
let is_quiet_bits b = is_nan_bits b && b land 0x400000 <> 0

(* [nan_bits ~quiet] draws the code of a float32 NaN of either sign, quiet or
   signalling. *)
let nan_bits ~quiet =
  Gen.(
    let+ negative = bool and+ payload = int_range 1 0x3fffff in
    0x7f800000
    lor (if quiet then 0x400000 else 0)
    lor payload
    lor if negative then 0x80000000 else 0)

let nan_case ~quiet =
  Gen.(
    with_pp
      (fun ppf (dt, b) -> Format.fprintf ppf "%s 0x%08x" (alias dt) b)
      (pair (of_list narrows) (nan_bits ~quiet)))

let f2f =
  group "f2f"
    [
      cases "widening every code is exact" ~id:alias narrows (fun dt ->
          let u = widening dt in
          List.iter
            (fun c ->
              let want = decode dt c in
              equal
                ~msg:(Printf.sprintf "0x%x (%h)" c want)
                hex (bits want)
                (as_int (at_code u c)))
            (List.filter (fun c -> not (is_nan_code dt c)) (every_code dt)));
      (* The examples reach every case of every type, the draws mantissas
         between them; the covers hold on the examples alone. *)
      prop ~count:1000 "narrowing a float32 rounds it as Dtype folds it"
        ~examples:
          (List.concat_map
             (fun dt ->
               (float32_boundaries dt
               @ if eight_bit dt then neighbourhood dt else [])
               |> List.sort_uniq compare
               |> List.map (fun b -> (dt, b)))
             narrows)
        float32_codes
        (fun (dt, b) ->
          cover (alias dt) true;
          List.iter (fun (label, holds) -> cover label holds) (regimes dt b);
          equal
            ~msg:(Printf.sprintf "%h" (f32 b))
            (code_of dt)
            (encode dt (`Float (f32 b)))
            (narrowed_code dt b));
      cases
        "without saturation, an 8-bit float overflows to its infinity or NaN"
        ~id:alias Dtype.fp8s (fun dt ->
          equal hex
            (encode dt (`Float Float.infinity))
            (as_int (at_code (narrowing ~sat:false dt) (bits 1e10))));
      cases
        "a 16-bit float is an infinity from its greatest value plus half an ulp"
        ~id:alias
        Dtype.[ Float16; Bfloat16 ]
        (fun dt ->
          overflows_at_the_edge dt;
          overflows_at_the_edge ~sat:false dt);
      cases
        "without saturation, an 8-bit float is an infinity or its NaN past its \
         greatest value plus half an ulp"
        ~id:alias Dtype.fp8s
        (overflows_at_the_edge ~sat:false);
      cases
        "with saturation, an 8-bit float's overflow is its greatest value, and \
         an infinity is its infinity or its NaN"
        ~id:alias Dtype.fp8s (fun dt ->
          List.iter
            (fun s ->
              let greatest = encode dt (`Float (s *. top dt)) in
              let x = narrowed dt in
              equal ~msg:"far above" hex greatest (x (s *. 1e10));
              equal ~msg:"the edge" hex greatest (x (s *. edge dt));
              equal ~msg:"just above" hex greatest (x (s *. above (top dt)));
              equal ~msg:"just below" hex greatest (x (s *. below (top dt)));
              equal ~msg:"an infinity" (code_of dt)
                (encode dt (`Float (infinity_of s)))
                (x (infinity_of s)))
            [ 1.; -1. ]);
      cases "widening keeps a NaN a NaN, and a quiet one quiet" ~id:alias
        narrows (fun dt ->
          let u = widening dt in
          List.iter
            (fun c ->
              let b = as_int (at_code u c) in
              let msg = Printf.sprintf "0x%x is 0x%08x" c b in
              satisfies ~msg ~claim:"a NaN" hex is_nan_bits b;
              if is_quiet dt c then
                satisfies ~msg ~claim:"a quiet NaN" hex is_quiet_bits b)
            (nan_codes dt));
      prop "narrowing keeps a quiet NaN quiet"
        ~examples:
          (List.concat_map
             (fun dt -> List.map (fun b -> (dt, b)) (both_signs quiet_nans))
             narrows)
        (nan_case ~quiet:true)
        (fun (dt, b) ->
          cover (alias dt) true;
          satisfies ~claim:"a quiet NaN" (code_of dt) (is_quiet dt)
            (narrowed_code dt b));
      prop "narrowing keeps a signalling NaN a NaN"
        ~examples:
          (List.concat_map
             (fun dt ->
               List.map (fun b -> (dt, b)) (both_signs signalling_nans))
             narrows)
        (nan_case ~quiet:false)
        (fun (dt, b) ->
          cover (alias dt) true;
          satisfies ~claim:"a NaN" (code_of dt) (is_nan_code dt)
            (narrowed_code dt b));
      (* e4m3's NaNs are 0x7f and 0xff; the input's bits are moved *)
      cases "narrowing an infinity or a NaN to e4m3 gives the NaN of its sign"
        ~id:(Printf.sprintf "0x%08x")
        [
          0x7f800000; 0xff800000; 0x7fc00000; 0xffc00000; 0x7f800001; 0xff800001;
        ] (fun b ->
          equal hex
            (if b land 0x80000000 <> 0 then 0xff else 0x7f)
            (narrowed_code Fp8e4m3 b));
      test "f2f converts only between a narrow float and a float32" (fun () ->
          let v = Shape.param 0 Dtype.Uint16 in
          rejects (fun () -> Decomp_dtype.f2f v Float16 Bfloat16);
          rejects (fun () -> Decomp_dtype.f2f v Float16 Float64);
          rejects (fun () ->
              Decomp_dtype.f2f (Shape.param 0 Uint32) Float32 Float32);
          rejects (fun () ->
              Decomp_dtype.f2f (Shape.param 0 Uint64) Float64 Float16);
          rejects (fun () -> Decomp_dtype.f2f v Int16 Float32);
          rejects (fun () ->
              Decomp_dtype.f2f (Shape.param 0 Uint32) Float32 Float64);
          rejects (fun () ->
              Decomp_dtype.f2f (Shape.param 0 Uint64) Float64 Float32));
    ]

(* exact *)

let committed =
  Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ]

let floats = narrows @ Dtype.[ Float32; Float64 ]
let precision dt = 1 + mantissa dt
let integer_of it = Gen.map as_z (value_of it)

(* A float, an integer type and a range within it, each end drawn about [-2^p]
   or [2^p] one time in three. *)
let exact_case =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp floats in
  let* it = of_list ~pp:Dtype.pp committed in
  let lo, hi = int_bounds it in
  let limit = Bigint.shift_left Bigint.one (precision dt) in
  let inside z = Bigint.(max lo (min hi z)) in
  let near s =
    map (fun d -> inside Bigint.((s * limit) + of_int d)) (int_range (-2) 2)
  in
  let point =
    frequency
      [ (1, integer_of it); (1, near Bigint.one); (1, near Bigint.minus_one) ]
  in
  let+ a = point and+ b = point in
  (dt, it, (Bigint.min a b, Bigint.max a b))

let exact =
  group "exact"
    [
      prop "exact x dt holds iff x's bounds lie within 2^p, p dt's precision"
        (Gen.with_pp
           (fun ppf (dt, it, (lo, hi)) ->
             Format.fprintf ppf "%a in [%a, %a] to %a" Dtype.pp it
               Bigint.pp_print lo Bigint.pp_print hi Dtype.pp dt)
           exact_case)
        (fun (dt, it, (lo, hi)) ->
          let limit = Bigint.shift_left Bigint.one (precision dt) in
          let within = Bigint.(geq lo (neg limit) && leq hi limit) in
          cover "within" within;
          cover "beyond" (not within);
          let x = Ops.variable ~dtype:it "x" (`Int lo) (`Int hi) in
          equal bool within (Decomp_dtype.exact x dt);
          (* and then the bounds cast to it exactly *)
          if within then
            List.iter
              (fun z ->
                equal ~msg:"a bound cast" float_exact (Bigint.to_float z)
                  (as_float (Dtype.truncate dt (`Int z))))
              [ lo; hi ]);
      cases "a bool is exact in every float" ~id:alias floats (fun dt ->
          equal bool true (Decomp_dtype.exact (Shape.param 0 Bool) dt));
      cases "exact refuses a type that is no float" ~id:alias
        Dtype.[ Int32; Bool; Uint64 ]
        (fun dt ->
          rejects (fun () -> Decomp_dtype.exact (Shape.param 0 Int32) dt));
    ]

(* narrow *)

(* [to_odd dt ~nearest ~beyond] is a value rounded to odd in [dt], given its
   nearest float [nearest] of [dt] and whether that lies beyond it, farther from
   zero: [nearest], or the float towards zero from it, with its last bit set. *)
let to_odd dt ~nearest ~beyond =
  let towards_zero =
    if not beyond then nearest
    else if dt = Dtype.Float32 then
      Float.copy_sign (below (Float.abs nearest)) nearest
    else Float.copy_sign (Float.pred (Float.abs nearest)) nearest
  in
  if dt = Dtype.Float32 then f32 (bits towards_zero lor 1)
  else Int64.float_of_bits (Int64.logor (Int64.bits_of_float towards_zero) 1L)

let narrow =
  let double = Shape.param 0 Dtype.Float64 in
  let whole it =
    let lo, hi = int_bounds it in
    Ops.variable ~dtype:it "x" (`Int lo) (`Int hi)
  in
  group "narrow"
    [
      prop "narrow x Float32 of a double is x rounded to odd"
        ~examples:
          [
            1. +. 0x1p-24 +. 0x1p-40;
            0x1p-150;
            0x1p-151;
            -0x1p-151;
            0x1p-149;
            0x1.fffffffffp127;
            0x1p128;
            1e300;
            -1e300;
            Float.infinity;
            Float.neg_infinity;
            0.;
            -0.;
            0x1.ffffffp-1;
          ]
        Gen.(with_pp (fun ppf x -> Format.fprintf ppf "%h" x) float)
        (fun x ->
          let nearest = round32 x in
          let inexact = nearest <> x in
          cover "inexact" inexact;
          let want =
            if not inexact then nearest
            else
              to_odd Float32 ~nearest ~beyond:(Float.abs nearest > Float.abs x)
          in
          equal ~msg:(Printf.sprintf "%h" x) float_exact want
            (as_float
               (Interpreter.eval
                  ~params:[ (0, `Float x) ]
                  (Decomp_dtype.narrow double Float32))));
      prop
        "narrow x of an integer of 32 or 64 bits is x rounded to odd, to \
         Float32 or Float64"
        ~examples:
          (List.map
             (fun (it, z) -> (it, Bigint.of_string z))
             Dtype.
               [
                 (Int32, "16842753");
                 (Int32, "2147483647");
                 (Int32, "-2147483648");
                 (Uint32, "4294967295");
                 (Int64, "1103806595073");
                 (Int64, "9223372036854775807");
                 (Int64, "-9007199791611905");
                 (Uint64, "18446744073709551615");
               ])
        Gen.(
          with_pp
            (fun ppf (it, z) ->
              Format.fprintf ppf "%a %a" Dtype.pp it Bigint.pp_print z)
            (let* it =
               of_list ~pp:Dtype.pp Dtype.[ Int32; Uint32; Int64; Uint64 ]
             in
             map (fun z -> (it, z)) (integer_of it)))
        (fun (it, z) ->
          let to_ = if Dtype.bitsize it = 64 then Dtype.Float64 else Float32 in
          cover (alias it) true;
          let nearest = as_float (Dtype.truncate to_ (`Int z)) in
          let rounded = Bigint.of_float nearest in
          let inexact = not (Bigint.equal rounded z) in
          cover "inexact" inexact;
          let want =
            if not inexact then nearest
            else to_odd to_ ~nearest ~beyond:Bigint.(gt (abs rounded) (abs z))
          in
          equal float_exact want
            (as_float
               (Interpreter.eval
                  ~vars:[ ("x", `Int z) ]
                  (Decomp_dtype.narrow (whole it) to_))));
      cases "narrow is a plain cast where nothing is dropped"
        ~id:(fun (x, dt) ->
          Format.asprintf "%a to %a" Dtype.pp (Ops.dtype x) Dtype.pp dt)
        [
          (Shape.param 0 Dtype.Float16, Dtype.Float32);
          ( Ops.variable ~dtype:Int32 "x"
              (`Int (Bigint.of_int (-1000)))
              (`Int (Bigint.of_int 1000)),
            Dtype.Float32 );
          (Shape.param 0 Dtype.Int32, Dtype.Float64);
          (Shape.param 0 Dtype.Int64, Dtype.Float16);
        ]
        (fun (x, dt) ->
          equal Uops.uop (Ops.cast x dt) (Decomp_dtype.narrow x dt));
    ]

(* computes and emulates *)

let names dts =
  match dts with [] -> "none" | l -> String.concat "," (List.map alias l)

let computes_group =
  group "computes"
    [
      cases "a target computes what it supports and what the pass emulates"
        ~id:names
        Dtype.
          [
            [];
            [ Float64 ];
            [ Float16; Int64; Fp8e4m3 ];
            [ Float64; Uint64; Bfloat16 ];
          ]
        (fun lacks ->
          equal (list dtype)
            (List.filter
               (fun dt ->
                 List.mem dt (narrows @ longs) || not (List.mem dt lacks))
               Dtype.all)
            (Decomp_dtype.computes (lacking lacks)));
    ]

let emulates_group =
  group "emulates"
    [
      cases
        "a type is emulated where the target lacks it or the setting names it, \
         an unsigned 64-bit integer as the signed one"
        ~id:(fun (lacks, named, dt, _) ->
          Printf.sprintf "lacking %s, named [%s]: %s" (names lacks)
            (String.concat "," named) (alias dt))
        Dtype.
          [
            ([ Float16 ], [], Float16, true);
            ([ Float16 ], [], Bfloat16, false);
            ([], [ "half" ], Float16, true);
            ([], [ "half" ], Float32, false);
            ([], [], Fp8e4m3, false);
            ([ Int64 ], [], Uint64, true);
            ([], [ "long" ], Uint64, true);
            ([], [ "ulong" ], Uint64, false);
            ([ Uint64 ], [], Uint64, false);
          ]
        (fun (lacks, named, dt, want) ->
          equal bool want
            (told named (fun () -> Decomp_dtype.emulates (lacking lacks) dt)));
      test "the setting is read at emulates r" (fun () ->
          let before = Decomp_dtype.emulates everything in
          let named =
            told [ "half" ] (fun () -> Decomp_dtype.emulates everything)
          in
          equal (pair bool bool) (false, true)
            (told [ "half" ] (fun () -> before Float16), named Float16));
    ]

(* Kernels *)

(* [kernel ins out n f] is the kernel that stores, for each [r] below [n], [f]
   of the elements [r] of the storage of [ins], in slots 0, 1, and so on, at
   element [r] of the [out] storage in the next slot. *)
let kernel ins out n f =
  let r = Ops.range ~axis_type:Weak (Int n) [ 0 ] in
  let at slot dt = Ops.index (Shape.param ~shape:[ Int n ] slot dt) [ r ] in
  let xs = List.mapi (fun slot dt -> Ops.load (at slot dt) []) ins in
  Ops.sink [ Ops.end_ (Ops.store (at (List.length ins) out) (f xs)) [ r ] ]

let cast_to dt xs = Ops.cast (List.hd xs) dt

(* [outputs k inputs] is what the kernel [k] writes to its output given the
   storage [inputs], in element order; [written ~on k inputs] is that of [k]
   emulated on the target [on]. *)
let outputs k inputs =
  let out = List.length inputs in
  Interpreter.writes
    ~buffers:(List.mapi (fun slot a -> (slot, Array.of_list a)) inputs)
    k
  |> List.filter_map (fun (slot, _, v) -> if slot = out then Some v else None)

let on_narrows = lacking narrows
let written ~on k inputs = outputs (emulate on k) inputs

(* [cast_one from to_ x] is the code that the emulated cast of the [from] value
   [x] to the narrow float [to_] writes; its kernel is emulated once. *)
let cast_one =
  let emulated = Hashtbl.create 64 in
  fun from to_ x ->
    let k =
      match Hashtbl.find_opt emulated (from, to_) with
      | Some k -> k
      | None ->
          let k = emulate on_narrows (kernel [ from ] to_ 1 (cast_to to_)) in
          Hashtbl.add emulated (from, to_) k;
          k
    in
    match outputs k [ [ x ] ] with
    | [ v ] -> as_int v
    | vs -> failf "%d values written" (List.length vs)

(* Doubles near ties: of every boundary tie, the tie itself, one ulp each side,
   and scaled by 1 +- 2^-40, which a float32 rounds onto the tie; doubles just
   below powers of two that a float32 rounds up across its binade; below half
   float32's least subnormal; past float32's greatest value, at 2^128, 1e300. *)
let double_examples dt =
  let ulps m =
    List.map
      (fun d -> Int64.float_of_bits (Int64.add (Int64.bits_of_float m) d))
      [ -1L; 1L ]
  in
  List.concat_map
    (fun m -> m :: (m *. (1. +. 0x1p-40)) :: (m *. (1. -. 0x1p-40)) :: ulps m)
    (ties dt)
  @ List.concat_map
      (fun x -> [ x; -.x ])
      ([
         0x1p-151;
         0x1.8p-151;
         0x1p-152;
         0x1.fffffffffp127;
         0x1p128;
         1e300;
         Float.infinity;
       ]
      @ List.map
          (fun e -> Float.ldexp (1. -. 0x1p-30) e)
          (List.filter
             (fun e -> e mod 3 = 0)
             (List.init 40 (fun k -> snd (Float.frexp (top dt)) - k))))

(* A narrow float, a tie between two of its values, and a double offset from it
   by none, one or 2^29 ulps of a double, or by drawn low bits: closer to the
   tie than a float32 ulp, where rounding twice would show. *)
let double_case =
  let open Gen in
  let* dt = of_list narrows in
  let* c = int_range 0 (greatest dt - 1) in
  let m = (decode dt c +. decode dt (c + 1)) /. 2. in
  let* offset =
    frequency
      [
        (1, constant 0L);
        (2, int64_range (-2L) 2L);
        ( 1,
          of_list [ Int64.shift_left 1L 29; Int64.neg (Int64.shift_left 1L 29) ]
        );
        (2, int64_range (-0x1fffffffL) 0x1fffffffL);
      ]
  in
  let+ negative = bool in
  let x = Int64.float_of_bits (Int64.add (Int64.bits_of_float m) offset) in
  (dt, if negative then -.x else x)

let pp_double ppf (dt, x) = Format.fprintf ppf "%s %h" (alias dt) x

(* [near_a_tie dt x] is [true] iff the double [x] lies strictly between two
   float32s that round to different codes of [dt]. *)
let near_a_tie dt x =
  let x = Float.abs x in
  let r = round32 x in
  r <> x && Float.is_finite r
  &&
  let lo, hi = if r < x then (r, above r) else (below r, r) in
  encode dt (`Float lo) <> encode dt (`Float hi)

(* A narrow float, a 32 or 64-bit, or narrower, integer type, and a value of it
   by its bounds, or about an integer tie of the narrow float. *)
let integer_case =
  let open Gen in
  let* dt = of_list narrows in
  let* from = of_list ~pp:Dtype.pp committed in
  let lo, hi = int_bounds from in
  let* c = int_range 0 (greatest dt - 1) in
  let m = (decode dt c +. decode dt (c + 1)) /. 2. in
  let* near =
    if Float.is_integer m && m >= 1. then
      map (fun d -> Bigint.(of_float m + of_int d)) (int_range (-1) 1)
    else integer_of from
  in
  let* negative = bool in
  let near = if negative then Bigint.neg near else near in
  let+ z = frequency [ (1, integer_of from); (2, constant near) ] in
  (dt, from, Bigint.(max lo (min hi z)))

let integer_examples dt from =
  let lo, hi = int_bounds from in
  let at_ties =
    List.concat_map
      (fun m ->
        if Float.is_integer m && Float.abs m >= 1. then
          List.map (fun d -> Bigint.(of_float m + of_int d)) [ -1; 0; 1 ]
        else [])
      (ties dt)
  in
  let every =
    if Dtype.bitsize from = 8 then
      List.init
        (Bigint.to_int Bigint.(hi - lo) + 1)
        (fun k -> Bigint.(lo + of_int k))
    else []
  in
  Bigint.[ lo; succ lo; pred hi; hi; zero; one ] @ at_ties @ every
  |> List.filter (fun z -> Bigint.(leq lo z && leq z hi))
  |> List.sort_uniq Bigint.compare
  |> List.map (fun z -> (dt, from, z))

let pairs =
  List.concat_map
    (fun fr ->
      List.filter_map
        (fun to_ -> if fr = to_ then None else Some (fr, to_))
        narrows)
    narrows

(* The codes of [fr] a cast to [to_] is checked at: every code of an 8-bit
   float; a 16-bit one's boundary codes, and its codes nearest each tie of [to_]
   and their neighbours, where a second rounding would show. *)
let source_codes fr to_ =
  if eight_bit fr then every_code fr
  else
    boundary_codes fr
    @ List.concat_map
        (fun m ->
          let c = encode fr (`Float m) in
          List.filter (fun c -> c >= 0 && c < size fr) [ c - 1; c; c + 1 ])
        (ties to_)
    |> List.sort_uniq compare

(* [undefined_casts ~on k inputs] is the elements at which the kernel [k],
   emulated on [on], converts a float to an integer type that cannot hold it,
   which C leaves undefined; given the storage [inputs]. *)
let undefined_casts ~on k inputs =
  let nodes = Ops.toposort ~calls:Enter (emulate on k) in
  let r = List.find (fun u -> Ops.op u = Range) nodes in
  let casts =
    List.filter
      (fun u ->
        Ops.op u = Cast
        && Dtype.is_int (Ops.dtype u)
        && Dtype.is_float (Ops.dtype (Ops.nth u 0)))
      nodes
  in
  let buffers = List.mapi (fun slot a -> (slot, Array.of_list a)) inputs in
  let undefined vars c =
    match Interpreter.eval ~vars ~buffers (Ops.nth c 0) with
    | `Float x ->
        let lo, hi = int_bounds (Ops.dtype c) in
        Float.is_nan x
        || Float.trunc x < Bigint.to_float lo
        || Float.trunc x > Bigint.to_float hi
    | _ -> false
  in
  List.filter
    (fun i ->
      List.exists
        (undefined [ (Option.get (Interpreter.name r), code i) ])
        casts)
    (List.init (List.length (List.hd inputs)) Fun.id)

let conversions =
  [
    prop "an emulated cast of a float32 rounds it as Dtype folds it"
      ~examples:
        (List.concat_map
           (fun dt ->
             List.map
               (fun b -> (dt, b))
               (float32_boundaries ~exponents:(in_range dt) dt))
           narrows)
      float32_codes
      (fun (dt, b) ->
        cover (alias dt) true;
        List.iter (fun (label, holds) -> cover label holds) (regimes dt b);
        equal
          ~msg:(Printf.sprintf "%h" (f32 b))
          (code_of dt)
          (encode dt (`Float (f32 b)))
          (cast_one Float32 dt (`Float (f32 b))));
    prop "an emulated cast of a double rounds it once"
      ~examples:
        (List.concat_map
           (fun dt -> List.map (fun x -> (dt, x)) (double_examples dt))
           narrows)
      (Gen.with_pp pp_double double_case)
      (fun (dt, x) ->
        cover (alias dt) true;
        cover "between two float32s next to a tie" (near_a_tie dt x);
        equal ~msg:(Printf.sprintf "%h" x) (code_of dt)
          (encode dt (`Float x))
          (cast_one Float64 dt (`Float x)));
    prop "an emulated cast of an integer rounds it once"
      ~examples:
        (List.concat_map
           (fun dt -> List.concat_map (integer_examples dt) committed)
           narrows)
      (Gen.with_pp
         (fun ppf (dt, from, z) ->
           Format.fprintf ppf "%s from %a %a" (alias dt) Dtype.pp from
             Bigint.pp_print z)
         integer_case)
      (fun (dt, from, z) ->
        cover (alias dt) true;
        cover (alias from) true;
        cover "next to a tie"
          (encode dt (`Int (Bigint.pred z)) <> encode dt (`Int (Bigint.succ z)));
        equal (code_of dt) (encode dt (`Int z)) (cast_one from dt (`Int z)));
    cases
      "an emulated cast of an integer converts no float to an integer that \
       cannot hold it"
      ~id:alias narrows (fun dt ->
        List.iter
          (fun from ->
            let values =
              List.map (fun (_, _, z) -> `Int z) (integer_examples dt from)
            in
            let k = kernel [ from ] dt (List.length values) (cast_to dt) in
            equal ~msg:(alias from) (list value) []
              (List.map (List.nth values)
                 (undefined_casts ~on:on_narrows k [ values ])))
          Dtype.[ Int32; Uint32; Int64; Uint64 ]);
    prop "an emulated cast between two narrow floats rounds once"
      ~examples:
        (List.concat_map
           (fun (fr, to_) ->
             List.map (fun c -> (fr, to_, c)) (source_codes fr to_))
           pairs)
      Gen.(
        with_pp
          (fun ppf (fr, to_, c) ->
            Format.fprintf ppf "%s 0x%x (%h) to %s" (alias fr) c (decode fr c)
              (alias to_))
          (let* fr, to_ = of_list pairs in
           map (fun c -> (fr, to_, c)) (int_range 0 (size fr - 1))))
      (fun (fr, to_, c) ->
        let v = decode fr c in
        cover "next to a tie of the target"
          (Float.is_finite v
          && encode to_ (`Float (Float.pred v))
             <> encode to_ (`Float (Float.succ v)));
        equal
          ~msg:(Printf.sprintf "%s to %s" (alias fr) (alias to_))
          (code_of to_)
          (encode to_ (`Float v))
          (cast_one fr to_ (code c)));
    cases "an emulated cast to a float32 is exact" ~id:alias narrows (fun dt ->
        let cs = List.filter (is_finite_code dt) (boundary_codes dt) in
        let k = kernel [ dt ] Float32 (List.length cs) (cast_to Float32) in
        equal (list hex)
          (List.map (fun c -> bits (decode dt c)) cs)
          (List.map
             (fun v -> bits (as_float v))
             (written ~on:on_narrows k [ List.map code cs ])));
  ]

(* Arithmetic and moves *)

let arithmetic =
  [
    ("add", Op.Add, Ops.O.( + ));
    ("sub", Op.Sub, Ops.O.( - ));
    ("mul", Op.Mul, Ops.O.( * ));
    ("max", Op.Max, Ops.maximum);
  ]

(* [float32 op a b] is [op] of the floats [a] and [b] rounded to float32, as
   Ops.exec_alu computes it, one operation at a time. *)
let float32 op a b = as_float (Ops.exec_alu op Float32 [ `Float a; `Float b ])

(* The emulated kernels of each operation on each narrow float, made once. *)
let binaries =
  lazy
    (List.concat_map
       (fun dt ->
         List.map
           (fun (name, _, f) ->
             let k =
               kernel [ dt; dt ] dt 1 (fun xs ->
                   f (List.nth xs 0) (List.nth xs 1))
             in
             ((dt, name), emulate on_narrows k))
           arithmetic)
       narrows)

let operand dt =
  Gen.such_that
    (fun c -> not (is_nan_code dt c))
    (Gen.int_range 0 (size dt - 1))

(* A narrow float's extremes: its infinity, or greatest value where it has none,
   and its least subnormal, of both signs. *)
let extremes dt =
  let inf = encode dt (`Float Float.infinity) in
  let big = if is_nan_code dt inf then greatest dt else inf in
  [ big; big + (size dt / 2); 1; 1 + (size dt / 2) ]

let computes_in_float32 =
  prop ~count:600
    "emulated arithmetic computes in float32 and rounds once, at the store"
    ~examples:
      (List.concat_map
         (fun dt ->
           List.concat_map
             (fun (name, _, _) ->
               List.concat_map
                 (fun a -> List.map (fun b -> (dt, name, a, b)) (extremes dt))
                 (extremes dt))
             arithmetic)
         narrows)
    Gen.(
      with_pp
        (fun ppf (dt, name, a, b) ->
          Format.fprintf ppf "%s %s 0x%x 0x%x" (alias dt) name a b)
        (let* dt = of_list narrows in
         let+ name, _, _ = of_list arithmetic
         and+ a = operand dt
         and+ b = operand dt in
         (dt, name, a, b)))
    (fun (dt, name, a, b) ->
      cover (alias dt) true;
      cover name true;
      let op =
        List.assoc name (List.map (fun (n, o, _) -> (n, o)) arithmetic)
      in
      let x = float32 op (decode dt a) (decode dt b) in
      cover "a NaN" (Float.is_nan x);
      cover "past the greatest value" (Float.abs x > top dt);
      let k = List.assoc (dt, name) (Lazy.force binaries) in
      equal
        (list (code_of dt))
        [ encode dt (`Float x) ]
        (List.map as_int (outputs k [ [ code a ]; [ code b ] ])))

(* [(a * b) + c] stored, and cast to a float32: the kernels of each narrow
   float, made once. *)
let product_sums =
  lazy
    (List.map
       (fun dt ->
         let f xs = Ops.O.((List.nth xs 0 * List.nth xs 1) + List.nth xs 2) in
         ( dt,
           ( emulate on_narrows (kernel [ dt; dt; dt ] dt 1 f),
             emulate on_narrows
               (kernel [ dt; dt; dt ] Float32 1 (fun xs ->
                    Ops.cast (f xs) Float32)) ) ))
       narrows)

let reads_unrounded =
  prop
    ~examples:(List.map (fun dt -> (dt, 1, 1, 1)) narrows)
    "a later operation, or a cast to another type, reads an emulated result \
     unrounded"
    Gen.(
      with_pp
        (fun ppf (dt, a, b, c) ->
          Format.fprintf ppf "%s 0x%x * 0x%x + 0x%x" (alias dt) a b c)
        (let* dt = of_list narrows in
         let+ a = operand dt and+ b = operand dt and+ c = operand dt in
         (dt, a, b, c)))
    (fun (dt, a, b, c) ->
      cover (alias dt) true;
      (* the product rounded to float32, then the sum: never one fused step *)
      let product = float32 Mul (decode dt a) (decode dt b) in
      let sum = float32 Add product (decode dt c) in
      cover "the product unheld by the narrow float"
        (Float.is_finite product
        && decode dt (encode dt (`Float product)) <> product);
      let stored, cast = List.assoc dt (Lazy.force product_sums) in
      let inputs = [ [ code a ]; [ code b ]; [ code c ] ] in
      equal ~msg:"stored"
        (list (code_of dt))
        [ encode dt (`Float sum) ]
        (List.map as_int (outputs stored inputs));
      equal ~msg:"cast" (list value) [ `Float sum ] (outputs cast inputs))

let comparisons =
  lazy
    (List.map
       (fun dt ->
         ( dt,
           emulate on_narrows
             (kernel [ dt; dt ] Bool 1 (fun xs ->
                  Ops.O.(List.nth xs 0 < List.nth xs 1))) ))
       narrows)

let compares_values =
  prop "an emulated comparison of narrow floats compares their values"
    ~examples:
      (List.concat_map
         (fun dt ->
           let nan = List.hd (nan_codes dt) and half = size dt / 2 in
           [
             (dt, 0, half); (dt, half, 0); (dt, 5, 5); (dt, nan, 5); (dt, 5, nan);
           ])
         narrows)
    Gen.(
      with_pp
        (fun ppf (dt, a, b) ->
          Format.fprintf ppf "%s 0x%x < 0x%x" (alias dt) a b)
        (let* dt = of_list narrows in
         let code = int_range 0 (size dt - 1) in
         let+ a = code and+ b = frequency [ (3, code); (1, constant (-1)) ] in
         (dt, a, if b < 0 then a else b)))
    (fun (dt, a, b) ->
      cover (alias dt) true;
      cover "equal" (a = b);
      cover "a NaN" (is_nan_code dt a || is_nan_code dt b);
      equal (list value)
        [ `Bool (decode dt a < decode dt b) ]
        (outputs
           (List.assoc dt (Lazy.force comparisons))
           [ [ code a ]; [ code b ] ]))

let copies_every_code =
  cases "an emulated copy keeps every code, a signalling NaN's included"
    ~id:alias narrows (fun dt ->
      let cs = every_code dt in
      let k = kernel [ dt ] dt (List.length cs) List.hd in
      equal (list hex) cs
        (List.map as_int (written ~on:on_narrows k [ List.map code cs ])))

let selects_every_code =
  cases "an emulated selection keeps every code, a signalling NaN's included"
    ~id:alias narrows (fun dt ->
      let cs = every_code dt in
      let others = List.rev cs in
      let conds = List.mapi (fun i _ -> i mod 2 = 0) cs in
      let k =
        kernel [ Bool; dt; dt ] dt (List.length cs) (fun xs ->
            Ops.where (List.nth xs 0) (List.nth xs 1) (List.nth xs 2))
      in
      equal (list hex)
        (List.map2
           (fun c (x, y) -> if c then x else y)
           conds (List.combine cs others))
        (List.map as_int
           (written ~on:on_narrows k
              [
                List.map (fun c -> `Bool c) conds;
                List.map code cs;
                List.map code others;
              ])))

(* A bit reinterpretation of a narrow float reads its storage, every code kept,
   a signalling NaN's included. *)
let reinterprets =
  cases "a bit reinterpretation reads a narrow float's storage" ~id:alias
    narrows (fun dt ->
      let st = storage dt in
      let cs = List.sort_uniq compare (boundary_codes dt @ nan_codes dt) in
      let read =
        kernel [ dt ] st (List.length cs) (fun xs ->
            Ops.bitcast (List.hd xs) st)
      in
      equal (list hex) cs
        (List.map as_int (written ~on:on_narrows read [ List.map code cs ])))

let computed_then_reinterpreted =
  cases "a narrow float computed, then reinterpreted, is the code of the result"
    ~id:alias narrows (fun dt ->
      let st = storage dt in
      let cs = List.filter (is_finite_code dt) (boundary_codes dt) in
      let k =
        kernel [ dt ] st (List.length cs) (fun xs ->
            Ops.bitcast Ops.O.(List.hd xs + float 1.) st)
      in
      equal
        (list (code_of dt))
        (List.map
           (fun c -> encode dt (`Float (float32 Add (decode dt c) 1.)))
           cs)
        (List.map as_int (written ~on:on_narrows k [ List.map code cs ])))

(* Float32 values within every narrow float's range, rounded once: today's bug
   guards. *)
let in_every_range = [ 13.7; 1.3; 0.1; 3.3; -2.7; 0.0123; 4113. ]

let round_trips =
  lazy
    (List.map
       (fun dt ->
         ( dt,
           emulate on_narrows
             (kernel [ Float32 ] Float32 1 (fun xs ->
                  Ops.cast (Ops.cast (List.hd xs) dt) Float32)) ))
       narrows)

let round_trip =
  prop "a cast to an emulated float and back is the value rounded once"
    ~examples:
      (List.concat_map
         (fun dt -> List.map (fun x -> (dt, bits x)) in_every_range)
         narrows)
    (Gen.with_pp
       (fun ppf (dt, b) -> Format.fprintf ppf "%s %h" (alias dt) (f32 b))
       float32_codes)
    (fun (dt, b) ->
      cover (alias dt) true;
      let x = f32 b in
      equal value
        (Dtype.truncate dt (`Float x))
        (List.hd
           (outputs (List.assoc dt (Lazy.force round_trips)) [ [ `Float x ] ])))

let nans_in_kernels =
  [
    cases "an emulated cast of a quiet float32 or double NaN is a quiet NaN"
      ~id:alias narrows (fun dt ->
        let values =
          List.map (fun b -> `Float (f32 b)) (both_signs quiet_nans)
        in
        List.iter
          (fun from ->
            let k = kernel [ from ] dt (List.length values) (cast_to dt) in
            List.iter
              (fun got ->
                satisfies ~msg:(alias from) ~claim:"a quiet NaN" (code_of dt)
                  (is_quiet dt) (as_int got))
              (written ~on:on_narrows k [ values ]))
          [ Dtype.Float32; Float64 ]);
    cases "an emulated cast of a NaN code to a float32 is a NaN" ~id:alias
      narrows (fun dt ->
        let cs = nan_codes dt in
        let k = kernel [ dt ] Float32 (List.length cs) (cast_to Float32) in
        List.iter
          (fun got ->
            satisfies ~claim:"a NaN" Dtypes.value
              (fun v -> Float.is_nan (as_float v))
              got)
          (written ~on:on_narrows k [ List.map code cs ]));
  ]

(* Narrow float conversions are IEEE conversions *)

(* [casts ~from dt table] checks that the emulated cast of each value of
   [table], of type [from], to [dt] writes the code beside it. *)
let casts ?(from = Dtype.Float32) dt table =
  let values = List.map fst table in
  let k = kernel [ from ] dt (List.length values) (cast_to dt) in
  equal
    (list (pair value hex))
    table
    (List.combine values
       (List.map as_int (written ~on:on_narrows k [ values ])))

let integer s = `Int (Bigint.of_string s)

let least_subnormals =
  Dtype.
    [
      (Float16, 0x1p-24);
      (Bfloat16, 0x1p-133);
      (Fp8e4m3, 0x1p-9);
      (Fp8e5m2, 0x1p-16);
      (Fp8e4m3fnuz, 0x1p-10);
      (Fp8e5m2fnuz, 0x1p-17);
    ]

let ieee_facets =
  [
    cases "an emulated narrow float keeps its subnormals, both ways"
      ~id:(fun (dt, _) -> alias dt)
      least_subnormals
      (fun (dt, least) ->
        casts dt
          [
            (`Float least, 0x01);
            (`Float (least *. 0.75), 0x01);
            (`Float (least *. 0.5), 0x00);
            (`Float (least *. 1.5), 0x02);
            (`Float (-.least), (size dt / 2) + 0x01);
          ];
        let k = kernel [ dt ] Float32 1 (cast_to Float32) in
        equal ~msg:"widened" value (`Float least)
          (List.hd (written ~on:on_narrows k [ [ code 1 ] ])));
    cases
      "a 16-bit float overflows from its greatest value plus half an ulp, the \
       tie to infinity"
      ~id:(fun (dt, _) -> alias dt)
      Dtype.
        [
          ( Float16,
            [
              (`Float 65504., 0x7bff);
              (`Float (below 65520.), 0x7bff);
              (`Float 65520., 0x7c00);
              (`Float (-65520.), 0xfc00);
            ] );
          ( Bfloat16,
            [
              (`Float (below (0x1p128 -. 0x1p119)), 0x7f7f);
              (`Float (0x1p128 -. 0x1p119), 0x7f80);
              (`Float (-.(0x1p128 -. 0x1p119)), 0xff80);
            ] );
        ]
      (fun (dt, table) -> casts dt table);
    (* An infinity narrows to the NaN of its sign in e4m3, whose NaNs are 0x7f
       and 0xff; an fnuz format has the one NaN 0x80. *)
    cases
      "an infinity stays one in e5m2 and is the NaN of e4m3 and the fnuz \
       formats, and a finite overflow saturates"
      ~id:(fun (dt, _) -> alias dt)
      Dtype.
        [
          (Fp8e4m3, [ 0x7f; 0xff; 0x7e; 0xfe ]);
          (Fp8e5m2, [ 0x7c; 0xfc; 0x7b; 0xfb ]);
          (Fp8e4m3fnuz, [ 0x80; 0x80; 0x7f; 0xff ]);
          (Fp8e5m2fnuz, [ 0x80; 0x80; 0x7f; 0xff ]);
        ]
      (fun (dt, codes) ->
        casts dt
          (List.combine
             [
               `Float Float.infinity;
               `Float Float.neg_infinity;
               `Float 1e10;
               `Float (-1e10);
             ]
             codes));
    cases "a double, or an integer more precise than a float32, rounds once"
      ~id:(fun (from, dt, _) -> alias from ^ " to " ^ alias dt)
      Dtype.
        [
          (Float64, Bfloat16, [ (`Float (1. +. 0x1p-8 +. 0x1p-40), 0x3f81) ]);
          (Float64, Float16, [ (`Float (65520. -. 0x1p-20), 0x7bff) ]);
          ( Int32,
            Bfloat16,
            [
              (integer "16842753", 0x4b81);
              (integer "-16842753", 0xcb81);
              (integer "2147483647", 0x4f00);
              (integer "-2147483648", 0xcf00);
            ] );
          (Uint32, Bfloat16, [ (integer "4294967295", 0x4f80) ]);
          ( Int64,
            Bfloat16,
            [
              (integer "1103806595073", 0x5381);
              (integer "9223372036854775807", 0x5f00);
              (integer "-9223372036854775808", 0xdf00);
            ] );
          ( Uint64,
            Bfloat16,
            [
              (integer "9259400833873739777", 0x5f01);
              (integer "18446744073709551615", 0x5f80);
            ] );
          (Uint32, Float16, [ (integer "4294967295", 0x7c00) ]);
          (Int32, Fp8e4m3, [ (integer "2147483647", 0x7e) ]);
        ]
      (fun (from, dt, table) -> casts ~from dt table);
    cases "an fnuz format stores an underflow to negative zero as positive zero"
      ~id:alias Dtype.fp8_fnuz (fun dt ->
        let least = List.assoc dt least_subnormals in
        casts dt
          [
            (`Float (-0.), 0x00);
            (`Float (-1e-30), 0x00);
            (`Float (-.least *. 0.5), 0x00);
          ]);
    cases "an OCP 8-bit float keeps its negative zero" ~id:alias
      Dtype.[ Fp8e4m3; Fp8e5m2 ]
      (fun dt -> casts dt [ (`Float (-0.), 0x80); (`Float (-1e-30), 0x80) ]);
  ]

let narrow_floats =
  group "narrow floats"
    (conversions
    @ [
        computes_in_float32;
        reads_unrounded;
        compares_values;
        copies_every_code;
        selects_every_code;
        reinterprets;
        computed_then_reinterpreted;
        round_trip;
      ]
    @ nans_in_kernels @ ieee_facets)

(* 64-bit integers *)

let on_32_bits = lacking longs
let word_of = function Dtype.Int64 -> Dtype.Int32 | _ -> Dtype.Uint32

(* [stored dt vs] is the storage of the values [vs] of [dt] on a target without
   64-bit integers: two 32-bit words for each 64-bit integer, low first. *)
let stored dt vs =
  if List.mem dt longs then
    List.concat_map
      (fun v ->
        List.map
          (fun k ->
            Dtype.truncate (word_of dt)
              (`Int (Bigint.extract (as_z v) (32 * k) 32)))
          [ 0; 1 ])
      vs
  else vs

let rec of_words dt = function
  | lo :: hi :: rest ->
      Dtype.truncate dt
        (`Int Bigint.(extract (as_z lo) 0 32 + shift_left (as_z hi) 32))
      :: of_words dt rest
  | [] -> []
  | [ _ ] -> failf "a 64-bit integer lacks its high word"

(* [computed ins out f] is the function that the emulated kernel of [f] from
   [ins] to [out] computes, on one element. *)
let computed ins out f =
  let k = lazy (emulate on_32_bits (kernel ins out 1 f)) in
  fun args ->
    let got =
      outputs (Lazy.force k) (List.map2 (fun dt v -> stored dt [ v ]) ins args)
    in
    match if List.mem out longs then of_words out got else got with
    | [ v ] -> v
    | vs -> failf "%d values written" (List.length vs)

let wrap dt z = Dtype.truncate dt (`Int z)

(* [law name shape args oracle] is the property that, for each 64-bit type [dt],
   the emulated kernel [shape dt], [(ins, out, f)], computes [oracle dt] on the
   arguments [args dt] draws. *)
let law ?(count = 100) ?(examples = []) ?(covers = fun _ _ -> []) name shape
    args oracle =
  let runs =
    List.map
      (fun dt ->
        let ins, out, f = shape dt in
        (dt, computed ins out f))
      longs
  in
  let pp ppf (dt, vs) =
    Format.fprintf ppf "%s %a" (alias dt)
      (Format.pp_print_list ~pp_sep:Format.pp_print_space (Testable.pp value))
      vs
  in
  prop ~count ~examples name
    Gen.(
      with_pp pp
        (let* dt = of_list longs in
         map (fun vs -> (dt, vs)) (args dt)))
    (fun (dt, vs) ->
      cover (alias dt) true;
      List.iter (fun (label, holds) -> cover label holds) (covers dt vs);
      equal value (oracle dt vs) (List.assoc dt runs vs))

let pair_of g = Gen.map (fun (a, b) -> [ a; b ]) (Gen.pair g g)
let operands dt = pair_of (value_of dt)

let alu2 op dt =
  ([ dt; dt ], dt, fun xs -> Ops.alu (List.nth xs 0) op [ List.nth xs 1 ])

let binary ?count ?examples ?covers ?(b = value_of) name op oracle =
  law ?count ?examples ?covers name (alu2 op)
    (fun dt -> Gen.map (fun (a, b) -> [ a; b ]) (Gen.pair (value_of dt) (b dt)))
    (fun dt -> function
      | [ a; b ] -> wrap dt (oracle (as_z a) (as_z b)) | _ -> assert false)

let ints dt l = List.map (fun z -> wrap dt (Bigint.of_string z)) l

(* Pairs that meet the word-wise cases: equal values, one high word with low
   words apart, high words one apart, and each type's extremes. *)
let word_pairs =
  List.concat_map
    (fun dt ->
      List.map
        (fun (a, b) -> (dt, ints dt [ a; b ]))
        [
          ("12345678901", "12345678901");
          ("4294967296", "4294967297");
          ("8589934591", "4294967295");
          ("-1", "0");
          ("-9223372036854775808", "9223372036854775807");
          ("9223372036854775807", "-1");
        ])
    longs

let high_word_equal _ = function
  | [ a; b ] ->
      [
        ( "one high word",
          Bigint.(equal (shift_right (as_z a) 32) (shift_right (as_z b) 32)) );
        ("equal", Bigint.equal (as_z a) (as_z b));
      ]
  | _ -> []

let nonzero dt =
  Gen.such_that (fun v -> not (Bigint.equal (as_z v) Bigint.zero)) (value_of dt)

(* Pairs that reach each path of the division: the least value by -1, which
   wraps, and by 1 and itself; quotients of 2^32 and more; divisors of 2^32 and
   more; a dividend below its divisor; and each pair of signs. *)
let division_examples =
  List.concat_map
    (fun dt ->
      List.map
        (fun (a, b) -> (dt, ints dt [ a; b ]))
        [
          ("-9223372036854775808", "-1");
          ("-9223372036854775808", "1");
          ("-9223372036854775808", "-9223372036854775808");
          ("-1", "-9223372036854775808");
          ("18446744073709551615", "4294967296");
          ("18446744073709551615", "4294967297");
          ("5", "1099511627776");
          ("-7", "2");
          ("7", "-2");
          ("-7", "-2");
          ("7", "2");
          ("123456789012345", "123456789012345");
        ])
    longs

(* A dividend of each type over a divisor of 0. *)
let by_zero =
  List.concat_map
    (fun dt ->
      List.map (fun a -> (dt, ints dt [ a; "0" ])) [ "-5"; "12345678901" ])
    longs

let division_covers dt = function
  | [ a; b ] ->
      let a = as_z a and b = as_z b in
      let two32 = Bigint.shift_left Bigint.one 32 in
      let negative z =
        Dtype.is_unsigned dt = false && Bigint.lt z Bigint.zero
      in
      [
        ("a quotient of 2^32 or more", Bigint.(geq (abs (div a b)) two32));
        ("a divisor of 2^32 or more", Bigint.(geq (abs b) two32));
        ("|a| < |b|", Bigint.(lt (abs a) (abs b)));
        ("exact", Bigint.(equal (rem a b) zero));
        ("both non-negative", (not (negative a)) && not (negative b));
        ("a negative dividend", negative a && not (negative b));
        ("a negative divisor", (not (negative a)) && negative b);
        ("both negative", negative a && negative b);
      ]
  | _ -> []

(* The floats beyond a 32-bit word's range, which the low word of an emulated
   cast to a 64-bit integer must not convert to 32 bits directly. *)
let wide_floats dt =
  let xs = [ 0x1p31; 0x1p32 +. 512.; 0x1p40; 0x1p62 ] in
  List.map
    (fun x -> `Float x)
    (if dt = Dtype.Int64 then xs @ List.map Float.neg xs else xs)

(* [float32_in dt] draws float32s whose truncation [dt] holds. *)
let float32_in dt =
  let lo, hi = int_bounds dt in
  Gen.such_that
    (fun x -> Bigint.to_float lo <= x && x < Bigint.to_float hi)
    (Gen.map (fun x -> round32 x) finite_float)

(* Shifts by 0, 5, 31, 32 and 62 or 63, across and within the words. *)
let shift_examples =
  let of_values dt values counts =
    List.map2
      (fun v n -> (dt, [ wrap dt v; `Int (Bigint.of_int n) ]))
      values counts
  in
  [
    of_values Int64
      (List.map Bigint.of_int [ -0x1234; 0x80000001; -1; 0x1234; 1 ])
      [ 0; 5; 31; 32; 62 ];
    of_values Uint64
      (List.map Bigint.of_int [ 0x80000001; 0x80000001; 1; 0xFEDC; 1 ])
      [ 0; 5; 31; 32; 62 ];
    of_values Int64
      (List.map Bigint.of_int
         [ -(1 lsl 40); -1; -(1 lsl 50); -(1 lsl 40); 0x123456789ABCDEF ])
      [ 0; 5; 31; 32; 63 ];
    of_values Uint64
      (List.init 5 (fun _ -> Bigint.of_string "0xFEDCBA9876543210"))
      [ 0; 5; 31; 32; 63 ];
  ]
  |> List.concat

let shift_count _ =
  Gen.map (fun n -> `Int (Bigint.of_int n)) (Gen.int_range 0 63)

(* 64-bit integers whose leading word rounds up to a power of two as a float32:
   a high word of [2^25 - 1], and a low word of [2^31 - 1] under a zero high
   word. *)
let leading_word_examples =
  List.concat_map
    (fun dt ->
      let zs =
        Bigint.[ (of_int 0x1ffffff lsl 32) + of_int 12345; of_int 0x7fffffff ]
      in
      let zs = if dt = Dtype.Int64 then zs @ List.map Bigint.neg zs else zs in
      List.map (fun z -> (dt, [ `Int z ])) zs)
    longs

let unary_of dt = Gen.map (fun v -> [ v ]) (value_of dt)

(* [converts name from to_] is the law that the emulated cast from [from dt] to
   [to_ dt] wraps the value it casts. *)
let converts name from to_ =
  law name
    (fun dt -> ([ from dt ], to_ dt, cast_to (to_ dt)))
    (fun dt -> unary_of (from dt))
    (fun dt -> function
      | [ `Bool b ] -> wrap (to_ dt) (if b then Bigint.one else Bigint.zero)
      | [ v ] -> wrap (to_ dt) (as_z v)
      | _ -> assert false)

let other dt = if dt = Dtype.Int64 then Dtype.Uint64 else Dtype.Int64
let long_ dt = dt

let sixty_four_bits =
  group "64-bit integers"
    [
      binary "an emulated add is exact, wrapped" Add Bigint.add;
      binary "an emulated sub is exact, wrapped" Sub Bigint.sub;
      binary "an emulated mul is exact, wrapped" Mul Bigint.mul;
      binary "an emulated and is exact" And Bigint.logand;
      binary "an emulated or is exact" Or Bigint.logor;
      binary "an emulated xor is exact" Xor Bigint.logxor;
      binary ~examples:word_pairs ~covers:high_word_equal
        "an emulated max is the greater" Max Bigint.max;
      binary ~examples:division_examples ~covers:division_covers ~b:nonzero
        "an emulated truncating division rounds towards zero" Truncdiv
        Bigint.div;
      binary ~examples:division_examples ~covers:division_covers ~b:nonzero
        "an emulated truncating remainder has the dividend's sign" Truncmod
        Bigint.rem;
      binary ~count:20 ~examples:by_zero
        ~b:(fun _ -> Gen.constant (`Int Bigint.zero))
        "an emulated quotient by 0 is 0" Truncdiv
        (fun _ _ -> Bigint.zero);
      binary ~count:20 ~examples:by_zero
        ~b:(fun _ -> Gen.constant (`Int Bigint.zero))
        "an emulated remainder by 0 is the dividend" Truncmod
        (fun a _ -> a);
      binary
        ~examples:(List.filteri (fun i _ -> i < 10) shift_examples)
        ~b:shift_count "an emulated shl shifts left by a count below 64" Shl
        (fun a n -> Bigint.shift_left a (Bigint.to_int n));
      binary
        ~examples:(List.filteri (fun i _ -> i >= 10) shift_examples)
        ~b:shift_count "an emulated shr shifts right by a count below 64" Shr
        (fun a n -> Bigint.shift_right a (Bigint.to_int n));
      law ~examples:word_pairs ~covers:high_word_equal
        "an emulated cmplt compares values"
        (fun dt ->
          ( [ dt; dt ],
            Bool,
            fun xs -> Ops.alu (List.nth xs 0) Cmplt [ List.nth xs 1 ] ))
        operands
        (fun _ -> function
          | [ a; b ] -> `Bool Bigint.(lt (as_z a) (as_z b)) | _ -> assert false);
      law ~examples:word_pairs ~covers:high_word_equal
        "an emulated cmpeq compares values"
        (fun dt ->
          ( [ dt; dt ],
            Bool,
            fun xs -> Ops.alu (List.nth xs 0) Cmpeq [ List.nth xs 1 ] ))
        operands
        (fun _ -> function
          | [ a; b ] -> `Bool Bigint.(equal (as_z a) (as_z b))
          | _ -> assert false);
      law ~examples:word_pairs ~covers:high_word_equal
        "an emulated cmpne compares values"
        (fun dt ->
          ( [ dt; dt ],
            Bool,
            fun xs -> Ops.alu (List.nth xs 0) Cmpne [ List.nth xs 1 ] ))
        operands
        (fun _ -> function
          | [ a; b ] -> `Bool (not Bigint.(equal (as_z a) (as_z b)))
          | _ -> assert false);
      law "an emulated negation is exact, wrapped"
        (fun dt -> ([ dt ], dt, fun xs -> Ops.alu (List.hd xs) Neg []))
        unary_of
        (fun dt -> function
          | [ a ] -> wrap dt (Bigint.neg (as_z a)) | _ -> assert false);
      law "an emulated selection picks its operand"
        (fun dt ->
          ( [ Bool; dt; dt ],
            dt,
            fun xs -> Ops.where (List.nth xs 0) (List.nth xs 1) (List.nth xs 2)
          ))
        (fun dt ->
          Gen.map
            (fun (c, (a, b)) -> [ `Bool c; a; b ])
            (Gen.pair Gen.bool (Gen.pair (value_of dt) (value_of dt))))
        (fun _ -> function
          | [ `Bool c; a; b ] -> if c then a else b | _ -> assert false);
      converts "an emulated cast from int32 wraps it"
        (fun _ -> Dtype.Int32)
        long_;
      converts "an emulated cast from uint32 wraps it"
        (fun _ -> Dtype.Uint32)
        long_;
      converts "an emulated cast from int8 wraps it" (fun _ -> Dtype.Int8) long_;
      converts "an emulated cast from bool wraps it" (fun _ -> Dtype.Bool) long_;
      converts "an emulated cast to int32 wraps it" long_ (fun _ -> Dtype.Int32);
      converts "an emulated cast to int8 wraps it" long_ (fun _ -> Dtype.Int8);
      converts "an emulated cast to uint16 wraps it" long_ (fun _ ->
          Dtype.Uint16);
      converts "an emulated cast to the other 64-bit integer wraps it" long_
        other;
      law
        "an emulated bit reinterpretation as the other 64-bit integer wraps it"
        (fun dt ->
          ([ dt ], other dt, fun xs -> Ops.bitcast (List.hd xs) (other dt)))
        unary_of
        (fun dt -> function
          | [ v ] -> wrap (other dt) (as_z v) | _ -> assert false);
      law
        ~examples:
          (List.concat_map
             (fun dt -> List.map (fun v -> (dt, [ v ])) (wide_floats dt))
             longs)
        "an emulated cast from a float32 rounds towards zero"
        (fun dt -> ([ Float32 ], dt, cast_to dt))
        (fun dt -> Gen.map (fun x -> [ `Float x ]) (float32_in dt))
        (fun dt -> function
          | [ `Float x ] -> wrap dt (Bigint.of_float (Float.trunc x))
          | _ -> assert false);
      law ~examples:leading_word_examples
        "an emulated cast to a float32 is the nearest"
        (fun dt -> ([ dt ], Float32, cast_to Float32))
        unary_of
        (fun _ -> function
          | [ v ] -> Dtype.truncate Float32 v | _ -> assert false);
      law "an emulated cast to a double is the nearest"
        (fun dt -> ([ dt ], Float64, cast_to Float64))
        unary_of
        (fun _ -> function
          | [ v ] -> Dtype.truncate Float64 v | _ -> assert false);
      cases
        "an emulated cast of a float32 converts no float to a word that cannot \
         hold it"
        ~id:alias longs (fun dt ->
          let values = wide_floats dt in
          let k = kernel [ Float32 ] dt (List.length values) (cast_to dt) in
          equal (list value) []
            (List.map (List.nth values)
               (undefined_casts ~on:on_32_bits k [ values ])));
      test "an emulated 64-bit integer converts to a float32 once" (fun () ->
          let converts from to_ z =
            let k = kernel [ from ] to_ 1 (cast_to to_) in
            outputs
              (emulate (lacking (longs @ narrows)) k)
              [ stored from [ integer z ] ]
          in
          equal (list value)
            [ `Float 4505803482464256. ]
            (converts Uint64 Float32 "4505803214028801");
          equal (list value)
            [ `Float (-9007200328482816.) ]
            (converts Int64 Float32 "-9007199791611905");
          equal (list hex) [ 0xdebb ]
            (List.map as_int (converts Int64 Bfloat16 "-6719370644036780033")));
    ]

(* Goldens *)

(* A golden holds a kernel and what the pass makes of it, as its generator
   records it, with each call of f2f held as a placeholder: a call of code in
   the target's language on its operand, whose code names f2f and its arguments
   (gen/codegen/decomp/decomp_dtype.py). The conversions are IEEE's. *)

let dtype_named s = Result.get_ok (Dtype.of_string s)

let sat_named = function
  | "True" -> true
  | "False" -> false
  | s -> failf "%S is not a boolean" s

let conversion u =
  match Ops.src u with
  | [ f; v ] when Ops.op f = Custom_function -> (
      let code = match Ops.arg f with Custom_function f -> f.name | _ -> "" in
      match String.split_on_char ' ' code with
      | [ "f2f"; fr; to_; sat ] ->
          Some
            (Decomp_dtype.f2f ~sat:(sat_named sat) v (dtype_named fr)
               (dtype_named to_))
      | _ -> None)
  | _ -> None

(* [with_conversions u] is [u] with its placeholders replaced by the conversions
   they stand for, the weak constants of the conversions committed as the pass
   commits them. *)
let with_conversions u =
  let restore =
    PM.v (fun () ->
        [ PM.rule (Ops.Upat.op Call ~name:"c") (fun m -> conversion (m "c")) ])
  in
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
    (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
       (After_sources restore))
    (After_sources Uop_weak.pm_commit_weak)

let narrow_kernels =
  [
    "add";
    "mulsub";
    "maximum";
    "where";
    "exp2";
    "sqrt";
    "sum";
    "lt";
    "from_float";
    "to_float";
    "from_char";
    "to_char";
    "flip";
    "gather";
    "pad";
    "bitcast_to_storage";
    "bitcast_from_storage";
    "bitcast_sum";
  ]

let long_kernels dt =
  [
    "add";
    "sub";
    "neg";
    "mul";
    (if dt = Dtype.Int64 then "div" else "mod");
    "shl";
    "shr";
    "shl_by";
    "xor";
    "and_or";
    "lt";
    "eq";
    "maximum";
    "where";
    "sum";
    "pad";
    "add_const";
    "from_int";
    "from_uint";
    "from_bool";
    "to_char";
    "bitcast";
  ]

(* Each golden, with the data types the setting names and those the other target
   lacks. *)
let goldens =
  List.concat_map
    (fun dt ->
      List.map (fun k -> (alias dt ^ "_" ^ k, [ dt ], [ dt ])) narrow_kernels)
    narrows
  @ Dtype.
      [
        ("half_fp8e4m3_cast", [ Float16; Fp8e4m3 ], [ Float16; Fp8e4m3 ]);
        ("bfloat16_half_cast", [ Float16; Bfloat16 ], [ Float16; Bfloat16 ]);
      ]
  @ List.concat_map
      (fun dt ->
        List.map
          (fun k -> (alias dt ^ "_" ^ k, [ Dtype.Int64 ], longs))
          (long_kernels dt))
      longs
  @ [ ("ulong_named_alone", [ Dtype.Uint64 ], [ Dtype.Uint64 ]) ]

(* [golden name] is the sink of the golden [name], read once. *)
let golden =
  let read = Hashtbl.create 160 in
  fun name ->
    match Hashtbl.find_opt read name with
    | Some g -> g
    | None ->
        let g = Golden.sink (name ^ ".golden") in
        Hashtbl.add read name g;
        g

(* [same_graph want got] checks that [got] is [want], which hash-consing makes
   one node, and fails with the diff of their graphs. *)
let same_graph want got =
  if not (Ops.equal want got) then
    equal text (Graph.to_string want) (Graph.to_string got)

(* [rewrites_as_golden (name, named, lacks)] checks the golden [name] on the
   target that lacks [lacks]. *)
let rewrites_as_golden (name, _, lacks) =
  let file = name ^ ".golden" in
  let upstream = lazy (Golden.upstream file) in
  let recorded () =
    if Golden.updating then Lazy.force upstream else golden name
  in
  let input () = Ops.nth (recorded ()) 0 in
  let actual () = emulate (lacking lacks) (input ()) in
  let expected () = with_conversions (Ops.nth (recorded ()) 1) in
  if Golden.updating && not (Ops.equal (expected ()) (actual ())) then
    Golden.graph file (fun () -> Ops.sink [ input (); actual () ])
  else test file (fun () -> same_graph (expected ()) (actual ()))

(* [told_as_lacking (name, named, lacks)] checks that the target told to emulate
   [named] rewrites the golden's kernel as the one that lacks [lacks]: a type's
   emulation is decided once per kernel. *)
let told_as_lacking (name, named, lacks) =
  test (name ^ ".golden") (fun () ->
      let k = Ops.nth (golden name) 0 in
      same_graph
        (emulate (lacking lacks) k)
        (told (List.map alias named) (fun () ->
             emulate
               (lacking (List.filter (fun dt -> not (List.mem dt named)) lacks))
               k)))

(* One kernel of each type, and the kernels of several: half_fp8e4m3_cast told
   to emulate half alone, the target lacking e4m3. *)
let told_goldens =
  List.map (fun dt -> (alias dt ^ "_add", [ dt ], [ dt ])) (narrows @ longs)
  @ Dtype.
      [
        ("half_fp8e4m3_cast", [ Float16 ], [ Float16; Fp8e4m3 ]);
        ("bfloat16_half_cast", [ Float16; Bfloat16 ], [ Float16; Bfloat16 ]);
        ("ulong_named_alone", [ Uint64 ], [ Uint64 ]);
      ]

(* The kernels of the goldens compute, emulated, what they compute natively. The
   kernels that compute in a narrow float are left out: emulated, they compute
   in float32 and round once, at the store. *)
let computes_natively =
  [
    "flip";
    "gather";
    "pad";
    "where";
    "maximum";
    "lt";
    "sum";
    "from_float";
    "to_float";
    "from_char";
    "to_char";
    "bitcast_to_storage";
    "bitcast_from_storage";
  ]

let natively_comparable =
  List.filter
    (fun (name, _, _) ->
      List.exists
        (fun dt ->
          List.exists (fun k -> name = alias dt ^ "_" ^ k) computes_natively)
        narrows
      || List.exists
           (fun dt -> String.starts_with ~prefix:(alias dt ^ "_") name)
           longs)
    goldens

let params k =
  List.filter_map
    (fun u ->
      match (Ops.op u, Ops.arg u) with
      | Param, Param a -> Some (a, Shape.max_shape u)
      | _ -> None)
    (Ops.toposort ~calls:Enter k)
  |> List.sort_uniq (fun ((a : Ops.param_arg), _) (b, _) ->
      Int.compare a.slot b.slot)

let contains affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

(* [element name ~last dt] draws an element of a storage of type [dt] of the
   golden [name], the last of its kernel's storage if [last]: the value its
   native kernel reads, and the values its emulated kernel reads in its place.
   Each is within what its kernel reads: an index within the gathered storage, a
   shift count below 64, a nonzero divisor, a float that a 64-bit integer holds,
   and a finite narrow float. *)
let element name ~last dt =
  let open Gen in
  let same g = map (fun v -> (v, [ v ])) g in
  match dt with
  | _ when List.mem dt narrows ->
      map
        (fun c -> (`Float (decode dt c), [ code c ]))
        (such_that (is_finite_code dt) (int_range 0 (size dt - 1)))
  | Dtype.Int64 | Uint64 ->
      let z =
        if contains "shl_by" name && last then
          map (fun n -> `Int (Bigint.of_int n)) (int_range 0 63)
        else if (contains "div" name || contains "mod" name) && last then
          map (fun n -> `Int (Bigint.of_int n)) (int_range 1 0x7fffffff)
        else value_of dt
      in
      map (fun v -> (v, stored dt [ v ])) z
  | Float32 when contains "long" name ->
      let lo = if contains "ulong" name then 0. else -0x1p61 in
      same (map (fun x -> `Float (round32 x)) (float_range lo (lo +. 0x1p62)))
  | Float32 ->
      same
        (map
           (fun (e, m, negative) ->
             `Float
               (f32 ((e lsl 23) lor m lor if negative then 0x80000000 else 0)))
           (triple (int_range 0 254) (int_range 0 0x7fffff) bool))
  | Int32 when contains "gather" name ->
      same (map (fun n -> `Int (Bigint.of_int n)) (int_range 0 15))
  | dt -> same (value_of dt)

(* [all gens] draws a list, one element from each of [gens]. *)
let all gens =
  List.fold_right
    (fun g acc ->
      Gen.(
        let+ x = g and+ xs = acc in
        x :: xs))
    gens (Gen.constant [])

(* [storages name] draws every storage of the golden [name]'s kernel. *)
let storages name =
  let slots = params (Ops.nth (golden name) 0) in
  all
    (List.map
       (fun ((a : Ops.param_arg), shape) ->
         let n = List.fold_left ( * ) 1 shape in
         let last = a.slot = List.length slots - 1 in
         Gen.map
           (fun xs -> (a.slot, a.dtype, xs))
           (all (List.init n (fun _ -> element name ~last a.dtype))))
       slots)

let emulated_goldens = Hashtbl.create 128

let emulated_golden (name, named, lacks) =
  match Hashtbl.find_opt emulated_goldens name with
  | Some k -> k
  | None ->
      let k = Ops.nth (golden name) 0 in
      let e =
        told (List.map alias named) (fun () -> emulate (lacking lacks) k)
      in
      Hashtbl.add emulated_goldens name e;
      e

(* [writes_natively g storage] checks that the golden [g]'s kernel writes,
   emulated, what it writes natively, its storage drawn. *)
let writes_natively ((name, _, _) as g) storage =
  let k = Ops.nth (golden name) 0 in
  let native =
    Interpreter.writes
      ~buffers:
        (List.map
           (fun (s, _, xs) -> (s, Array.of_list (List.map fst xs)))
           storage)
      k
  in
  let emulated =
    Interpreter.writes
      ~buffers:
        (List.map
           (fun (s, _, xs) -> (s, Array.of_list (List.concat_map snd xs)))
           storage)
      (emulated_golden g)
  in
  let dtype_of slot =
    let _, dt, _ = List.find (fun (s, _, _) -> s = slot) storage in
    dt
  in
  let rec decoded = function
    | (s, i, lo) :: (_, _, hi) :: rest when List.mem (dtype_of s) longs ->
        (s, i / 2, List.hd (of_words (dtype_of s) [ lo; hi ])) :: decoded rest
    | (s, i, `Int c) :: rest when List.mem (dtype_of s) narrows ->
        (s, i, `Float (decode (dtype_of s) (Bigint.to_int c))) :: decoded rest
    | w :: rest -> w :: decoded rest
    | [] -> []
  in
  equal ~msg:name (list (triple int int value)) native (decoded emulated)

let pp_storages ppf cases =
  List.iter
    (fun (name, storage) ->
      Format.fprintf ppf "@[<v 2>%s:" name;
      List.iter
        (fun (s, _, xs) ->
          Format.fprintf ppf "@ slot %d: %a" s
            (Format.pp_print_list
               ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
               (Testable.pp value))
            (List.map fst xs))
        storage;
      Format.fprintf ppf "@]@ ")
    cases

(* A case draws the storage of every natively comparable golden, so that each
   run checks each kernel at fresh values. *)
let writes_as_natively =
  prop ~count:4 "each golden's kernel writes, emulated, what it writes natively"
    (Gen.with_pp pp_storages
       (all
          (List.map
             (fun (name, _, _) -> Gen.map (fun s -> (name, s)) (storages name))
             natively_comparable)))
    (fun cases ->
      List.iter2
        (fun g (_, storage) -> writes_natively g storage)
        natively_comparable cases)

let goldens_group =
  group "rewrites each kernel as its golden states, f2f's conversions in place"
    (List.map rewrites_as_golden goldens)

(* The pass *)

let decomps on k =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Decomp_dtype.ctx on) k
    (After_sources Decomp_dtype.pm_dtype_decomps)

let add dt =
  kernel [ dt; dt ] dt 4 (fun xs -> Ops.O.(List.nth xs 0 + List.nth xs 1))

let pass =
  group "pm_dtype_decomps"
    [
      goldens_group;
      group "emulates a type told to as one the target lacks"
        (List.map told_as_lacking told_goldens);
      writes_as_natively;
      narrow_floats;
      sixty_four_bits;
      test "a target with every data type keeps its kernels" (fun () ->
          List.iter
            (fun dt ->
              equal ~msg:(alias dt) Uops.uop (add dt)
                (decomps everything (add dt)))
            (narrows @ longs));
      test "a kernel without narrow floats or 64-bit integers is kept"
        (fun () ->
          List.iter
            (fun dt ->
              equal ~msg:(alias dt) Uops.uop (add dt)
                (decomps (lacking (narrows @ longs)) (add dt)))
            Dtype.[ Float32; Float64; Int32; Uint32; Int8; Bool ]);
      test "unsigned 64-bit integers are emulated when the signed ones are"
        (fun () ->
          equal Uops.uop
            (decomps (lacking longs) (add Uint64))
            (decomps (lacking [ Int64 ]) (add Uint64)));
      test "64-bit storage holds two 32-bit words per element" (fun () ->
          List.iter
            (fun dt ->
              let k = decomps on_32_bits (kernel [ dt ] dt 7 List.hd) in
              equal ~msg:(alias dt)
                (list (pair dtype (list int)))
                [ (word_of dt, [ 14 ]); (word_of dt, [ 14 ]) ]
                (List.map
                   (fun ((a : Ops.param_arg), shape) -> (a.dtype, shape))
                   (params k)))
            longs);
      test "64-bit storage of no known size keeps none" (fun () ->
          let p = Shape.param ~addrspace:Global 0 Int64 in
          let k =
            Ops.sink
              [ Ops.store (Ops.index p [ Ops.int 0 ]) (Ops.int ~dtype:Int64 3) ]
          in
          equal
            (list (pair dtype (list int)))
            [ (Int32, []) ]
            (List.map
               (fun ((a : Ops.param_arg), shape) -> (a.dtype, shape))
               (params (decomps on_32_bits k))));
      test "a 64-bit integer variable cannot be emulated" (fun () ->
          let c =
            Ops.variable ~dtype:Int64 "c" (`Int Bigint.zero)
              (`Int (Bigint.shift_left Bigint.one 40))
          in
          let k = kernel [] Int64 1 (fun _ -> c) in
          rejects (fun () -> decomps on_32_bits k));
      test "a setting that names no data type is refused" (fun () ->
          told [ "nonsense" ] (fun () ->
              rejects (fun () -> decomps everything (add Float16))));
    ]

let () =
  exit
    (run "Tolk.Decomp_dtype"
       [ f2f; exact; narrow; computes_group; emulates_group; pass ])
