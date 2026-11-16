(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Codes

   A float32's code is held in the low 32 bits of an int64, a float64's in all
   64. [wide] picks float64. *)

let low32 = 0xFFFFFFFFL

let code wide v =
  if wide then Int64.bits_of_float v
  else Int64.logand (Int64.of_int32 (Int32.bits_of_float v)) low32

let value wide c =
  if wide then Int64.float_of_bits c else Int32.float_of_bits (Int64.to_int32 c)

let sign wide = if wide then Int64.min_int else 0x80000000L
let mmask wide = if wide then 0x000FFFFFFFFFFFFFL else 0x007FFFFFL
let pick wide (a, b) = if wide then b else a
let is_nan wide c = Float.is_nan (value wide c)

(* The zeros, the least and greatest subnormals, the least normals, the greatest
   finite floats and the infinities of both signs, then NaNs: quiet, negative,
   with a payload and signalling. *)
let edges wide =
  let pos =
    if wide then
      [
        0L;
        1L;
        0x000FFFFFFFFFFFFFL;
        0x0010000000000000L;
        0x7FEFFFFFFFFFFFFFL;
        0x7FF0000000000000L;
      ]
    else [ 0L; 1L; 0x007FFFFFL; 0x00800000L; 0x7F7FFFFFL; 0x7F800000L ]
  in
  let nans =
    if wide then
      [
        0x7FF8000000000000L;
        0xFFF8000000000000L;
        0x7FF80000DEADBEEFL;
        0x7FF0000000000001L;
      ]
    else [ 0x7FC00000L; 0xFFC00000L; 0x7FC12345L; 0x7F800001L ]
  in
  List.concat_map (fun c -> [ c; Int64.logor c (sign wide) ]) pos @ nans

(* Kinds

   A spec gives one operand's draws at one dtype: the header's switch points and
   edges of the kind, positive, and the bounds of its uniform ranges. A cover
   names a class of an element's operands. *)

type spec = { points : float list; ranges : float list }
type f1 = { f1 : 's 'd. (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t }
type fn = One of f1

type kind = {
  name : string;
  fn : fn;
  specs : (spec * spec) array;  (** Per operand, at float32 and float64. *)
  covers : (string * (bool -> float array -> bool)) list;
}

let name k = k.name
let arity k = Array.length k.specs
let spec wide k i = pick wide k.specs.(i)

(* Trigonometric kinds: nx_kinds.h's switch to Payne and Hanek's reduction, 2^15
   or 2^28, and the cutoff below which sin and tan are x, 2^-12 or 2^-27;
   multiples of pi/2 on either side of the switch, where the reduced argument
   cancels, among them 355 and 103993 times pi/2, whose ratios approach pi's;
   and large values to the greatest finite. *)

let half_pi = Float.pi /. 2.
let trig_big = (0x1p15, 0x1p28)
let trig_tiny = (0x1p-12, 0x1p-27)

let trig_spec wide =
  let big = pick wide trig_big in
  let below, past = if wide then (1e8, 2e8) else (20000., 20861.) in
  {
    points =
      ([
         pick wide trig_tiny; big; 2. *. big; below *. half_pi; past *. half_pi;
       ]
      @ List.map (fun k -> k *. half_pi) [ 1.; 2.; 3.; 4.; 5.; 355.; 103993. ]
      @
      if wide then [ 0x1.6ac5b262ca1ffp+849; 1e22; 1e300 ]
      else [ 105414350.; 2.1904577e38; 1e30 ]);
    ranges = [ Float.pi; (if wide then 1e9 else 1e5) ];
  }

let trig_covers =
  let mag wide e = Float.abs (value wide (code wide e.(0))) in
  [
    ( "below the cutoff, not zero",
      fun w e -> e.(0) <> 0. && mag w e < pick w trig_tiny );
    ( "from the cutoff to the switch",
      fun w e -> mag w e >= pick w trig_tiny && mag w e < pick w trig_big );
    ( "past the switch, finite",
      fun w e -> mag w e >= pick w trig_big && Float.is_finite e.(0) );
    ( "beside a multiple of pi/2",
      fun _ e ->
        let x = e.(0) in
        let q = Float.round (x /. half_pi) in
        Float.abs x < 1e9
        && q <> 0.
        && Float.abs (x -. (q *. half_pi)) <= 0x1p-20 *. Float.abs x );
  ]

let trig name f =
  {
    name;
    fn = One f;
    specs = [| (trig_spec false, trig_spec true) |];
    covers = trig_covers;
  }

let sin = trig "sin" { f1 = Nx.sin }
let cos = trig "cos" { f1 = Nx.cos }
let tan = trig "tan" { f1 = Nx.tan }

(* Cases *)

type case = { f32 : int64 array array; f64 : int64 array array }

(* Elements per dtype in a case: one compiled program per dtype and kind. *)
let n = 256

(* One operand's code at [wide], drawn from five classes by weight out of 12: a
   switch point stepped up to 3 ranks either way, of either sign (4); an edge
   (2); a uniform bit pattern (2); a uniform value of a range (3); a subnormal
   of either sign (1).

   Rates. Windtrap repeats an earlier draw of a generator with probability 1/8,
   so at least 7/8 of the elements are fresh draws. An edge, each of 16 equally
   likely, is then drawn with probability at least 7/8 * 2/12 / 16 = 1/110: a
   block of 256 elements misses it with probability (1 - 1/110)^256 < 0.1, and
   12 cases all miss it with probability below 1e-12. A subnormal (1/12 at
   least) and a cover of a switch point's side (with p points, 7/8 * 4/12 / p *
   3/7 at least, int_range drawing its uniform values and one corner in ten
   among -3, 3, 0, -1 and 1) are drawn more often for every kind's p <= 16. *)
let operand wide s =
  let open Gen in
  let signed g =
    let+ c = g and+ neg = bool in
    if neg then Int64.logor c (sign wide) else c
  in
  let point =
    signed
      (let+ p = of_list s.points and+ d = int_range (-3) 3 in
       Int64.add (code wide p) (Int64.of_int d))
  in
  let bits =
    if wide then int64_range Int64.min_int (Int64.pred Int64.max_int)
    else int64_range 0L (Int64.pred low32)
  in
  let range =
    let* r = of_list s.ranges in
    map (code wide) (float_range (-.r) r)
  in
  let sub = signed (int64_range 1L (mmask wide)) in
  frequency
    [ (4, point); (2, of_list (edges wide)); (2, bits); (3, range); (1, sub) ]

let element wide k =
  let ops = List.init (arity k) (fun i -> operand wide (spec wide k i)) in
  Gen.map Array.of_list
    (List.fold_right
       (fun g acc ->
         Gen.(
           let+ v = g and+ vs = acc in
           v :: vs))
       ops (Gen.constant []))

let gen k =
  let block wide = Gen.array ~size:(Gen.constant n) (element wide k) in
  Gen.with_pp
    (fun ppf _ ->
      Format.fprintf ppf "%d elements of %s at float32 and float64" n k.name)
    Gen.(
      let+ f32 = block false and+ f64 = block true in
      { f32; f64 })

(* The example's operand codes: the edges, then each point and its neighbours up
   to 3 ranks of both signs, then codes spread over the code space, [n] in all;
   elements pair the codes of each operand at the same position, the second
   operand's list turned by one. *)
let example_codes wide s =
  let around =
    List.concat_map
      (fun p ->
        List.concat_map
          (fun d ->
            let c = Int64.add (code wide p) (Int64.of_int d) in
            [ c; Int64.logor c (sign wide) ])
          [ -3; -2; -1; 0; 1; 2; 3 ])
      s.points
  in
  let fixed = Array.of_list (edges wide @ around) in
  let m = Array.length fixed in
  let spread i =
    let step = if wide then Int64.div Int64.max_int 128L else 0x02000000L in
    Int64.mul (Int64.of_int ((2 * i) + 1)) step
  in
  Array.init n (fun i -> if i < m then fixed.(i) else spread (i - m))

let example k =
  let block wide =
    let cols =
      Array.init (arity k) (fun j ->
          let c = example_codes wide (spec wide k j) in
          Array.init n (fun i -> c.((i + j) mod n)))
    in
    Array.init n (fun i -> Array.map (fun c -> c.(i)) cols)
  in
  { f32 = block false; f64 = block true }

(* Laws *)

let show wide c =
  Printf.sprintf "%h (%#Lx)" (value wide c)
    (if wide then c else Int64.logand c low32)

let same wide a b = Int64.equal a b || (is_nan wide a && is_nan wide b)

let compiled (type s d) (module S : Nx.Devices with type d = d) k
    (xs : (float, s, Nx.host) Nx.t list) =
  let before = Rune.evaluations () in
  let on x = Nx.place S.on x in
  let y =
    match (k.fn, xs) with
    | One { f1 }, [ x ] ->
        Rune.jit Nx.Ptree.(tensor @-> returns tensor) f1 (on x)
    | _ -> invalid_arg "Kinds: operands of another arity"
  in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  Nx.place Nx.Host.on y

let eager k xs =
  match (k.fn, xs) with
  | One { f1 }, [ x ] -> f1 x
  | _ -> invalid_arg "Kinds: operands of another arity"

(* A block's operands as host tensors and a result's codes. *)
let tensor32 cs =
  Nx.bitcast Nx.float32
    (Nx.create Nx.int32 [| Array.length cs |] (Array.map Int64.to_int32 cs))

let codes32 y =
  Array.map
    (fun c -> Int64.logand (Int64.of_int32 c) low32)
    (Nx.to_array (Nx.bitcast Nx.int32 y))

let tensor64 cs =
  Nx.bitcast Nx.float64 (Nx.create Nx.int64 [| Array.length cs |] cs)

let codes64 y = Nx.to_array (Nx.bitcast Nx.int64 y)
let column es j = Array.map (fun e -> e.(j)) es

(* The elements where compiled and eager differ, at most eight, then their
   count. *)
let misses k wide es ~eager ~compiled =
  let out = ref [] and count = ref 0 in
  Array.iteri
    (fun i e ->
      if not (same wide eager.(i) compiled.(i)) then begin
        incr count;
        if !count <= 8 then
          out :=
            Printf.sprintf "%s(%s) at %s: eager %s, compiled %s" k.name
              (String.concat ", " (Array.to_list (Array.map (show wide) e)))
              (if wide then "float64" else "float32")
              (show wide eager.(i))
              (show wide compiled.(i))
            :: !out
      end)
    es;
  if !count = 0 then []
  else
    List.rev !out
    @ [ Printf.sprintf "%d of %d elements differ" !count (Array.length es) ]

let run (type d) (module S : Nx.Devices with type d = d) k wide es =
  let cols = List.init (arity k) (column es) in
  if wide then
    let xs = List.map tensor64 cols in
    misses k wide es
      ~eager:(codes64 (eager k xs))
      ~compiled:(codes64 (compiled (module S) k xs))
  else
    let xs = List.map tensor32 cols in
    misses k wide es
      ~eager:(codes32 (eager k xs))
      ~compiled:(codes32 (compiled (module S) k xs))

(* Classes of an operand every kind's law covers. *)
let generic =
  let any p wide e = Array.exists (fun v -> p wide v) e in
  let payload wide v =
    Float.is_nan v
    && Int64.logand (code wide v) (Int64.shift_right_logical (mmask wide) 1)
       <> 0L
  in
  let sub wide v =
    v <> 0. && Float.abs v < if wide then Float.min_float else 0x1p-126
  in
  let greatest wide v =
    Float.abs v = if wide then Float.max_float else 0x1.fffffep127
  in
  [
    ("a NaN", any (fun _ v -> Float.is_nan v));
    ("a NaN with a payload", any payload);
    ("+0.", any (fun _ v -> v = 0. && not (Float.sign_bit v)));
    ("-0.", any (fun _ v -> v = 0. && Float.sign_bit v));
    ("+inf", any (fun _ v -> v = Float.infinity));
    ("-inf", any (fun _ v -> v = Float.neg_infinity));
    ("a subnormal", any sub);
    ("a greatest finite magnitude", any greatest);
  ]

let covers k wide es =
  let vs = Array.map (Array.map (value wide)) es in
  let dt = if wide then "float64: " else "float32: " in
  List.iter
    (fun (label, f) -> cover (dt ^ label) (Array.exists (f wide) vs))
    (generic @ k.covers)

let law (type d) (module S : Nx.Devices with type d = d) ~f64 k c =
  covers k false c.f32;
  if f64 then covers k true c.f64;
  let bad =
    run (module S) k false c.f32
    @ if f64 then run (module S) k true c.f64 else []
  in
  equal (list string) [] bad

let law (module S : Nx.Devices) = law (module S)

(* Narrow floats *)

type narrow = Narrow : string * (float, 's) Nx.dtype * int -> narrow

let narrows =
  Nx.
    [
      Narrow ("float16", float16, 16);
      Narrow ("bfloat16", bfloat16, 16);
      Narrow ("float8_e4m3fn", float8_e4m3fn, 8);
      Narrow ("float8_e5m2", float8_e5m2, 8);
      Narrow ("float4_e2m1fn", float4_e2m1fn, 4);
    ]

(* float4_e2m1fn's codes are its 16 values. *)
let float4 = [ 0.; 0.5; 1.; 1.5; 2.; 3.; 4.; 6. ]
let float4 = Array.of_list (float4 @ List.map Float.neg float4)

(* [codes dt bits is] is the codes of [dt] numbered [is]. *)
let narrow_codes (type s) (dt : (float, s) Nx.dtype) bits is =
  let m = Array.length is in
  match bits with
  | 16 -> Nx.bitcast dt (Nx.create Nx.uint16 [| m |] is)
  | 8 -> Nx.bitcast dt (Nx.create Nx.uint8 [| m |] is)
  | _ ->
      Nx.cast dt
        (Nx.create Nx.float64 [| m |] (Array.map (Array.get float4) is))

(* Every code, or for two operands every pair of codes, but for the 16-bit
   floats a grid: every 257th code and the zeros, ones and infinities of both
   signs. *)
let numbers k bits =
  let all = Array.init (if bits = 16 then 0 else 1 lsl bits) Fun.id in
  let grid =
    Array.of_list
      (List.init 256 (fun i -> i * 257)
      @ [
          0x8000; 0x3C00; 0xBC00; 0x3F80; 0xBF80; 0x7C00; 0xFC00; 0x7F80; 0xFF80;
        ])
  in
  match arity k with
  | 1 -> [ (if bits = 16 then Array.init 0x10000 Fun.id else all) ]
  | _ ->
      let base = if bits = 16 then grid else all in
      let m = Array.length base in
      [
        Array.init (m * m) (fun i -> base.(i / m));
        Array.init (m * m) (fun i -> base.(i mod m));
      ]

let narrow (type d) (module S : Nx.Devices with type d = d) k =
  let check (Narrow (label, dt, bits)) =
    let xs = List.map (narrow_codes dt bits) (numbers k bits) in
    let floats y = Nx.to_array (Nx.cast Nx.float64 y) in
    let e = floats (eager k xs) and c = floats (compiled (module S) k xs) in
    let ins = List.map floats xs in
    let out = ref [] and count = ref 0 in
    Array.iteri
      (fun i v ->
        let w = c.(i) in
        let ok =
          Int64.equal (Int64.bits_of_float v) (Int64.bits_of_float w)
          || (Float.is_nan v && Float.is_nan w)
        in
        if not ok then begin
          incr count;
          if !count <= 4 then
            out :=
              Printf.sprintf "%s(%s) at %s: eager %h, compiled %h" k.name
                (String.concat ", "
                   (List.map (fun a -> Printf.sprintf "%h" a.(i)) ins))
                label v w
              :: !out
        end)
      e;
    if !count = 0 then []
    else
      List.rev !out
      @ [
          Printf.sprintf "%d of %d elements differ at %s" !count
            (Array.length e) label;
        ]
  in
  equal (list string) [] (List.concat_map check narrows)

let narrow (module S : Nx.Devices) = narrow (module S)

(* Folded constants *)

(* The constants of one operand: each point and its neighbours, of both signs,
   and the edges but the NaNs. *)
let constants wide s =
  let around =
    List.concat_map
      (fun p ->
        List.concat_map
          (fun d ->
            let v = value wide (Int64.add (code wide p) (Int64.of_int d)) in
            [ v; -.v ])
          [ -1; 0; 1 ])
      s.points
  in
  List.filter_map
    (fun c ->
      let v = value wide c in
      if Float.is_nan v then None else Some v)
    (edges wide)
  @ around @ [ Float.nan ]

let folded_at (type s d) (module S : Nx.Devices with type d = d) k
    (dt : (float, s) Nx.dtype) wide =
  let cs = List.init (arity k) (fun i -> constants wide (spec wide k i)) in
  let tuples =
    match cs with
    | [ a ] -> List.map (fun v -> [ v ]) a
    | [ a; b ] ->
        let b = Array.of_list b in
        List.mapi (fun i v -> [ v; b.(i mod Array.length b) ]) a
    | _ -> invalid_arg "Kinds: operands of another arity"
  in
  let at x vs =
    let full v = Nx.full_like x v in
    let y =
      match (k.fn, vs) with
      | One { f1 }, [ a ] -> f1 (full a)
      | _ -> invalid_arg "Kinds: operands of another arity"
    in
    Nx.mul x y
  in
  let f x = Nx.concatenate ~axis:0 (List.map (at x) tuples) in
  let x = Nx.full dt [| 1 |] 1. in
  let e = Nx.to_array (f x) in
  let before = Rune.evaluations () in
  let c = Nx.to_array (Nx.place Nx.Host.on (Rune.jit' f (Nx.place S.on x))) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  let code v = code wide v in
  List.concat
    (List.mapi
       (fun i vs ->
         if same wide (code e.(i)) (code c.(i)) then []
         else
           [
             Printf.sprintf "%s(%s) folded at %s: eager %h, compiled %h" k.name
               (String.concat ", " (List.map (Printf.sprintf "%h") vs))
               (if wide then "float64" else "float32")
               e.(i) c.(i);
           ])
       tuples)

let folded (type d) (module S : Nx.Devices with type d = d) ~f64 k =
  let bad =
    folded_at (module S) k Nx.float32 false
    @ if f64 then folded_at (module S) k Nx.float64 true else []
  in
  equal (list string) [] bad

let folded (module S : Nx.Devices) = folded (module S)
