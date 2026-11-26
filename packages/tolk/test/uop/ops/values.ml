(* Evaluating: bounds and arithmetic on constants. *)

open Windtrap
open Tolk
open Common

let str pp x = Format.asprintf "%a" pp x
let alu dt o args = Ops.exec_alu o dt args
let exact dt o args = Ops.exec_alu ~truncate_output:false o dt args

let int_dtypes =
  Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64; Weak_int ]

let gen_int_dtype = Gen.of_list ~pp:(Testable.pp dtype) int_dtypes
let small = Gen.map (fun n -> `Int (Bigint.of_int n)) (Gen.int_range (-300) 300)
let gen_int = Gen.map (fun n -> `Int n) Dtypes.integer

(* The value of a shift, by what a count means in a type of [w] bits: past the
   width, a left shift is 0 and a right shift the sign fill; below it, [x 2^k]
   modulo [2^w] or the floor of [x / 2^k]. *)
let shift_reference dt o x k : Dtype.value =
  let width = Bigint.of_int (Dtype.bitsize dt) in
  let past = Bigint.geq k width in
  match (o : Op.t) with
  | Shl when past -> `Int Bigint.zero
  | Shl -> Dtype.truncate dt (`Int (Bigint.shift_left x (Bigint.to_int k)))
  | _ when past ->
      `Int (if Bigint.sign x < 0 then Bigint.minus_one else Bigint.zero)
  | _ -> `Int (Bigint.shift_right x (Bigint.to_int k))

(* A type's every operand at its extremes, -1 to 2, and counts either side of
   the width and of 2^8, 2^16, 2^32, 2^63 and 2^64. *)
let gen_shift =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp Dtype.ints in
  let in_type v d =
    Dtype.Value.to_z (Dtype.truncate dt Dtype.Value.(v + of_int d))
  in
  let mn = Dtype.min dt and mx = Dtype.max dt in
  let xs =
    [ in_type mn 0; in_type mn 1; in_type mx (-1); in_type mx 0 ]
    @ List.map (fun n -> in_type (Dtype.Value.of_int n) 0) [ -1; 0; 1; 2 ]
  in
  let powers =
    List.map (fun n -> Bigint.shift_left Bigint.one n) [ 8; 16; 32; 62; 63; 64 ]
  in
  let around c = Bigint.[ pred c; c; succ c ] in
  let width = Bigint.of_int (Dtype.bitsize dt) in
  let counts = List.concat_map around (width :: powers) in
  let z = Testable.pp Dtypes.z in
  let+ o = of_list ~pp:Op.pp Op.[ Shl; Shr ]
  and+ x = of_list ~pp:z xs
  and+ k = of_list ~pp:z counts in
  (dt, o, x, k)

(* Bounds *)

let variable dt lo hi name = Ops.variable ~dtype:dt name lo hi

let binary_bounds cell =
  let o = op_of_cell (cell "op") and dt = dtype_of_cell (cell "dtype") in
  let x =
    variable dt (value_of_cell (cell "a_lo")) (value_of_cell (cell "a_hi")) "a"
  in
  let y =
    variable dt (value_of_cell (cell "b_lo")) (value_of_cell (cell "b_hi")) "b"
  in
  check_bounds (Ops.alu x o [ y ])
    (value_of_cell (cell "vmin"), value_of_cell (cell "vmax"))

let cast_bounds cell =
  let x =
    variable
      (dtype_of_cell (cell "from"))
      (value_of_cell (cell "lo"))
      (value_of_cell (cell "hi"))
      "x"
  in
  check_bounds
    (Ops.cast x (dtype_of_cell (cell "to")))
    (value_of_cell (cell "vmin"), value_of_cell (cell "vmax"))

let load_bounds cell =
  let dt = dtype_of_cell (cell "dtype") in
  let hex = cell "bytes" in
  let data =
    String.init
      (String.length hex / 2)
      (fun k -> Char.chr (int_of_string ("0x" ^ String.sub hex (2 * k) 2)))
  in
  let table = Ops.v ~arg:(Bytes data) Op.Binary in
  let table = if Dtype.equal dt Uint8 then table else Ops.bitcast table dt in
  let r =
    Ops.range ~axis_type:Weak
      (Int (String.length data / Dtype.itemsize dt))
      [ 0 ]
  in
  check_bounds
    (Ops.load (Ops.index table [ r ]) [])
    (value_of_cell (cell "vmin"), value_of_cell (cell "vmax"))

(* The reference interpreter: the value of an integer expression over variables,
   each operation computed exactly. It discards a case where an operation leaves
   its type, where the result is undefined. *)
let rec eval env u =
  let fits dt v =
    match v with
    | `Int n when Dtype.is_int dt && not (Dtype.equal dt Weak_int) -> (
        match (Dtype.min dt, Dtype.max dt) with
        | `Int lo, `Int hi -> Bigint.leq lo n && Bigint.leq n hi
        | _ -> true)
    | _ -> true
  in
  let v =
    match (Ops.op u, Ops.arg u) with
    | Op.Param, Param { name = Some name; _ } -> List.assoc name env
    | Op.Const, Const (#Dtype.value as c) -> c
    | Op.Cast, Dtype dt -> (
        match eval env (Ops.nth u 0) with
        | `Bool b ->
            if Dtype.is_bool dt then `Bool b
            else `Int (if b then Bigint.one else Bigint.zero)
        | v -> v)
    | o, _ -> (
        match
          exact (Ops.dtype u) o
            (List.map (fun s -> (eval env s :> Dtype.const)) (Ops.src u))
        with
        | #Dtype.value as v -> v
        | `Invalid -> fail "Invalid in an expression without one")
  in
  assume (fits (Ops.dtype u) v);
  v

let compare_value (v0 : Dtype.value) (v1 : Dtype.value) =
  match (v0, v1) with
  | `Int a, `Int b -> Bigint.compare a b
  | `Bool a, `Bool b -> Bool.compare a b
  | `Float a, `Float b -> Float.compare a b
  | _ -> invalid_arg "values of different kinds"

let ordered_value = Testable.with_compare compare_value value
let gen_point = Gen.pair (Gen.int_range (-8) 8) (Gen.int_range 1 5)

(* [held u v] is that [u]'s bounds hold its value [v]. *)
let held u v =
  at_most ordered_value ~than:v (Ops.vmin u);
  at_least ordered_value ~than:v (Ops.vmax u)

(* An operation on a committed integer, as compiled code computes it, wraps at
   the type's width. [x o y], then [o'] with [x] again, over narrow variables of
   61 values starting anywhere in their type, at a point of each. *)
let gen_wrapping =
  let narrow = Dtype.[ Int8; Uint8; Int16; Uint16 ] in
  let ops = Op.[ Add; Sub; Mul; Xor; Shl ] in
  Gen.triple
    (Gen.of_list ~pp:(Testable.pp dtype) narrow)
    (Gen.pair (Gen.of_list ~pp:Op.pp ops) (Gen.of_list ~pp:Op.pp ops))
    (Gen.triple (Gen.int_range 0 65535) (Gen.int_range 0 65535)
       (Gen.int_range 0 60))

let wrapping_case (dt, (o0, o1), (a, b, k)) =
  let lo = Bigint.to_int (Dtype.Value.to_z (Dtype.min dt))
  and hi = Bigint.to_int (Dtype.Value.to_z (Dtype.max dt)) in
  let start n = min hi (lo + (n mod (hi - lo + 1))) in
  let x0 = start a and y0 = start b in
  let x1 = min hi (x0 + 60) and y1 = min hi (y0 + 60) in
  let x = variable dt (i x0) (i x1) "x" and y = variable dt (i y0) (i y1) "y" in
  let apply u o v =
    if o = Op.Shl then Ops.O.(u lsl Ops.int ~dtype:dt 3) else Ops.alu u o [ v ]
  in
  let u = apply (apply x o0 y) o1 x in
  let vars = [ ("x", i (min x1 (x0 + k))); ("y", i (max y0 (y1 - k))) ] in
  (u, vars)

(* A float sum, difference or product of variables over intervals of a float
   type's values, at a point of each, or a selection of one by comparing them
   ([Op.Where]: [x] where [x < y], else [x] where [y < x], else [y]). Magnitudes
   span the type's exponents, so that sums overflow and products fall to the
   subnormals. *)
let gen_float_bounds =
  let open Gen in
  let* dt =
    of_list ~pp:(Testable.pp dtype)
      Dtype.[ Float16; Bfloat16; Float32; Float64 ]
  in
  let e, m = Dtype.finfo dt in
  let emax = 1 lsl (e - 1) in
  let magnitude =
    let+ x = float_range (-2.) 2. and+ k = int_range (-emax - m) emax in
    Float.ldexp x k
  in
  let+ o = of_list ~pp:Op.pp Op.[ Add; Sub; Mul; Where ]
  and+ ends = quad magnitude magnitude magnitude magnitude
  and+ at = pair (float_range 0. 1.) (float_range 0. 1.) in
  (dt, o, ends, at)

(* The interval ops.mli states for [x o y], [x] in [a0, a1] and [y] in [b0, b1]:
   the exact interval of the operands' bounds, each end within the subnormals
   taken as itself or 0, as a target that flushes them reads it, widened by a
   relative [2^-m] and the type's least normal. Both this reference and the
   bounds are computed in doubles: a corner, its widening and their sum are each
   rounded by at most [2^-53] of a magnitude below [2 (|corner| + least
   normal)], so each errs by under [2^-50] of [|corner| + least normal], and a
   further [2^-48] of it covers both. The slack exceeds Float64's [2^-52], so
   Float64's tightness is checked only to a relative [2^-48]; the narrower
   types' [2^-m] lie far above the slack. *)
let stated_interval dt o (a0, a1) (b0, b1) =
  let e, m = Dtype.finfo dt in
  let tiny = Float.ldexp 1. (2 - (1 lsl (e - 1))) in
  let ends lo hi =
    ( (if 0. < lo && lo < tiny then 0. else lo),
      if -.tiny < hi && hi < 0. then 0. else hi )
  in
  let (a0, a1), (b0, b1) = (ends a0 a1, ends b0 b1) in
  let lo, hi =
    match (o : Op.t) with
    | Add -> (a0 +. b0, a1 +. b1)
    | Sub -> (a0 -. b1, a1 -. b0)
    | _ ->
        let corners = [ a0 *. b0; a0 *. b1; a1 *. b0; a1 *. b1 ] in
        ( List.fold_left Float.min Float.infinity corners,
          List.fold_left Float.max Float.neg_infinity corners )
  in
  let widen x sign =
    let slack = (Float.abs x +. tiny) *. Float.ldexp 1. (-48) in
    x +. (sign *. ((Float.abs x *. Float.ldexp 1. (-m)) +. tiny +. slack))
  in
  (widen lo (-1.), widen hi 1.)

let in_type dt x =
  match Dtype.truncate dt (`Float x) with
  | `Float y -> Float.is_finite y
  | _ -> false

let float_bounds_case (dt, o, (a0, a1, b0, b1), (s, t)) =
  let value x =
    match Dtype.truncate dt (`Float x) with `Float y -> y | _ -> Float.nan
  in
  let interval x0 x1 = (value (Float.min x0 x1), value (Float.max x0 x1)) in
  let (a0, a1), (b0, b1) = (interval a0 a1, interval b0 b1) in
  assume (List.for_all Float.is_finite [ a0; a1; b0; b1 ]);
  let point lo hi s =
    value (Float.min hi (Float.max lo ((lo *. (1. -. s)) +. (hi *. s))))
  in
  let x = variable dt (f a0) (f a1) "x" and y = variable dt (f b0) (f b1) "y" in
  let u =
    if Op.equal o Where then
      Ops.where (Ops.lt x y) x (Ops.where (Ops.lt y x) x y)
    else Ops.alu x o [ y ]
  in
  (match
     Interpreter.eval
       ~vars:[ ("x", f (point a0 a1 s)); ("y", f (point b0 b1 t)) ]
       u
   with
  | #Dtype.value as v -> held u v
  | `Invalid -> fail "Invalid in an expression without one");
  if Op.equal o Where then begin
    (* A selection's bounds lie within its branches'. *)
    at_least ordered_value ~than:(f (Float.min a0 b0)) (Ops.vmin u);
    at_most ordered_value ~than:(f (Float.max a1 b1)) (Ops.vmax u)
  end
  else
    let lo, hi = stated_interval dt o (a0, a1) (b0, b1) in
    let overflows = not (in_type dt lo && in_type dt hi) in
    cover "an operation that may overflow its type" overflows;
    cover "an operation within its type" (not overflows);
    if overflows then check_bounds u (Dtype.min dt, Dtype.max dt)
    else begin
      at_least ordered_value ~than:(f lo) (Ops.vmin u);
      at_most ordered_value ~than:(f hi) (Ops.vmax u)
    end

(* A cast or a bit reinterpretation of a variable over an interval of its type,
   at a point of it, an end as often as not. Integer ends are drawn at every
   magnitude, wrapping into the type, and as often near 0, where a narrower type
   holds them; float ends span the type's exponents and the infinities. *)
let cast_types =
  (Dtype.Bool :: int_dtypes) @ Dtype.[ Float16; Bfloat16; Float32; Float64 ]

let gen_in dt =
  let open Gen in
  if Dtype.is_bool dt then map (fun b -> `Bool b) bool
  else if Dtype.is_float dt then
    let e, _ = Dtype.finfo dt in
    let emax = 1 lsl (e - 1) in
    let finite =
      let+ x = float_range (-2.) 2. and+ k = int_range (-emax) emax in
      match Dtype.truncate dt (`Float (Float.ldexp x k)) with
      | `Float y when Float.is_finite y -> `Float y
      | _ -> `Float 0.
    in
    frequency
      [
        (6, finite);
        (1, of_list [ `Float Float.infinity; `Float Float.neg_infinity ]);
      ]
  else
    map
      (fun n -> (Dtype.truncate dt (`Int n) :> Dtype.value))
      (frequency
         [ (1, Dtypes.integer); (1, map Bigint.of_int (int_range (-200) 200)) ])

let pp_interval ppf (lo, v, hi) =
  Format.fprintf ppf "%a in [%a, %a]" Dtype.pp_const v Dtype.pp_const lo
    Dtype.pp_const hi

let gen_interval dt =
  let open Gen in
  with_pp pp_interval
    (let+ a = gen_in dt
     and+ b = gen_in dt
     and+ c = gen_in dt
     and+ at = int_range 0 3 in
     match List.sort compare_value [ a; b; c ] with
     | [ lo; v; hi ] -> (lo, (match at with 0 -> lo | 1 -> hi | _ -> v), hi)
     | _ -> invalid_arg "three values")

let gen_cast =
  let open Gen in
  let* src = of_list ~pp:(Testable.pp dtype) cast_types in
  let+ dst = of_list ~pp:(Testable.pp dtype) (Dtype.Weak_float :: cast_types)
  and+ ends = gen_interval src in
  (src, dst, ends)

(* An integer type of one width, signed or unsigned, to the other. *)
let gen_bitcast =
  let open Gen in
  let widths =
    Dtype.[ (Int8, Uint8); (Int16, Uint16); (Int32, Uint32); (Int64, Uint64) ]
  in
  let pp ppf (a, b) = Format.fprintf ppf "%a, %a" Dtype.pp a Dtype.pp b in
  let* signed, unsigned = of_list ~pp widths in
  let* to_unsigned = bool in
  let src, dst =
    if to_unsigned then (signed, unsigned) else (unsigned, signed)
  in
  let+ ends = gen_interval src in
  (src, dst, ends)

(* A float that leaves an integer type, a weak one's 800 bits included, has no
   value there. *)
let defined_cast dst (v : Dtype.value) =
  match v with
  | `Float x when Dtype.is_int dst ->
      Float.is_finite x
      &&
      let t = `Int (Bigint.of_float (Float.trunc x)) in
      compare_value (Dtype.min dst) t <= 0
      && compare_value t (Dtype.max dst) <= 0
  | _ -> true

(* Bounds may be of another kind than the value: a cast of a boolean to a float
   keeps its operand's [`Bool] bounds. They compare by magnitude. *)
let by_magnitude = Testable.with_compare Dtype.Value.compare value

let bounds_hold u v =
  match Interpreter.eval ~vars:[ ("x", v) ] u with
  | #Dtype.value as w ->
      at_most by_magnitude ~than:w (Ops.vmin u);
      at_least by_magnitude ~than:w (Ops.vmax u)
  | `Invalid -> fail "Invalid in an expression without one"

let full dt = (Dtype.min dt, Dtype.max dt)

let bounds_group =
  let bounds_of = pair Dtypes.value Dtypes.value in
  let cond = Ops.O.(weak_var "a" 0 8 < int 4) in
  group "vmin and vmax"
    [
      group
        "bound each operation of two intervals as binary_bounds.golden states"
        [
          Golden.cases "binary_bounds.golden"
            ~key:[ "op"; "dtype"; "a_lo"; "a_hi"; "b_lo"; "b_hi" ]
            binary_bounds;
        ];
      group "bound each load of a constant table as load_bounds.golden states"
        [ Golden.cases "load_bounds.golden" load_bounds ];
      group "bound each cast as cast_bounds.golden states"
        [
          Golden.cases "cast_bounds.golden"
            ~key:[ "from"; "lo"; "hi"; "to" ]
            cast_bounds;
        ];
      prop "hold every value an expression of integers takes"
        (Gen.pair Nodes.gen_recipe gen_point) (fun (r, (x0, x1)) ->
          let u = Nodes.build (Nodes.leaves ()) r in
          held u (eval [ ("v0", i x0); ("v1", i x1) ] u));
      prop "hold the value a committed integer wraps to" gen_wrapping
        (fun case ->
          let u, vars = wrapping_case case in
          match Interpreter.eval ~vars u with
          | #Dtype.value as v -> held u v
          | `Invalid -> fail "Invalid in an expression without one");
      prop "hold a shift's value, for a count of any size" gen_shift
        (fun (dt, o, x, k) ->
          let v = `Int x in
          let u =
            Ops.alu (Ops.variable ~dtype:dt "x" v v) o [ Ops.const (`Int k) ]
          in
          let w = shift_reference dt o x k in
          at_most by_magnitude ~than:w (Ops.vmin u);
          at_least by_magnitude ~than:w (Ops.vmax u));
      prop "hold every value a cast takes"
        ~examples:
          [
            (Dtype.Int16, Dtype.Int8, (i 0, i 5, i 100));
            (Dtype.Uint16, Dtype.Uint8, (i 0, i 300, i 1000));
            (Dtype.Float32, Dtype.Int32, (f (-2.5), f 0., f 7.25));
          ]
        gen_cast
        (fun (src, dst, (lo, v, hi)) ->
          let u = Ops.cast (variable src lo hi "x") dst in
          assume (defined_cast dst v);
          let integers = Dtype.is_int src && Dtype.is_int dst in
          cover "an integer interval another integer type holds"
            (integers
            && (not (Dtype.equal src dst))
            && (not (Dtype.equal dst Weak_int))
            && compare_value (Dtype.min dst) lo <= 0
            && compare_value hi (Dtype.max dst) <= 0);
          cover "an integer interval another integer type wraps"
            (integers
            && (not (Dtype.equal dst Weak_int))
            && compare_value (Dtype.max dst) hi < 0);
          cover "a float cast to an integer type"
            (Dtype.is_float src && Dtype.is_int dst);
          bounds_hold u v);
      prop "hold every value a bitcast between integers takes"
        ~examples:[ (Dtype.Int8, Dtype.Uint8, (i 0, i 3, i 100)) ]
        gen_bitcast
        (fun (src, dst, (lo, v, hi)) ->
          cover "an interval both types hold"
            (compare_value (i 0) lo <= 0
            && compare_value hi (Dtype.max src) <= 0
            && compare_value hi (Dtype.max dst) <= 0);
          bounds_hold (Ops.bitcast (variable src lo hi "x") dst) v);
      prop
        "of a float sum, difference, product or selection hold every value it \
         rounds to, within the interval ops.mli states"
        ~examples:
          [
            (Dtype.Float16, Op.Add, (6e4, 6.5e4, 6e4, 6.5e4), (0.5, 0.5));
            (Dtype.Float32, Op.Mul, (1e-30, 2e-30, 1e-10, 3e-10), (0.5, 0.5));
          ]
        gen_float_bounds float_bounds_case;
      test
        "of a float operation of an operand within the subnormals hold the \
         operand flushed" (fun () ->
          let x = variable Float32 (f 1e5) (f 2e5) "x" in
          let u = Ops.alu x Op.Mul [ Ops.float ~dtype:Float32 1e-40 ] in
          at_most ordered_value ~than:(f 0.) (Ops.vmin u);
          at_least ordered_value ~than:(f 2e-35) (Ops.vmax u));
      test
        "of a float operation of an unbounded operand, or that may overflow, \
         are its type's" (fun () ->
          let one dt = Ops.float ~dtype:dt 1. in
          let h = variable Float16 (f 0.) (f 65504.) "h" in
          let w = variable Weak_float (f 0.) (f 1.) "w" in
          check_bounds
            (Ops.alu (Shape.param 0 Float32) Op.Add [ one Float32 ])
            (full Float32);
          check_bounds (Ops.alu h Op.Add [ h ]) (full Float16);
          check_bounds (Ops.alu h Op.Mul [ h ]) (full Float16);
          check_bounds (Ops.alu w Op.Add [ one Weak_float ]) (full Weak_float));
      test
        "of a float selection by a < b narrow the branch it picks to where \
         that holds" (fun () ->
          let x = variable Float32 (f (-10.)) (f 10.) "x" in
          let p = Shape.param 0 Float32 in
          let c = Ops.float ~dtype:Float32 2. in
          let minus_c = Ops.float ~dtype:Float32 (-2.) in
          check_bounds (Ops.where (Ops.lt x c) x minus_c) (f (-10.), f 2.);
          check_bounds (Ops.where (Ops.lt minus_c x) x c) (f (-2.), f 10.);
          check_bounds (Ops.where (Ops.lt x c) c x) (f (-10.), f 10.);
          let above = Ops.where (Ops.lt minus_c p) p minus_c in
          check_bounds above (f (-2.), f Float.infinity);
          check_bounds (Ops.where (Ops.lt above c) above c) (f (-2.), f 2.));
      test "of a float truncation and negation map their operand's" (fun () ->
          let x = variable Float32 (f (-2.5)) (f 3.75) "x" in
          check_bounds (Ops.alu x Op.Trunc []) (f (-2.), f 3.);
          check_bounds (Ops.alu x Op.Neg []) (f (-3.75), f 2.5));
      test
        "of a committed float selection hold a weak constant at its type, \
         where it may round to zero" (fun () ->
          let p = Shape.param 0 Float32 in
          let inf = Ops.float ~dtype:Float32 Float.infinity in
          let tiny = Ops.float 0x0.0000000000001p-1022 in
          let u = Ops.where (Ops.lt p (Ops.float ~dtype:Float32 0.)) inf tiny in
          match Interpreter.eval ~params:[ (0, f 1.) ] u with
          | #Dtype.value as v ->
              equal value (f 0.) v;
              held u v
          | `Invalid -> fail "Invalid in an expression without one");
      prop "vmin is at most vmax of an expression of integers" Nodes.gen_recipe
        (fun r ->
          let u = Nodes.build (Nodes.leaves ()) r in
          at_most ordered_value ~than:(Ops.vmax u) (Ops.vmin u));
      cases "of a constant are the constant"
        ~id:(fun (what, _, _) -> what)
        (let edge what dt v =
           (what, Ops.const ~dtype:dt (v :> Dtype.const), (v, v))
         in
         [
           ("42", Ops.int 42, int_bounds 42 42);
           ("true", Ops.bool true, (`Bool true, `Bool true));
           ("1.5", Ops.float ~dtype:Float32 1.5, (f 1.5, f 1.5));
           ("-0.", Ops.float (-0.), (f (-0.), f (-0.)));
           edge "int64's greatest" Int64 (Dtype.max Int64);
           edge "int64's least" Int64 (Dtype.min Int64);
           edge "uint64's greatest" Uint64 (Dtype.max Uint64);
         ])
        (fun (_, u, expected) -> check_bounds u expected);
      cases "of a NaN constant and of invalid are their type's"
        ~id:(fun (what, _) -> what)
        [
          ("a NaN", Ops.float ~dtype:Float32 Float.nan); ("invalid", Ops.invalid);
        ]
        (fun (_, u) -> check_bounds u (full (Ops.dtype u)));
      test "of a node nothing bounds are its type's" (fun () ->
          check_bounds (Shape.param 7 Uint64) (full Uint64));
      test "are exact integers, whatever their size" (fun () ->
          let huge = Bigint.shift_left Bigint.one 200 in
          let c n = Ops.const (`Int n) in
          check_bounds Ops.O.(c huge - c (Bigint.pred huge)) (i 1, i 1);
          check_bounds
            Ops.O.(c huge * c huge)
            (`Int (Bigint.mul huge huge), `Int (Bigint.mul huge huge));
          check_bounds
            Ops.O.(c huge lsl int 100)
            ( `Int (Bigint.shift_left huge 100),
              `Int (Bigint.shift_left huge 100) ));
      cases "of a stack span its elements', its Invalid ones left out"
        ~id:(fun (what, _, _) -> what)
        [
          ( "integers and Invalid",
            [ i 0; i 4; `Invalid; `Invalid ],
            int_bounds 0 4 );
          ("one integer", [ i 42 ], int_bounds 42 42);
          ("integers", [ i 10; i 20; i (-5); i 7 ], int_bounds (-5) 20);
          ( "booleans",
            [ `Bool true; `Bool false; `Bool false ],
            (`Bool false, `Bool true) );
          ( "floats",
            [ f 1.5; f (-3.2); f 0. ],
            (Dtype.truncate Float32 (f (-3.2)), f 1.5) );
        ]
        (fun (_, values, expected) ->
          check_bounds (Shape.consts values) expected);
      (* [vmin] is above [vmax] at an end of 0: such a node takes no value. *)
      cases "of a range or hardware index run from 0 to its end less 1"
        ~id:(fun (what, _, _) -> what)
        [
          ( "a range to 5",
            Ops.range ~axis_type:Weak (Int 5) [ 0 ],
            int_bounds 0 4 );
          ( "a range to 0",
            Ops.range ~axis_type:Weak (Int 0) [ 0 ],
            int_bounds 0 (-1) );
          ( "an index to a variable of 1 to 10",
            Ops.special (Sym (var "i" 1 10)) "gidx0",
            int_bounds 0 9 );
          ("an index to 0", Ops.special (Int 0) "lidx0", int_bounds 0 (-1));
        ]
        (fun (_, u, expected) -> check_bounds u expected);
      test "of a comparison of constants are its value" (fun () ->
          let c = Ops.int 42 in
          check_bounds (Ops.ne c (Ops.int 42)) (`Bool false, `Bool false);
          check_bounds (Ops.ne c (Ops.int 43)) (`Bool true, `Bool true);
          check_bounds (Ops.ne c (Ops.int 41)) (`Bool true, `Bool true));
      test "of a variable offset or scaled move with it" (fun () ->
          let x = weak_var "x" 10 20 in
          check_bounds Ops.O.(x + int 5) (int_bounds 15 25);
          check_bounds Ops.O.(x - int 5) (int_bounds 5 15);
          check_bounds Ops.O.(int 5 - x) (int_bounds (-15) (-5));
          check_bounds Ops.O.(weak_var "y" (-3) 4 * int 2) (int_bounds (-6) 8);
          check_bounds
            Ops.O.(weak_var "y" 2 5 * int (-3))
            (int_bounds (-15) (-6));
          check_bounds
            Ops.O.(weak_var "y" (-2) 5 * int (-3))
            (int_bounds (-15) 6));
      test "of a mask of a variable lie within the mask" (fun () ->
          let x = weak_var "x" 10 20 in
          check_bounds Ops.O.(x land int 5) (int_bounds 0 5);
          check_bounds Ops.O.(x land int 15) (int_bounds 0 15);
          check_bounds Ops.O.(x land int 32) (int_bounds 0 0);
          check_bounds
            Ops.O.(var "x" (-100) 100 land int 0x7FFFFFFF)
            (int_bounds 0 0x7FFFFFFF));
      test "of a negative mask of a non-negative operand lie within the operand"
        (fun () ->
          check_bounds Ops.O.(weak_var "x" 0 20 land int (-4)) (int_bounds 0 20);
          check_bounds Ops.O.(weak_var "x" 3 20 land int (-4)) (int_bounds 0 20));
      test
        "of an integer selection of the lesser of two values are their \
         minimum's" (fun () ->
          let a = weak_var "a" 0 10 and b = weak_var "b" 4 6 in
          check_bounds (Ops.where Ops.O.(a < b) a b) (int_bounds 0 6);
          check_bounds (Ops.where Ops.O.(b < a) a b) (int_bounds 0 10));
      test "of a maximum then a minimum clamp" (fun () ->
          check_bounds
            (Ops.minimum
               (Ops.maximum (weak_var "x" 0 10) (Ops.int 5))
               (Ops.int 8))
            (int_bounds 5 8));
      test "of a selection span both branches" (fun () ->
          let x = weak_var "x" 0 10 in
          check_bounds
            (Ops.where
               Ops.O.(x < int 5)
               (weak_var "y" 1 11) (weak_var "z" 2 12))
            (int_bounds 1 12));
      test "of a shift by a negative constant are its type's" (fun () ->
          let x = var "x" 0 10 in
          let count = Ops.cast (Ops.int (-1)) Int32 in
          check_bounds Ops.O.(x lsl count) (full Int32);
          check_bounds Ops.O.(x lsr count) (full Int32));
      test "of a product with an unbounded float are infinite, never NaN"
        (fun () ->
          let y =
            Ops.load
              (Ops.index
                 (Shape.param ~shape:(ints [ 1 ]) 0 Float32)
                 [ Ops.int 0 ])
              []
          in
          check_bounds
            Ops.O.(float 0. * y)
            (f Float.neg_infinity, f Float.infinity));
      test "of a quotient of a load of an integer buffer follow its type's"
        (fun () ->
          let v =
            Ops.load
              (Ops.index
                 (Shape.param ~shape:(ints [ 1 ]) 1 Int32)
                 [ Ops.int 0 ])
              []
          in
          check_bounds Ops.O.(v // int 32) (int_bounds (-67108864) 67108863));
      test "of a load from constant bytes span the bytes" (fun () ->
          let table = Ops.v ~arg:(Bytes "\x03\x09\x01") Op.Binary in
          let r = Ops.range ~axis_type:Weak (Int 3) [ 0 ] in
          check_bounds (Ops.load (Ops.index table [ r ]) []) (int_bounds 1 9));
      test "of a pad take in its zeros" (fun () ->
          let x = Shape.expand (Ops.int ~dtype:Int32 5) (ints [ 2 ]) in
          check_bounds (Shape.pad x [ Some (Int 1, Int 1) ]) (int_bounds 0 5));
      test "of a copy and a contiguous are their source's" (fun () ->
          let src = Ops.O.(Shape.placeholder ~slot:0 [ 4 ] Int32 land int 3) in
          check_bounds (Ops.contiguous src) (int_bounds 0 3);
          check_bounds (Ops.copy_to_device src "NULL") (int_bounds 0 3));
      test "of a cast of a variable to a float, a bool or an unsigned type"
        (fun () ->
          let x = var "x" (-10) 10 in
          check_bounds (Ops.cast x Float32) (f (-10.), f 10.);
          check_bounds (Ops.cast x Bool) (`Bool false, `Bool true);
          check_bounds (Ops.cast x Uint32) (full Uint32));
      test
        "of a typed integer constant outside its type are its wrapped value, a \
         non-finite one's its type's" (fun () ->
          check_bounds (Ops.int ~dtype:Int8 300) (int_bounds 44 44);
          check_bounds (Ops.int ~dtype:Uint8 (-1)) (int_bounds 255 255);
          check_bounds (Ops.cast (Ops.float Float.infinity) Int32) (full Int32);
          check_bounds
            (Ops.cast (Ops.float ~dtype:Float32 4.5) Int32)
            (int_bounds 4 4));
      test
        "of a value of a float type without infinities, which may be NaN, are \
         infinite" (fun () ->
          let unknown = (f Float.neg_infinity, f Float.infinity) in
          List.iter
            (fun dt ->
              check_bounds (Shape.param 0 dt) unknown;
              check_bounds (Ops.cast (Shape.param 0 dt) Float32) unknown;
              check_bounds (Ops.cast (Shape.param 0 Float32) dt) unknown;
              let x = variable Float32 (f (-1.)) (f 2.) "x" in
              check_bounds (Ops.cast x dt) (f (-1.), f 2.);
              let greatest = Dtype.max dt in
              let x = variable Float32 (f (-1e6)) (f 1e6) "x" in
              check_bounds (Ops.cast x dt)
                (Dtype.Value.( ~- ) greatest, greatest))
            Dtype.[ Fp8e4m3; Fp8e4m3fnuz; Fp8e5m2fnuz ]);
      test "of a committed integer that may leave its type are its type's"
        (fun () ->
          let u = var ~dtype:Uint8 "u" 0 255 and y = var ~dtype:Int8 "y" 0 50 in
          check_bounds Ops.O.(u + int 1) (full Uint8);
          check_bounds Ops.O.(u - int 1) (full Uint8);
          check_bounds Ops.O.(y + int 100) (full Int8);
          check_bounds Ops.O.(y * int 4) (full Int8);
          check_bounds Ops.O.(y lsl int 2) (full Int8);
          check_bounds Ops.O.(u lxor int (-1)) (full Uint8);
          check_bounds Ops.O.(y + int 50) (int_bounds 50 100);
          check_bounds Ops.O.(weak_var "w" 0 50 * int 4) (int_bounds 0 200));
      test "of an integer cast to a signed type it leaves wrap" (fun () ->
          let w = var "w" 0 255 in
          check_bounds (Ops.cast w Int8) (full Int8);
          check_bounds (Ops.cast (var "v" 0 100) Int8) (int_bounds 0 100);
          check_bounds (Ops.cast (Ops.cast w Uint8) Int32) (int_bounds 0 255));
      test "of a constant table holding a NaN are its type's" (fun () ->
          let bits x =
            let b = Bytes.create 4 in
            Bytes.set_int32_le b 0 (Int32.bits_of_float x);
            Bytes.to_string b
          in
          let table =
            Ops.bitcast
              (Ops.v
                 ~arg:
                   (Bytes
                      (String.concat "" (List.map bits [ 1.; Float.nan; 2. ])))
                 Op.Binary)
              Float32
          in
          let r = Ops.range ~axis_type:Weak (Int 3) [ 0 ] in
          check_bounds (Ops.load (Ops.index table [ r ]) []) (full Float32));
      test "of an unrolled arange's index follow each step" (fun () ->
          let g = weak_var "gidx0" 0 2559 in
          let alu0 = Ops.O.(g * int (-1)) in
          let quotient = Ops.O.((alu0 + int 2559) // int (-4)) in
          equal bounds_of (i 0, i 2559) (bounds g);
          equal bounds_of (i (-2559), i 0) (bounds alu0);
          equal bounds_of (i 0, i 2559) (bounds Ops.O.(alu0 + int 2559));
          equal bounds_of (i (-640), i 0) (bounds quotient);
          equal bounds_of (i 0, i 640) (bounds Ops.O.(quotient * int (-1))));
      test
        "of selections between float constants span them, one of a NaN \
         unbounded" (fun () ->
          let cond = Ops.O.(weak_var "s" 0 3 < int 2) in
          let w =
            Ops.where cond (Ops.float 0.)
              (Ops.where cond (Ops.float 1.) (Ops.float 3.))
          in
          equal bounds_of (`Float 0., `Float 3.) (bounds w);
          equal bounds_of (i 0, i 3) (bounds (Ops.cast w Int32));
          equal bounds_of
            (`Float Float.neg_infinity, `Float Float.infinity)
            (bounds (Ops.where cond (Ops.float Float.nan) (Ops.float 1.)));
          let i32 =
            Ops.cast
              (Ops.where cond (Ops.float Float.neg_infinity) (Ops.float 2.7))
              Int32
          in
          equal bounds_of
            (`Int (Bigint.of_int32 Int32.min_int), i 2)
            (bounds i32));
      test "of a gated load of a constant table take in its alternative"
        (fun () ->
          let table = Ops.v Op.Binary ~arg:(Ops.Bytes "\002\003\005") in
          let at = Ops.index table [ weak_var "t" 0 2 ] in
          let alt n = Ops.int ~dtype:Uint8 n in
          equal bounds_of (i 2, i 5) (bounds (Ops.load at []));
          equal bounds_of (i 0, i 5) (bounds (Ops.load at [ alt 0; cond ]));
          equal bounds_of (i 2, i 9) (bounds (Ops.load at [ alt 9; cond ])));
    ]

(* operand_bounds *)

let operand_bounds_group =
  let x8 = var ~dtype:Int8 "x8" 0 10 in
  group "operand_bounds"
    [
      cases
        "commits a weak integer operand of a committed integer operation, \
         wrapping one value, or the type's bounds"
        ~id:(fun (what, _, _) -> what)
        [
          ( "an operand within the type",
            Ops.O.(x8 + weak_var "w" 1 5),
            int_bounds 1 5 );
          ("one value past the type", Ops.O.(x8 + int 300), int_bounds 44 44);
          ("values past the type", Ops.O.(x8 + weak_var "w" 200 210), full Int8);
          ( "the operand of a comparison, at its operands' type",
            Ops.O.(x8 < int 300),
            int_bounds 44 44 );
        ]
        (fun (_, u, expected) ->
          equal (pair value value) expected (Ops.operand_bounds u (Ops.nth u 1)));
      cases
        "commits a weak float operand of a committed float operation, rounding \
         its bounds"
        ~id:(fun (what, _, _) -> what)
        [
          ("0.1 at float32", Dtype.Float32, 0.1);
          ("1e6 at float16", Float16, 1e6);
        ]
        (fun (_, dt, c) ->
          let rounded = Dtype.truncate dt (f c) in
          equal (pair value value) (rounded, rounded)
            (Ops.operand_bounds
               Ops.O.(fvar ~dtype:dt "x" + float c)
               (Ops.float c)));
      test "keeps a shift's count's own bounds" (fun () ->
          let count = weak_var "k" 0 100 in
          let u = Ops.O.(x8 lsl count) in
          equal (pair value value) (int_bounds 0 100)
            (Ops.operand_bounds u count));
    ]

let overflowing =
  group "overflows"
    [
      test "is whether a node's bounds reach outside the type's" (fun () ->
          let rows =
            [
              ("0 to 200 at int8", weak_var "x" 0 200, Dtype.Int8);
              ("-128 to 127 at int8", weak_var "x" (-128) 127, Dtype.Int8);
              ("-129 to 0 at int8", weak_var "x" (-129) 0, Dtype.Int8);
              ("-1 to 3 at uint8", weak_var "x" (-1) 3, Dtype.Uint8);
              ("0 to 255 at uint8", weak_var "x" 0 255, Dtype.Uint8);
            ]
          in
          equal (list string)
            [ "0 to 200 at int8"; "-129 to 0 at int8"; "-1 to 3 at uint8" ]
            (List.filter_map
               (fun (what, u, dt) ->
                 if Ops.overflows u dt then Some what else None)
               rows));
    ]

let no_ints =
  group "fits_no_int"
    [
      test "is true of known bounds that neither int64 nor uint64 holds"
        (fun () ->
          let bounded lo hi = Ops.variable "b" (`Int lo) (`Int hi) in
          let two n = Bigint.shift_left Bigint.one n in
          let r = Ops.range ~axis_type:Weak (Ops.Int 4) [ 0 ] in
          let unknown = Ops.reduce (bounded Bigint.zero (two 3)) Op.Add [ r ] in
          let rows =
            [
              ( "0 to 2^64 - 1, a uint64's",
                bounded Bigint.zero (Bigint.pred (two 64)) );
              ( "-2^63 to 2^63 - 1, an int64's",
                bounded (Bigint.neg (two 63)) (Bigint.pred (two 63)) );
              ("-1 to 2^63", bounded Bigint.minus_one (two 63));
              ("0 to 2^64", bounded Bigint.zero (two 64));
              ("a reduction nothing bounds", unknown);
              ("a sum over a reduction nothing bounds", Ops.O.(unknown + int 1));
            ]
          in
          equal (list string)
            [ "-1 to 2^63"; "0 to 2^64" ]
            (List.filter_map
               (fun (what, u) -> if Ops.fits_no_int u then Some what else None)
               rows));
    ]

let exactness =
  group "exact"
    [
      test "is whether each value lies within a committed integer type"
        (fun () ->
          let huge = `Int (Bigint.shift_left Bigint.one 100) in
          let rows =
            [
              ("int8's bounds", Dtype.Int8, [ i (-128); i 127 ]);
              ("128 at int8", Int8, [ i 0; i 128 ]);
              ("-1 at uint8", Uint8, [ i (-1) ]);
              ("2^64 - 1 at uint64", Uint64, [ Dtype.max Uint64 ]);
              ("2^100 at a weak integer", Weak_int, [ i (-1); huge ]);
              ("2^100 at float32", Float32, [ huge ]);
              ("nothing at int8", Int8, []);
            ]
          in
          equal (list string)
            [
              "int8's bounds";
              "2^64 - 1 at uint64";
              "2^100 at a weak integer";
              "2^100 at float32";
              "nothing at int8";
            ]
            (List.filter_map
               (fun (what, dt, vs) ->
                 if Ops.exact dt vs then Some what else None)
               rows));
    ]

(* exec_alu *)

let binary_ops =
  Op.
    [
      Add;
      Sub;
      Mul;
      Truncdiv;
      Truncmod;
      Div;
      Mod;
      Max;
      Xor;
      Or;
      And;
      Cmplt;
      Cmpne;
      Cmpeq;
      Shl;
      Shr;
    ]

let alu_row cell =
  let o = op_of_cell (cell "op") and dt = dtype_of_cell (cell "dtype") in
  let args = consts_of_cell (cell "operands") in
  let compute () = Ops.exec_alu o dt args in
  if raised (cell "result") then rejects compute
  else equal const (const_of_cell (cell "result")) (compute ())

(* The operations exec_alu does not compute: those that are not arithmetic, and
   Threefry. *)
let not_computed =
  Op.Threefry
  :: List.filter
       (fun o -> not (Op.Set.mem o Op.Set.alu))
       (Op.Set.to_list Op.Set.all)

let past_shift = i (Ops.max_exact_shift + 1)

let exec_alu =
  group "exec_alu"
    [
      group "computes each operation as exec_alu_values.golden states"
        [
          Golden.cases "exec_alu_values.golden"
            ~key:[ "op"; "dtype"; "operands" ]
            ~differs:
              [
                (* tinygrad has no FDIV rule, and raises KeyError. *)
                ("D416", [ "op=Ops.FDIV" ]);
              ]
            alu_row;
        ];
      test
        "max is its second operand only if that is greater, so a NaN wins first"
        (fun () ->
          equal const (f Float.nan) (alu Float32 Op.Max [ f Float.nan; f 1. ]);
          equal const (f 1.) (alu Float32 Op.Max [ f 1.; f Float.nan ]));
      test
        "an invalid operation's NaN is the canonical positive quiet NaN, \
         whatever the host gives" (fun () ->
          let inf = Float.infinity in
          (* The bits, since NaNs compare equal as values. x86 gives
             0xfff8000000000000. *)
          let bits = function
            | `Float x -> Printf.sprintf "%Lx" (Int64.bits_of_float x)
            | c -> str Dtype.pp_const c
          in
          List.iter
            (fun dt ->
              let nan = bits (Dtype.truncate dt (f Dtype.nan) :> Dtype.const) in
              List.iter
                (fun (o, args) ->
                  equal ~msg:(str Op.pp o) string nan (bits (alu dt o args)))
                Op.
                  [
                    (Add, [ f inf; f (-.inf) ]);
                    (Sub, [ f inf; f inf ]);
                    (Mul, [ f 0.; f inf ]);
                    (Fdiv, [ f 0.; f 0. ]);
                    (Fdiv, [ f inf; f (-.inf) ]);
                    (Sqrt, [ f (-1.) ]);
                    (Log2, [ f (-1.) ]);
                    (Sin, [ f inf ]);
                    (Pow, [ f (-2.); f 0.5 ]);
                  ])
            Dtype.[ Float32; Float64 ];
          equal string "7ff8000000000000" (bits (f Dtype.nan)));
      prop
        "a selection keeps a NaN's bits and a negation flips only its sign, a \
         signalling NaN's included"
        (Gen.triple
           (Gen.of_list ~pp:Dtype.pp
              Dtype.[ Float16; Bfloat16; Float32; Float64 ])
           Gen.bool Gen.int64)
        (fun (dt, quiet, drawn) ->
          (* A NaN of any sign and payload. The payload's leading bit is set in
             a signalling NaN, so that every format holds a nonzero payload. *)
          let quiet_bit = 0x8_0000_0000_0000L in
          let lead =
            if quiet then quiet_bit else Int64.shift_right quiet_bit 1
          in
          let nan =
            Int64.(
              logor
                (logand drawn (logor min_int 0x7_FFFF_FFFF_FFFFL))
                (logor 0x7FF0_0000_0000_0000L lead))
          in
          let x =
            match Dtype.const dt (f (Int64.float_of_bits nan)) with
            | `Float x -> x
            | c -> failf "%a is not a float" Dtype.pp_const c
          in
          let bits = function
            | `Float x -> Printf.sprintf "%Lx" (Int64.bits_of_float x)
            | c -> str Dtype.pp_const c
          in
          cover ~share:0.3 "a signalling NaN"
            (Int64.logand (Int64.bits_of_float x) quiet_bit = 0L);
          equal ~msg:"where true" string
            (bits (f x))
            (bits (alu dt Op.Where [ `Bool true; f x; f 1. ]));
          equal ~msg:"where false" string
            (bits (f x))
            (bits (alu dt Op.Where [ `Bool false; f 1.; f x ]));
          equal ~msg:"neg" string
            (bits (f (Float.neg x)))
            (bits (alu dt Op.Neg [ f x ])));
      test "a float multiply-add folds rounded once" (fun () ->
          let e = Float.ldexp 1. in
          equal ~msg:"float32" const
            (f (e (-24)))
            (alu Float32 Op.Mulacc
               [ f (1. +. e (-12)); f (1. +. e (-12)); f (-1. -. e (-11)) ]);
          equal ~msg:"float64" const
            (f (e (-54)))
            (alu Float64 Op.Mulacc
               [ f (1. +. e (-27)); f (1. +. e (-27)); f (-1. -. e (-26)) ]);
          (* [1 + 3 2^-24 - 2^-70] is a double's midpoint of two float32s before
             it is a float32: rounded twice it ties to the even one. *)
          equal ~msg:"float32, past a double's rounding" const
            (f (1. +. e (-23)))
            (alu Float32 Op.Mulacc
               [ f (1. +. e (-23)); f (e (-24) -. e (-47)); f (1. +. e (-23)) ]));
      prop "a shift by a count of any size is the count's shift, in the type"
        gen_shift (fun (dt, o, x, k) ->
          cover "a count at or past the width"
            (Bigint.geq k (Bigint.of_int (Dtype.bitsize dt)));
          cover "a count past an OCaml int" (not (Bigint.fits_int k));
          equal
            ~msg:
              (Format.asprintf "%a %a by %s" Dtype.pp dt Op.pp o
                 (Bigint.to_string k))
            const
            (shift_reference dt o x k :> Dtype.const)
            (alu dt o [ `Int x; `Int k ]));
      test "a right shift at no width is the floor, for a count of any size"
        (fun () ->
          let huge = `Int Bigint.(shift_left one 70) in
          equal const (i 0) (alu Weak_int Shr [ i 3; huge ]);
          equal const (i (-1)) (alu Weak_int Shr [ i (-3); huge ]));
      test "a left shift of 0 at no width is 0, for a count of any size"
        (fun () ->
          equal const (i 0)
            (alu Weak_int Shl [ i 0; `Int Bigint.(shift_left one 70) ]));
      test
        "an integer power wraps at its type, and a negative power truncates \
         toward 0" (fun () ->
          equal const (i 19487171) (alu Int32 Op.Pow [ i 11; i 7 ]);
          equal const (i 689956897) (alu Int32 Op.Pow [ i 3; i 40 ]);
          equal const (i 243) (alu Uint8 Op.Pow [ i 3; i 5 ]);
          equal const
            (`Int (Bigint.shift_left Bigint.one 64))
            (alu Weak_int Op.Pow [ i 2; i 64 ]);
          equal const (i 0) (alu Int32 Op.Pow [ i 2; i (-2) ]);
          equal const (i 0) (alu Int32 Op.Pow [ i 0; i (-1) ]);
          equal const (i 1) (alu Int32 Op.Pow [ i 1; i (-5) ]);
          equal const (i (-1)) (alu Int32 Op.Pow [ i (-1); i (-3) ]);
          equal const (i 1) (alu Int32 Op.Pow [ i (-1); i (-4) ]);
          (* exponents whose powers have billions of bits *)
          equal const (i (-1431655765)) (alu Int32 Op.Pow [ i 3; i 2147483647 ]);
          equal const (i 3)
            (alu Int64 Op.Pow
               [ i 3; `Int (Bigint.succ (Bigint.shift_left Bigint.one 62)) ]));
      test "an overflowing power is an infinity of the base's sign" (fun () ->
          equal const (f Float.infinity) (alu Float64 Op.Pow [ f 2.; f 10000. ]);
          equal const (f Float.neg_infinity)
            (alu Float64 Op.Pow [ f (-2.); f 10001. ]);
          equal const (f Float.infinity) (alu Float64 Op.Pow [ f 1e300; f 2. ]));
      test "sqrt of zero is zero" (fun () ->
          equal const (f 0.) (alu Float32 Op.Sqrt [ f 0. ]));
      prop "truncating is Dtype.truncate of the exact result"
        (Gen.triple gen_int_dtype
           (Gen.of_list ~pp:Op.pp
              Op.[ Add; Sub; Mul; Truncdiv; Div; Mod; Xor; And; Or; Max ])
           (Gen.pair gen_int gen_int))
        (fun (dt, o, (a, b)) ->
          match exact dt o [ a; b ] with
          | #Dtype.value as v ->
              equal const
                (Dtype.truncate dt v :> Dtype.const)
                (alu dt o [ a; b ])
          | `Invalid -> fail "no Invalid operand");
      prop "Invalid poisons every binary operation"
        (Gen.pair (Gen.of_list ~pp:Op.pp binary_ops) small)
        (fun (o, x) ->
          let dt =
            if Op.Set.mem o Op.Set.comparison then Dtype.Bool else Int32
          in
          equal const `Invalid (alu dt o [ `Invalid; x ]);
          equal const `Invalid (alu dt o [ x; `Invalid ]));
      prop "truncated division and remainder recompose the dividend"
        (Gen.pair gen_int gen_int) (fun (a, b) ->
          assume (b <> `Int Bigint.zero);
          let q = exact Weak_int Op.Truncdiv [ a; b ]
          and r = exact Weak_int Op.Truncmod [ a; b ] in
          equal const a
            (exact Weak_int Op.Add [ exact Weak_int Op.Mul [ q; b ]; r ]));
      prop
        "floor division and remainder recompose the dividend, the remainder \
         signed as the divisor"
        (Gen.pair gen_int gen_int) (fun (a, b) ->
          assume (b <> `Int Bigint.zero);
          let q = exact Weak_int Op.Div [ a; b ]
          and r = exact Weak_int Op.Mod [ a; b ] in
          equal const a
            (exact Weak_int Op.Add [ exact Weak_int Op.Mul [ q; b ]; r ]);
          let sign = function
            | `Int z -> Bigint.sign z
            | c -> failf "%a is no integer" Dtype.pp_const c
          in
          satisfies ~claim:"0 or of the divisor's sign" int
            (fun s -> s = 0 || s = sign b)
            (sign r));
      cases "a quotient by 0 is 0 and a remainder by 0 the dividend"
        ~id:(fun (o, _) -> str Op.pp o)
        Op.[ (Truncdiv, i 0); (Div, i 0); (Truncmod, i 7); (Mod, i 7) ]
        (fun (o, expected) ->
          equal const expected (exact Weak_int o [ i 7; i 0 ]);
          equal const expected (alu Int32 o [ i 7; i 0 ]));
      cases "raises for an operation it does not compute, Threefry included"
        ~id:(str Op.pp) not_computed (fun o ->
          rejects (fun () -> alu Uint64 o [ i 1; i 2 ]));
      cases "raises for operands not as many as the operation's"
        ~id:(fun (o, args) ->
          Format.asprintf "%a of %d" Op.pp o (List.length args))
        Op.
          [
            (Add, [ i 1 ]);
            (Add, [ i 1; i 2; i 3 ]);
            (Neg, [ i 1; i 2 ]);
            (Where, [ `Bool true; i 1 ]);
          ]
        (fun (o, args) -> rejects (fun () -> alu Int32 o args));
      cases "raises for a shift by a negative count, which has no value"
        ~id:(str Op.pp)
        Op.[ Shl; Shr ]
        (fun o ->
          raises_match (Exn.invalid_arg ~sub:"a shift by a negative count, -1")
            (fun () -> alu Int32 o [ i 8; i (-1) ]));
      cases
        "raises for a left shift of a non-zero integer past max_exact_shift \
         that nothing truncates"
        ~id:fst
        [
          ("at a weak type", fun () -> alu Weak_int Shl [ i 1; past_shift ]);
          ( "by a count past an OCaml int",
            fun () -> alu Weak_int Shl [ i 1; `Int Bigint.(shift_left one 70) ]
          );
          ( "without truncate_output",
            fun () -> exact Int32 Shl [ i (-1); past_shift ] );
        ]
        (fun (_, f) -> raises_match (Exn.invalid_arg ~sub:"has no value") f);
      cases
        "raises for a power that nothing truncates and exact_power leaves out"
        ~id:fst
        [
          ("at a weak type", fun () -> alu Weak_int Pow [ i 3; i 10_000_000 ]);
          ( "without truncate_output",
            fun () -> exact Int32 Pow [ i 3; i 10_000_000 ] );
        ]
        (fun (_, f) -> rejects f);
    ]

(* is_number *)

let numbers =
  let x = var "x" 0 4 and k = weak_var "k" 0 4 in
  group "is_number"
    [
      test
        "holds of a shift's count and an integer power's weak integer exponent \
         alone" (fun () ->
          let rows =
            [
              ("a left shift's count", Ops.O.(x lsl k), 1);
              ("a right shift's count", Ops.O.(x lsr k), 1);
              ("a shift's operand", Ops.O.(x lsl k), 0);
              ("an integer power's weak exponent", Ops.pow x k, 1);
              ("an integer power's base", Ops.pow x k, 0);
              ( "an integer power's committed exponent",
                Ops.alu x Op.Pow [ var "e" 0 4 ],
                1 );
              ("a float power's exponent", Ops.pow (fvar "y") (fvar "z"), 1);
              ("a sum's second operand", Ops.O.(x + k), 1);
            ]
          in
          equal (list string)
            [
              "a left shift's count";
              "a right shift's count";
              "an integer power's weak exponent";
            ]
            (List.filter_map
               (fun (what, u, n) ->
                 if Ops.is_number u n then Some what else None)
               rows));
    ]

(* exact_power *)

(* A base and an exponent: often [0], [1] or [-1], or a base of up to 300 bits
   with an exponent below, at or past where [e] times the base's bits passes
   [max_exact_shift]. *)
let gen_power =
  Gen.(
    let* b =
      frequency
        [
          (1, map Bigint.of_int (int_range (-1) 1));
          (3, map Bigint.of_int (int_range (-20) 20));
          ( 2,
            map
              (fun (k, d) ->
                Bigint.add (Bigint.shift_left Bigint.one k) (Bigint.of_int d))
              (pair (int_range 2 300) (int_range (-2) 2)) );
        ]
    in
    let bits = max 1 (Bigint.numbits (Bigint.abs b)) in
    let edge = Ops.max_exact_shift / bits in
    let+ e =
      frequency
        [
          (1, int_range (-3) (-1));
          (2, int_range 0 40);
          (3, map (fun d -> edge + d) (int_range (-1) 2));
        ]
    in
    (b, e))
  |> Gen.with_pp (fun ppf (b, e) ->
      Format.fprintf ppf "(%s, %d)" (Bigint.to_string b) e)

(* The rule ops.mli states: a base of 0, 1 or -1, a negative exponent, or [e]
   times the bits of [|b|] at most [max_exact_shift]. *)
let stated_exact b e =
  Bigint.leq (Bigint.abs b) Bigint.one
  || e < 0
  || e * Bigint.numbits (Bigint.abs b) <= Ops.max_exact_shift

let verdict exact = if exact then "exact" else "not exact"

let exact_powers =
  group "exact_power"
    [
      (* A power with a few thousand bits is computed and compared; a larger one
         is left to the rule, since computing it costs seconds. *)
      prop
        "follows its rule, and exec_alu computes exactly the powers it admits"
        ~examples:
          [ (Bigint.of_int 3, 10_000_000); (Bigint.of_int 2, 8_388_608) ]
        gen_power
        (fun (b, e) ->
          let admitted = stated_exact b e in
          let bits = e * Bigint.numbits (Bigint.abs b) in
          cover "admitted at the bound"
            (admitted && bits > Ops.max_exact_shift / 2);
          cover "left out past the bound" (not admitted);
          equal string (verdict admitted)
            (verdict (Ops.exact_power (`Int b) (i e)));
          let power () = exact Weak_int Op.Pow [ `Int b; i e ] in
          if not admitted then rejects power
          else if e >= 0 && bits <= 4096 then
            equal const (`Int (Bigint.pow b e)) (power ()));
    ]

let shifts =
  group "max_exact_shift"
    [
      test "is the most bits a left shift that nothing truncates computes"
        (fun () ->
          let n = Ops.max_exact_shift in
          equal const
            (`Int (Bigint.shift_left Bigint.one n))
            (alu Weak_int Shl [ i 1; i n ]);
          raises_match (Exn.invalid_arg ~sub:"has no value") (fun () ->
              alu Weak_int Shl [ i 1; i (n + 1) ]));
    ]

let groups =
  [
    bounds_group;
    operand_bounds_group;
    overflowing;
    no_ints;
    exactness;
    exec_alu;
    numbers;
    exact_powers;
    shifts;
  ]
