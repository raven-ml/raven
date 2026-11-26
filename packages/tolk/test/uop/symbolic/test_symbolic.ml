(* Tests of Tolk.Symbolic: each rewrite its interface states, the laws of
   simplification, and every simplification tinygrad's tests make. *)

open Windtrap
open Tolk
open Simplification

(* The value of a shift: past the width a left shift is 0 and a right shift the
   sign fill; below it, [x 2^k] modulo [2^w] or the floor of [x / 2^k]. *)
let shift_reference dt o x k : Dtype.value =
  let past = Bigint.geq k (Bigint.of_int (Dtype.bitsize dt)) in
  match (o : Op.t) with
  | Shl when past -> `Int Bigint.zero
  | Shl -> Dtype.truncate dt (`Int (Bigint.shift_left x (Bigint.to_int k)))
  | _ when past ->
      `Int (if Bigint.sign x < 0 then Bigint.minus_one else Bigint.zero)
  | _ -> `Int (Bigint.shift_right x (Bigint.to_int k))

(* The power of constants by an exponent of any size: squares and products at
   the type's width, a negative exponent truncating towards 0. *)
let pow_reference dt b e : Dtype.value =
  let two = Bigint.of_int 2 in
  if Bigint.sign e >= 0 then begin
    let m = Bigint.shift_left Bigint.one (Dtype.bitsize dt) in
    let rec go acc b e =
      if Bigint.sign e = 0 then acc
      else
        let acc = if Bigint.divisible e two then acc else Bigint.(rem (mul acc b) m) in
        go acc Bigint.(rem (mul b b) m) (Bigint.shift_right e 1)
    in
    Dtype.truncate dt (`Int (go Bigint.one (Bigint.rem b m) e))
  end
  else if Bigint.equal b Bigint.one then `Int Bigint.one
  else if Dtype.mem dt Dtype.sints && Bigint.equal b Bigint.minus_one then
    `Int (if Bigint.divisible e two then Bigint.one else Bigint.minus_one)
  else `Int Bigint.zero

(* A base at a type's extremes, 0, 1, -1 and 2, and exponents around 0, -1, the
   width and the larger powers of two. *)
let gen_constant_pow =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp Dtype.ints in
  let in_type v d =
    Dtype.Value.to_z (Dtype.truncate dt Dtype.Value.(v + of_int d))
  in
  let bases =
    [ in_type (Dtype.min dt) 0; in_type (Dtype.max dt) 0 ]
    @ List.map (fun n -> in_type (Dtype.Value.of_int n) 0) [ -1; 0; 1; 2; 3 ]
  in
  let exponents =
    List.concat_map
      (fun c -> Bigint.[ pred c; c; succ c ])
      (Bigint.zero :: Bigint.minus_one
      :: Bigint.of_int (Dtype.bitsize dt)
      :: List.map (fun n -> Bigint.shift_left Bigint.one n) [ 8; 16; 32; 40 ])
  in
  let+ b = of_list bases and+ e = of_list exponents in
  (dt, b, e)

(* Constant operands at a type's extremes, counts either side of its width and
   of the larger powers of two. *)
let gen_constant_shift =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp Dtype.ints in
  let in_type v d =
    Dtype.Value.to_z (Dtype.truncate dt Dtype.Value.(v + of_int d))
  in
  let xs =
    [
      in_type (Dtype.min dt) 0;
      in_type (Dtype.min dt) 1;
      in_type (Dtype.max dt) 0;
    ]
    @ List.map (fun n -> in_type (Dtype.Value.of_int n) 0) [ -1; 0; 1; 7 ]
  in
  let powers =
    List.map (fun n -> Bigint.shift_left Bigint.one n) [ 8; 16; 32; 63; 64 ]
  in
  let around c = Bigint.[ pred c; c; succ c ] in
  let width = Bigint.of_int (Dtype.bitsize dt) in
  let z = Testable.pp Dtypes.z in
  let+ o = of_list ~pp:Op.pp Op.[ Shl; Shr ]
  and+ x = of_list ~pp:z xs
  and+ k = of_list ~pp:z (List.concat_map around (width :: powers)) in
  (dt, o, x, k)

(* [simplifies_to ~by input expected] is the claim that [by] (default [sym])
   rewrites [input] to [expected], the operands of [expected] in the order
   [commutative] gives them, as the rewrite does. *)
let simplifies_to ?(by = sym) input expected =
  equal uop (rewrite Shape.commutative expected) (by input)

let a = var "a" 0 8
let b = var "b" 0 8
let x = var ~dtype:Int32 "x" 0 8
let y = var ~dtype:Int32 "y" 0 8
let cond = Ops.O.(a < int 4)
let other = Ops.O.(b < int 2)
let f = Shape.param 0 Dtype.Float32
let g = Shape.param 1 Dtype.Float32
let buf = Shape.param ~shape:[ Int 16 ] 2 Dtype.Float32
let r0 = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
let r1 = Ops.range ~axis_type:Weak (Int 4) [ 1 ]
let reduce_range = Ops.range ~axis_type:Weak (Int 4) [ 2 ]
let int32 n = Ops.int ~dtype:Int32 n

(* A node built as given, where the constructors would fold it: a cast to its
   own type, a group of one node. *)
let raw ?arg op src = Ops.v ?arg ~src op

(* symbolic_simple *)

let constants =
  group "constants"
    [
      prop "a shift of constants folds by its count, of any size"
        gen_constant_shift
        (fun (dt, o, x, k) ->
          let u =
            Ops.alu (Ops.const ~dtype:dt (`Int x)) o [ Ops.const (`Int k) ]
          in
          cover "a count at or past the width"
            (Bigint.geq k (Bigint.of_int (Dtype.bitsize dt)));
          cover "a count an OCaml int does not hold" (not (Bigint.fits_int k));
          let folded = sym u in
          (match Ops.op folded with
          | Const | Cast -> ()
          | _ -> failf "%a does not fold" Ops.pp u);
          equal
            ~msg:(Format.asprintf "%a" Ops.pp u)
            Dtypes.const (shift_reference dt o x k :> Dtype.const)
            (Ops.value folded));
      prop "a power of constants folds by its exponent, of any size"
        gen_constant_pow
        (fun (dt, b, e) ->
          let u =
            Ops.alu (Ops.const ~dtype:dt (`Int b)) Op.Pow [ Ops.const (`Int e) ]
          in
          cover "an exponent past the width"
            (Bigint.gt e (Bigint.of_int (Dtype.bitsize dt)));
          cover "a negative exponent" (Bigint.sign e < 0);
          let folded = sym u in
          (match Ops.op folded with
          | Const | Cast -> ()
          | _ -> failf "%a does not fold" Ops.pp u);
          equal
            ~msg:(Format.asprintf "%a" Ops.pp u)
            Dtypes.const (pow_reference dt b e :> Dtype.const)
            (Ops.value folded));
      test "a float power of a constant reads its exponent in the float"
        (fun () ->
          (* 16777217 rounds to the even 16777216 in a float32. *)
          let u =
            Ops.alu
              (Ops.const ~dtype:Float32 (`Float (-1.)))
              Op.Pow
              [ Ops.const (`Int (Bigint.of_int 16777217)) ]
          in
          equal Dtypes.const (`Float 1.) (Ops.value (sym u)));
      test "a power no width truncates stays a power past what it can hold"
        (fun () ->
          let e = Bigint.(succ (shift_left one 23)) in
          let u = Ops.alu (Ops.const (`Int (Bigint.of_int 2))) Op.Pow [ Ops.const (`Int e) ] in
          equal string "POW" (Op.name (Ops.op (sym u))));
      cases "a power no width truncates folds for a base of 0, 1 or -1 at any size"
        ~id:string_of_int [ 0; 1; -1 ]
        (fun b ->
          let e = Bigint.(succ (shift_left one 70)) in
          let u =
            Ops.alu (Ops.const (`Int (Bigint.of_int b))) Op.Pow [ Ops.const (`Int e) ]
          in
          equal Dtypes.const (`Int (Bigint.of_int b)) (Ops.value (sym u)));
      test "a masked right shift past the width keeps its mask's sign" (fun () ->
          (* The rule that drops a mask reads the bits below the count, and past
             the width that is every bit, the sign included. *)
          let x = var ~dtype:Int32 "x" (-5) 5 in
          let high = Ops.O.(x land int32 0x7FFFFFFF) in
          let shifted k = Ops.O.(high lsr Ops.const (`Int k)) in
          let keeps k =
            let e = shifted k in
            let at x = [ ("x", `Int (Bigint.of_int x)) ] in
            equal Dtypes.const
              (Interpreter.eval ~vars:(at (-1)) e)
              (Interpreter.eval ~vars:(at (-1)) (sym e))
          in
          keeps (Bigint.of_int 32);
          keeps (Bigint.shift_left Bigint.one 63));
      test "a bare constant shifted by a committed count commits at its type"
        (fun () ->
          let by_u32 o x k =
            Ops.value (sym (Ops.alu (Ops.int x) o [ Ops.int ~dtype:Uint32 k ]))
          in
          equal Dtypes.const (`Int (Bigint.of_int 0x7FFFFFFF))
            (by_u32 Op.Shr (-1) 1);
          equal Dtypes.const (`Int Bigint.zero) (by_u32 Op.Shl 1 40));
      test "a left shift of constants at no width by a huge count stays"
        (fun () ->
          let huge = Bigint.shift_left Bigint.one 80 in
          let e = Ops.alu (Ops.int 3) Shl [ Ops.const (`Int huge) ] in
          equal uop e (sym e));
    ]

(* A power leaves committed constants, [1.0 * x], that symbolic_simple alone
   does not uncast: its claims hold under symbolic. *)
let powers =
  let by_symbolic input expected =
    equal uop (rewrite Shape.commutative expected) (symbolic input)
  in
  group "powers"
    [
      test "x ** infinity and x ** NaN are left to xpow" (fun () ->
          List.iter
            (fun e ->
              let p = Ops.pow f (Ops.float e) in
              by_symbolic p p;
              equal uop (sym (Transcendental.xpow f (Ops.float e))) (sym p))
            [ Float.infinity; Float.nan ]);
      test "sym computes an integer power by products, without exp2 or log2"
        (fun () ->
          let p = sym (Ops.alu x Op.Pow [ y ]) in
          equal (list string) []
            (List.filter_map
               (fun u ->
                 match Ops.op u with
                 | (Pow | Exp2 | Log2) as o -> Some (Op.name o)
                 | _ -> None)
               (Ops.toposort ~calls:Skip p)));
    ]

let symbolic_simple = group "symbolic_simple" [ constants; powers ]

(* Integers wrap at a committed width, so a rewrite that computes as
   unbounded integers do applies there only where nothing wraps. Each graph is
   evaluated before and after [sym] at bindings where something wraps. *)

let keeps_machine_value ?(by = sym) u points =
  let after = by u in
  List.iter
    (fun vars ->
      let msg = Format.asprintf "at %a" pp_binding vars in
      equal ~msg Dtypes.const (Interpreter.eval ~vars u)
        (Interpreter.eval ~vars after))
    points

let wrapping =
  let u = var ~dtype:Uint8 "u" 0 255 and y = var ~dtype:Int8 "y" 0 50 in
  let at_u n = [ ("u", i n) ] and at_y n = [ ("y", i n) ] in
  group "integers wrap"
    [
      test "an unsigned quotient by a weak -1 is a quotient by the greatest"
        (fun () ->
          keeps_machine_value Ops.O.(u // int (-1)) (List.map at_u [ 0; 1; 5; 255 ]));
      test "a negation crosses a comparison only where it does not wrap"
        (fun () ->
          let x = var ~dtype:Int8 "x" (-128) 127
          and z = var ~dtype:Int8 "z" (-128) 127 in
          let points =
            List.concat_map
              (fun a -> List.map (fun b -> [ ("x", i a); ("z", i b) ]) [ -128; 3 ])
              [ -128; -3; 0; 5 ]
          in
          keeps_machine_value Ops.O.(x * int (-1) < z * int (-1)) points;
          keeps_machine_value Ops.O.(x * int (-1) < int 3) points;
          keeps_machine_value
            Ops.O.(x * int (-1) < Ops.int ~dtype:Int8 (-128))
            points;
          keeps_machine_value Ops.O.(x * int (-1) < int 128) points;
          (* a weak x negated, compared at int8 *)
          let w = var "w" (-128) 0 in
          keeps_machine_value
            Ops.O.(w * int (-1) < Ops.int ~dtype:Int8 3)
            (List.map (fun a -> [ ("w", i a) ]) [ -128; -3; 0 ]));
      test "an offset crosses a comparison only where neither side wraps"
        (fun () ->
          let points = List.map at_u [ 0; 5; 255 ] in
          keeps_machine_value Ops.O.(u - int 1 < int 255) points;
          keeps_machine_value
            Ops.O.(u - Ops.int ~dtype:Uint8 1 < Ops.int ~dtype:Uint8 255)
            points;
          keeps_machine_value Ops.O.(u + int 1 < int 1) points;
          keeps_machine_value
            Ops.O.(y + int 100 < int 0)
            (List.map at_y [ 10; 40 ]));
      test "a comparison folds from bounds only where they do not wrap"
        (fun () ->
          let points = List.map at_y [ 10; 40 ] in
          keeps_machine_value Ops.O.(y * int 4 < int 0) points;
          keeps_machine_value Ops.O.(y lsl int 2 < int 0) points;
          keeps_machine_value
            Ops.O.(Ops.cast (var "w" 0 255) Int8 < int 0)
            [ [ ("w", i 100) ]; [ ("w", i 200) ] ];
          keeps_machine_value
            Ops.O.(u lxor int (-1) < int 0)
            (List.map at_u [ 0; 5 ]));
      test "a chain of integer casts is one cast only where the value fits"
        (fun () ->
          keeps_machine_value
            (Ops.cast (Ops.cast Ops.O.(y + int 100) Uint8) Int32)
            (List.map at_y [ 10; 40 ]));
      (* At u = 255 the products r * u over r < 4 are 0, 255, 254 and 253:
         their maximum is 255, where 3 * u is 253. *)
      test "a factor leaves a maximum only where no product wraps" (fun () ->
          let max_of v =
            Ops.reduce
              Ops.O.(Ops.cast reduce_range (Ops.dtype v) * v)
              Max [ reduce_range ]
          in
          keeps_machine_value (max_of u) (List.map at_u [ 0; 5; 255 ]);
          let w = var ~dtype:Int32 "w" 0 (Int32.to_int Int32.max_int) in
          keeps_machine_value (max_of w)
            [ [ ("w", i 1000) ]; [ ("w", i (Int32.to_int Int32.max_int)) ] ]);
      (* At p = q = 12 the int8 product p * q wraps to -112: the maximum of r *
         -112 over r < 2 is 0, where 1 * -112 is -112. *)
      test "a factor leaves a maximum only where it is not negative" (fun () ->
          let p = var ~dtype:Int8 "p" 0 50 and q = var ~dtype:Int8 "q" 0 50 in
          let r = Ops.range ~axis_type:Weak (Int 2) [ 3 ] in
          let red = Ops.reduce Ops.O.(Ops.cast r Int8 * p * q) Max [ r ] in
          keeps_machine_value red
            [ [ ("p", i 2); ("q", i 3) ]; [ ("p", i 12); ("q", i 12) ] ]);
      (* c * c wraps to 0 at each width where c is 2^(bits / 2) or the
         greatest power of two, the least signed value: the square is 0. *)
      test
        "a square of a product keeps its value, its constants folded at an \
         unsigned width"
        (fun () ->
          List.iter
            (fun dtype ->
              let bits = Dtype.bitsize dtype in
              let w = var ~dtype "w" 0 7 in
              let power k = Bigint.shift_left Bigint.one k in
              let top =
                match Dtype.min dtype with
                | `Int z when Bigint.sign z < 0 -> z
                | _ -> power (bits - 1)
              in
              List.iter
                (fun c ->
                  let c = Ops.const ~dtype (`Int c) in
                  let u = Ops.O.(w * c * (w * c)) in
                  keeps_machine_value u
                    (List.map (fun n -> [ ("w", i n) ]) [ 0; 1; 7 ]);
                  if Dtype.is_unsigned dtype then
                    equal uop (Ops.const ~dtype (`Int Bigint.zero)) (sym u))
                [ power (bits / 2); top ])
            Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]);
      (* Like terms whose weak factors sum past the width, as the greatest
         value and 3: the combined factor is read at the width, so the sum,
         and a quotient, remainder or comparison of it, which reads its
         factor's divisors, keeps its value. *)
      test "like terms combine their factors at their width" (fun () ->
          List.iter
            (fun dtype ->
              let bits = Dtype.bitsize dtype in
              let _, hi = Dtypes.int_bounds dtype in
              let w = var ~dtype "w" 0 7 and z = var ~dtype "z" 0 3 in
              let half = Bigint.shift_left Bigint.one (bits - 1) in
              let k n = Ops.const (`Int n) in
              let points =
                List.concat_map
                  (fun n ->
                    [ [ ("w", i n); ("z", i 0) ]; [ ("w", i n); ("z", i 3) ] ])
                  [ 0; 1; 7 ]
              in
              List.iter
                (fun (c0, c1) ->
                  List.iter
                    (fun s ->
                      List.iter
                        (fun u -> keeps_machine_value u points)
                        Ops.O.
                          [
                            s; s // int 2; s // int 3; s % int 4; s // int 16;
                            s < int 16; (s + int 3) // int 8; Ops.ne s (int 0);
                          ])
                    Ops.O.
                      [
                        (w * k c0) + (w * k c1);
                        z + (w * k c0) + (w * k c1);
                        (w * k c0) + w;
                        z + w + (w * k c0);
                        z + (w * k c0) + w;
                      ])
                Bigint.[ (succ half, pred half); (hi, of_int 3); (hi, one) ])
            Dtype.[ Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64 ]);
      (* An int32 shift by 32 or more, and the least int32's remainder by -1,
         have no value in C, where their int64 forms do: [-1 >> 32] is [-1]
         and [-2^31 % -1] is [0] at int64. *)
      test "an int64 operation whose int32 form has no value stays int64"
        (fun () ->
          let ops_at_int32 u =
            List.filter_map
              (fun n ->
                if
                  Op.Set.mem (Ops.op n) Op.Set.binary
                  && Dtype.equal (Ops.dtype n) Int32
                then Some n
                else None)
              (Ops.toposort ~calls:Skip (sym u))
          in
          let long x = Ops.cast x Int64 in
          let small = var ~dtype:Int8 "s" (-128) 127 in
          let shifted = Ops.O.(long small lsr Ops.int ~dtype:Int64 32) in
          equal (list Uops.uop) [] (ops_at_int32 shifted);
          let least =
            Ops.variable ~dtype:Int32 "a" (Dtype.min Int32) (Dtype.max Int32)
          in
          let divisor = var ~dtype:Int8 "d" (-2) (-1) in
          List.iter
            (fun rem ->
              equal (list Uops.uop) []
                (ops_at_int32 (rem (long least) (long divisor))))
            [ Ops.O.( % ); (fun x y -> Ops.alu x Truncmod [ y ]) ]);
    ]

(* A committed integer constant holds its type's value. A fold reads a weak
   operand of an operation on a committed integer, and writes its result, at
   that type's width, and so do the bounds. *)
let committed_constants =
  let v ?(dtype = Dtype.Uint8) name lo hi = var ~dtype name lo hi in
  let b = v "b" 0 1 and d = v "d" 1 2 and a = v "a" 0 1 in
  group "committed constants hold their type's value"
    [
      test "a weak operand is read at the width of the operation" (fun () ->
          let b32 = v ~dtype:Uint32 "b" 0 1 in
          keeps_machine_value
            Ops.O.(Ops.maximum (int 1) b32 // int (-2))
            [ [ ("b", i 0) ]; [ ("b", i 1) ] ];
          keeps_machine_value
            Ops.O.(Ops.where (int (-3) < b) a (int (-3)) // d)
            [ [ ("a", i 0); ("b", i 1); ("d", i 2) ] ];
          keeps_machine_value
            (Ops.maximum (Ops.int (-3)) a)
            [ [ ("a", i 0) ]; [ ("a", i 1) ] ]);
      test "a folded result is written at its width" (fun () ->
          let c = v "c" 2 2 in
          keeps_machine_value
            Ops.O.(Ops.maximum ~-(Ops.maximum (b % int 2) ~-c) b)
            [ [ ("b", i 0); ("c", i 2) ]; [ ("b", i 1); ("c", i 2) ] ]);
      test "nested maxima fold their constants at the width of the operation"
        (fun () ->
          let w = v "w" 0 255 in
          keeps_machine_value
            Ops.O.(Ops.maximum (Ops.maximum w (int 1)) (int (-3)))
            [ [ ("w", i 0) ]; [ ("w", i 254) ] ]);
    ]

(* Floats keep IEEE's values, signed zeros, infinities, NaN and subnormals
   included. Each graph is evaluated before and after the rewrite, with [f], [g]
   and [h] bound to the given float32 values, at points where a rewrite tolk
   restricts to exact values changes the value. *)

let keeps_float_bits ?(by = sym) u points =
  let after = by u in
  List.iter
    (fun values ->
      let params =
        List.mapi (fun k v -> (List.nth [ 0; 1; 3 ] k, float32 v)) values
      in
      let msg =
        Format.asprintf "at %a"
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
             Format.pp_print_float)
          values
      in
      equal ~msg same_float
        (Interpreter.eval ~params u)
        (Interpreter.eval ~params after))
    points

let floats =
  let h = Shape.param 3 Dtype.Float32 and inf = Float.infinity in
  let cond = Ops.O.(f < float 1.) in
  group "floats keep IEEE values"
    [
      test "float folds: identities that IEEE does not keep" (fun () ->
          keeps_float_bits Ops.O.(f + float 0.) [ [ -0. ] ];
          keeps_float_bits
            Ops.O.(f * float 0.)
            [ [ inf ]; [ -1. ]; [ Float.nan ] ];
          keeps_float_bits Ops.O.(f / f) [ [ 0. ]; [ inf ] ];
          keeps_float_bits Ops.O.(f * g / g) [ [ 1e10; 1e30 ] ];
          keeps_float_bits Ops.O.(f / g / h) [ [ 1e20; 1e20; 1e20 ] ]);
      test "signed zeros: a negated sum and complementary selections" (fun () ->
          keeps_float_bits Ops.O.(float (-1.) * (f + float 3.)) [ [ -3. ] ];
          keeps_float_bits Ops.O.(float (-1.) * (f + g)) [ [ 1.; -1. ] ];
          keeps_float_bits
            Ops.O.(Ops.where cond g (float 0.) + Ops.where cond (float 0.) h)
            [ [ 0.; -0.; 5. ] ]);
      test "reassociation: constants, like terms and selections of a sum"
        (fun () ->
          let tiny = Float.ldexp 1. (-24) in
          keeps_float_bits Ops.O.(f + float 1e8 + float (-1e8)) [ [ 1. ] ];
          keeps_float_bits Ops.O.(f * float 1e30 * float 1e-30) [ [ 1e10 ] ];
          keeps_float_bits Ops.O.(f + float 1e8 + g) [ [ 3.; 3. ] ];
          keeps_float_bits Ops.O.(g + f + f) [ [ tiny; 1. ] ];
          keeps_float_bits Ops.O.((f * float 1.1) + (f * float 2.2)) [ [ 1. ] ];
          keeps_float_bits Ops.O.(f + (f * float (-1.))) [ [ -1. ]; [ inf ] ];
          keeps_float_bits
            Ops.O.(g + (f * float 4.) + (f * float (-3.)))
            [ [ 1e38; 1. ] ];
          keeps_float_bits
            Ops.O.(f * float (-1.) * g)
            [ [ 3.; 1e30 ]; [ -0.; 2. ]; [ inf; 0. ]; [ Float.nan; 1. ] ];
          keeps_float_bits
            Ops.O.(
              g
              + Ops.where cond (float tiny) (float 0.)
              + Ops.where cond (float tiny) (float 0.))
            [ [ 0.; 1. ] ];
          let r = Ops.range ~axis_type:Weak (Int 3) [ 2 ] in
          keeps_float_bits
            (Ops.reduce Ops.O.(Ops.cast r Float32 * f * g) Add [ r ])
            [ [ 2e38; 0.25 ] ]);
      test "reciprocal and sigmoid forms stay" (fun () ->
          let d = Ops.reciprocal Ops.O.(float 1. + f) in
          keeps_float_bits (Ops.reciprocal Ops.O.(f * f)) [ [ 1e20 ] ];
          keeps_float_bits (Ops.reciprocal Ops.O.(f * float 1e10)) [ [ 1e30 ] ];
          keeps_float_bits Ops.O.(f * d) [ [ 1e-8 ]; [ inf ] ]);
      test "maxima: a NaN operand and the order of zeros" (fun () ->
          let nan = Ops.O.(float inf + float (-.inf)) in
          keeps_float_bits
            (Ops.maximum nan (Ops.maximum f (Ops.float 0.)))
            [ [ 1. ] ];
          keeps_float_bits
            (Ops.where Ops.O.(f < float (-0.)) (Ops.float 0.) f)
            [ [ -1e-45 ] ];
          keeps_float_bits (Ops.maximum f (Ops.float inf)) [ [ Float.nan ] ]);
      test "a Dekker split and product keep their low parts" (fun () ->
          let split x =
            let c = Ops.O.(x * float 4097.) in
            let hi = Ops.O.(c - (c - x)) in
            (hi, Ops.O.(x - hi))
          in
          let fh, fl = split f and gh, gl = split g in
          let p = Ops.O.(f * g) in
          let err = Ops.O.((fh * gh) - p + (fh * gl) + (fl * gh) + (fl * gl)) in
          let points = [ [ 1.1; 3.14159 ]; [ 0.1; 7.3 ]; [ 1e10; 1e-10 ] ] in
          List.iter (fun u -> keeps_float_bits u points) [ fh; fl; err ];
          is_true
            (Interpreter.eval ~params:[ (0, float32 1.1) ] (sym fl) <> `Float 0.));
    ]

(* Conditions *)

let given_valid =
  let c = var "c" 0 3 in
  group "uop_given_valid"
    [
      test "a clause bounds an expression" (fun () ->
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid Ops.O.(a < int 5) Ops.O.(a // int 5));
          equal uop (Ops.int 1)
            (Symbolic.uop_given_valid Ops.O.(int 4 < a) Ops.O.(a // int 5)));
      test "two clauses bound one expression from both sides" (fun () ->
          equal uop (Ops.int 1)
            (Symbolic.uop_given_valid
               Ops.O.((a < int 5) land (int 2 < a))
               Ops.O.(a // int 3)));
      test "a negated clause bounds an expression from below" (fun () ->
          equal uop (Ops.bool false)
            (Symbolic.uop_given_valid
               Ops.O.(lnot (a < int 5))
               Ops.O.(a < int 3)));
      test "every clause of a conjunction holds" (fun () ->
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid
               Ops.O.((a < int 5) land (b < int 5))
               Ops.O.((a + b) // int 9)));
      test "a negated clause with a constant on the left bounds from above"
        (fun () ->
          equal uop (Ops.bool true)
            (Symbolic.uop_given_valid
               Ops.O.(lnot (int 2 < a))
               Ops.O.(a < int 3));
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid
               Ops.O.(lnot (int 1 < a))
               Ops.O.(a // int 2)));
      test "a clause on a sum with a constant term tries no term of it"
        (fun () ->
          (* The 1 is no term to try at least 1: each lane [base + k]
             simplifies [base] alike. *)
          let a = var "a" 0 7 and b = var "b" 0 7 and d = var "d" 0 1 in
          let valid =
            Ops.O.(
              lnot (a + b + int 1 < int 1)
              land lnot (c + d < int 1))
          in
          let base =
            Ops.O.(
              (((a * int 8) + b + (c * int 128) + (d * int 64) + int 1000)
              % int 1024)
              * int 4)
          in
          let lane k =
            Ops.nth (Symbolic.uop_given_valid valid Ops.O.(base + int k)) 0
          in
          List.iter
            (fun k -> equal ~msg:(string_of_int k) uop (lane 1) (lane k))
            [ 2; 3 ]);
      test "a clause on a sum tries each term at least 1" (fun () ->
          let valid = Ops.O.(lnot (a + b < int 1)) in
          let both_zero = Ops.O.((a < int 1) land (b < int 1)) in
          equal uop (Ops.bool false) (Symbolic.uop_given_valid valid both_zero);
          equal uop both_zero
            (Symbolic.uop_given_valid ~try_simplex:false valid both_zero));
      test "a clause on a sum simplifies each lane of a pair on its own"
        (fun () ->
          let valid = Ops.O.(lnot (a + b < int 1)) in
          let both_zero = Ops.O.((a < int 1) land (b < int 1)) in
          let pair = raw Stack [ both_zero; Ops.O.(a < int 1) ] in
          equal uop
            (raw Stack [ Ops.bool false; Ops.O.(a < int 1) ])
            (Symbolic.uop_given_valid valid pair));
      test "a clause on a loaded value bounds it" (fun () ->
          let x = var "x" 0 100 in
          let inside = Ops.O.((int 29 < x) land (x < int 80)) in
          let loaded =
            Ops.cast
              (Ops.index
                 (Shape.param ~shape:[ Int 100 ] 1 Dtype.Int32)
                 [ Shape.valid Ops.O.(x + int (-30)) inside ])
              Weak_int
          in
          let valid =
            Ops.O.(inside land (int (-1) < loaded) land (loaded < int 100))
          in
          equal uop loaded
            (Symbolic.uop_given_valid valid Ops.O.(loaded % int 100)));
      test "a clause that is not a bound is ignored" (fun () ->
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid
               Ops.O.((a <> int 1) land (a < int 5))
               Ops.O.(a // int 5)));
      test "a clause on a sum with coefficients does not try each term"
        (fun () ->
          let valid = Ops.O.(lnot ((a * int 2) + b < int 1)) in
          let e = Ops.O.((a * int 2 < int 1) land (b < int 1)) in
          equal uop e (Symbolic.uop_given_valid valid e));
      test "a clause on a sum leaves what its terms disagree on" (fun () ->
          let valid = Ops.O.(lnot (a + b < int 1)) in
          let e = Ops.O.((a < int 1) lor (b < int 1)) in
          equal uop e (Symbolic.uop_given_valid valid e));
      test "a clause that bounds nothing leaves the expression" (fun () ->
          let e = Ops.O.(c + a) in
          equal uop e (Symbolic.uop_given_valid Ops.O.(cond lor other) e));
    ]

let simplify_valid =
  let some = option uop in
  group "simplify_valid"
    [
      test "a single clause is unchanged" (fun () ->
          equal some None (Symbolic.simplify_valid Ops.O.(a < int 5)));
      test "a clause implied by a tighter one is true" (fun () ->
          equal some
            (Some (Ops.uprod Ops.O.(a < int 3) [ Ops.bool true ]))
            (Symbolic.simplify_valid Ops.O.((a < int 5) land (a < int 3))));
      test "duplicate clauses are one" (fun () ->
          let c = Ops.O.(a < int 5) in
          equal some (Some c) (Symbolic.simplify_valid Ops.O.(c land c)));
      test "a clause on an expression others read is applied first" (fun () ->
          let x = var "x" 0 100 and y = var "y" 0 1 in
          let valid = Ops.O.((x + y < int 3) land (x < int 2)) in
          equal some
            (Some (Ops.uprod Ops.O.(x < int 2) [ Ops.bool true ]))
            (Symbolic.simplify_valid valid);
          equal uop Ops.O.(x < int 2) (sym valid));
      test "reordering the clauses alone is no change" (fun () ->
          equal some None
            (Symbolic.simplify_valid Ops.O.((a + b < int 5) land (a < int 2))));
      test "the clauses simplify whatever their order" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 2) [ 0 ] in
          let first = Ops.O.(r < int 1)
          and second = Ops.O.(((r * int 5) + int 1) % int 6 < int 5) in
          let expected = Some (Ops.uprod first [ Ops.bool true ]) in
          equal some expected
            (Symbolic.simplify_valid Ops.O.(second land first));
          equal some expected
            (Symbolic.simplify_valid Ops.O.(first land second)));
      test "a condition that involves an index is left" (fun () ->
          let load = Ops.index buf [ a ] in
          equal some None
            (Symbolic.simplify_valid
               Ops.O.((load < float 1.) land (a < int 3) land (a < int 5))));
    ]

let simplify_valid_pm =
  let rewrite = rewrite Symbolic.pm_simplify_valid in
  group "pm_simplify_valid"
    [
      test "a conjunction is simplified" (fun () ->
          equal uop
            (Ops.uprod Ops.O.(a < int 3) [ Ops.bool true ])
            (rewrite Ops.O.((a < int 5) land (a < int 3))));
      test "a gated index is simplified knowing its gate holds" (fun () ->
          equal uop
            (Shape.valid (Ops.int 0) Ops.O.(a < int 5))
            (rewrite (Shape.valid Ops.O.(a // int 5) Ops.O.(a < int 5))));
      test "a gated committed value is left" (fun () ->
          let gated = Shape.valid Ops.O.(x // int32 5) Ops.O.(x < int32 5) in
          equal uop gated (rewrite gated));
    ]

let drop_and_clauses =
  let rewrite = rewrite Symbolic.pm_drop_and_clauses in
  group "pm_drop_and_clauses"
    [
      test "clauses that run in none of the value's ranges are dropped"
        (fun () ->
          let gate = Ops.O.((r0 < int 3) land (r1 < int 2) land cond) in
          equal uop
            (Shape.valid r0 (Ops.uprod (Ops.bool true) [ Ops.O.(r0 < int 3) ]))
            (rewrite (Shape.valid r0 gate)));
      test "a gate whose clauses all run in the value's ranges is left"
        (fun () ->
          let gated =
            Shape.valid Ops.O.(r0 + r1) Ops.O.((r0 < int 3) land (r1 < int 2))
          in
          equal uop gated (rewrite gated));
    ]

let move_where_on_load =
  let rewrite = rewrite Symbolic.pm_move_where_on_load in
  let index idx = Ops.index buf [ idx ] in
  group "pm_move_where_on_load"
    [
      test "a clause moves into the index's gate" (fun () ->
          equal uop
            (Ops.where (Ops.bool true)
               (index (Shape.valid a (Ops.uprod (Ops.bool true) [ cond ])))
               (Ops.float ~dtype:Float32 0.))
            (rewrite (Ops.where cond (index a) (Ops.float 0.))));
      test "a selection of 0 first moves the negated condition" (fun () ->
          equal uop
            (Ops.where (Ops.bool true)
               (index
                  (Shape.valid a
                     (Ops.uprod (Ops.bool true) [ Ops.logical_not cond ])))
               (Ops.float ~dtype:Float32 0.))
            (rewrite (Ops.where cond (Ops.float 0.) (index a))));
      test "a clause the index already requires is dropped" (fun () ->
          let idx = Shape.valid a cond in
          equal uop
            (Ops.where (Ops.bool true)
               (index (Shape.valid a (Ops.uprod cond [ other ])))
               (Ops.float ~dtype:Float32 0.))
            (rewrite
               (Ops.where Ops.O.(cond land other) (index idx) (Ops.float 0.))));
      test "a clause on a range the index lacks stays" (fun () ->
          let sel =
            Ops.where Ops.O.(r0 < int 2) (index a) (Ops.float ~dtype:Float32 0.)
          in
          equal uop sel (rewrite sel));
      test "a constant clause stays" (fun () ->
          let sel =
            Ops.where (Ops.bool true) (index a) (Ops.float ~dtype:Float32 0.)
          in
          equal uop sel (rewrite sel));
      test "a clause that reads another index stays" (fun () ->
          let other_load = Ops.O.(Ops.index buf [ b ] < float 1.) in
          let sel =
            Ops.where other_load (index a) (Ops.float ~dtype:Float32 0.)
          in
          equal uop sel (rewrite sel));
    ]

let clean_up_group_sink =
  let rewrite = rewrite Symbolic.pm_clean_up_group_sink in
  group "pm_clean_up_group_sink"
    [
      test "a sink splices the sources of its sinks, groups, stacks and noops"
        (fun () ->
          equal uop
            (Ops.sink [ a; b; f; g ])
            (rewrite (Ops.sink [ Ops.sink [ a; b ]; Ops.group [ f; g ] ]));
          equal uop (Ops.sink [ a ]) (rewrite (Ops.sink [ raw Noop [ a ] ])));
      test "a group keeps the groups it is ordered after" (fun () ->
          let nested = Ops.group [ Ops.group [ a; b ]; f ] in
          equal uop nested (rewrite nested));
    ]

let conditions =
  group "conditions"
    [
      given_valid;
      simplify_valid;
      simplify_valid_pm;
      drop_and_clauses;
      move_where_on_load;
      clean_up_group_sink;
    ]

(* sym *)

let sym_group =
  let index = Ops.index buf [ a ] in
  group "sym"
    [
      test "a power is computed from exp2 and log2" (fun () ->
          simplifies_to (Ops.pow f g) (sym (Transcendental.xpow f g)));
      test "storing what a load of the same index reads does nothing" (fun () ->
          simplifies_to (Ops.store index (Ops.load index [])) (raw Noop []);
          simplifies_to
            (Ops.store index Ops.O.(Ops.load index [] + float (-0.)))
            (raw Noop []));
      test "storing a changed load stays" (fun () ->
          let st = Ops.store index Ops.O.(Ops.load index [] + float 1.) in
          simplifies_to st st;
          let st = Ops.store index Ops.O.(Ops.load index [] + float 0.) in
          simplifies_to st st);
      test "storing a selection of the loaded value stores where it differs"
        (fun () ->
          simplifies_to
            (Ops.store index (Ops.where cond f (Ops.load index [])))
            (Ops.store (Ops.index buf [ Shape.valid a cond ]) f));
      test "storing a selection of the loaded value and another stores where \
            it does not hold" (fun () ->
          simplifies_to
            (Ops.store index (Ops.where cond (Ops.load index []) f))
            (Ops.store
               (Ops.index buf [ Shape.valid a (Ops.logical_not cond) ])
               f));
      test "storing nested selections of the loaded value stores where the \
            outer holds and the inner does not" (fun () ->
          let l = Ops.load index [] in
          let st = Ops.store index (Ops.where cond (Ops.where other l f) l) in
          let gate = Ops.O.(cond land Ops.logical_not other) in
          simplifies_to st (Ops.store (Ops.index buf [ Shape.valid a gate ]) f));
      test "storing invalid does nothing" (fun () ->
          simplifies_to (Ops.store index Ops.invalid) (raw Noop []));
      test "storing a gated value stores where its gate holds" (fun () ->
          simplifies_to
            (Ops.store index (Shape.valid f cond))
            (Ops.store (Ops.index buf [ Shape.valid a cond ]) f));
      test "reciprocals of products stay" (fun () ->
          let d = Ops.reciprocal Ops.O.(float 1. + f) in
          List.iter
            (fun u -> simplifies_to u u)
            [
              Ops.reciprocal Ops.O.(f * f);
              Ops.reciprocal Ops.O.(f * f * f);
              Ops.reciprocal Ops.O.(f * float 2.);
              Ops.O.(f * d);
              Ops.O.(f * (d * g));
              Ops.O.(f * (d + g));
            ]);
      test "factors independent of a sum's ranges move out of it" (fun () ->
          let body = Ops.O.(Ops.cast reduce_range Int32 * x) in
          simplifies_to
            (Ops.reduce body Add [ reduce_range ])
            Ops.O.(
              Ops.reduce (Ops.cast reduce_range Int32) Add [ reduce_range ] * x));
      test "a product wholly independent of a sum's ranges moves out whole"
        (fun () ->
          simplifies_to
            (Ops.reduce Ops.O.(x * y) Add [ reduce_range ])
            Ops.O.(
              Ops.reduce (Ops.int ~dtype:Int32 1) Add [ reduce_range ] * (x * y)));
      test "factors stay in a float sum or maximum" (fun () ->
          List.iter
            (fun o ->
              let red =
                Ops.reduce
                  Ops.O.(Ops.cast reduce_range Float32 * f)
                  o [ reduce_range ]
              in
              simplifies_to red red)
            [ Add; Max ]);
      test "factors stay in a reduction other than a sum or a maximum"
        (fun () ->
          let red =
            Ops.reduce
              Ops.O.(Ops.cast reduce_range Float32 * f)
              Mul [ reduce_range ]
          in
          simplifies_to red red);
      test "only non-negative factors move out of a maximum" (fun () ->
          let body = Ops.O.(Ops.cast reduce_range Int32 * x) in
          simplifies_to
            (Ops.reduce body Max [ reduce_range ])
            Ops.O.(
              Ops.reduce (Ops.cast reduce_range Int32) Max [ reduce_range ] * x);
          let v = var ~dtype:Int32 "v" (-3) 3 in
          let body = Ops.O.(Ops.cast reduce_range Int32 * v) in
          let red = Ops.reduce body Max [ reduce_range ] in
          simplifies_to red red);
      test "-(x + y) is -x + -y for integers, and stays for floats"
        (fun () ->
          simplifies_to
            Ops.O.(int (-1) * (a + b))
            Ops.O.((a * int (-1)) + (b * int (-1)));
          let neg = Ops.O.((f + g) * float (-1.)) in
          simplifies_to neg neg);
      test "a negated sum with a scaled term folds each term's coefficient"
        (fun () ->
          let x = var "px" (-10) 10 and y = var "py" (-10) 10 in
          simplifies_to
            Ops.O.((x + (y * int 5)) * int (-1))
            Ops.O.((x * int (-1)) + (y * int (-5))));
      test "(x + y) * c is x * c + y * c for weak integers" (fun () ->
          simplifies_to
            Ops.O.((a + b) * int 3)
            Ops.O.((a * int 3) + (b * int 3)));
      test "(x + y) * c stays for floats" (fun () ->
          let e = Ops.O.((f + g) * float 3.) in
          simplifies_to e e);
    ]

(* tinygrad's tests that make no simplification of their own: they test the
   nodes, bounds and evaluation that simplification relies on. *)

let other_tests =
  group "tinygrad's other tests"
    [
      test "an index simplified under its gate keeps its value where it holds"
        (fun () ->
          let r0 = Ops.range ~axis_type:Weak (Int 30) [ 0 ]
          and r1 = Ops.range ~axis_type:Weak (Int 7) [ 1 ]
          and r2 = Ops.range ~axis_type:Weak (Int 2) [ 2 ] in
          let alu11 = Ops.O.(r1 + r2) in
          let idx =
            Ops.O.(
              ((alu11 + int 1) // int 7 * int (-31))
              + ((((alu11 + int 218) // int 224) + r0) % int 30 * int 1568))
          in
          let gated = Shape.valid idx Ops.O.((r2 < int 1) land (r1 < int 6)) in
          let simplified = sym gated in
          equal uop
            (Shape.valid
               Ops.O.(r0 * int 1568)
               Ops.O.((r2 < int 1) land (r1 < int 6)))
            simplified;
          ignore (keeps_value ~count:64 ~name:"mobilenet" gated simplified));
    ]

(* Laws *)

let laws =
  let open Expressions in
  let scenario_law name law = prop name scenario (fun s -> law (node s)) in
  group "laws"
    [
      scenario_law
        "sym keeps the value of an integer expression where nothing wraps"
        (fun e ->
          let checked = keeps_value ~name:"sym" e (sym e) in
          cover ~share:0.9 "a binding was checked" (checked > 0));
      prop ~count:1000
        "sym keeps the value of an integer expression at a committed width, \
         wrapping included"
        committed_scenario (fun s ->
          let e = node s in
          let checked = keeps_value ~wrapping:true ~name:"sym" e (sym e) in
          cover ~share:0.9 "a binding was checked" (checked > 0));
      prop "sym keeps a float expression's value bit for bit at special values"
        float_scenario (fun f ->
          let e = float_node f in
          keeps_float_value ~name:"sym" e (sym e));
      scenario_law "sym is idempotent" (fun e -> Law.idempotent uop sym e);
    ]

(* Cost *)

(* [selections name n] is [n] selections in sequence over a value read from
   memory. Each condition is in its true branch, so the where-closure rule looks
   at every selection, and keeps it, since the value reaches an INDEX. *)
let selections name n =
  let rec link x k =
    if k = n then x
    else
      let c = Ops.O.(x < Ops.float ~dtype:Float32 (float_of_int k)) in
      let t = Ops.O.(x + Ops.cast c Dtype.Float32) in
      link (Ops.where c t Ops.O.(x * Ops.float ~dtype:Float32 2.)) (k + 1)
  in
  link (Ops.load (Ops.index buf [ var name 0 15 ]) []) 0

(* [far_conditions name n] is [n] selections in sequence over a value read from
   memory, all on one condition built before the chain: asking whether a branch
   holds it walks the chain below the branch. The branches reach an INDEX, which
   rejects each selection without that walk. *)
let far_conditions name n =
  let x = Ops.load (Ops.index buf [ var name 0 15 ]) [] in
  let c = Ops.O.(x < Ops.float ~dtype:Float32 3.) in
  let rec link x acc k =
    if k = n then acc
    else
      let x = Ops.O.(x + Ops.float ~dtype:Float32 1.) in
      link x (Ops.where c x acc) (k + 1)
  in
  link x x 0

(* [loaded_conditions name n] is [far_conditions] over a load of four elements,
   whose address is a shrink of the buffer: the condition reads memory without
   an INDEX. *)
let loaded_conditions name n =
  let x =
    Ops.load (Ops.v Op.Shrink ~src:[ buf; var name 0 12; Ops.int 4 ]) []
  in
  let lanes u = Ops.broadcast u 4 in
  let c = Ops.O.(x < lanes (Ops.float ~dtype:Float32 3.)) in
  let rec link x acc k =
    if k = n then acc
    else
      let x = Ops.O.(x + lanes (Ops.float ~dtype:Float32 1.)) in
      link x (Ops.where c x acc) (k + 1)
  in
  link x x 0

(* [words f] is the words [f ()] allocates. A full collection first, and a
   minor heap that holds every word [f] allocates, keep the collector from
   freeing, during [f], nodes [f] would otherwise find interned: the count is
   then [f]'s own, whenever it runs. *)
let words f =
  let gc = Gc.get () in
  Gc.full_major ();
  Gc.set { gc with minor_heap_size = 8 lsl 20 };
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  let words = Gc.minor_words () -. before in
  Gc.set gc;
  words

let cost =
  group "cost"
    [
      test "sym's work on a chain of selections is linear in its length"
        (fun () ->
          let work n =
            let e = selections ("chain" ^ string_of_int n) n in
            words (fun () -> sym e)
          in
          let short = work 250 and long = work 500 in
          less float_exact ~than:(2.5 *. short) long);
      test
        "sym's work on a chain under one far condition is linear in its length"
        (fun () ->
          let work n =
            let e = far_conditions ("far" ^ string_of_int n) n in
            words (fun () -> sym e)
          in
          (* Linear work is [a * n + b] with [b >= 0]: at most twice the work,
             with a tenth of slack for tables that double their capacity at
             different lengths. *)
          let short = work 250 and long = work 500 in
          less float_exact ~than:(2.2 *. short) long);
      test
        "sym's work on a chain under one condition over a load of several \
         elements is linear in its length"
        (fun () ->
          let work n =
            let e = loaded_conditions ("loaded" ^ string_of_int n) n in
            words (fun () -> sym e)
          in
          let short = work 250 and long = work 500 in
          less float_exact ~than:(2.2 *. short) long);
    ]

let () =
  exit
    (run "Tolk.Symbolic"
       [
         symbolic_simple;
         wrapping;
         committed_constants;
         floats;
         conditions;
         sym_group;
         other_tests;
         laws;
         cost;
         group "tinygrad" [ Recorded.tests; Recorded.random_expressions ];
       ])
