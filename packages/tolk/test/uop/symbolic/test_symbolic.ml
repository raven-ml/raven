(* Tests of Tolk.Symbolic: each rewrite its interface states, the laws of
   simplification, and every simplification tinygrad's tests make. *)

open Windtrap
open Tolk
open Common

(* [folds_to input expected] is the claim that symbolic_simple rewrites [input]
   to [expected]. *)
let folds_to input expected =
  equal uop expected (rewrite Shape.symbolic_simple input)

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
let r0 = Ops.range (Int 4) [ 0 ]
let r1 = Ops.range (Int 4) [ 1 ]
let reduce_range = Ops.range ~axis_type:Reduce (Int 4) [ 2 ]
let int32 n = Ops.int ~dtype:Int32 n
let cast_bool dt = Ops.cast cond dt

(* A node built as given, where the constructors would fold it: a cast to its
   own type, a group of one node. *)
let raw ?arg op src = Ops.v ?arg ~src op

(* Invalid values *)

let invalid_values =
  let gated = Shape.valid a cond in
  group "invalid values"
    [
      test "invalid_gate matches a gated value, naming its sources" (fun () ->
          let by_name =
            List.sort (fun (n0, _) (n1, _) -> String.compare n0 n1)
          in
          equal
            (list (list (pair string uop)))
            [ [ ("cond", cond); ("i", Ops.invalid); ("x", a) ] ]
            (List.map by_name (Ops.Upat.match_ Shape.invalid_gate gated)));
      test "invalid_gate does not match a selection of a valid value" (fun () ->
          equal int 0
            (List.length
               (Ops.Upat.match_ Shape.invalid_gate (Ops.where cond a b))));
      test "a binary operation moves inside the gate of its first operand"
        (fun () ->
          folds_to Ops.O.(gated * int 10) (Shape.valid Ops.O.(a * int 10) cond);
          folds_to Ops.O.(gated < int 3) (Shape.valid Ops.O.(a < int 3) cond));
      test "a binary operation moves inside the gate of its second operand"
        (fun () ->
          folds_to Ops.O.(int 10 * gated) (Shape.valid Ops.O.(int 10 * a) cond));
      test "a multiply-add moves inside the gate of each operand (D25)"
        (fun () ->
          let mulacc x y z = Ops.alu x Mulacc [ y; z ] in
          folds_to (mulacc (Shape.valid f cond) g g)
            (Shape.valid (mulacc f g g) cond);
          folds_to (mulacc f g (Shape.valid g cond))
            (Shape.valid (mulacc f g g) cond));
      test "a multiply-add of invalid is invalid (D25)" (fun () ->
          folds_to (raw Mulacc [ f; Ops.invalid; g ]) Ops.invalid);
      test "a cast moves inside the gate" (fun () ->
          let gated = Shape.valid x Ops.O.(x < int32 4) in
          folds_to
            (Ops.cast gated Dtype.Float16)
            (Shape.valid (Ops.cast x Dtype.Float16) Ops.O.(x < int32 4)));
      test "an arithmetic operation of invalid is invalid" (fun () ->
          folds_to Ops.O.(Ops.invalid * int 2) Ops.invalid;
          folds_to Ops.O.(Ops.invalid + a) Ops.invalid;
          folds_to (Ops.cast Ops.invalid Dtype.Int32) Ops.invalid);
      test "a stack of invalid keeps its width" (fun () ->
          let lanes = raw Stack [ Ops.invalid; Ops.invalid ] in
          folds_to lanes lanes);
      test "a selection between invalid and invalid is invalid" (fun () ->
          folds_to (Ops.where cond Ops.invalid Ops.invalid) Ops.invalid);
      test "a rule that computes with the value of invalid does not apply"
        (fun () ->
          let stays e =
            equal uop (rewrite Shape.commutative e) (symbolic e)
          in
          stays (Ops.where Ops.O.(a < Ops.invalid) (Ops.int 0) a);
          stays Ops.O.(a * int 2 < Ops.invalid);
          stays Ops.O.(a // int 2 < Ops.invalid));
      test "a comparison of invalid is left" (fun () ->
          let lt = Ops.O.(Ops.invalid < a) in
          equal uop lt (symbolic lt));
      test "a selection by invalid is invalid" (fun () ->
          folds_to (raw Where [ Ops.invalid; a; b ]) Ops.invalid);
      test "a gate on a condition moves out of the selection" (fun () ->
          let gated_cond = Shape.valid other cond in
          folds_to (Ops.where gated_cond a b)
            (Shape.valid (Ops.where other a b) cond));
      test "invalid in the true branch moves to the false branch" (fun () ->
          folds_to
            (Ops.where cond Ops.invalid a)
            (Shape.valid a (Ops.logical_not cond)));
      test "a gate in a branch lifts out of the selection" (fun () ->
          folds_to
            (Ops.where other (Shape.valid a cond) b)
            (Shape.valid (Ops.where other a b)
               Ops.O.(Ops.logical_not other lor cond));
          folds_to
            (Ops.where other b (Shape.valid a cond))
            (Shape.valid (Ops.where other b a) Ops.O.(other lor cond)));
      test "a gate independent of a reduction's ranges moves out of it"
        (fun () ->
          let body = Shape.valid a cond in
          folds_to
            (Ops.reduce body Add [ reduce_range ])
            (Shape.valid (Ops.reduce a Add [ reduce_range ]) cond));
      test "a gate clause on a reduction's range stays in it" (fun () ->
          let on_range = Ops.O.(reduce_range < int 2) in
          let body = Shape.valid a Ops.O.(cond land on_range) in
          folds_to
            (Ops.reduce body Add [ reduce_range ])
            (Shape.valid
               (Ops.reduce (Shape.valid a on_range) Add [ reduce_range ])
               cond));
      test "a store to an invalid index does nothing" (fun () ->
          folds_to (Ops.store (Ops.index buf [ Ops.invalid ]) f) (raw Noop []));
      test "a load from an invalid index is its alternative, or 0" (fun () ->
          let index = Ops.index buf [ Ops.invalid ] in
          folds_to (Ops.load index []) (Ops.float ~dtype:Float32 0.);
          folds_to (Ops.load index [ g ]) g);
    ]

let remove_invalid =
  let rewrite = rewrite Shape.pm_remove_invalid in
  group "pm_remove_invalid"
    [
      test "a gate's invalid is 0 of the gate's type" (fun () ->
          equal uop (Ops.where cond x (int32 0)) (rewrite (Shape.valid x cond)));
      test "a stack's invalid elements are 0" (fun () ->
          let stack = raw Stack [ a; Ops.invalid; b ] in
          equal uop (raw Stack [ a; Ops.int 0; b ]) (rewrite stack));
      test "a float gate's invalid is 0.0 of its type" (fun () ->
          equal uop
            (Ops.where cond f (Ops.float ~dtype:Float32 0.))
            (rewrite (Shape.valid f cond)));
    ]

(* symbolic_simple *)

let identities =
  group "identities"
    [
      test "x + 0, x lxor 0 and x lor 0 are x" (fun () ->
          folds_to Ops.O.(a + int 0) a;
          folds_to Ops.O.(x lxor int 0) x;
          folds_to Ops.O.(x lor int 0) x);
      test "shifts by 0 are x" (fun () ->
          folds_to Ops.O.(x lsl int 0) x;
          folds_to Ops.O.(x lsr int 0) x);
      test "x * 1 and x // 1 are x" (fun () ->
          folds_to Ops.O.(a * int 1) a;
          folds_to Ops.O.(int 1 * a) a;
          folds_to Ops.O.(a // int 1) a);
      test "x // -1 is -x" (fun () ->
          folds_to Ops.O.(a // int (-1)) Ops.O.(~-a));
      test "x // x is 1" (fun () -> folds_to Ops.O.(a // a) (Ops.int 1));
      test "(x lxor y) lxor y is x" (fun () ->
          folds_to Ops.O.(x lxor y lxor y) x);
      test "(x % y) % y is x % y" (fun () ->
          folds_to Ops.O.(a % b % b) Ops.O.(a % b));
      test "a boolean and a constant fold by the constant" (fun () ->
          folds_to Ops.O.(cond land bool true) cond;
          folds_to Ops.O.(cond land bool false) (Ops.bool false);
          folds_to Ops.O.(cond lor bool true) (Ops.bool true);
          folds_to Ops.O.(cond lor bool false) cond);
      test "x <> false is x" (fun () ->
          folds_to Ops.O.(cond <> bool false) cond);
      test "a double negation is x" (fun () ->
          folds_to (Ops.logical_not (Ops.logical_not cond)) cond);
      test "where x true false is x, where x false true is not x" (fun () ->
          folds_to (Ops.where cond (Ops.bool true) (Ops.bool false)) cond;
          folds_to
            (Ops.where cond (Ops.bool false) (Ops.bool true))
            (Ops.logical_not cond));
      test "an idempotent operation of x with itself is x" (fun () ->
          folds_to (Ops.maximum a a) a;
          folds_to Ops.O.(cond land cond) cond;
          folds_to Ops.O.(cond lor cond) cond);
      test "a boolean cast to an integer and compared to 0 is the boolean"
        (fun () ->
          folds_to Ops.O.(cast_bool Int32 <> int 0) cond;
          folds_to Ops.O.(cast_bool Weak_int <> int 0) cond);
      test "a boolean cast to an integer and compared to 1 is its negation"
        (fun () ->
          folds_to Ops.O.(cast_bool Int32 <> int 1) (Ops.logical_not cond));
      test "a boolean cast to an integer differs from any other integer"
        (fun () ->
          folds_to Ops.O.(cast_bool Int32 <> int 2) (Ops.bool true);
          folds_to Ops.O.(cast_bool Int32 <> int (-1)) (Ops.bool true));
      test "the truncation of an integer or a boolean is itself" (fun () ->
          folds_to (Ops.trunc x) x;
          folds_to (Ops.trunc a) a;
          folds_to (Ops.trunc cond) cond);
      test "the truncation of a float stays" (fun () ->
          folds_to (Ops.trunc f) (Ops.trunc f));
    ]

let recombination =
  let g = var "g" 0 124 in
  group "recombination"
    [
      test "x % d + (x // d) * d is x" (fun () ->
          folds_to Ops.O.((g % int 4) + (g // int 4 * int 4)) g;
          folds_to Ops.O.((g // int 4 * int 4) + (g % int 4)) g);
      test "a scaled remainder and quotient recombine into x times the scale"
        (fun () ->
          folds_to
            Ops.O.((g % int 4 * int 2) + (g // int 4 * int 8))
            Ops.O.(g * int 2));
      test "a recombination keeps the other terms of the sum" (fun () ->
          folds_to Ops.O.(a + (g % int 4) + (g // int 4 * int 4)) Ops.O.(g + a));
      test "a remainder of the quotient recombines into a wider remainder"
        (fun () ->
          folds_to
            Ops.O.((g % int 4) + (g // int 4 % int 3 * int 4))
            Ops.O.(g % int 12));
      test "a quotient of another divisor does not recombine" (fun () ->
          let sum = Ops.O.((g % int 4) + (g // int 3 * int 4)) in
          folds_to sum sum);
      test
        "a quotient by a variable, of another base, or merged with another \
         divisor does not recombine" (fun () ->
          let stays sum = folds_to sum sum in
          stays Ops.O.((g % int 4) + (g // b * int 4));
          stays Ops.O.((g % int 4) + (a // int 4 * int 4));
          stays Ops.O.((g // int 2 % int 4) + (g // int 16 * int 4));
          stays Ops.O.((g % int 4) + ((g + int 1) // int 4 * int 4)));
      test "a remainder of the quotient by a negative number does not recombine"
        (fun () ->
          let sum = Ops.O.((g % int 4) + (g // int 4 % int (-3) * int 4)) in
          folds_to sum sum);
      test "a committed sum does not recombine" (fun () ->
          let sum = Ops.O.((x % int32 4) + (x // int32 4 * int32 4)) in
          folds_to sum sum);
    ]

let zeros =
  let u = var ~dtype:Uint32 "u" 0 255 in
  group "zeros"
    [
      test "x < x is false" (fun () -> folds_to Ops.O.(a < a) (Ops.bool false));
      test "x <> x is false for integers and booleans" (fun () ->
          folds_to Ops.O.(a <> a) (Ops.bool false);
          folds_to Ops.O.(cond <> cond) (Ops.bool false));
      test "x <> x stays for floats, which may be NaN" (fun () ->
          folds_to Ops.O.(f <> f) Ops.O.(f <> f));
      test "x % x, x lxor x and x land 0 are 0" (fun () ->
          folds_to Ops.O.(a % a) (Ops.int 0);
          folds_to Ops.O.(x lxor x) (int32 0);
          folds_to Ops.O.(x land int 0) (int32 0));
      test "a mask of the bits a right shift drops is removed" (fun () ->
          folds_to Ops.O.((u land int (-4)) lsr int 2) Ops.O.(u lsr int 2));
      test "a mask of the bits a division by a power of two drops is removed"
        (fun () -> folds_to Ops.O.(u land int (-4) // int 4) Ops.O.(u // int 4));
      test "a mask of kept bits stays" (fun () ->
          let shifted = Ops.O.((u land int (-8)) lsr int 2) in
          folds_to shifted shifted;
          let above = Ops.O.(u land int (-8) // int 4) in
          folds_to above above;
          let by_three = Ops.O.(u land int (-4) // int 3) in
          folds_to by_three by_three);
    ]

let constants =
  group "constants"
    [
      test "an operation on weak constants is its exact value" (fun () ->
          let big = Bigint.shift_left Bigint.one 40 in
          folds_to
            Ops.O.(Ops.const (`Int big) + Ops.const (`Int big))
            (Ops.const (`Int (Bigint.shift_left Bigint.one 41)));
          folds_to Ops.O.(float 1.25 + float 2.5) (Ops.float 3.75));
      test "an operation on committed constants computes at their width"
        (fun () ->
          folds_to Ops.O.(int32 3 * int32 4) (int32 12);
          folds_to
            Ops.O.(Ops.float ~dtype:Float32 1.5 + Ops.float ~dtype:Float32 2.)
            (Ops.float ~dtype:Float32 3.5));
      (* Uncasting a committed constant reads it at its width, where
         tinygrad keeps the literal unwrapped. *)
      test "a comparison reads a committed constant at its width" (fun () ->
          let x = var ~dtype:Uint8 "x" 0 255 in
          let lt = Ops.O.(x < Ops.cconst Uint8 (i 300)) in
          equal uop Ops.O.(x < int 44) (rewrite Shape.pm_uncast_const lt);
          equal Dtypes.const (`Bool false)
            (Interpreter.eval ~vars:[ ("x", i 100) ] (symbolic lt)));
      (* A committed constant is read at its width, by an operation that
         folds and by a cast of it, where tinygrad reads it unwrapped. *)
      test "an operation reads committed constants at their width" (fun () ->
          let max = Ops.int ~dtype:Uint32 0xFFFF_FFFF in
          let one = Ops.int ~dtype:Uint32 1 in
          folds_to Ops.O.((max + one) lsr one) (Ops.int ~dtype:Uint32 0);
          let half = `Int (Bigint.shift_left Bigint.one 31) in
          folds_to
            (Ops.cast (Ops.cconst Int32 half) Int64)
            (Ops.int ~dtype:Int64 (-2147483648));
          let e = var ~dtype:Uint8 "e" 3 3 in
          equal Dtypes.const (i 0)
            (Interpreter.eval
               ~vars:[ ("e", i 3) ]
               (symbolic Ops.O.(Ops.cast (int (-1)) Uint8 % e))));
      test "NaN is unequal to itself when constants fold, as IEEE says"
        (fun () ->
          let nan = Ops.float ~dtype:Float32 Float.nan in
          folds_to Ops.O.(nan <> nan) (Ops.bool true);
          folds_to Ops.O.(nan < nan) (Ops.bool false));
      test "a truncating division of constants rounds toward zero" (fun () ->
          let tdiv x y = Ops.alu (Ops.int x) Cdiv [ Ops.int y ] in
          let tmod x y = Ops.alu (Ops.int x) Cmod [ Ops.int y ] in
          folds_to (tdiv 7 (-3)) (Ops.int (-2));
          folds_to (tdiv (-50) 6) (Ops.int (-8));
          folds_to (tmod 5 (-3)) (Ops.int 2));
      test "a shift of constants by a negative count stays" (fun () ->
          let stays e =
            folds_to e e;
            equal uop e (symbolic e)
          in
          stays Ops.O.(int 3 lsl int (-1));
          stays Ops.O.(int32 3 lsr int32 (-2));
          let u = var ~dtype:Uint32 "u" 0 255 in
          stays Ops.O.((u land int (-4)) lsr int (-2)));
      test "a threefry of constants stays" (fun () ->
          let tf =
            Ops.alu (Ops.int ~dtype:Uint32 0) Threefry
              [ Ops.int ~dtype:Uint32 1 ]
          in
          folds_to tf tf);
      test "a division of constants by zero is 0, a remainder the dividend"
        (fun () ->
          folds_to Ops.O.(int 7 // int 0) (Ops.int 0);
          folds_to (Ops.alu (Ops.int 7) Cdiv [ Ops.int 0 ]) (Ops.int 0);
          folds_to (Ops.alu (Ops.int 7) Cmod [ Ops.int 0 ]) (Ops.int 7);
          folds_to Ops.O.(int 7 % int 0) (Ops.int 7));
      test "a where, a comparison and a negation of stacks fold lane by lane"
        (fun () ->
          let stack l = raw Stack l in
          folds_to
            (Ops.where
               (stack [ Ops.bool true; Ops.bool false ])
               (stack [ Ops.int 3; Ops.int 4 ])
               (stack [ Ops.int 30; Ops.int 40 ]))
            (stack [ Ops.int 3; Ops.int 40 ]);
          folds_to
            Ops.O.(stack [ int 1; int 4 ] < stack [ int 2; int 3 ])
            (stack [ Ops.bool true; Ops.bool false ]);
          folds_to
            Ops.O.(~-(stack [ int 1; int (-2) ]))
            (stack [ Ops.int (-1); Ops.int 2 ]));
      test "an operation on stacks of constants folds lane by lane" (fun () ->
          folds_to
            Ops.O.(
              raw Stack [ int32 1; int32 2 ] + raw Stack [ int32 3; int32 4 ])
            (raw Stack [ int32 4; int32 6 ]));
      test "a unary operation of a weak float constant stays weak" (fun () ->
          folds_to (Ops.sqrt (Ops.float 4.)) (Ops.float 2.);
          folds_to (Ops.exp2 (Ops.float 3.)) (Ops.float 8.));
      test "weak constants commit to the type of a committed peer" (fun () ->
          folds_to Ops.O.(int32 3 + int 4) (int32 7));
      test "a cast of an infinity or a NaN to an integer stays a cast"
        (fun () ->
          List.iter
            (fun v ->
              let c = Ops.cast (Ops.float ~dtype:Float32 v) Int32 in
              folds_to c c)
            [ Float.infinity; Float.neg_infinity; Float.nan ]);
      test "a cast of a constant is the constant of the cast's type" (fun () ->
          folds_to (Ops.cast (Ops.int 3) Float32) (Ops.float ~dtype:Float32 3.);
          folds_to
            (Ops.cast (Ops.cast (Ops.int 3) Int64) Float32)
            (Ops.float ~dtype:Float32 3.));
      test "0 / 0 is NaN" (fun () ->
          let zero = Ops.float 0. in
          let nan =
            rewrite ~order:before Shape.symbolic_simple Ops.O.(zero / zero)
          in
          match Ops.arg nan with
          | Const (`Float v) when Float.is_nan v -> ()
          | _ -> failf "0 / 0 is@ %a" (Testable.pp uop) nan);
      test "x / x and (x * y) / y stay, for floats" (fun () ->
          folds_to Ops.O.(f / f) Ops.O.(f / f);
          folds_to Ops.O.(f * g / g) Ops.O.(f * g / g));
      test "x * 0 is 0 for integers and booleans, and stays for floats"
        (fun () ->
          folds_to Ops.O.(a * int 0) (Ops.int 0);
          folds_to Ops.O.(cond * bool false) (Ops.bool false);
          folds_to Ops.O.(f * float 0.) Ops.O.(f * float 0.));
    ]

let booleans =
  group "booleans"
    [
      test "a boolean product is a conjunction" (fun () ->
          folds_to Ops.O.(cond * other) Ops.O.(cond land other));
      test "a boolean sum and maximum are a disjunction" (fun () ->
          folds_to Ops.O.(cond + other) Ops.O.(cond lor other);
          folds_to (Ops.maximum cond other) Ops.O.(cond lor other));
    ]

let casts =
  let i8 = var ~dtype:Int8 "i" (-4) 4 in
  group "casts"
    [
      test "a cast or bitcast to its operand's type is its operand" (fun () ->
          folds_to (raw ~arg:(Dtype Int32) Cast [ x ]) x;
          folds_to (raw ~arg:(Dtype Int32) Bitcast [ x ]) x);
      test "a bitcast of a constant has the same bits" (fun () ->
          folds_to
            (Ops.bitcast (int32 (-1)) Uint32)
            (Ops.int ~dtype:Uint32 0xFFFF_FFFF);
          folds_to
            (Ops.bitcast (Ops.float ~dtype:Float32 1.) Int32)
            (int32 0x3f80_0000));
      test "a cast through a type that holds every value back is its operand"
        (fun () -> folds_to (Ops.cast (Ops.cast i8 Int32) Int8) i8);
      test "a cast through a narrower type back stays" (fun () ->
          let there_and_back = Ops.cast (Ops.cast x Int8) Int32 in
          folds_to there_and_back there_and_back);
      test "two bitcasts are one" (fun () ->
          folds_to
            (Ops.bitcast (Ops.bitcast x Float32) Uint32)
            (Ops.bitcast x Uint32);
          folds_to (Ops.bitcast (Ops.bitcast x Float32) Int32) x);
      (* A folded bitcast reads a float constant's bits as its type stores
         them, where tinygrad converts it first and a NaN loses its bits. *)
      cases
        "a bitcast round trip of every 8- and 16-bit word folds to its value"
        ~name:(Format.asprintf "%a" Dtype.pp)
        Dtype.[ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz; Float16; Bfloat16 ]
        (fun mid ->
          let w = if Dtype.itemsize mid = 1 then Dtype.Uint8 else Uint16 in
          for word = 0 to (1 lsl (8 * Dtype.itemsize mid)) - 1 do
            let u = Ops.bitcast (Ops.bitcast (Ops.int ~dtype:w word) mid) w in
            equal
              ~msg:(Printf.sprintf "0x%x" word)
              Dtypes.const (Interpreter.eval u)
              (Interpreter.eval (Shape.simplify u))
          done);
      test "a bitcast of a constant to another width stays" (fun () ->
          let widened = Ops.bitcast (int32 1) Int64 in
          folds_to widened widened);
      test "a cast to a boolean is x <> 0" (fun () ->
          folds_to (Ops.cast a Bool) Ops.O.(a <> int 0));
    ]

(* A power leaves committed constants, [1.0 * x], that symbolic_simple alone
   does not uncast: its claims hold under symbolic. *)
let powers =
  let by_symbolic input expected =
    equal uop (rewrite Shape.commutative expected) (symbolic input)
  in
  group "powers"
    [
      test "x ** 0 is 1" (fun () ->
          by_symbolic (Ops.pow f (Ops.float 0.)) (Ops.float ~dtype:Float32 1.));
      test "x ** 1 is x, x ** 2 is x * x" (fun () ->
          by_symbolic (Ops.pow f (Ops.float 1.)) f;
          by_symbolic (Ops.pow f (Ops.float 2.)) Ops.O.(f * f));
      test "x ** 0.5 is the square root of x, +0. at -0. and +inf at -inf"
        (fun () ->
          let special v r =
            Ops.where Ops.O.(f <> float v) r (Ops.float (Float.abs v))
          in
          by_symbolic
            (Ops.pow f (Ops.float 0.5))
            (special Float.neg_infinity (special 0. (Ops.sqrt f))));
      test "x ** -1 is the reciprocal of x" (fun () ->
          by_symbolic (Ops.pow f (Ops.float (-1.))) (Ops.reciprocal f));
      test "x ** c stays for c between -1 and 0" (fun () ->
          let p = Ops.pow f (Ops.float (-0.8)) in
          by_symbolic p p);
      test "a power of constants is its value" (fun () ->
          by_symbolic
            (Ops.pow (Ops.float ~dtype:Float32 3.) (Ops.float 2.))
            (Ops.float ~dtype:Float32 9.));
      test "x ** 1.3 stays" (fun () ->
          let p = Ops.pow f (Ops.float 1.3) in
          by_symbolic p p);
      test "a large integer exponent takes no square root" (fun () ->
          let p = symbolic (Ops.pow f (Ops.float (Float.ldexp 1. 60))) in
          equal (list string) []
            (List.filter_map
               (fun u -> if Ops.op u = Sqrt then Some "Ops.SQRT" else None)
               (Ops.toposort ~calls:Enter p)));
      test "x ** infinity and x ** NaN are left to xpow" (fun () ->
          List.iter
            (fun e ->
              let p = Ops.pow f (Ops.float e) in
              by_symbolic p p;
              equal uop (sym (Transcendental.xpow f (Ops.float e))) (sym p))
            [ Float.infinity; Float.nan ]);
      test "1 ** x is 1, and c ** x is exp2 (x * log2 c) for positive c"
        (fun () ->
          by_symbolic (Ops.pow (Ops.float ~dtype:Float32 1.) f) (Ops.float 1.);
          by_symbolic (Ops.pow (Ops.float ~dtype:Float32 2.) f) (Ops.exp2 f));
      test "c ** x computes in float for an integer exponent" (fun () ->
          let n = Shape.param 5 Dtype.Int32 in
          folds_to
            (Ops.pow (Ops.float 3.) n)
            (Ops.exp2 Ops.O.(Ops.cast n Weak_float * float (Float.log2 3.))));
      test "c ** x stays for c at most 0" (fun () ->
          by_symbolic
            (Ops.pow (Ops.float ~dtype:Float32 (-2.)) f)
            (Ops.pow (Ops.float (-2.)) f));
      test "a 64-bit integer packed from two halves unpacks to the half read"
        (fun () ->
          let hi = var ~dtype:Uint32 "hi" 0 7
          and lo = var ~dtype:Uint32 "lo" 0 7 in
          let wide dt = Ops.cast dt Uint64 in
          let packed = Ops.O.((wide hi lsl int 32) lor wide lo) in
          folds_to (Ops.cast packed Uint32) lo;
          folds_to Ops.O.(packed lsr int 32) (wide hi);
          let hi64 = var ~dtype:Uint64 "hi" 0 0xFFFF in
          let wide_hi = Ops.O.(((hi64 lsl int 32) lor wide lo) lsr int 32) in
          folds_to wide_hi wide_hi);
    ]

let selections =
  group "selections"
    [
      test "a selection between equal values is that value" (fun () ->
          folds_to (Ops.where cond a a) a);
      test "a selection by a constant is the branch it picks" (fun () ->
          folds_to (Ops.where (Ops.bool true) a b) a;
          folds_to (Ops.where (Ops.bool false) a b) b);
      test "a weak branch picked from a weak selection stays as it is"
        (fun () ->
          let v = var ~dtype:Weak_float "v" 0 3 in
          folds_to (raw Where [ Ops.bool true; Ops.int 1; v ]) (Ops.int 1));
      test "a selection by a broadcast constant is the branch it picks"
        (fun () ->
          let p = Shape.param ~shape:[ Int 4 ] 0 Dtype.Float32
          and q = Shape.param ~shape:[ Int 4 ] 1 Dtype.Float32 in
          let truth b = Shape.const_like ~dtype:Bool p (`Bool b) in
          folds_to (Ops.where (truth true) p q) p;
          folds_to (Ops.where (truth false) p q) q;
          (* Assuming the outer condition false within its false branch, and the
             inner one true within its true branch, would trade the two
             constants without end. *)
          simplifies_to
            (Ops.where (truth true) p
               (Ops.where (truth false) (Ops.where (truth true) p q) q))
            p);
      test "a selection by a padded constant is no constant's" (fun () ->
          let p = Shape.param ~shape:[ Int 6 ] 0 Dtype.Float32
          and q = Shape.param ~shape:[ Int 6 ] 1 Dtype.Float32 in
          let mask =
            Shape.pad
              (Shape.const_like ~dtype:Bool
                 (Shape.param ~shape:[ Int 4 ] 2 Dtype.Float32)
                 (`Bool true))
              [ Some (Ops.Int 1, Ops.Int 1) ]
          in
          let e = Ops.where mask p q in
          folds_to e e);
      test "a selection by a constant keeps the selection's type" (fun () ->
          let h = var ~dtype:Float16 "h" 0 3 in
          folds_to
            (Ops.where (Ops.bool true) (Ops.float 0.) h)
            (Ops.float ~dtype:Float16 0.);
          folds_to (Ops.where (Ops.bool true) (Ops.int 0) x) (int32 0));
    ]

let symbolic_simple =
  group "symbolic_simple"
    [
      identities;
      recombination;
      zeros;
      constants;
      booleans;
      casts;
      powers;
      selections;
    ]

(* commutative *)

let commutative =
  let canonical u = rewrite Shape.commutative u in
  group "commutative"
    [
      test "two sums of the same weak integer terms are the same node"
        (fun () ->
          equal uop (canonical Ops.O.(a + b)) (canonical Ops.O.(b + a)));
      test "a committed sum keeps its order" (fun () ->
          equal uop Ops.O.(y + x) (canonical Ops.O.(y + x)));
      test "operands that differ only by their tags keep their order" (fun () ->
          let sum = Ops.O.(a + Ops.rtag a) in
          equal uop sum (canonical sum));
      test "a non-commutative operation keeps its order" (fun () ->
          equal uop Ops.O.(b < a) (canonical Ops.O.(b < a)));
    ]

(* symbolic *)

let by_symbolic = simplifies_to ~by:symbolic

let terms =
  group "terms"
    [
      test "x lor not x is true" (fun () ->
          by_symbolic Ops.O.(cond lor Ops.logical_not cond) (Ops.bool true));
      test "like terms combine" (fun () ->
          by_symbolic Ops.O.((a * int 2) + (a * int 3)) Ops.O.(a * int 5);
          by_symbolic Ops.O.(a + (a * int 3)) Ops.O.(a * int 4);
          by_symbolic Ops.O.(a + a) Ops.O.(a * int 2));
      test "like terms combine as the last two terms of a sum" (fun () ->
          by_symbolic
            Ops.O.(b + (a * int 2) + (a * int 3))
            Ops.O.(b + (a * int 5));
          by_symbolic Ops.O.(b + a + (a * int 3)) Ops.O.(b + (a * int 4));
          by_symbolic Ops.O.(b + (a * int 3) + a) Ops.O.(b + (a * int 4));
          by_symbolic Ops.O.(b + a + a) Ops.O.(b + (a * int 2)));
      test "a term's new coefficient is a weak constant" (fun () ->
          let n = Shape.param 5 Dtype.Int32 in
          by_symbolic Ops.O.(n + n) Ops.O.(n * int 2);
          folds_to Ops.O.(n // int (-1)) Ops.O.(n * int (-1)));
      test "(x / y) / z stays" (fun () ->
          let h = Shape.param 3 Dtype.Float32 in
          by_symbolic Ops.O.(f / g / h) Ops.O.(f / g / h));
      test "-(x + c) is -x + -c for integers, and stays for floats"
        (fun () ->
          by_symbolic
            Ops.O.(int (-1) * (a + int 3))
            Ops.O.((a * int (-1)) + int (-3));
          let neg = Ops.O.(float (-1.) * (f + float 3.)) in
          by_symbolic neg neg);
      test "c * (x + c') is c * x + c * c'" (fun () ->
          by_symbolic Ops.O.(int 2 * (a + int 3)) Ops.O.((a * int 2) + int 6));
    ]

let symbolic_selections =
  let t = var "t" 0 3 and e = var "e" 0 3 in
  group "selections"
    [
      test "a selection by a negation swaps its branches" (fun () ->
          by_symbolic
            (Ops.where (Ops.logical_not cond) t e)
            (Ops.where cond e t);
          by_symbolic
            (Ops.where (raw Cmpne [ Ops.bool true; cond ]) t e)
            (Ops.where cond e t));
      test "the condition is true in the true branch and false in the other"
        (fun () ->
          let inner = Ops.where cond Ops.O.(~-a) a in
          by_symbolic
            (Ops.where cond Ops.O.(inner * int 2) Ops.O.(inner + int 1))
            (Ops.where cond Ops.O.(a * int (-2)) Ops.O.(a + int 1)));
      test "the condition is false in the false branch" (fun () ->
          let t = var "t" 0 3 in
          by_symbolic
            (Ops.where cond t (Ops.cast cond Weak_int))
            (Ops.where cond t
               (raw ~arg:(Dtype Weak_int) Cast [ Ops.bool false ])));
      test "another condition is not folded in the branches" (fun () ->
          let e' = Ops.where other t e in
          let sel = Ops.where cond e' (Ops.where other e t) in
          by_symbolic sel sel);
      test
        "a padded constant condition is not folded in the branches"
        (fun () ->
          let x = Shape.param ~shape:[ Ops.Int 6 ] 0 Dtype.Float32 in
          let y = Shape.param ~shape:[ Ops.Int 6 ] 1 Dtype.Float32 in
          let inner = Shape.param ~shape:[ Ops.Int 4 ] 2 Dtype.Float32 in
          (* One node for every use of the constant: assuming it true in a
             branch would rewrite it everywhere. *)
          let c =
            Shape.pad
              (Shape.const_like ~dtype:Bool inner (`Bool true))
              [ Some (Ops.Int 1, Ops.Int 1) ]
          in
          let sel = Ops.where c (Ops.where c x (Ops.where c y x)) y in
          equal uop sel (symbolic sel));
      test "a condition over an index is not folded in the branches" (fun () ->
          let load = Ops.index buf [ a ] in
          let c = Ops.O.(load < float 1.) in
          let sel = Ops.where c (Ops.where c f g) (Shape.param 3 Dtype.Float32) in
          equal uop sel (symbolic sel));
      test "where g x 0 <> 0 is g land (x <> 0)" (fun () ->
          by_symbolic
            Ops.O.(Ops.where cond b (int 0) <> int 0)
            Ops.O.(cond land (b <> int 0)));
      test "nested selections sharing a false branch merge by conjunction"
        (fun () ->
          by_symbolic
            (Ops.where cond (Ops.where other t e) e)
            (Ops.where Ops.O.(cond land other) t e));
      test "nested selections sharing a true branch merge by disjunction"
        (fun () ->
          by_symbolic
            (Ops.where cond t (Ops.where other t e))
            (Ops.where Ops.O.(cond lor other) t e));
      test "an operation on selections by one condition with constant branches"
        (fun () ->
          by_symbolic
            Ops.O.(Ops.where cond t (int 0) + Ops.where cond e (int 1))
            (Ops.where cond Ops.O.(t + e) (Ops.int 1));
          by_symbolic
            Ops.O.(b + Ops.where cond t (int 0) + Ops.where cond e (int 1))
            Ops.O.(b + Ops.where cond (t + e) (int 1)));
      test "a cast of a selection stays outside it" (fun () ->
          let sel = Ops.where cond other Ops.O.(a < int 2) in
          by_symbolic (Ops.cast sel Int32) (Ops.cast sel Int32));
      test "an operation on selections without a constant pair stays" (fun () ->
          let sum = Ops.O.(Ops.where cond t e + Ops.where cond e t) in
          by_symbolic sum sum;
          let sum = Ops.O.(b + Ops.where cond t e + Ops.where cond e t) in
          by_symbolic sum sum);
      test "complementary zero branches select directly" (fun () ->
          by_symbolic
            Ops.O.(Ops.where cond t (int 0) + Ops.where cond (int 0) e)
            (Ops.where cond t e));
    ]

let bounds =
  group "bounds"
    [
      test "a comparison whose bounds are equal is a constant" (fun () ->
          by_symbolic Ops.O.(a < int 9) (Ops.bool true);
          by_symbolic Ops.O.(a < int 0) (Ops.bool false);
          by_symbolic Ops.O.(a <> int 9) (Ops.bool true));
      test "a division and a remainder with one possible value are constants"
        (fun () ->
          by_symbolic Ops.O.(a // int 9) (Ops.int 0);
          by_symbolic Ops.O.(var "v" 10 14 // int 5) (Ops.int 2));
      test "a variable of one value is that value" (fun () ->
          by_symbolic (var "one" 1 1) (Ops.int 1));
      test "a hardware index of one value is 0" (fun () ->
          by_symbolic (Ops.special (Int 1) "gidx0") (Ops.int 0));
      test "a range of a constant end with one value is 0" (fun () ->
          by_symbolic (Ops.range (Int 1) [ 0 ]) (Ops.int 0));
      test "a range of a symbolic end is not folded" (fun () ->
          let r = Ops.range (Sym (Ops.cast cond Weak_int)) [ 0 ] in
          by_symbolic r r);
      test "a maximum of operands whose bounds do not overlap is the greater"
        (fun () ->
          by_symbolic (Ops.maximum a (Ops.int 10)) (Ops.int 10);
          by_symbolic (Ops.maximum (Ops.int (-1)) a) a);
      test "a selection that computes something else than a maximum stays"
        (fun () ->
          let v = var "v" (-10) 10 in
          let sel = Ops.where Ops.O.(v < int 0) (Ops.int 1) v in
          by_symbolic sel sel;
          let sel = Ops.where Ops.O.(int 0 < v) v (Ops.int 1) in
          by_symbolic sel sel);
      test "a selection that computes a maximum is a maximum" (fun () ->
          let v = var "v" (-10) 10 in
          by_symbolic
            (Ops.where Ops.O.(v < int 0) (Ops.int 0) v)
            (Ops.maximum v (Ops.int 0));
          by_symbolic
            (Ops.where Ops.O.(int 0 < v) v (Ops.int 0))
            (Ops.maximum v (Ops.int 0)));
    ]

let symbolic_constants =
  group "constants"
    [
      test "two associative operations on constants fold them" (fun () ->
          by_symbolic Ops.O.(a + int 3 + int 4) Ops.O.(a + int 7);
          by_symbolic Ops.O.(a * int 3 * int 4) Ops.O.(a * int 12);
          by_symbolic
            (Ops.maximum (Ops.maximum (var "v" 0 20) (Ops.int 10)) (Ops.int 11))
            (Ops.maximum (var "v" 0 20) (Ops.int 11)));
      test "(x // c1) // c2 is x // (c1 * c2) for positive c2" (fun () ->
          let v = var "v" 0 1800 in
          by_symbolic Ops.O.(v // int 10 // int 9) Ops.O.(v // int 90));
      test "constants move to the end of sums and products" (fun () ->
          by_symbolic Ops.O.(a + int 3 + b) Ops.O.(a + b + int 3);
          by_symbolic Ops.O.(a * int 3 * b) Ops.O.(a * b * int 3));
    ]

let comparisons =
  group "comparisons"
    [
      test "c0 + x < c1 is x < c1 - c0" (fun () ->
          by_symbolic Ops.O.(a + int 2 < int 5) Ops.O.(a < int 3));
      test "c0 * x < c1 divides by c0, rounding up" (fun () ->
          by_symbolic Ops.O.(a * int 4 < int 13) Ops.O.(a < int 4);
          by_symbolic Ops.O.(a * int 4 < int 16) Ops.O.(a < int 4));
      test "c0 * x < c1 flips x's sign for a negative c0" (fun () ->
          let v = var "v" (-5) 5 in
          by_symbolic Ops.O.(v * int (-4) < int 13) Ops.O.(v * int (-1) < int 4));
      test "x // d < c is x < c * d for positive d" (fun () ->
          let v = var "v" 0 24 in
          by_symbolic Ops.O.(v // int 4 < int 3) Ops.O.(v < int 12));
      test "x // d < c is c * d < x for negative d" (fun () ->
          let v = var "v" 0 24 in
          by_symbolic Ops.O.(v // int (-4) < int (-3)) Ops.O.(int 12 < v));
      test "a common divisor of c and the coefficients divides out" (fun () ->
          let small = var "s" 0 3 in
          by_symbolic Ops.O.((a * int 4) + small < int 16) Ops.O.(a < int 4));
      test "a divisor stays when the bound is at most 0" (fun () ->
          let signed = var "s" (-3) 3 and small = var "t" 0 3 in
          let lt = Ops.O.((signed * int 4) + small < int 0) in
          by_symbolic lt lt);
      test "a divisor stays when the other terms can reach it" (fun () ->
          let wide = var "w" 0 4 in
          let lt = Ops.O.((a * int 4) + wide < int 16) in
          by_symbolic lt lt);
      test "c0 + x < c1 stays for floats, where moving c0 rounds" (fun () ->
          let d = var ~dtype:Float64 "d" 0 2 in
          let lt = Ops.O.(float 1e16 + d < float (1e16 +. 2.)) in
          by_symbolic lt lt);
      test "-x < -y is y < x" (fun () ->
          by_symbolic Ops.O.(a * int (-1) < b * int (-1)) Ops.O.(b < a));
      test "not (x < 1) drops the positive coefficients of x" (fun () ->
          let lt1 e = Ops.O.(e < int 1 <> bool true) in
          by_symbolic
            (lt1 Ops.O.((a * int 3) + (b * int 4)))
            (lt1 Ops.O.(a + b)));
      test "not (x < 1) keeps the coefficients of a term that can be negative"
        (fun () ->
          let lt1 e = Ops.O.(e < int 1 <> bool true) in
          let v = var "v" (-3) 3 in
          let e = lt1 Ops.O.((a * int 3) + (v * int 4)) in
          by_symbolic e e;
          let e = lt1 Ops.O.((a * int (-3)) + (b * int 4)) in
          by_symbolic e e);
    ]

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
      test "(x // c1) // c2 stays where c1 * c2 wraps" (fun () ->
          let w =
            Ops.variable ~dtype:Int32 "w" (Dtype.min Int32) (Dtype.max Int32)
          in
          let u = Ops.O.(w // int 65536 // int 65536) in
          by_symbolic u u);
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
          keeps_float_bits
            Ops.O.(
              g
              + Ops.where cond (float tiny) (float 0.)
              + Ops.where cond (float tiny) (float 0.))
            [ [ 0.; 1. ] ];
          let r = Ops.range ~axis_type:Reduce (Int 3) [ 2 ] in
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
      test "a float selection that computes a maximum stays" (fun () ->
          let u = Ops.where Ops.O.(f < float 0.) (Ops.float 0.) f in
          simplifies_to ~by:symbolic u u;
          let u = Ops.where Ops.O.(float 0. < f) f (Ops.float 0.) in
          simplifies_to ~by:symbolic u u);
      test "powers keep pow's special values" (fun () ->
          let by = symbolic in
          keeps_float_bits ~by
            (Ops.pow f (Ops.float 2.5))
            [ [ -.inf ]; [ -0. ] ];
          keeps_float_bits ~by
            (Ops.pow f (Ops.float 0.5))
            [ [ -0. ]; [ -.inf ] ];
          keeps_float_bits ~by (Ops.pow f (Ops.float (-0.8))) [ [ 1e-40 ] ];
          keeps_float_bits ~by
            (Ops.pow (Ops.float ~dtype:Float32 inf) f)
            [ [ 0. ] ]);
    ]

let ranges =
  let r = Ops.range (Int 8) [ 0 ] in
  group "ranges"
    [
      test "a range modulo its end is the range" (fun () ->
          by_symbolic Ops.O.(r % int 8) r);
      test "a range divided by its end is 0" (fun () ->
          by_symbolic Ops.O.(r // int 8) (Ops.int 0));
      test "a range of a symbolic end modulo and divided by its end" (fun () ->
          let n = Ops.O.(var "n" 1 10 + int 2) in
          let r = Ops.range (Sym n) [ 0 ] in
          by_symbolic Ops.O.(r % n) r;
          by_symbolic Ops.O.(r // n) (Ops.int 0));
    ]

let symbolic_casts =
  let i8 = var ~dtype:Int8 "i" (-4) 4 in
  let byte = var ~dtype:Uint8 "b" 0 255 and shift = var ~dtype:Uint8 "s" 0 7 in
  group "casts"
    [
      test "a cast through a type that holds every value is one cast" (fun () ->
          by_symbolic
            (Ops.cast (Ops.cast i8 Int32) Float32)
            (Ops.cast i8 Float32));
      test "a cast of an integer that fits the intermediate type is one cast"
        (fun () ->
          by_symbolic (Ops.cast (Ops.cast x Int8) Int64) (Ops.cast x Int64));
      test "a cast of an integer that overflows the intermediate type stays"
        (fun () ->
          let big = var ~dtype:Int32 "big" 0 1000 in
          let c = Ops.cast (Ops.cast big Int8) Int64 in
          by_symbolic c c);
      test "64-bit arithmetic whose values fit 32 bits computes in 32 bits"
        (fun () ->
          let wide v = Ops.cast v Int64 in
          by_symbolic Ops.O.(wide x + wide y) (wide Ops.O.(x + y));
          by_symbolic Ops.O.(wide x * wide y) (wide Ops.O.(x * y)));
      test "64-bit arithmetic with a weak constant computes in 32 bits"
        (fun () ->
          by_symbolic
            Ops.O.(Ops.cast x Int64 + int 5)
            (Ops.cast Ops.O.(x + int 5) Int64));
      test "64-bit arithmetic that can overflow 32 bits stays" (fun () ->
          let wide = var ~dtype:Int64 "w" 0 2147483647 in
          by_symbolic Ops.O.(wide * int 8) Ops.O.(wide * int 8);
          let bounded = var ~dtype:Int64 "b" (-16) 16 in
          by_symbolic
            Ops.O.(bounded * int 8)
            (Ops.cast Ops.O.(Ops.cast bounded Int32 * int 8) Int64));
      test
        "a cast chain of a bounded integer is one cast, of an unbounded one two"
        (fun () ->
          let bounded = var ~dtype:Int64 "b" (-16) 16 in
          by_symbolic
            (Ops.cast (Ops.cast bounded Int32) Int16)
            (Ops.cast bounded Int16);
          let unbounded =
            Ops.variable ~dtype:Int64 "u" (i 0)
              (`Int (Bigint.of_string "4294967295"))
          in
          let chain = Ops.cast (Ops.cast unbounded Int32) Int64 in
          by_symbolic chain chain);
      test "a cast of x + c to a signed integer is the cast of x plus c"
        (fun () ->
          by_symbolic
            (Ops.cast Ops.O.(a + int 3) Int32)
            Ops.O.(Ops.cast a Int32 + int 3));
      (* An unpacked nibble, as a uint4 load reads its byte: (u lsr k) land 15,
         then widened. *)
      test "a widening cast of an unsigned mask masks the widened operand (D134)"
        (fun () ->
          by_symbolic
            (Ops.cast Ops.O.(byte land int 15) Uint32)
            Ops.O.(Ops.cast byte Uint32 land int 15));
      test
        "a widening cast of an unsigned shift by less than its width shifts \
         the widened operand (D134)" (fun () ->
          by_symbolic
            (Ops.cast Ops.O.(byte lsr shift) Uint32)
            Ops.O.(Ops.cast byte Uint32 lsr Ops.cast shift Uint32));
      test "a widened nibble is unpacked at the wide type (D134)" (fun () ->
          by_symbolic
            (Ops.cast Ops.O.(byte lsr int 4 land int 15) Uint32)
            Ops.O.(Ops.cast byte Uint32 lsr int 4 land int 15));
      test
        "a cast stays of a shift that may reach the width, of a signed \
         operand, or to a narrower type (D134)" (fun () ->
          let k = var ~dtype:Uint8 "k" 0 8 in
          let c = Ops.cast Ops.O.(byte lsr k) Uint32 in
          by_symbolic c c;
          let y = var ~dtype:Int8 "y" (-128) 127 in
          let c = Ops.cast Ops.O.(y land int 15) Int32 in
          by_symbolic c c;
          let w = var ~dtype:Uint32 "w" 0 65535 in
          let c = Ops.cast Ops.O.(w land int 15) Uint8 in
          by_symbolic c c);
      test "a widened mask or shift keeps every value (D134)" (fun () ->
          let points =
            List.concat_map
              (fun b -> List.map (fun k -> [ ("b", i b); ("s", i k) ]) [ 0; 3; 4; 7 ])
              [ 0; 1; 15; 16; 127; 128; 200; 255 ]
          in
          List.iter
            (fun dt ->
              keeps_machine_value ~by:symbolic
                (Ops.cast Ops.O.(byte lsr shift land int 15) dt)
                points;
              keeps_machine_value ~by:symbolic
                (Ops.cast Ops.O.(byte land (Ops.int ~dtype:Uint8 255 lsr shift)) dt)
                points)
            [ Dtype.Uint16; Uint32; Uint64 ]);
    ]

let ordering =
  let store = Ops.store (Ops.index buf [ a ]) f in
  group "ordering"
    [
      test "an after keeps its stores and ranges" (fun () ->
          let after = Ops.after f [ store; r0 ] in
          by_symbolic after after);
      test "an after waits on the sources of any other node" (fun () ->
          by_symbolic
            (Ops.after f [ Ops.O.(g + float 1.); store ])
            (Ops.after f [ store ]));
      test "an after or an end of nothing is its value" (fun () ->
          by_symbolic (raw After [ f ]) f;
          by_symbolic (raw End [ store ]) store);
      test "an end of constant ranges only is its store" (fun () ->
          by_symbolic (Ops.end_ store [ Ops.range (Int 1) [ 1 ] ]) store);
      test "a backedge keeps a constant condition" (fun () ->
          let edge =
            Ops.backedge store ~loop:(Ops.loop 3) ~cond:(Ops.bool false)
          in
          by_symbolic edge edge);
      test "an end drops the ranges that became constants" (fun () ->
          by_symbolic
            (Ops.end_ store [ r0; Ops.range (Int 1) [ 1 ] ])
            (Ops.end_ store [ r0 ]));
    ]

let symbolic_group =
  group "symbolic"
    [
      terms;
      symbolic_selections;
      bounds;
      symbolic_constants;
      comparisons;
      ranges;
      symbolic_casts;
      ordering;
      test "division and remainder are simplified" (fun () ->
          by_symbolic Ops.O.(((a * int 4) + int 3) // int 4) a;
          by_symbolic Ops.O.(((a * int 4) + int 3) % int 4) (Ops.int 3));
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
               Ops.O.(a < int 5 <> bool true)
               Ops.O.(a < int 3)));
      test "every clause of a conjunction holds" (fun () ->
          equal uop (Ops.int 0)
            (Symbolic.uop_given_valid
               Ops.O.((a < int 5) land (b < int 5))
               Ops.O.((a + b) // int 9)));
      test "a clause on a sum tries each term at least 1" (fun () ->
          let valid = Ops.O.(a + b < int 1 <> bool true) in
          let both_zero = Ops.O.((a < int 1) land (b < int 1)) in
          equal uop (Ops.bool false) (Symbolic.uop_given_valid valid both_zero);
          equal uop both_zero
            (Symbolic.uop_given_valid ~try_simplex:false valid both_zero));
      test "a clause on a sum simplifies each lane of a pair on its own"
        (fun () ->
          let valid = Ops.O.(a + b < int 1 <> bool true) in
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
          let valid = Ops.O.((a * int 2) + b < int 1 <> bool true) in
          let e = Ops.O.((a * int 2 < int 1) land (b < int 1)) in
          equal uop e (Symbolic.uop_given_valid valid e));
      test "a clause on a sum leaves what its terms disagree on" (fun () ->
          let valid = Ops.O.(a + b < int 1 <> bool true) in
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
          let r = Ops.range (Int 2) [ 0 ] in
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
      test "a group of one node is that node" (fun () ->
          equal uop a (rewrite (raw Group [ a ])));
      test "a sink splices the sources of its sinks, groups, stacks and noops"
        (fun () ->
          equal uop
            (Ops.sink [ a; b; f; g ])
            (rewrite (Ops.sink [ Ops.sink [ a; b ]; raw Group [ f; g ] ]));
          equal uop (Ops.sink [ a ]) (rewrite (Ops.sink [ raw Noop [ a ] ])));
      test "a group splices the sources of its sinks and groups" (fun () ->
          equal uop
            (raw Group [ a; b; f ])
            (rewrite (raw Group [ raw Group [ a; b ]; f ])));
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

(* Installation *)

let installation =
  group "installation"
    [
      test "simplify is symbolic's rewrite" (fun () ->
          let e = Ops.O.((a * int 4) + b < int 16 lor (a + a < int 3)) in
          equal uop (symbolic e) (Shape.simplify e));
      test "resolve decides from the symbolic rules" (fun () ->
          equal bool true (Shape.resolve Ops.O.(a < int 9));
          equal bool false (Shape.resolve ~default:true Ops.O.(a * int 2 < int 0)));
      test "a constant and a sink of constants and stacks are themselves"
        (fun () ->
          let consts =
            Ops.sink
              [ Ops.int 3; raw Stack []; raw Stack [ Ops.int 1; Ops.int 2 ] ]
          in
          equal uop consts (Shape.simplify consts));
    ]

(* tinygrad's tests that make no simplification of their own: they test the
   nodes, bounds and evaluation that simplification relies on. *)

let other_tests =
  let bounds u = (Ops.vmin u, Ops.vmax u) in
  let bounds_of = pair Dtypes.value Dtypes.value in
  group "tinygrad's other tests"
    [
      test "equal expressions are the same node, and operand order counts"
        (fun () ->
          let i1 = var "idx1" 0 3 and i2 = var "idx2" 0 3 in
          let same u0 u1 = is_true (Ops.equal u0 u1) in
          let differ u0 u1 = is_false (Ops.equal u0 u1) in
          same Ops.O.(i1 * int 4) Ops.O.(i1 * int 4);
          differ Ops.O.(i1 * int 4) Ops.O.(i1 * int 3);
          differ Ops.O.(i1 * int 4) Ops.O.(i1 + int 4);
          differ Ops.O.(i1 * int 4) Ops.O.(i2 * int 4);
          same Ops.O.(i1 + i2) Ops.O.(i1 + i2);
          differ Ops.O.(i1 + i2) Ops.O.(i2 + i1);
          differ Ops.O.(i1 * i2) Ops.O.(i2 * i1));
      test "divide_exact gives up on what does not divide" (fun () ->
          let a = var "a" 1 8 and b = var "b" 1 8 and x = var "x" (-20) 0 in
          let none u d = is_none ~pp:(Testable.pp uop) (Shape.divide_exact u d) in
          none a b;
          none Ops.O.(a + int 2) a;
          none Ops.O.(x * int (-1)) a;
          none Ops.O.(a * int 5) Ops.O.(a * int 10);
          none Ops.O.((a * int 10) - int 1) Ops.O.(a * int 10));
      test "variables lists each variable once, sorted by name" (fun () ->
          let a = var "a" 0 10 and b = var "b" 0 10 and c = var "c" 0 10 in
          let vars u = Shape.variables u in
          equal (list uop) [] (vars (Ops.int 0));
          equal (list uop) [ a ] (vars Ops.O.(a * int 3));
          equal (list uop) [ a; b; c ] (vars Ops.O.(a + b + c));
          equal (list uop) [ a; b; c ] (vars Ops.O.(a + (b * c)));
          equal (list uop) [ a; b ] (vars Ops.O.((a % int 3) + (b // int 5)));
          equal (list uop) [ a; b; c ] (vars Ops.O.(a + b + c - a));
          equal (list uop) [ a ] (vars Ops.O.(a * a));
          equal (list uop) [ a ] (vars Ops.O.((a // int 4) + (a // int 6))));
      test "sym_infer reads bits through bit reinterpretations" (fun () ->
          let a = var ~dtype:Int32 "a" 1 10
          and b = var ~dtype:Int32 "b" (-5) 5 in
          let c =
            Ops.variable ~dtype:Uint32 "c" (i 0)
              (`Int (Bigint.of_string "4294967295"))
          in
          let shifted =
            Ops.O.(Ops.bitcast (Ops.bitcast a Uint32 lsl int 1) Int32 + int 2)
          in
          equal int 6 (Shape.sym_infer (Sym shifted) [ ("a", 2) ]);
          equal int 0xFFFF_FFFF
            (Shape.sym_infer (Sym (Ops.bitcast b Uint32)) [ ("b", -1) ]);
          equal int (-1)
            (Shape.sym_infer (Sym (Ops.bitcast c Int32)) [ ("c", 0xFFFF_FFFF) ]);
          equal int 1069547520
            (Shape.sym_infer
               (Sym (Ops.bitcast (Ops.cast (Ops.float 1.5) Float32) Uint32))
               []));
      test "sym_infer evaluates an expression nested 200 deep" (fun () ->
          let a = var "a" 1 8192 and b = var "b" 0 8191 in
          let step e =
            Ops.O.((Ops.maximum (e * (b + a)) (int (-33554432)) * int (-1)) + a)
          in
          let rec nest n e = if n = 0 then e else nest (n - 1) (step e) in
          equal int 1 (Shape.sym_infer (Sym (nest 200 a)) [ ("a", 1); ("b", 0) ]));
      test "the bounds of an unrolled arange's index" (fun () ->
          let g = var "gidx0" 0 2559 in
          let alu0 = Ops.O.(g * int (-1)) in
          let quotient = Ops.O.((alu0 + int 2559) // int (-4)) in
          equal bounds_of (i 0, i 2559) (bounds g);
          equal bounds_of (i (-2559), i 0) (bounds alu0);
          equal bounds_of (i 0, i 2559) (bounds Ops.O.(alu0 + int 2559));
          equal bounds_of (i (-640), i 0) (bounds quotient);
          equal bounds_of (i 0, i 640) (bounds Ops.O.(quotient * int (-1))));
      test "the bounds of selections between float constants" (fun () ->
          let cond = Ops.O.(var "s" 0 3 < int 2) in
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
      test "log2 of -1 folds to NaN and the reciprocal of 0 to infinity"
        (fun () ->
          let is_nan u =
            match Ops.arg u with
            | Const (`Float v) -> Float.is_nan v
            | _ -> false
          in
          is_true (is_nan (Shape.simplify (Ops.log2 (Ops.float (-1.)))));
          equal uop (Ops.float Float.infinity)
            (Shape.simplify (Ops.reciprocal (Ops.float 0.))));
      test "an index simplified under its gate keeps its value where it holds"
        (fun () ->
          let r0 = Ops.range (Int 30) [ 0 ]
          and r1 = Ops.range (Int 7) [ 1 ]
          and r2 = Ops.range (Int 2) [ 2 ] in
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
          keeps_value ~count:64 ~name:"mobilenet" gated simplified);
    ]

(* Laws *)

let laws =
  let open Expressions in
  let scenario_law name law = prop name scenario (fun s -> law (node s)) in
  group "laws"
    [
      scenario_law
        "sym keeps the value of an integer expression where nothing wraps"
        (fun e -> keeps_value ~name:"sym" e (sym e));
      scenario_law
        "symbolic keeps the value of an integer expression where nothing wraps"
        (fun e -> keeps_value ~name:"symbolic" e (symbolic e));
      prop ~count:1000
        "sym keeps the value of an integer expression at a committed width, \
         wrapping included"
        committed_scenario (fun s ->
          let e = node s in
          keeps_value ~wrapping:true ~name:"sym" e (sym e));
      prop "sym keeps a float expression's value bit for bit at special values"
        float_scenario (fun f ->
          let e = float_node f in
          keeps_float_value ~name:"sym" e (sym e));
      scenario_law "sym is idempotent" (fun e -> Law.idempotent uop sym e);
      prop "a graph of constants folds to what the machine computes" scenario
        (fun s ->
          let e = constants s in
          keeps_value ~name:"constants" e (symbolic e));
      scenario_law "simplify is symbolic's rewrite" (fun e ->
          equal uop (symbolic e) (Shape.simplify e));
      prop "commutative orders the operands of a weak integer sum"
        Gen.(pair weak_scenario weak_scenario)
        (fun (s0, s1) ->
          Law.commutative uop
            (fun u0 u1 -> rewrite Shape.commutative Ops.O.(u0 + u1))
            (node s0, node s1));
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

(* [words f] is the words [f ()] allocates. *)
let words f =
  let before = Gc.minor_words () in
  ignore (Sys.opaque_identity (f ()));
  Gc.minor_words () -. before

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
    ]

let () =
  exit
    (run "Tolk.Symbolic"
       [
         invalid_values;
         remove_invalid;
         symbolic_simple;
         commutative;
         symbolic_group;
         wrapping;
         committed_constants;
         floats;
         conditions;
         sym_group;
         installation;
         other_tests;
         laws;
         cost;
         group "tinygrad" [ Recorded.tests; Recorded.random_expressions ];
       ])
