(* Shape's simplification rules: each rewrite shape.mli states, the values every
   rewrite keeps, and the helpers the rules share. *)

open Windtrap
open Tolk
open Simplification

(* [folds_to input expected] is the claim that symbolic_simple rewrites [input]
   to [expected]. *)
let folds_to input expected =
  equal uop expected (rewrite Shape.symbolic_simple input)

(* [by_symbolic input expected] is the claim that symbolic rewrites [input] to
   [expected], the operands of [expected] in the order [commutative] gives them,
   as the rewrite does. *)
let by_symbolic input expected =
  equal uop (rewrite Shape.commutative expected) (symbolic input)

let stays_under_symbolic u = by_symbolic u u
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
let reduce_range = Ops.range ~axis_type:Weak (Int 4) [ 2 ]
let int32 n = Ops.int ~dtype:Int32 n
let cast_bool dt = Ops.cast cond dt
let op = Common.op
let naming = list (list (pair string uop))

(* A node built as given, where the constructors would fold it: a cast to its
   own type, a group of one node. *)
let raw ?arg op src = Ops.v ?arg ~src op

(* Rows of tables of rewrites: a name, an input and what it rewrites to, or a
   name and an input that stays. *)
let id (name, _, _) = name
let folds (_, input, expected) = folds_to input expected
let simplifies (_, input, expected) = by_symbolic input expected
let stays (_, u) = folds_to u u
let stays_symbolic (_, u) = stays_under_symbolic u

(* [holding p candidates] is the candidates [p] holds of, in order. *)
let holding p candidates = List.filter p candidates

(* A constant, as a weak literal or a committed one: a literal cast to its
   type. *)
let constant =
  let is u =
    Ops.op u = Const || (Ops.op u = Cast && Ops.op (Ops.nth u 0) = Const)
  in
  fun u -> satisfies ~claim:"a constant" uop is u

(* Values *)

(* Integers wrap at a committed width, so a rewrite that computes as unbounded
   integers do applies there only where nothing wraps. *)
let keeps_machine_value u points =
  let after = symbolic u in
  List.iter
    (fun vars ->
      let msg = Format.asprintf "at %a" pp_binding vars in
      equal ~msg Dtypes.const (Interpreter.eval ~vars u)
        (Interpreter.eval ~vars after))
    points

(* Floats keep IEEE's values. [keeps_float_bits u points] evaluates [u] before
   and after symbolic, with the parameters of slots 0, 1 and 3 bound to each
   point's float32 values. *)
let keeps_float_bits u points =
  let after = symbolic u in
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

(* The integer types index arithmetic computes in. *)
let integer_types = Dtype.Weak_int :: Dtype.ints

(* [at_each_type law] is [law] drawn at each integer type, one property each, so
   that each type is reached on every run. *)
let at_each_type law =
  List.map
    (fun dt ->
      prop
        (Format.asprintf "at %a" Dtype.pp dt)
        (Expressions.scenario_of (Gen.constant dt))
        law)
    integer_types

let values =
  let inf = Float.infinity in
  group "values"
    [
      group
        "symbolic keeps the value of an integer expression where nothing wraps"
        (at_each_type (fun s ->
             let e = Expressions.node s in
             let r = symbolic e in
             cover "the rewrite changed the graph" (not (Ops.equal e r));
             let checked = keeps_value ~name:"symbolic" e r in
             cover ~share:0.4 "a binding was checked" (checked > 0)));
      group "symbolic folds a graph of constants to what the machine computes"
        (at_each_type (fun s ->
             let e = Expressions.constants s in
             let r = symbolic e in
             constant r;
             at_least ~msg:"bindings checked" Windtrap.int ~than:1
               (keeps_value ~name:"constants" e r)));
      test "(x // c1) // c2 stays where c1 * c2 wraps" (fun () ->
          let w =
            Ops.variable ~dtype:Int32 "w" (Dtype.min Int32) (Dtype.max Int32)
          in
          stays_under_symbolic Ops.O.(w // int 65536 // int 65536));
      cases "a float selection that computes a maximum stays" ~id:fst
        [
          ("where (f < 0) 0 f", Ops.where Ops.O.(f < float 0.) (Ops.float 0.) f);
          ("where (0 < f) f 0", Ops.where Ops.O.(float 0. < f) f (Ops.float 0.));
        ]
        stays_symbolic;
      cases "a float division by itself or by a factor of its dividend stays"
        ~id:fst
        [ ("f / f", Ops.O.(f / f)); ("(f * g) / g", Ops.O.(f * g / g)) ]
        stays;
      test "a power keeps pow's special values" (fun () ->
          keeps_float_bits (Ops.pow f (Ops.float 2.5)) [ [ -.inf ]; [ -0. ] ];
          keeps_float_bits (Ops.pow f (Ops.float 0.5)) [ [ -0. ]; [ -.inf ] ];
          keeps_float_bits (Ops.pow f (Ops.float (-0.8))) [ [ 1e-40 ] ];
          keeps_float_bits (Ops.pow (Ops.float ~dtype:Float32 inf) f) [ [ 0. ] ]);
    ]

(* invalid_gate *)

let invalid_gate =
  let gated = Shape.valid a cond in
  let by_name = List.sort (fun (n0, _) (n1, _) -> String.compare n0 n1) in
  let namings u = List.map by_name (Ops.Upat.match_ Shape.invalid_gate u) in
  group "invalid_gate"
    [
      test "matches a gated value, naming its sources" (fun () ->
          equal naming
            [ [ ("cond", cond); ("i", Ops.invalid); ("x", a) ] ]
            (namings gated));
      test "does not match a selection of a valid value" (fun () ->
          equal naming [] (namings (Ops.where cond a b)));
    ]

(* pm_remove_invalid *)

let pm_remove_invalid =
  let removes (_, input, expected) =
    equal uop expected (rewrite Shape.pm_remove_invalid input)
  in
  group "pm_remove_invalid"
    [
      cases "replaces each invalid a gate or a stack holds by 0 of its type" ~id
        [
          ("an int32 gate", Shape.valid x cond, Ops.where cond x (int32 0));
          ( "a float32 gate",
            Shape.valid f cond,
            Ops.where cond f (Ops.float ~dtype:Float32 0.) );
          ( "a stack",
            raw Stack [ a; Ops.invalid; b ],
            raw Stack [ a; Ops.int 0; b ] );
        ]
        removes;
    ]

(* fits *)

let fits =
  let candidates =
    [
      i (-129);
      i (-128);
      i 127;
      i 128;
      i 256;
      `Int (Bigint.shift_left Bigint.one 62);
    ]
  in
  group "fits"
    [
      cases "is true unless x is of a signed integer type that does not hold v"
        ~id
        [
          ("int8", Ops.int ~dtype:Int8 0, [ i (-128); i 127 ]);
          ("uint8, unsigned", Ops.int ~dtype:Uint8 0, candidates);
          ("the weak integer", a, candidates);
          ("float32", f, candidates);
        ]
        (fun (_, u, expected) ->
          equal (list Dtypes.value) expected (holding (Shape.fits u) candidates));
    ]

(* symbolic_simple *)

let invalid_values =
  let gated = Shape.valid a cond in
  let mulacc x y z = Ops.alu x Mulacc [ y; z ] in
  let on_range = Ops.O.(reduce_range < int 2) in
  let index = Ops.index buf [ Ops.invalid ] in
  group "invalid values"
    [
      cases "an operation on a gated value moves inside the gate" ~id
        [
          ( "a product's first operand",
            Ops.O.(gated * int 10),
            Shape.valid Ops.O.(a * int 10) cond );
          ( "a comparison's first operand",
            Ops.O.(gated < int 3),
            Shape.valid Ops.O.(a < int 3) cond );
          ( "a product's second operand",
            Ops.O.(int 10 * gated),
            Shape.valid Ops.O.(int 10 * a) cond );
          ( "a multiply-add's first operand",
            mulacc (Shape.valid f cond) g g,
            Shape.valid (mulacc f g g) cond );
          ( "a multiply-add's last operand",
            mulacc f g (Shape.valid g cond),
            Shape.valid (mulacc f g g) cond );
          ( "a cast",
            Ops.cast (Shape.valid x Ops.O.(x < int32 4)) Dtype.Float16,
            Shape.valid (Ops.cast x Dtype.Float16) Ops.O.(x < int32 4) );
        ]
        folds;
      cases "an operation on invalid is invalid" ~id
        [
          ("a product", Ops.O.(Ops.invalid * int 2), Ops.invalid);
          ("a sum with a variable", Ops.O.(Ops.invalid + a), Ops.invalid);
          ("a cast", Ops.cast Ops.invalid Dtype.Int32, Ops.invalid);
          ("a multiply-add", raw Mulacc [ f; Ops.invalid; g ], Ops.invalid);
          ( "a selection by invalid",
            raw Where [ Ops.invalid; a; b ],
            Ops.invalid );
          ( "a selection between invalid and invalid",
            Ops.where cond Ops.invalid Ops.invalid,
            Ops.invalid );
          ( "a stack of invalid, expanded to its width",
            raw Stack [ Ops.invalid; Ops.invalid ],
            Shape.expand Ops.invalid [ Int 2 ] );
        ]
        folds;
      test "a comparison of invalid with a node that is not constant is left"
        (fun () ->
          let lt = Ops.O.(Ops.invalid < a) in
          equal uop lt (symbolic lt));
      test "a comparison of invalid with a constant folds to invalid" (fun () ->
          equal Dtypes.const `Invalid
            (Ops.value (symbolic Ops.O.(Ops.invalid < int 3))));
      cases "a gate in a selection moves outermost" ~id
        [
          ( "a gate on the condition",
            Ops.where (Shape.valid other cond) a b,
            Shape.valid (Ops.where other a b) cond );
          ( "invalid in the true branch",
            Ops.where cond Ops.invalid a,
            Shape.valid a (Ops.logical_not cond) );
          ( "a gate in the true branch",
            Ops.where other (Shape.valid a cond) b,
            Shape.valid (Ops.where other a b)
              Ops.O.(Ops.logical_not other lor cond) );
          ( "a gate in the false branch",
            Ops.where other b (Shape.valid a cond),
            Shape.valid (Ops.where other b a) Ops.O.(other lor cond) );
        ]
        folds;
      test
        "a gate on a reduction's operand that does not depend on its ranges \
         moves out of the reduction" (fun () ->
          folds_to
            (Ops.reduce (Shape.valid a cond) Add [ reduce_range ])
            (Shape.valid (Ops.reduce a Add [ reduce_range ]) cond));
      cases
        "a gate on a reduction's operand that depends on its ranges selects \
         the reduction's identity where it fails"
        ~id
        [
          ( "beside a clause that does not",
            Ops.reduce
              (Shape.valid a Ops.O.(cond land on_range))
              Add [ reduce_range ],
            Shape.valid
              (Ops.reduce
                 (Ops.where on_range a (Ops.int 0))
                 Add [ reduce_range ])
              cond );
          ( "alone",
            Ops.reduce (Shape.valid a on_range) Max [ reduce_range ],
            Ops.reduce
              (Ops.where on_range a
                 (Shape.const_like a (Ops.identity_element Max (Ops.dtype a))))
              Max [ reduce_range ] );
        ]
        folds;
      test "a store to an invalid index does nothing after its sources"
        (fun () ->
          folds_to (Ops.store index f) (Ops.after (raw Noop []) [ index; f ]));
      cases "a load from an invalid index is its alternative value, or 0" ~id
        [
          ("no alternative", Ops.load index [], Ops.float ~dtype:Float32 0.);
          ("an alternative", Ops.load index [ g ], g);
        ]
        folds;
    ]

let identities =
  let p = var "p" 1 8 in
  let n = Shape.param 5 Dtype.Int32 in
  group "identities"
    [
      cases "x + 0, x lxor 0, x lor 0, x lsl 0, x lsr 0, x * 1 and x // 1 are x"
        ~id
        [
          ("a + 0", Ops.O.(a + int 0), a);
          ("x lxor 0", Ops.O.(x lxor int 0), x);
          ("x lor 0", Ops.O.(x lor int 0), x);
          ("x lsl 0", Ops.O.(x lsl int 0), x);
          ("x lsr 0", Ops.O.(x lsr int 0), x);
          ("a * 1", Ops.O.(a * int 1), a);
          ("1 * a", Ops.O.(int 1 * a), a);
          ("a // 1", Ops.O.(a // int 1), a);
        ]
        folds;
      test "a float x + -0. is x" (fun () -> folds_to Ops.O.(f + float (-0.)) f);
      test "a float x + 0. stays, since -0. + 0. is 0." (fun () ->
          folds_to Ops.O.(f + float 0.) Ops.O.(f + float 0.));
      cases "x // -1 is -x for a signed or weak x" ~id
        [
          ("weak", Ops.O.(a // int (-1)), Ops.O.(~-a));
          ("int32", Ops.O.(n // int (-1)), Ops.O.(n * int (-1)));
        ]
        folds;
      test "x // x is 1 where x's bounds hold no 0" (fun () ->
          folds_to Ops.O.(p // p) (Ops.int 1));
      test "x // x stays where x's bounds hold 0" (fun () ->
          folds_to Ops.O.(a // a) Ops.O.(a // a));
      test "(x lxor y) lxor y is x for a weak y" (fun () ->
          folds_to Ops.O.(a lxor b lxor b) a);
      test "(x lxor y) lxor y stays for a committed y" (fun () ->
          folds_to Ops.O.(x lxor y lxor y) Ops.O.(x lxor y lxor y));
      test "(x % y) % y is x % y" (fun () ->
          folds_to Ops.O.(a % b % b) Ops.O.(a % b));
      cases "a boolean x land c and x lor c with c constant are x or c" ~id
        [
          ("x land true", Ops.O.(cond land bool true), cond);
          ("x land false", Ops.O.(cond land bool false), Ops.bool false);
          ("x lor true", Ops.O.(cond lor bool true), Ops.bool true);
          ("x lor false", Ops.O.(cond lor bool false), cond);
        ]
        folds;
      cases
        "x = true, x <> false, a double negation, where x true false and an \
         idempotent operation of x with itself are x"
        ~id
        [
          ("x = true", Ops.eq cond (Ops.bool true), cond);
          ("x <> false", Ops.O.(cond <> bool false), cond);
          ("not (not x)", Ops.logical_not (Ops.logical_not cond), cond);
          ( "where x true false",
            Ops.where cond (Ops.bool true) (Ops.bool false),
            cond );
          ("max a a", Ops.maximum a a, a);
          ("x land x", Ops.O.(cond land cond), cond);
          ("x lor x", Ops.O.(cond lor cond), cond);
        ]
        folds;
      test "where x false true is the negation of x" (fun () ->
          folds_to
            (Ops.where cond (Ops.bool false) (Ops.bool true))
            (Ops.logical_not cond));
      cases
        "a boolean cast to an integer and equal to 0 or 1 is its negation or \
         itself, to any other integer false, and unequal the negation of that"
        ~id
        [
          ("= 0", Ops.eq (cast_bool Int32) (Ops.int 0), Ops.logical_not cond);
          ("weak, = 1", Ops.eq (cast_bool Weak_int) (Ops.int 1), cond);
          ("= 2", Ops.eq (cast_bool Int32) (Ops.int 2), Ops.bool false);
          ("<> 0", Ops.O.(cast_bool Int32 <> int 0), cond);
          ("weak, <> 0", Ops.O.(cast_bool Weak_int <> int 0), cond);
          ("<> 1", Ops.O.(cast_bool Int32 <> int 1), Ops.logical_not cond);
          ("<> 2", Ops.O.(cast_bool Int32 <> int 2), Ops.bool true);
          ("<> -1", Ops.O.(cast_bool Int32 <> int (-1)), Ops.bool true);
        ]
        folds;
      cases "the truncation of an integer or a boolean is itself" ~id
        [
          ("int32", Ops.trunc x, x);
          ("weak", Ops.trunc a, a);
          ("boolean", Ops.trunc cond, cond);
        ]
        folds;
      test "the truncation of a float stays" (fun () ->
          folds_to (Ops.trunc f) (Ops.trunc f));
    ]

let recombination =
  let g = var "g" 0 124 in
  group "recombination"
    [
      cases "a remainder and a quotient by d recombine into their base" ~id
        [
          ("x % 4 + (x // 4) * 4", Ops.O.((g % int 4) + (g // int 4 * int 4)), g);
          ("(x // 4) * 4 + x % 4", Ops.O.((g // int 4 * int 4) + (g % int 4)), g);
          ( "scaled by 2",
            Ops.O.((g % int 4 * int 2) + (g // int 4 * int 8)),
            Ops.O.(g * int 2) );
          ( "beside another term",
            Ops.O.(a + (g % int 4) + (g // int 4 * int 4)),
            Ops.O.(g + a) );
        ]
        folds;
      test
        "a remainder of the quotient by k recombines into a remainder by d * k"
        (fun () ->
          folds_to
            Ops.O.((g % int 4) + (g // int 4 % int 3 * int 4))
            Ops.O.(g % int 12));
      cases "terms of no common base, divisor or sign do not recombine" ~id:fst
        [
          ( "a quotient by another divisor",
            Ops.O.((g % int 4) + (g // int 3 * int 4)) );
          ("a quotient by a variable", Ops.O.((g % int 4) + (g // b * int 4)));
          ( "a quotient of another base",
            Ops.O.((g % int 4) + (a // int 4 * int 4)) );
          ( "a quotient merged with another divisor",
            Ops.O.((g // int 2 % int 4) + (g // int 16 * int 4)) );
          ( "a quotient of a shifted base",
            Ops.O.((g % int 4) + ((g + int 1) // int 4 * int 4)) );
          ( "a remainder of the quotient by a negative k",
            Ops.O.((g % int 4) + (g // int 4 % int (-3) * int 4)) );
          ("a committed sum", Ops.O.((x % int32 4) + (x // int32 4 * int32 4)));
        ]
        stays;
    ]

let zeros =
  let u = var ~dtype:Uint32 "u" 0 255 in
  group "zeros"
    [
      test "x < x is false" (fun () -> folds_to Ops.O.(a < a) (Ops.bool false));
      cases "x = x is true, and x <> x false, for integers and booleans" ~id
        [
          ("weak a = a", Ops.eq a a, Ops.bool true);
          ("int32 x = x", Ops.eq x x, Ops.bool true);
          ("weak a <> a", Ops.O.(a <> a), Ops.bool false);
          ("boolean <> itself", Ops.O.(cond <> cond), Ops.bool false);
        ]
        folds;
      test "x <> x stays for floats, which may be NaN" (fun () ->
          folds_to Ops.O.(f <> f) Ops.O.(f <> f));
      cases "x % x, x lxor x, x land 0 and an integer or boolean x * 0 are 0"
        ~id
        [
          ("a % a", Ops.O.(a % a), Ops.int 0);
          ("x lxor x", Ops.O.(x lxor x), int32 0);
          ("x land 0", Ops.O.(x land int 0), int32 0);
          ("a * 0", Ops.O.(a * int 0), Ops.int 0);
          ("boolean * false", Ops.O.(cond * bool false), Ops.bool false);
        ]
        folds;
      test "a float x * 0. stays" (fun () ->
          folds_to Ops.O.(f * float 0.) Ops.O.(f * float 0.));
      test "a mask that clears only bits a right shift drops is removed"
        (fun () ->
          folds_to Ops.O.((u land int (-4)) lsr int 2) Ops.O.(u lsr int 2));
      cases "a mask that clears a bit the result keeps stays" ~id:fst
        [
          ("shifted by 2", Ops.O.((u land int (-8)) lsr int 2));
          ("divided by 4", Ops.O.(u land int (-8) // int 4));
          ("divided by 3", Ops.O.(u land int (-4) // int 3));
        ]
        stays;
    ]

(* An arithmetic operation on two constants of one type. *)
type operation = { dt : Dtype.t; o : Op.t; v0 : Dtype.value; v1 : Dtype.value }

let pp_operation ppf c =
  Format.fprintf ppf "%a %a %a at %a" Dtype.pp_const
    (c.v0 :> Dtype.const)
    Op.pp c.o Dtype.pp_const
    (c.v1 :> Dtype.const)
    Dtype.pp c.dt

let int_ops =
  Op.
    [
      Add;
      Sub;
      Mul;
      Div;
      Mod;
      Truncdiv;
      Truncmod;
      Max;
      Cmplt;
      Cmpne;
      Cmpeq;
      Xor;
      Or;
      And;
      Shl;
      Shr;
    ]

let float_ops = Op.[ Add; Sub; Mul; Fdiv; Max; Cmplt; Cmpne; Cmpeq ]

(* Operands at a type's bounds and their neighbours. *)
let gen_operation =
  let open Gen in
  with_pp pp_operation
    (let* dt =
       of_list ~pp:Dtype.pp
         Dtype.
           [
             Int8;
             Uint8;
             Int16;
             Uint16;
             Int32;
             Uint32;
             Int64;
             Uint64;
             Float16;
             Bfloat16;
             Float32;
             Float64;
           ]
     in
     let+ o =
       of_list ~pp:Op.pp (if Dtype.is_float dt then float_ops else int_ops)
     and+ v0 = Dtypes.value_of dt
     and+ v1 = Dtypes.value_of dt in
     { dt; o; v0; v1 })

let folds_to_its_value c =
  let operand v = Ops.const ~dtype:c.dt (v :> Dtype.const) in
  let u = Ops.alu (operand c.v0) c.o [ operand c.v1 ] in
  let r = rewrite Shape.symbolic_simple u in
  let args = [ (c.v0 :> Dtype.const); (c.v1 :> Dtype.const) ] in
  match Ops.exec_alu c.o (Ops.dtype u) args with
  | v ->
      cover "a value" true;
      constant r;
      equal same_float v (Ops.value r)
  | exception Invalid_argument _ ->
      cover "no value" true;
      equal uop u r

let operation_examples =
  [
    (* a shift by a negative count, which has no value *)
    { dt = Int32; o = Shl; v0 = i 3; v1 = i (-1) };
    { dt = Int32; o = Shr; v0 = i 3; v1 = i (-2) };
    (* division and remainder by 0: 0, and the dividend (op.mli) *)
    { dt = Int32; o = Div; v0 = i 7; v1 = i 0 };
    { dt = Int32; o = Mod; v0 = i 7; v1 = i 0 };
    { dt = Int32; o = Truncdiv; v0 = i 7; v1 = i 0 };
    { dt = Int32; o = Truncmod; v0 = i 7; v1 = i 0 };
    (* rounding toward 0 *)
    { dt = Int32; o = Truncdiv; v0 = i (-50); v1 = i 6 };
    { dt = Int32; o = Truncmod; v0 = i 5; v1 = i (-3) };
    (* NaN is unequal to itself and less than nothing *)
    { dt = Float32; o = Cmpne; v0 = `Float Float.nan; v1 = `Float Float.nan };
    { dt = Float32; o = Cmplt; v0 = `Float Float.nan; v1 = `Float Float.nan };
    (* a committed integer computes at its width *)
    { dt = Uint32; o = Add; v0 = i 0xFFFF_FFFF; v1 = i 1 };
    { dt = Float32; o = Add; v0 = `Float 1.5; v1 = `Float 2. };
  ]

let constants =
  let big = Bigint.shift_left Bigint.one 40 in
  let stack l = raw Stack l in
  let u16 x = Ops.const ~dtype:Uint16 (`Int (Bigint.of_int x)) in
  group "constants"
    [
      prop
        "an operation on constants of one type is its value, and stays where \
         it has none"
        ~examples:operation_examples gen_operation folds_to_its_value;
      test "a threefry of constants stays" (fun () ->
          let tf =
            Ops.alu (Ops.int ~dtype:Uint32 0) Threefry
              [ Ops.int ~dtype:Uint32 1 ]
          in
          folds_to tf tf);
      cases "an operation on weak constants keeps their mathematical value" ~id
        [
          ( "2^40 + 2^40",
            Ops.O.(Ops.const (`Int big) + Ops.const (`Int big)),
            Ops.const (`Int (Bigint.shift_left Bigint.one 41)) );
          ("1.25 + 2.5", Ops.O.(float 1.25 + float 2.5), Ops.float 3.75);
          ("sqrt 4.", Ops.sqrt (Ops.float 4.), Ops.float 2.);
          ("exp2 3.", Ops.exp2 (Ops.float 3.), Ops.float 8.);
        ]
        folds;
      (* A committed constant is read at its width, by an operation that folds
         and by a cast of it, where tinygrad reads it unwrapped. *)
      cases "an operation reads a committed constant at its width" ~id
        [
          ( "(2^32 - 1 + 1) >> 1 at uint32",
            Ops.O.(
              (Ops.int ~dtype:Uint32 0xFFFF_FFFF + Ops.int ~dtype:Uint32 1)
              lsr Ops.int ~dtype:Uint32 1),
            Ops.int ~dtype:Uint32 0 );
          ( "2^31 at int32, cast to int64",
            Ops.cast
              (Ops.cconst Int32 (`Int (Bigint.shift_left Bigint.one 31)))
              Int64,
            Ops.int ~dtype:Int64 (-2147483648) );
        ]
        folds;
      test "a weak operand of a committed operation is read at its width"
        (fun () ->
          (* -1 at uint8 is 255, and 255 % 3 is 0 *)
          let e = var ~dtype:Uint8 "e" 3 3 in
          equal Dtypes.const (i 0)
            (Interpreter.eval
               ~vars:[ ("e", i 3) ]
               (symbolic Ops.O.(Ops.cast (int (-1)) Uint8 % e))));
      test "weak constants commit to the type of a committed peer" (fun () ->
          folds_to Ops.O.(int32 3 + int 4) (int32 7));
      cases "an operation on stacks of constants folds lane by lane" ~id
        [
          ( "a selection",
            Ops.where
              (stack [ Ops.bool true; Ops.bool false ])
              (stack [ Ops.int 3; Ops.int 4 ])
              (stack [ Ops.int 30; Ops.int 40 ]),
            stack [ Ops.int 3; Ops.int 40 ] );
          ( "a comparison",
            Ops.O.(stack [ int 1; int 4 ] < stack [ int 2; int 3 ]),
            stack [ Ops.bool true; Ops.bool false ] );
          ( "a negation",
            Ops.O.(~-(stack [ int 1; int (-2) ])),
            stack [ Ops.int (-1); Ops.int 2 ] );
          ( "a sum of int32",
            Ops.O.(stack [ int32 1; int32 2 ] + stack [ int32 3; int32 4 ]),
            stack [ int32 4; int32 6 ] );
          ( "a shift, each lane's bare count read whole",
            Ops.alu
              (stack [ u16 3; u16 3 ])
              Shl
              [ stack [ Ops.int 65536; Ops.int 1 ] ],
            stack [ u16 0; u16 6 ] );
        ]
        folds;
      cases
        "a cast of an infinity or a NaN to an integer, which has no value, \
         stays"
        ~id:(Printf.sprintf "%h")
        [ Float.infinity; Float.neg_infinity; Float.nan ] (fun v ->
          let c = Ops.cast (Ops.float ~dtype:Float32 v) Int32 in
          folds_to c c);
      cases "a cast of a constant is the constant of the cast's type" ~id
        [
          ( "int to float32",
            Ops.cast (Ops.int 3) Float32,
            Ops.float ~dtype:Float32 3. );
          ( "through int64",
            Ops.cast (Ops.cast (Ops.int 3) Int64) Float32,
            Ops.float ~dtype:Float32 3. );
        ]
        folds;
      test "0 / 0 is NaN" (fun () ->
          let zero = Ops.float 0. in
          let r =
            rewrite ~order:before Shape.symbolic_simple Ops.O.(zero / zero)
          in
          equal same_float (`Float Float.nan) (Ops.value r));
      cases
        "a folded selection keeps a signalling NaN's bits, and a negation \
         flips only its sign"
        ~id:(Format.asprintf "%a" Dtype.pp)
        Dtype.[ Float16; Bfloat16; Float32; Float64 ]
        (fun dt ->
          (* A signalling NaN whose leading payload bit every format holds,
             moved and never computed, so its bits are portable. *)
          let snan = Int64.float_of_bits 0x7FF4_0000_0000_0000L in
          let c = Ops.float ~dtype:dt snan in
          let bits u =
            match Ops.value (rewrite Shape.symbolic_simple u) with
            | `Float v -> Int64.bits_of_float v
            | _ -> failf "%a is not a float constant" (Testable.pp uop) u
          in
          let int64 =
            Testable.make
              ~pp:(fun ppf b -> Format.fprintf ppf "0x%Lx" b)
              ~equal:Int64.equal
          in
          let held = bits c in
          equal ~msg:"the constant is signalling" int64 0L
            (Int64.logand held 0x8_0000_0000_0000L);
          let one = Ops.float ~dtype:dt 1. in
          equal ~msg:"where" int64 held (bits (Ops.where (Ops.bool true) c one));
          equal ~msg:"where, other branch" int64 held
            (bits (Ops.where (Ops.bool false) one c));
          equal ~msg:"neg" int64
            (Int64.logxor Int64.min_int held)
            (bits (raw Neg [ c ])));
    ]

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

(* A shift's count and a power's exponent are numbers, read as they are. *)
let numbers =
  group "counts and exponents"
    [
      prop "a shift of constants folds by its count, of any size"
        gen_constant_shift (fun (dt, o, x, k) ->
          let u =
            Ops.alu (Ops.const ~dtype:dt (`Int x)) o [ Ops.const (`Int k) ]
          in
          cover "a count at or past the width"
            (Bigint.geq k (Bigint.of_int (Dtype.bitsize dt)));
          cover "a count an OCaml int does not hold" (not (Bigint.fits_int k));
          let folded = symbolic u in
          constant folded;
          equal
            ~msg:(Format.asprintf "%a" Ops.pp u)
            Dtypes.const
            (shift_reference dt o x k :> Dtype.const)
            (Ops.value folded));
      test "a masked right shift past the width keeps its mask's sign"
        (fun () ->
          (* The rule that drops a mask reads the bits below the count, and past
             the width that is every bit, the sign included. *)
          let x = var ~dtype:Int32 "x" (-5) 5 in
          let high = Ops.O.(x land int32 0x7FFFFFFF) in
          let keeps k =
            let e = Ops.O.(high lsr Ops.const (`Int k)) in
            let at = [ ("x", i (-1)) ] in
            equal Dtypes.const
              (Interpreter.eval ~vars:at e)
              (Interpreter.eval ~vars:at (symbolic e))
          in
          keeps (Bigint.of_int 32);
          keeps (Bigint.shift_left Bigint.one 63));
      test "a bare constant shifted by a committed count commits at its type"
        (fun () ->
          let by_u32 o x k =
            Ops.value
              (symbolic (Ops.alu (Ops.int x) o [ Ops.int ~dtype:Uint32 k ]))
          in
          equal Dtypes.const (i 0x7FFFFFFF) (by_u32 Op.Shr (-1) 1);
          equal Dtypes.const (i 0) (by_u32 Op.Shl 1 40));
      test "a left shift of constants at no width by a huge count stays"
        (fun () ->
          let huge = Bigint.shift_left Bigint.one 80 in
          let e = Ops.alu (Ops.int 3) Shl [ Ops.const (`Int huge) ] in
          equal uop e (symbolic e));
      test "a power no width truncates stays a power past what it can hold"
        (fun () ->
          let e = Bigint.(succ (shift_left one 23)) in
          let u = Ops.alu (Ops.int 2) Op.Pow [ Ops.const (`Int e) ] in
          equal op Op.Pow (Ops.op (symbolic u)));
      cases
        "a power no width truncates folds for a base of 0, 1 or -1 at any size"
        ~id:string_of_int [ 0; 1; -1 ] (fun b ->
          let e = Bigint.(succ (shift_left one 70)) in
          let u = Ops.alu (Ops.int b) Op.Pow [ Ops.const (`Int e) ] in
          equal Dtypes.const (i b) (Ops.value (symbolic u)));
    ]

let booleans =
  group "booleans"
    [
      cases "a boolean * is land, and a boolean + or maximum is lor" ~id
        [
          ("*", Ops.O.(cond * other), Ops.O.(cond land other));
          ("+", Ops.O.(cond + other), Ops.O.(cond lor other));
          ("maximum", Ops.maximum cond other, Ops.O.(cond lor other));
        ]
        folds;
    ]

(* [round_trips mid word] is the claim that the bitcast of [word] to the float
   [mid] and back folds to [word]: a folded bitcast reads a float constant's
   bits as its type stores them, where tinygrad converts it first and a NaN
   loses its bits. *)
let round_trips mid word =
  let w = if Dtype.itemsize mid = 1 then Dtype.Uint8 else Uint16 in
  let u = Ops.bitcast (Ops.bitcast (Ops.int ~dtype:w word) mid) w in
  equal
    ~msg:(Printf.sprintf "0x%x" word)
    Dtypes.const (Interpreter.eval u)
    (Interpreter.eval (Shape.simplify u))

(* Every word of an 8-bit float; of a 16-bit one, the words whose exponent is
   all zeros (zeros and subnormals) or all ones (infinities and NaNs, signalling
   ones included). *)
let special_words mid =
  if Dtype.itemsize mid = 1 then List.init 256 Fun.id
  else
    let exponent = if mid = Dtype.Float16 then 0x7C00 else 0x7F80 in
    List.filter
      (fun w -> w land exponent = 0 || w land exponent = exponent)
      (List.init 0x10000 Fun.id)

let casts =
  let i8 = var ~dtype:Int8 "i" (-4) 4 in
  group "casts"
    [
      cases "a cast or bitcast to its operand's type is its operand" ~id
        [
          ("cast", raw ~arg:(Dtype Int32) Cast [ x ], x);
          ("bitcast", raw ~arg:(Dtype Int32) Bitcast [ x ], x);
        ]
        folds;
      cases
        "a bitcast of a constant to a type of its width is the constant with \
         the same bits"
        ~id
        [
          ( "-1 to uint32",
            Ops.bitcast (int32 (-1)) Uint32,
            Ops.int ~dtype:Uint32 0xFFFF_FFFF );
          ( "1.0 to int32",
            Ops.bitcast (Ops.float ~dtype:Float32 1.) Int32,
            int32 0x3f80_0000 );
        ]
        folds;
      test "a bitcast of a constant to another width stays" (fun () ->
          let widened = Ops.bitcast (int32 1) Int64 in
          folds_to widened widened);
      (* A boolean is true where its byte is not zero, and a bitcast back reads
         the byte, which no boolean constant holds. *)
      test "a bitcast to a boolean of a byte neither 0 nor 1 stays" (fun () ->
          let byte = Ops.bitcast (Ops.int ~dtype:Uint8 2) Bool in
          folds_to byte byte);
      cases
        "a bitcast round trip of a word through a float folds to the word: \
         every 8-bit word, and each 16-bit word of an exponent all zeros or \
         all ones"
        ~id:(Format.asprintf "%a" Dtype.pp)
        Dtype.[ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz; Float16; Bfloat16 ]
        (fun mid -> List.iter (round_trips mid) (special_words mid));
      prop
        "a bitcast round trip of any 16-bit word through a float folds to the \
         word"
        Gen.(
          pair
            (of_list ~pp:Dtype.pp Dtype.[ Float16; Bfloat16 ])
            (int_range 0 0xFFFF))
        (fun (mid, word) -> round_trips mid word);
      test
        "a cast through a type that holds every value of the result's type, \
         back to that type, is its operand" (fun () ->
          folds_to (Ops.cast (Ops.cast i8 Int32) Int8) i8);
      test "a cast through a narrower type back stays" (fun () ->
          let there_and_back = Ops.cast (Ops.cast x Int8) Int32 in
          folds_to there_and_back there_and_back);
      cases "two bitcasts are one" ~id
        [
          ( "to another type",
            Ops.bitcast (Ops.bitcast x Float32) Uint32,
            Ops.bitcast x Uint32 );
          ("back", Ops.bitcast (Ops.bitcast x Float32) Int32, x);
        ]
        folds;
      test "a cast to a boolean is x <> 0" (fun () ->
          folds_to (Ops.cast a Bool) Ops.O.(a <> int 0));
    ]

(* A power leaves committed constants, [1.0 * x], that symbolic_simple alone
   does not uncast: its claims hold under symbolic. *)
let powers =
  let special v r =
    Ops.where (Ops.eq f (Ops.float v)) (Ops.float (Float.abs v)) r
  in
  let p = Shape.param 5 Dtype.Int32 in
  let n = Ops.cast p Weak_float in
  group "powers"
    [
      cases
        "x ** c for a float x and c an integer or a half-integer, 0 or at \
         least 1 in magnitude, is a product of powers"
        ~id
        [
          ("x ** 0", Ops.pow f (Ops.float 0.), Ops.float ~dtype:Float32 1.);
          ("x ** 1", Ops.pow f (Ops.float 1.), f);
          ("x ** 2", Ops.pow f (Ops.float 2.), Ops.O.(f * f));
          ("x ** -1", Ops.pow f (Ops.float (-1.)), Ops.reciprocal f);
          ( "x ** 0.5, +0. at -0. and +inf at -inf",
            Ops.pow f (Ops.float 0.5),
            special Float.neg_infinity (special 0. (Ops.sqrt f)) );
          ( "3. ** 2.",
            Ops.pow (Ops.float ~dtype:Float32 3.) (Ops.float 2.),
            Ops.float ~dtype:Float32 9. );
        ]
        simplifies;
      cases "x ** c stays for c below 1 in magnitude, or not a half-integer"
        ~id:fst
        [
          ("x ** -0.8", Ops.pow f (Ops.float (-0.8)));
          ("x ** 1.3", Ops.pow f (Ops.float 1.3));
        ]
        stays_symbolic;
      test "x ** c for a large integer c takes no square root" (fun () ->
          let p = symbolic (Ops.pow f (Ops.float (Float.ldexp 1. 60))) in
          equal (list op) []
            (List.filter
               (fun o -> o = Op.Sqrt)
               (List.map Ops.op (Ops.toposort ~calls:Enter p))));
      test "an integer x ** c stays a power for a negative c" (fun () ->
          let p = symbolic (Ops.alu x Op.Pow [ Ops.int ~dtype:Int32 (-2) ]) in
          equal op Op.Pow (Ops.op p);
          equal uop x (Ops.nth p 0));
      cases
        "1 ** x is 1 of the power's type, and c ** x is exp2 (x * log2 c) for \
         a positive finite c and a float x"
        ~id
        [
          ( "1 ** x",
            Ops.pow (Ops.float ~dtype:Float32 1.) f,
            Ops.float ~dtype:Float32 1. );
          ("2 ** x", Ops.pow (Ops.float ~dtype:Float32 2.) f, Ops.exp2 f);
        ]
        simplifies;
      cases
        "c ** x for an integer x reads x as a float, and log2 c correctly \
         rounded"
        ~id
        [
          (* log2 3 correctly rounded *)
          ( "3 ** x",
            Ops.pow (Ops.float 3.) p,
            Ops.exp2 Ops.O.(n * float 1.584962500721156) );
          (* log2 0.86 correctly rounded, which glibc's log2 gives one unit
             away, as -0.2175914350726268. *)
          ( "0.86 ** x",
            Ops.pow (Ops.float 0.86) p,
            Ops.exp2 Ops.O.(n * float (-0.21759143507262677)) );
        ]
        folds;
      test "c ** x stays a power for an integer x" (fun () ->
          let p = symbolic (Ops.alu (Ops.int ~dtype:Int32 3) Op.Pow [ x ]) in
          equal op Op.Pow (Ops.op p);
          equal uop x (Ops.nth p 1));
      test "c ** x stays for c at most 0" (fun () ->
          by_symbolic
            (Ops.pow (Ops.float ~dtype:Float32 (-2.)) f)
            (Ops.pow (Ops.float (-2.)) f));
      cases "x ** infinity and x ** NaN stay" ~id:fst
        [
          ("x ** inf", Ops.pow f (Ops.float Float.infinity));
          ("x ** nan", Ops.pow f (Ops.float Float.nan));
        ]
        stays_symbolic;
      test
        "a 64-bit integer packed from two 32-bit halves unpacks to the half \
         read" (fun () ->
          let hi = var ~dtype:Uint32 "hi" 0 7
          and lo = var ~dtype:Uint32 "lo" 0 7 in
          let wide dt = Ops.cast dt Uint64 in
          let packed = Ops.O.((wide hi lsl int 32) lor wide lo) in
          folds_to (Ops.cast packed Uint32) lo;
          folds_to Ops.O.(packed lsr int 32) (wide hi);
          (* A 64-bit high half keeps its low 32 bits only. *)
          let hi64 = var ~dtype:Uint64 "hi" 0 0x1_0000_0000 in
          folds_to
            Ops.O.(((hi64 lsl int 32) lor wide lo) lsr int 32)
            (wide (Ops.cast hi64 Uint32)));
    ]

let selections =
  group "selections"
    [
      test "a selection between equal values is that value" (fun () ->
          folds_to (Ops.where cond a a) a);
      cases "a selection by a constant is the branch it picks" ~id
        [
          ("true", Ops.where (Ops.bool true) a b, a);
          ("false", Ops.where (Ops.bool false) a b, b);
          ( "a weak branch of a weak selection",
            raw Where
              [ Ops.bool true; Ops.int 1; var ~dtype:Weak_float "v" 0 3 ],
            Ops.int 1 );
        ]
        folds;
      cases "a selection by a constant keeps the selection's type" ~id
        [
          ( "float16",
            Ops.where (Ops.bool true) (Ops.float 0.)
              (var ~dtype:Float16 "h" 0 3),
            Ops.float ~dtype:Float16 0. );
          ("int32", Ops.where (Ops.bool true) (Ops.int 0) x, int32 0);
        ]
        folds;
      test "a selection by a padded constant, which is not one value, stays"
        (fun () ->
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
    ]

let movements =
  let p = Shape.param ~shape:[ Int 16 ] 0 Dtype.Float32 in
  let shrink u lo n = Shape.mop u (Shrink [ (Int lo, Int n) ]) in
  group "movements"
    [
      test "cleans up movements as mop_cleanup does" (fun () ->
          folds_to (shrink (shrink p 2 8) 3 4) (shrink p 5 4));
    ]

let symbolic_simple =
  group "symbolic_simple"
    [
      invalid_values;
      identities;
      recombination;
      zeros;
      constants;
      numbers;
      booleans;
      casts;
      powers;
      selections;
      movements;
    ]

(* commutative *)

let commutative =
  let canonical u = rewrite Shape.commutative u in
  group "commutative"
    [
      prop "two sums of the same weak integer terms are the same node"
        Gen.(pair Expressions.weak_scenario Expressions.weak_scenario)
        (fun (s0, s1) ->
          Law.commutative uop
            (fun u0 u1 -> canonical Ops.O.(u0 + u1))
            (Expressions.node s0, Expressions.node s1));
      test "a committed sum keeps its order" (fun () ->
          equal uop Ops.O.(y + x) (canonical Ops.O.(y + x)));
      test "a non-commutative operation keeps its order" (fun () ->
          equal uop Ops.O.(b < a) (canonical Ops.O.(b < a)));
    ]

(* symbolic *)

let terms =
  let n = Shape.param 5 Dtype.Int32 in
  group "terms"
    [
      test "x lor not x is true" (fun () ->
          by_symbolic Ops.O.(cond lor Ops.logical_not cond) (Ops.bool true));
      cases "like terms combine, also as the last two terms of a sum" ~id
        [
          ("2a + 3a", Ops.O.((a * int 2) + (a * int 3)), Ops.O.(a * int 5));
          ("a + 3a", Ops.O.(a + (a * int 3)), Ops.O.(a * int 4));
          ("a + a", Ops.O.(a + a), Ops.O.(a * int 2));
          ("n + n, int32", Ops.O.(n + n), Ops.O.(n * int 2));
          ( "b + 2a + 3a",
            Ops.O.(b + (a * int 2) + (a * int 3)),
            Ops.O.(b + (a * int 5)) );
          ("b + a + 3a", Ops.O.(b + a + (a * int 3)), Ops.O.(b + (a * int 4)));
          ("b + 3a + a", Ops.O.(b + (a * int 3) + a), Ops.O.(b + (a * int 4)));
          ("b + a + a", Ops.O.(b + a + a), Ops.O.(b + (a * int 2)));
        ]
        simplifies;
      cases "c * (x + c') is c * x + c * c' for a weak integer x" ~id
        [
          ( "2 * (a + 3)",
            Ops.O.(int 2 * (a + int 3)),
            Ops.O.((a * int 2) + int 6) );
          ( "-1 * (a + 3)",
            Ops.O.(int (-1) * (a + int 3)),
            Ops.O.((a * int (-1)) + int (-3)) );
        ]
        simplifies;
      cases "a float sum or quotient keeps its grouping" ~id:fst
        [
          ("-1 * (f + 3)", Ops.O.(float (-1.) * (f + float 3.)));
          ("(f / g) / h", Ops.O.(f / g / Shape.param 3 Dtype.Float32));
        ]
        stays_symbolic;
    ]

let symbolic_selections =
  let t = var "t" 0 3 and e = var "e" 0 3 in
  group "selections"
    [
      cases "a selection by a negation swaps its branches" ~id
        [
          ("not c", Ops.where (Ops.logical_not cond) t e, Ops.where cond e t);
          ( "false = c",
            Ops.where (raw Cmpeq [ Ops.bool false; cond ]) t e,
            Ops.where cond e t );
        ]
        simplifies;
      cases
        "the condition is true in the true branch and false in the false one"
        ~id
        [
          ( "both branches",
            (let inner = Ops.where cond Ops.O.(~-a) a in
             Ops.where cond Ops.O.(inner * int 2) Ops.O.(inner + int 1)),
            Ops.where cond Ops.O.(a * int (-2)) Ops.O.(a + int 1) );
          ( "the false branch",
            Ops.where cond t (Ops.cast cond Weak_int),
            Ops.where cond t (raw ~arg:(Dtype Weak_int) Cast [ Ops.bool false ])
          );
        ]
        simplifies;
      test "a range's bound keeps the condition whatever the branch" (fun () ->
          let bound = Ops.where cond (Ops.int 3) (Ops.int 0) in
          let r = Ops.range ~axis_type:Weak (Sym bound) [ 0 ] in
          let index = Ops.O.(r * int 2) in
          stays_under_symbolic
            (Ops.where cond index (Shape.const_like index `Invalid)));
      test "the condition holds elsewhere in a branch holding such a range"
        (fun () ->
          let bound = Ops.where cond (Ops.int 3) (Ops.int 0) in
          let r = Ops.range ~axis_type:Weak (Sym bound) [ 0 ] in
          let inner = Ops.where cond Ops.O.(r * int 2) a in
          by_symbolic (Ops.where cond inner e)
            (Ops.where cond Ops.O.(r * int 2) e));
      test "another condition is not assumed in the branches" (fun () ->
          stays_under_symbolic
            (Ops.where cond (Ops.where other t e) (Ops.where other e t)));
      cases "the condition is not assumed in the branches" ~id:fst
        [
          ( (* One node for every use of the constant: assuming it in a branch
               would rewrite it everywhere. *)
            "a broadcast constant",
            let x = Shape.param ~shape:[ Ops.Int 4 ] 0 Dtype.Float32 in
            let y = Shape.param ~shape:[ Ops.Int 4 ] 1 Dtype.Float32 in
            let c = Shape.const_like ~dtype:Bool x (`Bool false) in
            Ops.where c (Ops.where c x (Ops.where c y x)) y );
          ( "a padded constant",
            let x = Shape.param ~shape:[ Ops.Int 6 ] 0 Dtype.Float32 in
            let y = Shape.param ~shape:[ Ops.Int 6 ] 1 Dtype.Float32 in
            let inner = Shape.param ~shape:[ Ops.Int 4 ] 2 Dtype.Float32 in
            let c =
              Shape.pad
                (Shape.const_like ~dtype:Bool inner (`Bool true))
                [ Some (Ops.Int 1, Ops.Int 1) ]
            in
            Ops.where c (Ops.where c x (Ops.where c y x)) y );
          ( "a condition reaching an index",
            let c = Ops.O.(Ops.index buf [ a ] < float 1.) in
            Ops.where c (Ops.where c f g) (Shape.param 3 Dtype.Float32) );
          ( (* The access memory coalescing makes of four loads: its address is
               a shrink of the buffer, where one load's is an index. *)
            "a condition reaching a load of several elements",
            let load =
              Ops.load (Ops.v Op.Shrink ~src:[ buf; a; Ops.int 4 ]) []
            in
            let lanes u = Ops.broadcast u 4 in
            let c = Ops.O.(load < lanes (float 1.)) in
            Ops.where c (Ops.where c (lanes f) (lanes g)) load );
          ( "branches reaching a load of several elements",
            let load =
              Ops.load (Ops.v Op.Shrink ~src:[ buf; a; Ops.int 4 ]) []
            in
            let c = Ops.broadcast cond 4 in
            let inner = Ops.where c load (Ops.broadcast f 4) in
            Ops.where c Ops.O.(inner + load) inner );
          ( "a guard of a conversion of a float to an integer",
            let near = Ops.O.(f < float 4096.) in
            let q = Ops.cast (Ops.where near f (Ops.O.float 0.)) Dtype.Int32 in
            Ops.where near q x );
        ]
        (fun (_, sel) -> equal uop sel (symbolic sel));
      test "where g x 0 <> 0 is g land (x <> 0)" (fun () ->
          by_symbolic
            Ops.O.(Ops.where cond b (int 0) <> int 0)
            Ops.O.(cond land (b <> int 0)));
      cases "nested selections sharing a branch merge their conditions" ~id
        [
          ( "the false branch, by land",
            Ops.where cond (Ops.where other t e) e,
            Ops.where Ops.O.(cond land other) t e );
          ( "the true branch, by lor",
            Ops.where cond t (Ops.where other t e),
            Ops.where Ops.O.(cond lor other) t e );
          ( "the outer's false branch as the inner's true one, by land of the \
             inner's negation",
            Ops.where cond (Ops.where other e t) e,
            Ops.where Ops.O.(cond land Ops.logical_not other) t e );
        ]
        simplifies;
      cases
        "an operation on two selections by one condition with a constant pair \
         of branches selects between the operations"
        ~id
        [
          ( "a sum",
            Ops.O.(Ops.where cond t (int 0) + Ops.where cond e (int 1)),
            Ops.where cond Ops.O.(t + e) (Ops.int 1) );
          ( "the last two terms of a sum",
            Ops.O.(b + Ops.where cond t (int 0) + Ops.where cond e (int 1)),
            Ops.O.(b + Ops.where cond (t + e) (int 1)) );
          ( "where c t 0 + where c 0 f",
            Ops.O.(Ops.where cond t (int 0) + Ops.where cond (int 0) e),
            Ops.where cond t e );
        ]
        simplifies;
      cases "an operation on selections without a constant pair stays" ~id:fst
        [
          ("a sum", Ops.O.(Ops.where cond t e + Ops.where cond e t));
          ( "in a longer sum",
            Ops.O.(b + Ops.where cond t e + Ops.where cond e t) );
          ( "a cast of one selection",
            Ops.cast (Ops.where cond other Ops.O.(a < int 2)) Int32 );
        ]
        stays_symbolic;
    ]

let bounds =
  let v = var "v" (-10) 10 in
  let s = var ~dtype:Int32 "s" (-10) 10 in
  group "bounds"
    [
      cases "a node whose bounds are equal is that constant" ~id
        [
          ("a < 9", Ops.O.(a < int 9), Ops.bool true);
          ("a < 0", Ops.O.(a < int 0), Ops.bool false);
          ("a <> 9", Ops.O.(a <> int 9), Ops.bool true);
          ("a // 9", Ops.O.(a // int 9), Ops.int 0);
          (* 10 to 14 divided by 5 is 2 *)
          ("[10, 14] // 5", Ops.O.(var "v" 10 14 // int 5), Ops.int 2);
          (* a multiple of 4 is 0 modulo 2 *)
          ("4a % 2", Ops.O.(a * int 4 % int 2), Ops.int 0);
          ("a variable of one value", var "one" 1 1, Ops.int 1);
          ( "a hardware index of one value",
            Ops.special (Int 1) "gidx0",
            Ops.int 0 );
          ( "a range of one value",
            Ops.range ~axis_type:Weak (Int 1) [ 0 ],
            Ops.int 0 );
          ( "an after of one value",
            Ops.after (var "k" 2 2) [ Ops.v Op.Noop ],
            Ops.int 2 );
        ]
        simplifies;
      test "a range with a symbolic end is not folded" (fun () ->
          stays_under_symbolic
            (Ops.range ~axis_type:Weak (Sym (Ops.cast cond Weak_int)) [ 0 ]));
      cases
        "an integer maximum of operands that do not overlap is the greater, \
         committed to its type"
        ~id
        [
          ("max a 10", Ops.maximum a (Ops.int 10), Ops.int 10);
          ("max -1 a", Ops.maximum (Ops.int (-1)) a, a);
          ("max x 10, int32", Ops.maximum x (Ops.int 10), int32 10);
        ]
        simplifies;
      cases "an integer selection that computes a maximum is one" ~id
        [
          ( "where (v < 0) 0 v",
            Ops.where Ops.O.(v < int 0) (Ops.int 0) v,
            Ops.maximum v (Ops.int 0) );
          ( "where (0 < v) v 0",
            Ops.where Ops.O.(int 0 < v) v (Ops.int 0),
            Ops.maximum v (Ops.int 0) );
          ( "where (s < 3) 5 (s + 2)",
            Ops.where Ops.O.(s < int32 3) (int32 5) Ops.O.(s + int32 2),
            Ops.maximum Ops.O.(s + int 2) (Ops.int 5) );
        ]
        simplifies;
      cases "a selection that computes something other than a maximum stays"
        ~id:fst
        [
          ("where (v < 0) 1 v", Ops.where Ops.O.(v < int 0) (Ops.int 1) v);
          ("where (0 < v) v 1", Ops.where Ops.O.(int 0 < v) v (Ops.int 1));
        ]
        stays_symbolic;
    ]

let symbolic_constants =
  group "constants"
    [
      cases "two associative operations on constants fold the constants" ~id
        [
          ("a + 3 + 4", Ops.O.(a + int 3 + int 4), Ops.O.(a + int 7));
          ("a * 3 * 4", Ops.O.(a * int 3 * int 4), Ops.O.(a * int 12));
          ( "max (max v 10) 11",
            Ops.maximum (Ops.maximum (var "v" 0 20) (Ops.int 10)) (Ops.int 11),
            Ops.maximum (var "v" 0 20) (Ops.int 11) );
        ]
        simplifies;
      test "(x // c1) // c2 is x // (c1 * c2) for a positive c2" (fun () ->
          let v = var "v" 0 1800 in
          by_symbolic Ops.O.(v // int 10 // int 9) Ops.O.(v // int 90));
      cases "constants move to the end of integer sums and products" ~id
        [
          ("a + 3 + b", Ops.O.(a + int 3 + b), Ops.O.(a + b + int 3));
          ("a * 3 * b", Ops.O.(a * int 3 * b), Ops.O.(a * b * int 3));
        ]
        simplifies;
      test "a factor -1 moves to the end of a float product" (fun () ->
          by_symbolic Ops.O.(f * float (-1.) * g) Ops.O.(f * g * float (-1.)));
    ]

let comparisons =
  group "comparisons"
    [
      test "c0 + x < c1 is x < c1 - c0" (fun () ->
          by_symbolic Ops.O.(a + int 2 < int 5) Ops.O.(a < int 3));
      cases "c0 * x < c1 divides both sides by c0, rounding up" ~id
        [
          ("4a < 13", Ops.O.(a * int 4 < int 13), Ops.O.(a < int 4));
          ("4a < 16", Ops.O.(a * int 4 < int 16), Ops.O.(a < int 4));
          ( "a negative c0 flips x's sign",
            Ops.O.(var "v" (-5) 5 * int (-4) < int 13),
            Ops.O.(int (-4) < var "v" (-5) 5) );
        ]
        simplifies;
      test "x // d < c is x < c * d for a positive d" (fun () ->
          let v = var "v" 0 24 in
          by_symbolic Ops.O.(v // int 4 < int 3) Ops.O.(v < int 12));
      cases
        "d * y + p < c is y < ceil((c - p) / d) where that bound is one for \
         every value of p"
        ~id
        [
          ( "4a + [0, 3] < 16",
            Ops.O.((a * int 4) + var "s" 0 3 < int 16),
            Ops.O.(a < int 4) );
          ( "4s + [0, 3] < 0, s signed",
            Ops.O.((var "s" (-3) 3 * int 4) + var "t" 0 3 < int 0),
            Ops.O.(var "s" (-3) 3 < int 0) );
        ]
        simplifies;
      test "d * y + p < c stays where the bound differs across p's values"
        (fun () ->
          stays_under_symbolic Ops.O.((a * int 4) + var "w" 0 4 < int 16));
      test "c0 + x < c1 stays for floats, where moving c0 rounds" (fun () ->
          let d = var ~dtype:Float64 "d" 0 2 in
          stays_under_symbolic Ops.O.(float 1e16 + d < float (1e16 +. 2.)));
      test "-x < -y is y < x" (fun () ->
          by_symbolic Ops.O.(a * int (-1) < b * int (-1)) Ops.O.(b < a));
      test
        "not (x < 1) drops the coefficients of a sum of non-negative terms \
         with positive coefficients" (fun () ->
          let lt1 e = Ops.O.(lnot (e < int 1)) in
          by_symbolic
            (lt1 Ops.O.((a * int 3) + (b * int 4)))
            (lt1 Ops.O.(a + b)));
      cases "not (x < 1) keeps the coefficients otherwise" ~id:fst
        [
          ( "a term that can be negative",
            Ops.O.(lnot ((a * int 3) + (var "v" (-3) 3 * int 4) < int 1)) );
          ( "a negative coefficient",
            Ops.O.(lnot ((a * int (-3)) + (b * int 4) < int 1)) );
        ]
        stays_symbolic;
    ]

let ranges =
  let r = Ops.range ~axis_type:Weak (Int 8) [ 0 ] in
  let n = Ops.O.(var "n" 1 10 + int 2) in
  let rn = Ops.range ~axis_type:Weak (Sym n) [ 0 ] in
  group "ranges"
    [
      cases "a range modulo its end is the range" ~id
        [
          ("a constant end", Ops.O.(r % int 8), r);
          ("a symbolic end", Ops.O.(rn % n), rn);
        ]
        simplifies;
      cases "a range divided by its end is 0" ~id
        [
          ("a constant end", Ops.O.(r // int 8), Ops.int 0);
          ("a symbolic end", Ops.O.(rn // n), Ops.int 0);
        ]
        simplifies;
    ]

let symbolic_casts =
  let i8 = var ~dtype:Int8 "i" (-4) 4 in
  let byte = var ~dtype:Uint8 "b" 0 255 and shift = var ~dtype:Uint8 "s" 0 7 in
  let bounded = var ~dtype:Int64 "b" (-16) 16 in
  let wide v = Ops.cast v Int64 in
  group "casts"
    [
      test
        "a cast through a type that holds every value of the operand is one \
         cast" (fun () ->
          by_symbolic
            (Ops.cast (Ops.cast i8 Int32) Float32)
            (Ops.cast i8 Float32));
      cases "a cast of an integer that fits the intermediate type is one cast"
        ~id
        [
          ( "int32 to int8 to int64",
            Ops.cast (Ops.cast x Int8) Int64,
            Ops.cast x Int64 );
          ( "int64 to int32 to int16",
            Ops.cast (Ops.cast bounded Int32) Int16,
            Ops.cast bounded Int16 );
        ]
        simplifies;
      cases "a cast of an integer the intermediate type does not hold stays"
        ~id:fst
        [
          ( "int32 [0, 1000] through int8",
            Ops.cast (Ops.cast (var ~dtype:Int32 "big" 0 1000) Int8) Int64 );
          ( "int64 [0, 2^32 - 1] through int32",
            let u =
              Ops.variable ~dtype:Int64 "u" (i 0)
                (`Int (Bigint.of_string "4294967295"))
            in
            Ops.cast (Ops.cast u Int32) Int64 );
        ]
        stays_symbolic;
      cases
        "a binary operation on 64-bit integers whose values fit 32 bits \
         computes in int32"
        ~id
        [
          ("a sum", Ops.O.(wide x + wide y), wide Ops.O.(x + y));
          ("a product", Ops.O.(wide x * wide y), wide Ops.O.(x * y));
          ( "with a weak constant",
            Ops.O.(wide x + int 5),
            wide Ops.O.(x + int 5) );
          ( "a product by a constant",
            Ops.O.(bounded * int 8),
            wide Ops.O.(Ops.cast bounded Int32 * int 8) );
        ]
        simplifies;
      test "a binary operation on 64-bit integers that can pass 32 bits stays"
        (fun () ->
          let w = var ~dtype:Int64 "w" 0 2147483647 in
          stays_under_symbolic Ops.O.(w * int 8));
      test "a cast of x + c to a signed integer is the cast of x plus c"
        (fun () ->
          by_symbolic
            (Ops.cast Ops.O.(a + int 3) Int32)
            Ops.O.(Ops.cast a Int32 + int 3));
      test "a widened unsigned mask or shift keeps every value" (fun () ->
          let points =
            List.concat_map
              (fun b ->
                List.map (fun k -> [ ("b", i b); ("s", i k) ]) [ 0; 3; 4; 7 ])
              [ 0; 1; 15; 16; 127; 128; 200; 255 ]
          in
          List.iter
            (fun dt ->
              keeps_machine_value
                (Ops.cast Ops.O.((byte lsr shift) land int 15) dt)
                points;
              keeps_machine_value
                (Ops.cast
                   Ops.O.(byte land (Ops.int ~dtype:Uint8 255 lsr shift))
                   dt)
                points)
            [ Dtype.Uint16; Uint32; Uint64 ]);
    ]

let ordering =
  let store = Ops.store (Ops.index buf [ a ]) f in
  group "ordering"
    [
      test "an after keeps its stores and ranges" (fun () ->
          stays_under_symbolic (Ops.after f [ store; r0 ]));
      test
        "an after waits on a reduction's source, with the reduction's ranges \
         ended" (fun () ->
          let value = Ops.cast r0 Dtype.Float32 in
          by_symbolic
            (Ops.after f [ Ops.reduce value Op.Add [ r0 ] ])
            (Ops.after f [ Ops.end_ (Ops.after (Ops.v Op.Noop) [ r0 ]) [ r0 ] ]));
      test "an after waits on the sources of any other node" (fun () ->
          by_symbolic
            (Ops.after f [ Ops.O.(g + float 1.); store ])
            (Ops.after f [ store ]));
      cases "an after or an end of nothing is its value" ~id
        [ ("after", raw After [ f ], f); ("end", raw End [ store ], store) ]
        simplifies;
      cases "an end drops the ranges that became constants" ~id
        [
          ( "beside another range",
            Ops.end_ store [ r0; Ops.range ~axis_type:Weak (Int 1) [ 1 ] ],
            Ops.end_ store [ r0 ] );
          ( "alone, so the end is its value",
            Ops.end_ store [ Ops.range ~axis_type:Weak (Int 1) [ 1 ] ],
            store );
        ]
        simplifies;
      test "a backedge keeps a constant condition" (fun () ->
          stays_under_symbolic
            (Ops.backedge store ~loop:(Ops.loop 3) ~cond:(Ops.bool false)));
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
      cases "then simplifies division and remainder" ~id
        [
          ("(4a + 3) // 4", Ops.O.(((a * int 4) + int 3) // int 4), a);
          ("(4a + 3) % 4", Ops.O.(((a * int 4) + int 3) % int 4), Ops.int 3);
        ]
        simplifies;
    ]

(* Values of constants *)

let values_of_constants =
  let double = Ops.Upat.(var "x" * cvar "c") in
  let matcher rule = Ops.Pattern_matcher.v (fun () -> [ rule ]) in
  (* doubles the constant factor of a product, read with number *)
  let doubles m =
    let c = Dtype.Value.(Shape.number (Ops.value (m "c")) * of_int 2) in
    Some Ops.O.(m "x" * Ops.const (c :> Dtype.const))
  in
  group "values of constants"
    [
      cases "a rule that computes with the value of invalid does not apply"
        ~id:fst
        [
          ( "a < invalid in a selection",
            Ops.where Ops.O.(a < Ops.invalid) (Ops.int 0) a );
          ("2a < invalid", Ops.O.(a * int 2 < Ops.invalid));
          ("a // 2 < invalid", Ops.O.(a // int 2 < Ops.invalid));
        ]
        (fun (_, e) -> equal uop (rewrite Shape.commutative e) (symbolic e));
      test "number is the value of a constant" (fun () ->
          equal Dtypes.value (i 3) (Shape.number (i 3)));
      test "number raises Not_a_number on invalid" (fun () ->
          raises Shape.Not_a_number (fun () -> Shape.number `Invalid));
      test "rule declines where its function raises Not_a_number" (fun () ->
          let m = matcher (Shape.rule double doubles) in
          equal (option uop) None
            (Ops.Pattern_matcher.rewrite m () Ops.O.(a * Ops.invalid));
          equal (option uop)
            (Some Ops.O.(a * int 4))
            (Ops.Pattern_matcher.rewrite m () Ops.O.(a * int 2)));
      test "rule_ctx lets Not_a_number through" (fun () ->
          let m = matcher (Shape.rule_ctx double (fun () m -> doubles m)) in
          raises Shape.Not_a_number (fun () ->
              Ops.Pattern_matcher.rewrite m () Ops.O.(a * Ops.invalid)));
      test "with_ctx runs a matcher under any context" (fun () ->
          let m = Shape.with_ctx (matcher (Shape.rule double doubles)) in
          equal (option uop)
            (Some Ops.O.(a * int 4))
            (Ops.Pattern_matcher.rewrite m "any context" Ops.O.(a * int 2)));
      cases "equals is whether the constant holds the number" ~id
        [
          ("3", Ops.int 3, [ i 3 ]);
          ("a cast of 3", Ops.cconst Int32 (i 3), [ i 3 ]);
          ("invalid", Ops.invalid, []);
        ]
        (fun (_, u, expected) ->
          equal (list Dtypes.value) expected
            (holding (Shape.equals u) [ i 2; i 3; i 4 ]));
      test "equals rejects a node that is neither a constant nor a cast of one"
        (fun () -> Common.rejects (fun () -> Shape.equals a (i 0)));
      test "const_v is the number as a constant of the node's type" (fun () ->
          let p = Shape.param ~shape:[ Int 2; Int 3 ] 0 Float16 in
          equal uop (Shape.const_like p (i 5)) (Shape.const_v p (i 5)));
      cases "conj is the conjunction" ~id
        [
          ("two", Shape.conj [ cond; other ], Ops.O.(cond land other));
          ("one", Shape.conj [ cond ], cond);
        ]
        (fun (_, actual, expected) -> equal uop expected actual);
      test "conj rejects the empty list" (fun () ->
          Common.rejects (fun () -> Shape.conj []));
      test "invalid_pat matches invalid, named i, and no other constant"
        (fun () ->
          equal naming
            [ [ ("i", Ops.invalid) ] ]
            (Ops.Upat.match_ Shape.invalid_pat Ops.invalid);
          equal naming [] (Ops.Upat.match_ Shape.invalid_pat (Ops.int 0)));
    ]

(* Weak constants *)

(* A golden holds a graph and its rewrite by pm_uncast_const, with the setting
   DEFAULT_FLOAT at float. *)
let uncast name u =
  Golden.graph (name ^ ".golden") (fun () ->
      Setting.context
        [ B (Setting.default_float, "float") ]
        (fun () -> Ops.sink [ u; rewrite Shape.pm_uncast_const u ]))

let uncasts_as_goldens =
  let small ?(name = "x") dtype = var ~dtype name 0 10 in
  let i32 n = Ops.int ~dtype:Int32 n in
  group "pm_uncast_const rewrites each recorded graph as its golden states"
    [
      uncast "uncast_a_committed_literal" Ops.O.(small Int32 + i32 1);
      uncast "uncast_keeps_literals_with_no_committed_peer"
        Ops.O.(i32 1 + i32 2);
      uncast "uncast_keeps_a_shifted_literal"
        (Ops.v Op.Shl ~src:[ i32 1; var ~dtype:Uint32 "s" 0 31 ]);
      uncast "uncast_keeps_a_cast_of_an_expression"
        Ops.O.(small Int32 + Ops.cast (var "w" 0 4 + int 1) Int32);
      uncast "uncast_keeps_a_shifted_literal_of_its_peer_s_type"
        (Ops.v Op.Shl ~src:[ i32 1; var ~dtype:Int32 "s" 0 31 ]);
      uncast "uncast_keeps_a_literal_that_widens_a_comparison"
        (Ops.v Op.Cmplt ~src:[ small Int16; i32 1 ]);
      uncast "uncast_drops_a_cast_that_fits"
        Ops.O.(small ~name:"u" Uint8 < Ops.cconst Uint8 (i 44));
    ]

let weak_constants =
  group "weak constants"
    [
      test "is_weak holds of Dtype.weaks alone" (fun () ->
          equal (list Dtypes.dtype) Dtype.weaks
            (holding Shape.is_weak Dtypes.declared));
      test "weak is is_weak of the node's type" (fun () ->
          equal (list uop)
            [ a; Ops.float 1. ]
            (holding Shape.weak [ a; x; Ops.float 1.; f ]));
      test "is_const holds of a constant node" (fun () ->
          equal (list uop)
            [ Ops.int 3; Ops.invalid ]
            (holding Shape.is_const [ Ops.int 3; Ops.invalid; a; cond ]));
      test "unchanged holds of the node's sources, node for node" (fun () ->
          equal
            (list (list uop))
            [ [ a; b ] ]
            (holding
               (fun src -> Shape.unchanged src Ops.O.(a + b))
               [ [ a; b ]; [ b; a ]; [ a ]; [ a; b; a ] ]));
      cases "derived_dtypes is the operands' meet and the node's type" ~id
        [
          ( "an int32 and a weak operand",
            (Ops.O.(x + int 1), [ x; Ops.int 1 ]),
            Some (Dtype.Int32, Dtype.Int32) );
          ( "a comparison of int32",
            (Ops.O.(x < int 1), [ x; Ops.int 1 ]),
            Some (Dtype.Int32, Dtype.Bool) );
          ("weak operands", (Ops.O.(a + b), [ a; b ]), None);
        ]
        (fun (_, (u, src), expected) ->
          equal
            (option (pair Dtypes.dtype Dtypes.dtype))
            expected
            (Shape.derived_dtypes u src));
      uncasts_as_goldens;
      (* Uncasting a committed constant reads it at its width, where tinygrad
         keeps the literal unwrapped. *)
      test "pm_uncast_const reads a committed constant at its width" (fun () ->
          let x = var ~dtype:Uint8 "x" 0 255 in
          let lt = Ops.O.(x < Ops.cconst Uint8 (i 300)) in
          equal uop Ops.O.(x < int 44) (rewrite Shape.pm_uncast_const lt);
          equal Dtypes.const (`Bool false)
            (Interpreter.eval ~vars:[ ("x", i 100) ] (symbolic lt)));
    ]
