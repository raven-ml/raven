open Windtrap
open Tolk
open Dtypes

let x ?(slot = 0) dt = Shape.param slot dt
let int n = `Int (Bigint.of_int n)
let var name = Ops.variable ~dtype:Dtype.Int32 name (int 0) (int 10)

(* [rejects f u] asserts that [f u] raises [Invalid_argument]: [u] is built
   first, so that a refusal to build it fails the test. *)
let rejects f u = raises_match (Exn.invalid_arg ?sub:None) (fun () -> f u)

(* A variable of the committed integer type [dt] over all of its values. *)
let variable dt name =
  let lo, hi = int_bounds dt in
  Ops.variable ~dtype:dt name (`Int lo) (`Int hi)

(* A weak integer variable over the values of 64 bits. *)
let weak name =
  let lo, hi = int_bounds Dtype.Int64 in
  Ops.variable name (`Int lo) (`Int hi)

(* [overflow] prints a verdict of {!Interpreter.overflows}. *)
let overflow =
  Testable.make ~equal:Bool.equal ~pp:(fun ppf over ->
      Format.pp_print_string ppf
        (if over then "overflows" else "stays within its type"))

(* [float_bits ~params bits op] is the bits of the float32 [op f], where [f] is
   the float32 of [bits]: a moved NaN's bits are the same on every host. *)
let float_bits ?(params = []) bits op =
  let f = Ops.bitcast (x Dtype.Uint32) Dtype.Float32 in
  Interpreter.eval
    ~params:((0, int bits) :: params)
    (Ops.bitcast (op f) Dtype.Uint32)

(* Name *)

let name =
  cases "is a variable's, a hardware index's or a range's name, and no other's"
    ~id:(fun (id, _, _) -> id)
    [
      ("a variable", var "i", Some "i");
      ("a hardware index", Ops.special (Int 8) "gidx0", Some "gidx0");
      ("a range", Ops.range ~axis_type:Weak (Int 4) [ 0 ], Some "r0");
      ( "a range of two parts",
        Ops.range ~axis_type:Weak (Int 4) [ 1; 2 ],
        Some "r1_2" );
      ( "a range of a negative part",
        Ops.range ~axis_type:Weak (Int 4) [ -1 ],
        Some "rm1" );
      ("a constant", Ops.int ~dtype:Dtype.Int32 1, None);
      ("a scalar parameter", x Dtype.Int32, None);
    ]
    (fun (_, u, expected) ->
      equal (option string) expected (Interpreter.name u))

(* Eval *)

let leaves =
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
  and g = Ops.special (Int 8) "gidx0" in
  group "leaves"
    [
      test "a constant is its value" (fun () ->
          equal const (`Float 1.5)
            (Interpreter.eval (Ops.float ~dtype:Dtype.Float32 1.5)));
      test "a variable is the value vars binds to its name" (fun () ->
          let u = Ops.sub (var "i") (var "j") in
          equal const (int (-4))
            (Interpreter.eval ~vars:[ ("j", int 7); ("i", int 3) ] u));
      test
        "a range and a hardware index are the values vars binds to their names"
        (fun () ->
          equal const (int 21)
            (Interpreter.eval
               ~vars:[ ("r0", int 2); ("gidx0", int 1) ]
               Ops.O.((r * int 10) + g)));
      test "a bound variable vars does not bind is its bound value" (fun () ->
          equal const (int 2) (Interpreter.eval (Shape.bind (var "i") (int 2))));
      test "vars binds a bound variable over its bound value" (fun () ->
          equal const (int 5)
            (Interpreter.eval
               ~vars:[ ("i", int 5) ]
               (Shape.bind (var "i") (int 2))));
      test "a parameter is the value params gives its slot" (fun () ->
          equal const (int 7)
            (Interpreter.eval
               ~params:[ (0, int 3); (1, int 7) ]
               (x ~slot:1 Dtype.Int32)));
      (* params binds the other scalar parameters: a variable's slot binds
         nothing, so that variables of one slot stay apart *)
      test
        "a variable without a value is refused, even where params gives its \
         slot" (fun () ->
          let i = var "i" in
          let slot =
            match Ops.arg i with Param { slot; _ } -> slot | _ -> 0
          in
          rejects (Interpreter.eval ~params:[ (slot, int 3) ]) i);
      cases "a range or a hardware index without a value is refused" ~id:fst
        [ ("a range", r); ("a hardware index", g) ]
        (fun (_, u) -> rejects Interpreter.eval u);
      test "a parameter without a value is refused" (fun () ->
          rejects Interpreter.eval (x Dtype.Int32));
    ]

let operations =
  let x8 = variable Dtype.Uint8 "x" in
  let w = weak "w" in
  group "operations"
    [
      (* IEEE 754: 2049 lies halfway between the halves 2048 and 2050, and ties
         go to the even significand *)
      cases "an operation rounds its result to its type"
        ~id:(fun (id, _, _, _) -> id)
        [
          ( "half 2048 + 1 is 2048",
            Dtype.Float16,
            (`Float 2048., `Float 1.),
            `Float 2048. );
          ("int8 127 + 1 is -128", Dtype.Int8, (int 127, int 1), int (-128));
        ]
        (fun (_, dt, (a, b), expected) ->
          let sum = Ops.add (x dt) (x ~slot:1 dt) in
          equal const expected (Interpreter.eval ~params:[ (0, a); (1, b) ] sum));
      (* IEEE 754's fusedMultiplyAdd rounds the exact a * b + c once: 1.0625^2 =
         1 + 2^-3 + 2^-8 is the tie between two bfloat16s, and 2^-100 puts the
         sum above it. Rounded to a double first, 2^-100 is lost, and the tie
         goes to the even 1.125. The float32 row is the same with 1 + 2^-12. *)
      cases "a multiply-add of floats rounds once to its type"
        ~id:(fun (dt, _, _) -> Testable.to_string dtype dt)
        [
          (Dtype.Bfloat16, 1.0625, 1.1328125);
          (Float32, 1. +. ldexp 1. (-12), 1. +. ldexp 1. (-11) +. ldexp 1. (-23));
        ]
        (fun (dt, a, expected) ->
          let u = Ops.alu (x dt) Mulacc [ x ~slot:1 dt; x ~slot:2 dt ] in
          let params =
            [ (0, `Float a); (1, `Float a); (2, `Float (ldexp 1. (-100))) ]
          in
          equal const (`Float expected) (Interpreter.eval ~params u));
      (* -1 committed to a uint8 is 255 *)
      cases "a weak integer operand takes its committed peer's type and wraps"
        ~id:fst
        [
          ("max (w, x)", (Ops.maximum w x8, int 255));
          ("x < w", (Ops.lt x8 w, `Bool true));
          ("max (-1, x)", (Ops.maximum (Ops.int (-1)) x8, int 255));
          ("x < -1", (Ops.lt x8 (Ops.int (-1)), `Bool true));
        ]
        (fun (_, (u, expected)) ->
          equal const expected
            (Interpreter.eval ~vars:[ ("x", int 100); ("w", int (-1)) ] u));
      (* wrapped to a uint8, a count or an exponent of 256 would be 0, and give
         x and 1 *)
      cases
        "a shift's count and a power's exponent are numbers, read at no width"
        ~id:fst
        [
          ("1 << 256", (Ops.shl x8 (Ops.int 256), 1));
          ("128 >> 256", (Ops.shr x8 (Ops.int 256), 128));
          ("2 ** 256", (Ops.pow x8 (Ops.int 256), 2));
        ]
        (fun (_, (u, at)) ->
          equal const (int 0) (Interpreter.eval ~vars:[ ("x", int at) ] u));
      (* Ops.exec_alu's, and IEEE 754's negate: the sign flips, nothing else *)
      cases "a negation or a selection keeps a NaN's bits, as Ops.exec_alu does"
        ~id:(fun (id, _, _) -> id)
        [
          ( "a negation of a signalling NaN",
            (fun f -> Ops.v ~src:[ f ] Neg),
            (0x7f800001, 0xff800001) );
          ( "a negation of a quiet NaN",
            (fun f -> Ops.v ~src:[ f ] Neg),
            (0x7fc12345, 0xffc12345) );
          ( "a selection of a signalling NaN and a weak literal",
            (fun f -> Ops.where (x ~slot:1 Dtype.Bool) f (Ops.float 1.)),
            (0x7f800001, 0x7f800001) );
        ]
        (fun (_, op, (input, expected)) ->
          equal const (int expected)
            (float_bits ~params:[ (1, `Bool true) ] input op));
      test "an operation Ops.exec_alu refuses is refused" (fun () ->
          rejects
            (Interpreter.eval ~vars:[ ("x", int 1); ("w", int (-1)) ])
            (Ops.shl x8 w));
      test "a node that is not arithmetic is refused" (fun () ->
          rejects Interpreter.eval (Ops.sink [ Ops.int ~dtype:Dtype.Int32 1 ]));
    ]

(* Casts *)

(* [bits] compares values as {!Dtypes.const} does, floats bit for bit, NaNs
   included. *)
let bits =
  let float_bits = function
    | `Float x -> Some (Int64.bits_of_float x)
    | `Int _ | `Bool _ | `Invalid -> None
  in
  Testable.make
    ~pp:(fun ppf v ->
      match float_bits v with
      | Some b -> Format.fprintf ppf "%a (0x%016Lx)" (Testable.pp const) v b
      | None -> Testable.pp const ppf v)
    ~equal:(fun v w ->
      match (float_bits v, float_bits w) with
      | Some b, Some c -> Int64.equal b c
      | None, None -> Testable.equal const v w
      | _ -> false)

(* [cast_of from to_ v] is the cast of the value [v] of [from] to [to_], as
   Interpreter.mli states it: [v] itself at its own type, a NaN to a float
   truncated with its sign and payload, and otherwise [v] converted as a
   constant of [to_], then wrapped. *)
let cast_of from to_ (v : Dtype.value) : Dtype.const =
  if Dtype.equal from to_ then (v :> Dtype.const)
  else
    match v with
    | `Float x when Float.is_nan x && Dtype.is_float to_ ->
        (Dtype.truncate to_ v :> Dtype.const)
    | v -> (
        match Dtype.const to_ v with
        | #Dtype.value as c -> (Dtype.truncate to_ c :> Dtype.const)
        | `Invalid -> `Invalid)

(* [overflows_of from to_ v] is [true] iff the cast of [v] from [from] to the
   integer type [to_] gives it an exact value outside it, or [v] is an infinity
   or a NaN, which has none. *)
let overflows_of from to_ (v : Dtype.value) =
  (not (Dtype.equal from to_))
  && Dtype.is_int to_
  &&
  match v with
  | `Float x when not (Float.is_finite x) -> true
  | v ->
      let lo, hi = int_bounds to_ and z = Dtype.Value.to_z v in
      Bigint.lt z lo || Bigint.gt z hi

let stored_dtypes =
  List.filter
    (fun dt -> not (List.mem dt Dtype.[ Void; Weak_int; Weak_float ]))
    declared

let int_dtypes = List.filter Dtype.is_int stored_dtypes

(* The bounds of the integer types and their neighbours. *)
let bounds =
  List.concat_map
    (fun dt ->
      let lo, hi = int_bounds dt in
      [ Bigint.pred lo; lo; Bigint.succ lo; Bigint.pred hi; hi; Bigint.succ hi ])
    int_dtypes

let quiet_nan = Int64.float_of_bits 0x7ff8_1234_5000_0000L
let signalling_nan = Int64.float_of_bits 0x7ff4_0000_0000_0000L
let negative_nan = Int64.float_of_bits 0xfff8_0000_2000_0000L

let specials =
  [ quiet_nan; signalling_nan; negative_nan; -0.; infinity; neg_infinity ]

(* A source value, made a value of each type as a constant of it. *)
let source =
  Gen.with_pp (Testable.pp value)
    (Gen.frequency
       [
         (2, Gen.map (fun x -> `Float x) finite_float);
         (1, Gen.map (fun x -> `Float x) (Gen.of_list specials));
         (1, Gen.map (fun z -> `Int z) integer);
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
       ])

(* Every special float and every bound, and floats a cast to an integer rounds
   towards zero (-3.9), wraps (300.7) or keeps within an int8 (127.9). *)
let examples =
  List.map (fun x -> `Float x) (specials @ [ 300.7; -3.9; 127.9 ])
  @ List.map (fun z -> `Int z) bounds

let quiet_bit x = Int64.(logand (bits_of_float x) 0x0008_0000_0000_0000L) <> 0L

(* [each_cast s f] calls [f from to_ v] for each pair of stored types, [v] the
   source [s] as a constant of [from]. *)
let each_cast s f =
  List.iter
    (fun from ->
      match Dtype.const from s with
      | `Invalid -> ()
      | #Dtype.value as v -> List.iter (fun to_ -> f from to_ v) stored_dtypes)
    stored_dtypes

let cast_msg from to_ v =
  Format.asprintf "%a of %a to %a" (Testable.pp value) v (Testable.pp dtype)
    from (Testable.pp dtype) to_

let casts =
  group "casts"
    [
      prop
        "a cast converts as a constant, then wraps, and is its operand at its \
         own type"
        ~examples source (fun s ->
          (match s with
          | `Float f ->
              cover "a quiet NaN with a payload"
                (Float.is_nan f && quiet_bit f
                && Int64.bits_of_float f <> Int64.bits_of_float Float.nan);
              cover "a signalling NaN" (Float.is_nan f && not (quiet_bit f));
              cover "-0." (f = 0. && Float.sign_bit f);
              cover "infinity" (f = infinity);
              cover "-infinity" (f = neg_infinity)
          | `Int z ->
              cover "an integer type's bound or its neighbour"
                (List.exists (Bigint.equal z) bounds)
          | `Bool _ -> ());
          each_cast s (fun from to_ v ->
              equal ~msg:(cast_msg from to_ v) bits (cast_of from to_ v)
                (Interpreter.eval ~params:[ (0, v) ] (Ops.cast (x from) to_))));
      test "a cast to its operand's type keeps a signalling NaN's bits"
        (fun () ->
          equal const (int 0x7f800001)
            (float_bits 0x7f800001 (fun f ->
                 Ops.v ~src:[ f ] ~arg:(Dtype Dtype.Float32) Cast)));
      (* Dtype.truncate keeps the sign and the top ten bits of the payload,
         0x412345 lsr 13 = 0x209, and quiets the NaN *)
      test "a cast of a NaN to a narrower float keeps its sign and payload"
        (fun () ->
          let half f = Ops.bitcast (Ops.cast f Dtype.Float16) Dtype.Uint16 in
          let f = Ops.bitcast (x Dtype.Uint32) Dtype.Float32 in
          equal const (int 0xfe09)
            (Interpreter.eval ~params:[ (0, int 0xffc12345) ] (half f)));
      (* IEEE 754's binary32 encodings *)
      cases "a bit reinterpretation is Dtype.bitcast"
        ~id:(fun (v, _) -> Printf.sprintf "%h" v)
        [ (1., 0x3f800000); (-0., 0x80000000) ]
        (fun (v, expected) ->
          equal const (int expected)
            (Interpreter.eval
               ~params:[ (0, `Float v) ]
               (Ops.bitcast (x Dtype.Float32) Dtype.Uint32)));
    ]

(* Operations against Ops.exec_alu *)

(* [outcome f] is [f ()], or [Error ()] if it raises [Invalid_argument]. *)
let outcome f =
  match f () with v -> Ok v | exception Invalid_argument _ -> Error ()

let refused =
  Testable.make
    ~pp:(fun ppf () -> Format.pp_print_string ppf "refused")
    ~equal:(fun () () -> true)

(* A value of [dt]: any, or a special float or an integer bound made one as a
   constant of [dt]. *)
let operand dt =
  let as_dt v =
    match Dtype.const dt v with
    | #Dtype.value as v -> v
    | `Invalid -> invalid_arg "a value has no constant"
  in
  Gen.frequency
    [
      (3, value_of dt);
      (1, Gen.map as_dt (Gen.of_list examples));
      (1, Gen.map (fun n -> as_dt (int n)) (Gen.int_range (-3) 3));
    ]

let arithmetic =
  Op.
    [
      Add;
      Sub;
      Mul;
      Max;
      Cmplt;
      Cmpne;
      Cmpeq;
      And;
      Or;
      Xor;
      Div;
      Mod;
      Truncdiv;
      Truncmod;
    ]

let operation =
  let open Gen in
  with_pp
    (fun ppf (op, dt, a, b) ->
      Format.fprintf ppf "%a %a at %a, %a" Op.pp op (Testable.pp value) a
        (Testable.pp dtype) dt (Testable.pp value) b)
    (let* op = of_list arithmetic in
     let* dt =
       frequency [ (6, stored); (1, of_list Dtype.[ Weak_int; Weak_float ]) ]
     in
     (* A weak value is any of the widest type of its kind. *)
     let held = function
       | Dtype.Weak_int -> Dtype.Int64
       | Weak_float -> Float64
       | dt -> dt
     in
     let+ a = operand (held dt) and+ b = operand (held dt) in
     (op, dt, a, b))

(* [operands dt a b] is a node of two operands of [dt], and its evaluation where
   they are [a] and [b]: parameters, weak integer variables or weak float
   literals. *)
let operands op dt a b =
  match (dt, a, b) with
  | Dtype.Weak_int, _, _ ->
      ( Ops.alu (weak "a") op [ weak "b" ],
        fun u -> Interpreter.eval ~vars:[ ("a", a); ("b", b) ] u )
  | Weak_float, `Float a, `Float b ->
      (Ops.alu (Ops.float a) op [ Ops.float b ], fun u -> Interpreter.eval u)
  | _ ->
      ( Ops.alu (x dt) op [ x ~slot:1 dt ],
        fun u -> Interpreter.eval ~params:[ (0, a); (1, b) ] u )

(* An integer type's extremes, where a quotient or a remainder overflows and an
   unsigned order differs from the signed one. *)
let extremes =
  let lo dt = `Int (fst (int_bounds dt))
  and hi dt = `Int (snd (int_bounds dt)) in
  Op.
    [
      (Div, Dtype.Int8, lo Dtype.Int8, int (-1));
      (Truncdiv, Int8, lo Int8, int (-1));
      (Truncmod, Int64, lo Int64, int (-1));
      (Mod, Int64, lo Int64, int (-1));
      (Cmplt, Uint64, lo Uint64, hi Uint64);
      (Cmplt, Uint64, hi Uint64, lo Uint64);
      (Cmplt, Uint32, hi Uint32, int 1);
      (Max, Uint64, hi Uint64, int 1);
    ]

let exec_alu =
  group "operations as Ops.exec_alu computes them"
    [
      prop ~count:2000 ~examples:extremes
        "an operation of two operands of one type is Ops.exec_alu's, a NaN's \
         bits included"
        operation (fun (op, dt, a, b) ->
          let u, eval =
            match outcome (fun () -> operands op dt a b) with
            | Ok u -> u
            | Error () -> reject ()
          in
          let is_nan = function `Float x -> Float.is_nan x | _ -> false in
          cover "an integer type" (Dtype.is_int dt);
          cover "a float type" (Dtype.is_float dt);
          cover "a weak type" (Dtype.mem dt Dtype.weaks);
          cover "a NaN operand" (is_nan a || is_nan b);
          cover "a division by 0"
            (List.mem op Op.[ Div; Mod; Truncdiv; Truncmod ]
            && Dtype.Value.( = ) b (int 0));
          equal (result bits refused)
            (outcome (fun () ->
                 Ops.exec_alu op (Ops.dtype u)
                   [ (a :> Dtype.const); (b :> Dtype.const) ]))
            (outcome (fun () -> eval u)));
      prop ~count:500
        "a selection of a number and a weak literal is Ops.exec_alu's of the \
         literal as the number's type commits it"
        Gen.(
          let* dt =
            of_list (List.filter (fun dt -> dt <> Dtype.Bool) stored_dtypes)
          in
          let+ c = bool and+ a = operand dt and+ n = int_range (-300) 300 in
          (dt, c, a, n))
        (fun (dt, c, a, n) ->
          let literal, v =
            if Dtype.is_float dt then
              (Ops.float (Float.of_int n /. 4.), `Float (Float.of_int n /. 4.))
            else (Ops.int n, int n)
          in
          let u =
            match
              outcome (fun () ->
                  Ops.where (x ~slot:1 Dtype.Bool) (x dt) literal)
            with
            | Ok u -> u
            | Error () -> reject ()
          in
          let v =
            if Dtype.is_int dt then (Dtype.truncate dt v :> Dtype.value) else v
          in
          equal (result bits refused)
            (outcome (fun () ->
                 Ops.exec_alu Where (Ops.dtype u)
                   [ `Bool c; (a :> Dtype.const); (v :> Dtype.const) ]))
            (outcome (fun () ->
                 Interpreter.eval ~params:[ (0, a); (1, `Bool c) ] u)));
    ]

let selections =
  let c = x Dtype.Bool in
  group "selections"
    [
      test "a selection is the branch it picks" (fun () ->
          let u =
            Ops.where c
              (Ops.int ~dtype:Dtype.Int32 1)
              (Ops.int ~dtype:Dtype.Int32 2)
          in
          equal const (int 2) (Interpreter.eval ~params:[ (0, `Bool false) ] u));
      cases
        "a selection of committed operands keeps the bits of the NaN it selects"
        ~id:(Printf.sprintf "0x%08x") [ 0xffc12345; 0x7f800001 ] (fun nan ->
          let select f =
            Ops.where (Ops.ne f f) f (Ops.float ~dtype:Dtype.Float32 1.)
          in
          equal const (int nan) (float_bits nan select));
      test "a selection evaluates the branch it does not pick" (fun () ->
          let unbound = Ops.cast (var "i") Dtype.Int32 in
          rejects
            (Interpreter.eval ~params:[ (0, `Bool true) ])
            (Ops.where c (Ops.int ~dtype:Dtype.Int32 1) unbound));
    ]

let invalid =
  let i = var "i" in
  let gated = Shape.valid i Ops.O.(i < int 5) in
  let at v u = Interpreter.eval ~vars:[ ("i", int v) ] u in
  group "invalid"
    [
      test "a gated index is its value where its gate holds" (fun () ->
          equal const (int 3) (at 3 gated));
      test "a gated index is invalid where its gate fails" (fun () ->
          equal const `Invalid (at 7 gated));
      cases "an operation of an invalid operand is invalid" ~id:fst
        [
          ("a product by 0", Ops.O.(gated * int 0));
          ("a sum", Ops.O.(int 1 + gated));
          ("a cast", Ops.cast gated Dtype.Int64);
        ]
        (fun (_, u) -> equal const `Invalid (at 7 u));
      test "a selection by an invalid condition is invalid" (fun () ->
          equal const `Invalid
            (at 7 (Ops.where Ops.O.(gated < int 2) (Ops.int 1) (Ops.int 2))));
      test "a selection that picks an invalid branch is invalid" (fun () ->
          equal const `Invalid
            (at 7 (Ops.where Ops.O.(i < int 9) Ops.invalid (Ops.int 1))));
      test "a selection of an invalid branch it does not pick is valid"
        (fun () ->
          equal const (int 1)
            (at 7 (Ops.where Ops.O.(i < int 9) (Ops.int 1) Ops.invalid)));
    ]

(* A float reduction of four terms of [dt] over one range. *)
let float_reduction =
  let open Gen in
  with_pp
    (fun ppf (op, dt, terms) ->
      Format.fprintf ppf "%a at %a of %a" Op.pp op (Testable.pp dtype) dt
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (Testable.pp value))
        terms)
    (let* op = of_list Op.[ Add; Mul; Max ] in
     let* dt = of_list (List.filter Dtype.is_float stored_dtypes) in
     let+ terms = list ~size:(constant 4) (operand dt) in
     (op, dt, terms))

let reductions =
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
  and s = Ops.range ~axis_type:Weak (Int 3) [ 1 ]
  and o = Ops.range ~axis_type:Weak (Int 5) [ 2 ] in
  let over rs op u = Ops.reduce u op rs in
  let int32 u = Ops.cast u Dtype.Int32 in
  let op_id (op, _) = Format.asprintf "%a" Op.pp op in
  group "reductions"
    [
      (* 1 + 2 + 3 + 4, 1 * 2 * 3 * 4 and the greatest of them *)
      cases "a reduction folds its operation over each value of its range"
        ~id:op_id
        [ (Op.Add, 10); (Mul, 24); (Max, 4) ]
        (fun (op, expected) ->
          equal const (int expected)
            (Interpreter.eval (over [ r ] op (int32 Ops.O.(r + int 1)))));
      (* Ops.identity_element *)
      cases "a reduction over an empty range is its identity" ~id:op_id
        [ (Op.Add, 0); (Mul, 1); (Max, -2147483648) ]
        (fun (op, expected) ->
          let empty = Ops.range ~axis_type:Weak (Int 0) [ 3 ] in
          equal const (int expected)
            (Interpreter.eval (over [ empty ] op (int32 empty))));
      prop ~count:500
        "a float reduction folds Ops.exec_alu's operation from its identity"
        float_reduction (fun (op, dt, terms) ->
          let t = Ops.range ~axis_type:Weak (Int 4) [ 7 ] in
          let table = Shape.param ~shape:[ Int 4 ] 0 dt in
          let fold acc v = Ops.exec_alu op dt [ acc; (v :> Dtype.const) ] in
          equal bits
            (List.fold_left fold (Ops.identity_element op dt) terms)
            (Interpreter.eval
               ~buffers:[ (0, Array.of_list terms) ]
               (over [ t ] op (Ops.index table [ t ]))));
      test "a reduction over two ranges runs over every pair" (fun () ->
          equal const (int 66)
            (Interpreter.eval
               (over [ r; s ] Add (int32 Ops.O.((r * int 3) + s)))));
      (* In float32, 2^24 + 1 rounds to 2^24 (ties to even): in the order 2^24,
         1, -2^24, 1 the sum is 1; in 2^24, -2^24, 1, 1 it is 2. *)
      test "the last range of a reduction varies fastest" (fun () ->
          let t = Ops.range ~axis_type:Weak (Int 2) [ 4 ]
          and u = Ops.range ~axis_type:Weak (Int 2) [ 5 ] in
          let table = Shape.param ~shape:[ Int 4 ] 0 Dtype.Float32 in
          let terms =
            Array.map (fun v -> `Float v) [| 16777216.; 1.; -16777216.; 1. |]
          in
          let term = Ops.index table [ Ops.O.((t * int 2) + u) ] in
          equal const (`Float 1.)
            (Interpreter.eval ~buffers:[ (0, terms) ] (over [ t; u ] Add term)));
      (* t counts below r + 1: 1 + 2 + 3 + 4 steps *)
      test "a reduction's range is evaluated where its earlier ranges are bound"
        (fun () ->
          let t = Ops.range ~axis_type:Weak (Sym Ops.O.(r + int 1)) [ 6 ] in
          equal const (int 10)
            (Interpreter.eval
               (over [ r; t ] Add (Ops.int ~dtype:Dtype.Int32 1))));
      test "a reduction reads the ranges it runs inside" (fun () ->
          equal const (int 10)
            (Interpreter.eval
               ~vars:[ ("r2", int 1) ]
               (over [ r ] Add (int32 Ops.O.(r + o)))));
      test "a reduction binds its range, whatever vars binds" (fun () ->
          equal const (int 6)
            (Interpreter.eval
               ~vars:[ ("r0", int 3) ]
               (over [ r ] Add (int32 r))));
      test "a range's end may be a variable" (fun () ->
          let n = Ops.range ~axis_type:Weak (Sym (var "n")) [ 4 ] in
          equal const (int 10)
            (Interpreter.eval ~vars:[ ("n", int 5) ] (over [ n ] Add (int32 n))));
      test "a reduction of an invalid value is invalid" (fun () ->
          equal const `Invalid
            (Interpreter.eval
               (over [ r ] Add (Shape.valid (int32 r) Ops.O.(r < int 2)))));
      test "a reduction over a node that is not a range is refused" (fun () ->
          rejects Interpreter.eval
            (Ops.v
               ~src:[ int32 r; Ops.O.(r + int 1) ]
               ~arg:(Reduce { op = Add; num_axes = 0 })
               Reduce));
    ]

(* The [n] elements of [storage] from [offset], as a vector access. *)
let vector storage offset n =
  Ops.v ~src:[ storage; offset; Ops.int n ] Op.Shrink

(* [lane k u] is lane [k] of the vector [u], by an index that is no literal, as
   a kernel's are. *)
let lane k u = Ops.v ~src:[ u; Ops.int ~dtype:Dtype.Int32 k ] Index
let int32s ns = Shape.stack (List.map (Ops.int ~dtype:Dtype.Int32) ns)

let storage =
  let table = Shape.param ~shape:[ Int 4 ] 0 Dtype.Int32 in
  let elements = [ (0, Array.map int [| 10; 11; 12; 13 |]) ] in
  let i = var "i" in
  let at v u = Interpreter.eval ~vars:[ ("i", int v) ] ~buffers:elements u in
  let gated =
    Ops.load (Ops.index table [ i ])
      [ Ops.int ~dtype:Dtype.Int32 (-1); Ops.O.(i < int 4) ]
  in
  group "storage"
    [
      test "an index reads the element at its index" (fun () ->
          equal const (int 12) (at 2 (Ops.index table [ i ])));
      test "a load is the value it loads" (fun () ->
          equal const (int 13) (at 3 (Ops.load (Ops.index table [ i ]) [])));
      test "a gated load is what it loads where its gate holds" (fun () ->
          equal const (int 13) (at 3 gated));
      (* index 7 is outside the storage: a read of it would be refused *)
      test
        "a gated load is its alternative where its gate fails, its index unread"
        (fun () -> equal const (int (-1)) (at 7 gated));
      test "an index at an invalid index reads zero" (fun () ->
          equal const (int 0)
            (at 7 (Ops.index table [ Shape.valid i Ops.O.(i < int 4) ])));
      test "an index outside the storage is refused" (fun () ->
          rejects (at 4) (Ops.index table [ i ]));
      test "an index by two integers is refused" (fun () ->
          rejects (at 0) (Ops.v ~src:[ table; i; i ] Index));
      test "storage without elements is refused" (fun () ->
          rejects
            (Interpreter.eval ~vars:[ ("i", int 0) ])
            (Ops.index table [ i ]));
      (* the store orders the read and changes no element *)
      test "storage read after other nodes is that storage" (fun () ->
          let store =
            Ops.store
              (Ops.index table [ Ops.int 2 ])
              (Ops.int ~dtype:Dtype.Int32 99)
          in
          equal const (int 12)
            (at 2 (Ops.index (Ops.after table [ store ]) [ i ])));
      test "a lane of a vector load reads its element past the offset"
        (fun () ->
          let vector = Ops.load (vector table (Ops.int 1) 2) [] in
          equal const (int 12) (at 0 (Ops.index vector [ Ops.int 1 ])));
      test "a lane of a vector load at an invalid offset reads zero" (fun () ->
          let offset = Shape.valid i Ops.O.(i < int 2) in
          let vector = Ops.load (vector table offset 2) [] in
          equal const (int 0) (at 3 (Ops.index vector [ Ops.int 0 ])));
      test
        "a lane of a gated vector load is its alternative's where its gate \
         fails" (fun () ->
          let gated =
            Ops.load
              (vector table (Ops.int 0) 2)
              [ int32s [ -1; -2 ]; Ops.O.(i < int 4) ]
          in
          equal const (int (-2)) (at 7 (lane 1 gated)));
      test "a lane outside its vector is refused" (fun () ->
          let vector = Ops.load (vector table (Ops.int 0) 2) [] in
          rejects (at 0) (Ops.index vector [ Ops.int 2 ]));
      cases "a lane of a stack is its source" ~id:string_of_int [ 0; 2 ]
        (fun k ->
          equal const
            (int (5 + k))
            (Interpreter.eval (lane k (int32s [ 5; 6; 7 ]))));
      test "a lane outside its stack is refused" (fun () ->
          rejects Interpreter.eval (lane 3 (int32s [ 5; 6; 7 ])));
      test "a lane of an operation of stacks is the operation of their lanes"
        (fun () ->
          let sum =
            Ops.cast (Ops.add (int32s [ 1; 2 ]) (int32s [ 10; 20 ])) Dtype.Int64
          in
          equal const (int 22) (Interpreter.eval (lane 1 sum)));
      (* 2.0 is 0x40000000 in binary32 *)
      test "a lane of a bit reinterpretation of a stack reinterprets its lane"
        (fun () ->
          let floats =
            Shape.stack (List.map (Ops.float ~dtype:Dtype.Float32) [ 1.; 2. ])
          in
          equal const (int 0x40000000)
            (Interpreter.eval (lane 1 (Ops.bitcast floats Dtype.Uint32))));
    ]

(* Overflows *)

let cast_overflows =
  prop "a cast overflows where its value is past its integer type, or has none"
    ~examples source (fun s ->
      let past = ref false and unrepresentable = ref false in
      each_cast s (fun from to_ v ->
          let over = overflows_of from to_ v in
          (match v with
          | `Float f when not (Float.is_finite f) ->
              unrepresentable := !unrepresentable || over
          | _ -> past := !past || over);
          equal ~msg:(cast_msg from to_ v) overflow over
            (Interpreter.overflows ~params:[ (0, v) ] (Ops.cast (x from) to_)));
      cover "a cast of a number past its integer type" !past;
      cover "a cast of a NaN or an infinity to an integer type" !unrepresentable)

(* Exact integer arithmetic *)

let committed =
  Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ]

let binary =
  Op.[ Add; Sub; Mul; Truncdiv; Truncmod; Div; Mod; Max; And; Or; Xor ]

(* Shift counts around each width, and past every one. *)
let counts = [ 0; 1; 3; 7; 8; 15; 16; 31; 32; 63; 64; 65; 256 ]

(* [expression dt depth] draws an integer expression of [dt] over the variables
   [a] and [b] of each type and the weak variable [w], each operation built as
   given ({!Ops.alu}). *)
let rec expression dt depth =
  let open Gen in
  let leaf =
    frequency
      [
        ( 4,
          map
            (fun v -> variable dt (v ^ Dtypes.alias dt))
            (of_list [ "a"; "b" ]) );
        (1, map (fun n -> Ops.int ~dtype:dt n) (int_range (-3) 3));
        (1, constant (weak "w"));
      ]
  in
  if depth = 0 then leaf
  else
    let sub = expression dt (depth - 1) in
    frequency
      [
        (1, leaf);
        ( 5,
          let+ op = of_list binary and+ a = sub and+ b = sub in
          Ops.alu a op [ b ] );
        ( 2,
          let+ op = of_list Op.[ Shl; Shr ]
          and+ a = sub
          and+ k = of_list counts in
          Ops.alu a op [ Ops.int k ] );
        ( 1,
          let+ a = sub and+ e = int_range 0 3 in
          Ops.alu a Pow [ Ops.int e ] );
        ( 1,
          let+ a = sub and+ b = sub and+ y = sub and+ z = sub in
          Ops.alu (Ops.alu a Cmplt [ b ]) Where [ y; z ] );
        ( 1,
          let* from = of_list committed in
          let+ a = expression from (depth - 1) in
          Ops.cast a dt );
      ]

(* A value of the committed integer type [dt]: its bounds and their inner
   neighbours, small values, and any 64 bits wrapped to it. *)
let value_in dt =
  let open Gen in
  let wrap z = Dtype.Value.to_z (Dtype.truncate dt (`Int z)) in
  let lo, hi = int_bounds dt in
  frequency
    [
      (1, of_list [ lo; Bigint.succ lo; Bigint.pred hi; hi ]);
      (3, map (fun n -> wrap (Bigint.of_int n)) (int_range (-8) 8));
      (2, map (fun n -> wrap (Bigint.of_int64 n)) int64);
    ]

let variables u =
  List.filter_map
    (fun n ->
      match (Ops.op n, Interpreter.name n) with
      | Param, Some name -> Some (name, Ops.dtype n)
      | _ -> None)
    (Ops.toposort ~calls:Enter u)

(* An expression of a committed type and a value of each of its variables, a
   weak one's of 64 bits. *)
let program =
  let open Gen in
  let* dt, depth = pair (of_list committed) (int_range 1 2) in
  let* u = map (fun u -> Ops.cast u dt) (expression dt depth) in
  let value (name, vdt) =
    let dt = if Dtype.equal vdt Dtype.Weak_int then Dtype.Int64 else vdt in
    map (fun v -> (name, `Int v)) (value_in dt)
  in
  let+ vars =
    List.fold_right
      (fun v rest ->
        let+ v = value v and+ rest = rest in
        v :: rest)
      (variables u) (constant [])
  in
  (u, vars)

let pp_program ppf (u, vars) =
  Format.fprintf ppf "%a@ at %a" Ops.pp u
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (n, v) ->
         Format.fprintf ppf "%s = %a" n (Testable.pp value) v))
    vars

(* [exact vars u] is [u] in unbounded integers, each operation as Op states it:
   a division by 0 is 0 and its remainder the dividend, and a left shift by a
   committed type's width or more is 0. *)
let rec exact vars u =
  let arg i = exact vars (Ops.nth u i) in
  let by_zero () = Bigint.sign (arg 1) = 0 in
  match Ops.op u with
  | Const -> Dtype.Value.to_z (literal u)
  | Param ->
      Dtype.Value.to_z (List.assoc (Option.get (Interpreter.name u)) vars)
  | Cast -> arg 0
  | Add -> Bigint.add (arg 0) (arg 1)
  | Sub -> Bigint.sub (arg 0) (arg 1)
  | Mul -> Bigint.mul (arg 0) (arg 1)
  | Truncdiv -> if by_zero () then Bigint.zero else Bigint.div (arg 0) (arg 1)
  | Truncmod -> if by_zero () then arg 0 else Bigint.rem (arg 0) (arg 1)
  | Div -> if by_zero () then Bigint.zero else Bigint.fdiv (arg 0) (arg 1)
  | Mod ->
      if by_zero () then arg 0
      else Bigint.sub (arg 0) (Bigint.mul (arg 1) (Bigint.fdiv (arg 0) (arg 1)))
  | Max -> Bigint.max (arg 0) (arg 1)
  | And -> Bigint.logand (arg 0) (arg 1)
  | Or -> Bigint.logor (arg 0) (arg 1)
  | Xor -> Bigint.logxor (arg 0) (arg 1)
  | Pow -> Bigint.pow (arg 0) (Bigint.to_int (arg 1))
  | Shr -> Bigint.shift_right (arg 0) (Bigint.to_int (arg 1))
  | Shl ->
      let dt = Ops.dtype u and k = Bigint.to_int (arg 1) in
      if List.mem dt committed && k >= Dtype.bitsize dt then Bigint.zero
      else Bigint.shift_left (arg 0) k
  | Where -> if holds vars (Ops.nth u 0) then arg 1 else arg 2
  | op -> invalid_arg (Format.asprintf "exact: %a" Op.pp op)

and holds vars c =
  Bigint.lt (exact vars (Ops.nth c 0)) (exact vars (Ops.nth c 1))

and literal u =
  match Ops.value u with
  | #Dtype.value as v -> v
  | `Invalid -> invalid_arg "exact: an invalid literal"

(* What the cases of the law reach, where they do not overflow. *)
let reached vars u =
  let nodes = Ops.toposort ~calls:Enter u in
  let has p = List.exists p nodes in
  let past_width n =
    Ops.op n = Shl
    && List.mem (Ops.dtype n) committed
    && Bigint.to_int (exact vars (Ops.nth n 1)) >= Dtype.bitsize (Ops.dtype n)
  in
  let by_zero n =
    List.mem (Ops.op n) Op.[ Truncdiv; Truncmod; Div; Mod ]
    && Bigint.sign (exact vars (Ops.nth n 1)) = 0
  in
  let at_bound (name, v) =
    name <> "w"
    && List.exists
         (fun dt ->
           let lo, hi = int_bounds dt in
           String.ends_with ~suffix:(Dtypes.alias dt) name
           && List.exists (Bigint.equal (Dtype.Value.to_z v)) [ lo; hi ])
         committed
  in
  List.iter
    (fun op ->
      cover
        (Format.asprintf "%a computed exactly" Op.pp op)
        (has (fun n -> Ops.op n = op)))
    (binary @ Op.[ Shl; Shr; Pow; Where; Cast ]);
  List.iter
    (fun dt ->
      cover
        (Format.asprintf "an exact result of %a" (Testable.pp dtype) dt)
        (Dtype.equal dt (Ops.dtype u)))
    committed;
  cover "a weak operand committed exactly" (List.mem_assoc "w" vars);
  cover "a left shift by its type's width or more" (has past_width);
  cover "a division or a remainder by 0" (has by_zero);
  cover "a variable at its type's least or greatest value"
    (List.exists at_bound vars)

(* The law Symbolic's rewrites rest on: they are checked where nothing
   overflows. *)
let exactness =
  prop "where it is false, eval computes exact integer arithmetic" ~count:2000
    (Gen.with_pp pp_program program) (fun (u, vars) ->
      let over = Interpreter.overflows ~vars u in
      cover "an expression that overflows" over;
      if not over then begin
        reached vars u;
        equal const (`Int (exact vars u)) (Interpreter.eval ~vars u)
      end)

let overflows =
  let x8 = variable Dtype.Int8 "x" in
  let w = weak "w" in
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let max64 = snd (int_bounds Dtype.Int64) in
  let x64 = variable Dtype.Int64 "x" and xu64 = variable Dtype.Uint64 "x" in
  let at u ~vars = Interpreter.overflows ~vars u in
  group "overflows"
    [
      cast_overflows;
      exactness;
      (* exact values against the types' bounds *)
      cases
        "an operation overflows where its exact value leaves its committed type"
        ~id:(fun (id, _, _, _) -> id)
        [
          ("int8 127 + 1", Ops.O.(x8 + int 1), int 127, true);
          ("int8 126 + 1", Ops.O.(x8 + int 1), int 126, false);
          ("int8 -128 * -1", Ops.O.(x8 * int (-1)), int (-128), true);
          ( "int8 -128 / -1",
            Ops.alu x8 Truncdiv [ Ops.int (-1) ],
            int (-128),
            true );
          ("int64 greatest + 1", Ops.O.(x64 + int 1), `Int max64, true);
          ( "int64 greatest - 1 + 1",
            Ops.O.(x64 + int 1),
            `Int (Bigint.pred max64),
            false );
          ("uint64 0 - 1", Ops.alu xu64 Sub [ Ops.int 1 ], int 0, true);
          ("uint64 1 - 1", Ops.alu xu64 Sub [ Ops.int 1 ], int 1, false);
        ]
        (fun (_, u, v, expected) ->
          equal overflow expected (at u ~vars:[ ("x", v) ]));
      test "an overflow inside a larger expression overflows" (fun () ->
          equal overflow true
            (at Ops.O.(x8 + int 1 < int 0) ~vars:[ ("x", int 127) ]));
      test "an overflow in the branch a selection does not pick overflows"
        (fun () ->
          let u = Ops.where Ops.O.(x8 < int 0) Ops.O.(x8 + int 1) x8 in
          equal overflow true (at u ~vars:[ ("x", int 127) ]));
      (* 1000 * 2^62 is past every 64-bit type *)
      test "a weak integer does not overflow" (fun () ->
          equal overflow false
            (at (Ops.mul w (Ops.int (1 lsl 62))) ~vars:[ ("w", int 1000) ]));
      test "the commitment of a weak operand past its peer's type overflows"
        (fun () ->
          equal overflow true
            (at (Ops.maximum w x8) ~vars:[ ("x", int 0); ("w", int 1000) ]));
      (* Ops.v holds the literal -1 as the uint8 255 *)
      test "a literal's commitment within its peer's type does not overflow"
        (fun () ->
          let u8 = variable Dtype.Uint8 "x" in
          equal overflow false
            (at (Ops.maximum (Ops.int (-1)) u8) ~vars:[ ("x", int 0) ]));
      cases "a reduction overflows where a step leaves its type"
        ~id:(fun (id, _, _) -> id)
        [
          ( "an int8 sum of four 100s",
            Ops.reduce (Ops.int ~dtype:Dtype.Int8 100) Add [ r ],
            true );
          ( "an int32 sum of 0 to 3",
            Ops.reduce (Ops.cast r Dtype.Int32) Add [ r ],
            false );
        ]
        (fun (_, u, expected) -> equal overflow expected (at u ~vars:[]));
      (* Ops.exec_alu's value of a count past the width *)
      cases
        "a left shift by its type's width or more is 0 and does not overflow"
        ~id:(fun (dt, bound, k) ->
          Format.asprintf "%a %s << %d" (Testable.pp dtype) dt
            (if bound = `Least then "least" else "greatest")
            k)
        [
          (Dtype.Uint8, `Greatest, 8);
          (Uint8, `Greatest, 256);
          (Int32, `Least, 32);
          (Int32, `Greatest, 32);
          (Uint64, `Greatest, 64);
          (Uint64, `Greatest, 256);
        ]
        (fun (dt, bound, k) ->
          let lo, hi = int_bounds dt in
          let vars = [ ("x", `Int (if bound = `Least then lo else hi)) ] in
          let u = Ops.shl (variable dt "x") (Ops.int k) in
          equal (pair const overflow)
            (int 0, false)
            (Interpreter.eval ~vars u, at u ~vars));
      test "overflows raises as eval does" (fun () ->
          rejects Interpreter.overflows (var "i"));
    ]

(* Writes *)

let writes =
  let out = Shape.param ~shape:[ Int 16 ] 0 Dtype.Int32 in
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
  and s = Ops.range ~axis_type:Weak (Int 2) [ 1 ] in
  let store ?gate index value =
    Ops.store ?gate (Ops.index out [ index ]) value
  in
  let int32 u = Ops.cast u Dtype.Int32 in
  let seven = Ops.int ~dtype:Dtype.Int32 7 in
  let writes u = Interpreter.writes u in
  let write = triple Windtrap.int Windtrap.int value in
  group "writes"
    [
      test "a store writes at each value of the ranges it runs inside"
        (fun () ->
          equal (list write)
            [ (0, 0, int 0); (0, 2, int 1); (0, 4, int 2); (0, 6, int 3) ]
            (writes (Ops.end_ (store Ops.O.(r * int 2) (int32 r)) [ r ])));
      test "a store inside two ranges writes at every pair" (fun () ->
          let u = Ops.end_ (store Ops.O.((r * int 2) + s) (int32 s)) [ r; s ] in
          equal (list write)
            (List.init 8 (fun i -> (0, i, int (i mod 2))))
            (writes u));
      (* t counts below r + 1: r = 0 writes at 0, r = 1 at 4 and 5, r = 2 at 8,
         9 and 10 *)
      test "a store's range is evaluated where the ranges before it are bound"
        (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 3) [ 2 ] in
          let t = Ops.range ~axis_type:Weak (Sym Ops.O.(r + int 1)) [ 3 ] in
          let u = Ops.end_ (store Ops.O.((r * int 4) + t) (int32 t)) [ r; t ] in
          equal (list write)
            [
              (0, 0, int 0);
              (0, 4, int 0);
              (0, 5, int 1);
              (0, 8, int 0);
              (0, 9, int 1);
              (0, 10, int 2);
            ]
            (writes u));
      test "a store where its index is invalid writes nothing" (fun () ->
          let index = Shape.valid r Ops.O.(r < int 1) in
          equal (list write)
            [ (0, 0, int 0) ]
            (writes (Ops.end_ (store index (int32 r)) [ r ])));
      test "a store of an invalid value writes nothing" (fun () ->
          let value = Shape.valid (int32 r) Ops.O.(r < int 1) in
          equal (list write)
            [ (0, 0, int 0) ]
            (writes (Ops.end_ (store r value) [ r ])));
      test "a store where its gate fails writes nothing" (fun () ->
          let u = Ops.end_ (store ~gate:Ops.O.(r < int 1) r (int32 r)) [ r ] in
          equal (list write) [ (0, 0, int 0) ] (writes u));
      (* r // 2 is 0, 0, 1, 1 *)
      test "a write at several bindings is one write" (fun () ->
          let half = Ops.alu r Truncdiv [ Ops.int 2 ] in
          equal (list write)
            [ (0, 0, int 7); (0, 1, int 7) ]
            (writes (Ops.end_ (store half seven) [ r ])));
      test "a store of several values at one element writes each" (fun () ->
          equal (list write)
            [ (0, 0, int 0); (0, 0, int 1); (0, 0, int 2); (0, 0, int 3) ]
            (writes (Ops.end_ (store (Ops.int 0) (int32 r)) [ r ])));
      test "-0. and 0. at one element are two writes" (fun () ->
          let out = Shape.param ~shape:[ Int 1 ] 0 Dtype.Float32 in
          let zero =
            Ops.where
              Ops.O.(s < int 1)
              (Ops.float ~dtype:Dtype.Float32 0.)
              (Ops.float ~dtype:Dtype.Float32 (-0.))
          in
          let u =
            Ops.end_ (Ops.store (Ops.index out [ Ops.int 0 ]) zero) [ s ]
          in
          equal (list write)
            [ (0, 0, `Float (-0.)); (0, 0, `Float 0.) ]
            (writes u));
      (* the NaN of each payload is moved from its bits, the same on every
         host *)
      test "NaNs of two payloads at one element are one write" (fun () ->
          let out = Shape.param ~shape:[ Int 1 ] 0 Dtype.Float32 in
          let nan bits =
            Ops.bitcast (Ops.const ~dtype:Dtype.Uint32 (int bits)) Dtype.Float32
          in
          let either =
            Ops.where Ops.O.(s < int 1) (nan 0x7fc00001) (nan 0x7fc00002)
          in
          let u =
            Ops.end_ (Ops.store (Ops.index out [ Ops.int 0 ]) either) [ s ]
          in
          equal (list write) [ (0, 0, `Float Float.nan) ] (writes u));
      test "writes are sorted by slot, then index" (fun () ->
          let other = Shape.param ~shape:[ Int 4 ] 1 Dtype.Int32 in
          let u =
            Ops.sink
              [
                Ops.store (Ops.index other [ Ops.int 0 ]) seven;
                store (Ops.int 3) seven;
                store (Ops.int 1) seven;
              ]
          in
          equal (list write)
            [ (0, 1, int 7); (0, 3, int 7); (1, 0, int 7) ]
            (writes u));
      test "writes binds a store's ranges by name on top of vars" (fun () ->
          let i = Ops.cast (var "i") Dtype.Int32 in
          equal (list write)
            [ (0, 0, int 5); (0, 1, int 5) ]
            (Interpreter.writes
               ~vars:[ ("r1", int 9); ("i", int 5) ]
               (Ops.end_ (store s i) [ s ])));
      test "writes raises as eval does" (fun () ->
          rejects writes (store (Ops.int 0) (Ops.cast (var "i") Dtype.Int32)));
      (* a range of no step, which binds nothing *)
      test "a range only an End lists adds no binding" (fun () ->
          let empty = Ops.range ~axis_type:Weak (Int 0) [ 2 ] in
          let u = Ops.end_ (store (Ops.int 3) seven) [ empty ] in
          equal (list write) [ (0, 3, int 7) ] (writes u));
      test "a store that reads a range through an After runs inside it"
        (fun () ->
          let empty = Ops.range ~axis_type:Weak (Int 0) [ 2 ] in
          let index = Ops.after (Ops.int 3) [ empty ] in
          equal (list write) [] (writes (store index seven)));
      test "a store's value may read storage and reduce" (fun () ->
          let table = Shape.param ~shape:[ Int 4 ] 1 Dtype.Int32 in
          let total = Ops.reduce (Ops.index table [ s ]) Add [ s ] in
          equal (list write)
            [ (0, 0, int 5) ]
            (Interpreter.writes
               ~buffers:[ (1, Array.map int [| 2; 3; 9; 9 |]) ]
               (store (Ops.int 0) total)));
      test "a store's reductions read its ranges at each of their values"
        (fun () ->
          let scaled = Ops.reduce Ops.O.(int32 r * int32 s) Add [ s ] in
          let count = Ops.reduce (int32 s) Add [ s ] in
          equal (list write)
            [ (0, 0, int 1); (0, 1, int 2); (0, 2, int 3); (0, 3, int 4) ]
            (writes (Ops.end_ (store r Ops.O.(scaled + count)) [ r ])));
      test "a reduction's range ends read the store's ranges at each value"
        (fun () ->
          let t = Ops.range ~axis_type:Weak (Sym Ops.O.(r + int 1)) [ 2 ] in
          let count = Ops.reduce (Ops.int ~dtype:Dtype.Int32 1) Add [ t; r ] in
          equal (list write)
            [ (0, 0, int 4); (0, 1, int 8); (0, 2, int 12); (0, 3, int 16) ]
            (writes (Ops.end_ (store r count) [ r ])));
      test "a store through a vector writes each lane past the offset"
        (fun () ->
          equal (list write)
            [ (0, 4, int 7); (0, 5, int 8) ]
            (writes (Ops.store (vector out (Ops.int 4) 2) (int32s [ 7; 8 ]))));
      test "a store through a vector writes no invalid lane" (fun () ->
          let lanes = Shape.stack [ seven; Ops.invalid ] in
          equal (list write)
            [ (0, 4, int 7) ]
            (writes (Ops.store (vector out (Ops.int 4) 2) lanes)));
      test "a store through a vector at an invalid offset writes nothing"
        (fun () ->
          let offset = Shape.valid Ops.O.(r * int 2) Ops.O.(r < int 1) in
          let lanes = Shape.stack [ int32 r; int32 r ] in
          equal (list write)
            [ (0, 0, int 0); (0, 1, int 0) ]
            (writes (Ops.end_ (Ops.store (vector out offset 2) lanes) [ r ])));
      test "a store through a vector of a stack of another length is refused"
        (fun () ->
          rejects writes (Ops.store (vector out (Ops.int 4) 2) (int32s [ 7 ])));
      test "a store through a vector of a value that is no stack is refused"
        (fun () -> rejects writes (Ops.store (vector out (Ops.int 4) 2) seven));
      test "a store to a destination that is no index of storage is refused"
        (fun () -> rejects writes (Ops.v ~src:[ out; seven ] Store));
    ]

let () =
  exit
    (run "Interpreter"
       [
         group "name" [ name ];
         group "eval"
           [
             leaves;
             operations;
             exec_alu;
             casts;
             selections;
             invalid;
             reductions;
             storage;
           ];
         overflows;
         writes;
       ])
