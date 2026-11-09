open Windtrap
open Tolk
open Dtypes

let x ?(slot = 0) dt = Shape.param slot dt
let int n = `Int (Bigint.of_int n)
let eval ?vars ?params ?buffers u = Interpreter.eval ?vars ?params ?buffers u
let var name = Ops.variable ~dtype:Dtype.Int32 name (int 0) (int 10)
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

let interpreter =
  group "eval"
    [
      test "a constant is its value" (fun () ->
          equal const (`Float 1.5) (eval (Ops.float ~dtype:Dtype.Float32 1.5)));
      test "a parameter is the value given to its slot" (fun () ->
          equal const (int 7)
            (eval ~params:[ (0, int 3); (1, int 7) ] (x ~slot:1 Dtype.Int32)));
      test "a parameter without a value is refused" (fun () ->
          rejects (fun () -> eval (x Dtype.Int32)));
      test "a variable is the value bound to its name" (fun () ->
          let u = Ops.sub (var "i") (var "j") in
          equal const (int (-4)) (eval ~vars:[ ("j", int 7); ("i", int 3) ] u));
      test "a variable without a value is refused, even with its slot given"
        (fun () -> rejects (fun () -> eval ~params:[ (-1, int 3) ] (var "i")));
      test "an operation rounds its result to its type" (fun () ->
          let sum = Ops.add (x Dtype.Float16) (x ~slot:1 Dtype.Float16) in
          equal const (`Float 2048.)
            (eval ~params:[ (0, `Float 2048.); (1, `Float 1.) ] sum));
      test "a cast to an integer rounds towards zero, then wraps" (fun () ->
          equal const (int 44)
            (eval
               ~params:[ (0, `Float 300.7) ]
               (Ops.cast (x Dtype.Float32) Dtype.Uint8));
          equal const (int (-3))
            (eval
               ~params:[ (0, `Float (-3.9)) ]
               (Ops.cast (x Dtype.Float32) Dtype.Int32)));
      test "a bit reinterpretation keeps the bits" (fun () ->
          equal const (int 0x3f800000)
            (eval
               ~params:[ (0, `Float 1.) ]
               (Ops.bitcast (x Dtype.Float32) Dtype.Int32)));
      test "a selection evaluates its chosen arm" (fun () ->
          let c = x Dtype.Bool in
          let u =
            Ops.where c
              (Ops.int ~dtype:Dtype.Int32 1)
              (Ops.int ~dtype:Dtype.Int32 2)
          in
          equal const (int 2) (eval ~params:[ (0, `Bool false) ] u));
      cases "a selection keeps the bits of the NaN it selects"
        ~id:(Printf.sprintf "0x%08x") [ 0xffc12345; 0x7f800001 ] (fun bits ->
          let f = Ops.bitcast (x Dtype.Uint32) Dtype.Float32 in
          let u =
            Ops.where (Ops.ne f f) f (Ops.float ~dtype:Dtype.Float32 1.)
          in
          equal const (int bits)
            (eval ~params:[ (0, int bits) ] (Ops.bitcast u Dtype.Uint32)));
      test "a cast to its operand's type keeps a signalling NaN's bits"
        (fun () ->
          let f = Ops.bitcast (x Dtype.Uint32) Dtype.Float32 in
          let u = Ops.v ~src:[ f ] ~arg:(Dtype Dtype.Float32) Cast in
          equal const (int 0x7f800001)
            (eval ~params:[ (0, int 0x7f800001) ] (Ops.bitcast u Dtype.Uint32)));
      test "a cast of a NaN to a float keeps its sign" (fun () ->
          let u = Ops.cast (x Dtype.Float32) Dtype.Float16 in
          let nan = Float.neg Float.nan in
          match eval ~params:[ (0, `Float nan) ] u with
          | `Float y ->
              is_true ~msg:"a negative NaN" (Float.is_nan y && Float.sign_bit y)
          | v -> failf "%a is not a float" (Testable.pp const) v);
      test "a node that is not arithmetic is refused" (fun () ->
          rejects (fun () -> eval (Ops.sink [ Ops.int ~dtype:Dtype.Int32 1 ])));
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
   integer type [to_] gives it an exact value outside it: an infinity or a NaN
   has none, and takes a value of [to_]. *)
let overflows_of from to_ (v : Dtype.value) =
  (not (Dtype.equal from to_))
  && Dtype.is_int to_
  &&
  match v with
  | `Float x when not (Float.is_finite x) -> false
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

(* A source value, made a value of each type as a constant of it. Every
   special float and every bound is an example of each run. *)
let source =
  Gen.with_pp (Testable.pp value)
    (Gen.frequency
       [
         (2, Gen.map (fun x -> `Float x) finite_float);
         (1, Gen.map (fun x -> `Float x) (Gen.of_list specials));
         (1, Gen.map (fun z -> `Int z) integer);
         (1, Gen.map (fun b -> `Bool b) Gen.bool);
       ])

let examples =
  List.map (fun x -> `Float x) specials @ List.map (fun z -> `Int z) bounds

let quiet_bit x = Int64.(logand (bits_of_float x) 0x0008_0000_0000_0000L) <> 0L

let casts =
  group "casts"
    [
      prop
        "a cast between any two types converts as a constant, then wraps, \
         and overflows where the constant is past the type"
        ~examples source (fun s ->
          (match s with
          | `Float f ->
              cover "a quiet NaN with a payload"
                (Float.is_nan f && quiet_bit f
                && Int64.bits_of_float f <> Int64.bits_of_float Float.nan);
              cover "a signalling NaN"
                (Float.is_nan f && not (quiet_bit f));
              cover "-0." (f = 0. && Float.sign_bit f);
              cover "infinity" (f = infinity);
              cover "-infinity" (f = neg_infinity)
          | `Int z ->
              cover "an integer type's bound or its neighbour"
                (List.exists (Bigint.equal z) bounds)
          | `Bool _ -> ());
          let past = ref false in
          List.iter
            (fun from ->
              match Dtype.const from s with
              | `Invalid -> ()
              | #Dtype.value as v ->
                  List.iter
                    (fun to_ ->
                      let msg =
                        Format.asprintf "%a of %a to %a" (Testable.pp value) v
                          (Testable.pp dtype) from (Testable.pp dtype) to_
                      in
                      let u = Ops.cast (x from) to_ and params = [ (0, v) ] in
                      equal ~msg bits (cast_of from to_ v) (eval ~params u);
                      let over = overflows_of from to_ v in
                      past := !past || over;
                      equal ~msg bool over (Interpreter.overflows ~params u))
                    stored_dtypes)
            stored_dtypes;
          cover "a cast past its integer type" !past);
      test "a cast of a float past its integer type overflows" (fun () ->
          let at v =
            Interpreter.overflows
              ~params:[ (0, `Float v) ]
              (Ops.cast (x Dtype.Float32) Dtype.Int8)
          in
          equal ~msg:"300.7" bool true (at 300.7);
          equal ~msg:"127.9" bool false (at 127.9));
    ]

let leaves =
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
  and g = Ops.special (Int 8) "gidx0" in
  group "leaves"
    [
      test "a range and a hardware index are the values bound to their names"
        (fun () ->
          equal const (int 21)
            (eval
               ~vars:[ ("r0", int 2); ("gidx0", int 1) ]
               Ops.O.((r * int 10) + g)));
      test "a range is named after its identity" (fun () ->
          equal (option string) (Some "r1_2")
            (Interpreter.name (Ops.range ~axis_type:Weak (Int 4) [ 1; 2 ])));
      test "a bound variable is its bound value, unless vars binds it"
        (fun () ->
          let b = Shape.bind (var "i") (int 2) in
          equal const (int 2) (eval b);
          equal const (int 5) (eval ~vars:[ ("i", int 5) ] b));
      test "a range or a hardware index without a value is refused" (fun () ->
          rejects (fun () -> eval r);
          rejects (fun () -> eval g));
    ]

let invalid =
  let i = var "i" in
  let gated = Shape.valid i Ops.O.(i < int 5) in
  let at v u = eval ~vars:[ ("i", int v) ] u in
  group "invalid"
    [
      test "a gated index is its value where its gate holds" (fun () ->
          equal const (int 3) (at 3 gated));
      test "a gated index is invalid where its gate fails" (fun () ->
          equal const `Invalid (at 7 gated));
      test "an operation of an invalid operand is invalid" (fun () ->
          equal const `Invalid (at 7 Ops.O.(gated * int 0));
          equal const `Invalid (at 7 Ops.O.(int 1 + gated));
          equal const `Invalid (at 7 (Ops.cast gated Dtype.Int64)));
      test "a selection by an invalid condition is invalid" (fun () ->
          equal const `Invalid
            (at 7 (Ops.where Ops.O.(gated < int 2) (Ops.int 1) (Ops.int 2))));
      test "a selection of an invalid branch it does not pick is valid"
        (fun () ->
          equal const (int 1)
            (at 7 (Ops.where Ops.O.(i < int 9) (Ops.int 1) Ops.invalid)));
    ]

let overflows =
  let x8 = Ops.variable ~dtype:Dtype.Int8 "x" (int (-128)) (int 127) in
  let w = Ops.variable "w" (int 0) (int 1000) in
  let at name v u = Interpreter.overflows ~vars:[ (name, int v) ] u in
  group "overflows"
    [
      test "an operation past its committed type overflows" (fun () ->
          is_true (at "x" 127 Ops.O.(x8 + int 1));
          is_true (at "x" (-128) Ops.O.(x8 * int (-1))));
      test "an operation within its type does not" (fun () ->
          is_false (at "x" 126 Ops.O.(x8 + int 1)));
      test "an overflow inside a larger expression is found" (fun () ->
          is_true (at "x" 127 Ops.O.(x8 + int 1 < int 0)));
      test "a weak integer never overflows" (fun () ->
          is_false (at "w" 1000 Ops.O.(w * int 1_000_000_000_000)));
      test "a cast outside its type wraps in eval and overflows" (fun () ->
          let wide = Ops.variable ~dtype:Dtype.Int32 "y" (int 0) (int 1000) in
          let narrow = Ops.cast wide Dtype.Int8 in
          equal const (int (-56)) (eval ~vars:[ ("y", int 200) ] narrow);
          is_true (at "y" 200 narrow);
          is_false (at "y" 100 narrow));
    ]

let commits =
  let x8 = Ops.variable ~dtype:Dtype.Uint8 "x" (int 0) (int 255) in
  let at v u = eval ~vars:[ ("x", int v) ] u in
  group "commits"
    [
      test "a weak integer operand takes its committed peer's type" (fun () ->
          equal const (int 255) (at 0 (Ops.maximum (Ops.int (-1)) x8));
          equal const (`Bool true) (at 100 Ops.O.(x8 < int (-1))));
      test "a weak expression that leaves its peer's type overflows" (fun () ->
          let w = Ops.variable "w" (int 0) (int 1000) in
          is_true
            (Interpreter.overflows
               ~vars:[ ("x", int 0); ("w", int 1000) ]
               (Ops.maximum Ops.O.(w * int 1) x8)));
      test "a literal holds its peer's value and does not overflow" (fun () ->
          is_false
            (Interpreter.overflows
               ~vars:[ ("x", int 0) ]
               (Ops.maximum (Ops.int (-1)) x8)));
    ]

let reductions =
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
  and s = Ops.range ~axis_type:Weak (Int 3) [ 1 ]
  and o = Ops.range ~axis_type:Weak (Int 5) [ 2 ] in
  let over rs op u = Ops.reduce u op rs in
  let int32 u = Ops.cast u Dtype.Int32 in
  group "reductions"
    [
      test "a sum adds its value at each value of its range" (fun () ->
          equal const (int 6) (eval (over [ r ] Add (int32 r))));
      test "a product and a maximum fold their operation" (fun () ->
          let one_more = int32 Ops.O.(r + int 1) in
          equal const (int 24) (eval (over [ r ] Mul one_more));
          equal const (int 4) (eval (over [ r ] Max one_more)));
      test "a reduction over two ranges runs over every pair" (fun () ->
          equal const (int 66)
            (eval (over [ r; s ] Add (int32 Ops.O.((r * int 3) + s)))));
      test "a reduction reads the ranges it runs inside" (fun () ->
          equal const (int 10)
            (eval
               ~vars:[ ("r2", int 1) ]
               (over [ r ] Add (int32 Ops.O.(r + o)))));
      test "a reduction binds its range, whatever vars binds" (fun () ->
          equal const (int 6)
            (eval ~vars:[ ("r0", int 3) ] (over [ r ] Add (int32 r))));
      test "a reduction over an empty range is its identity" (fun () ->
          let empty = Ops.range ~axis_type:Weak (Int 0) [ 3 ] in
          equal const (int 0) (eval (over [ empty ] Add (int32 empty)));
          equal const (int (-2147483648))
            (eval (over [ empty ] Max (int32 empty))));
      test "a range's end may be a variable" (fun () ->
          let n = Ops.range ~axis_type:Weak (Sym (var "n")) [ 4 ] in
          equal const (int 10)
            (eval ~vars:[ ("n", int 5) ] (over [ n ] Add (int32 n))));
      test "a reduction of an invalid value is invalid" (fun () ->
          equal const `Invalid
            (eval (over [ r ] Add (Shape.valid (int32 r) Ops.O.(r < int 2)))));
      test "a reduction over a node that is not a range is refused" (fun () ->
          rejects (fun () ->
              eval
                (Ops.v
                   ~src:[ int32 r; Ops.O.(r + int 1) ]
                   ~arg:(Reduce { op = Add; num_axes = 0 })
                   Reduce)));
      test "an integer sum past its type overflows" (fun () ->
          let big = Ops.int ~dtype:Dtype.Int8 100 in
          is_true (Interpreter.overflows (over [ r ] Add big));
          is_false (Interpreter.overflows (over [ r ] Add (int32 r))));
    ]

(* The [n] elements of [storage] from [offset], as a vector access. *)
let vector storage offset n =
  Ops.v ~src:[ storage; offset; Ops.int n ] Op.Shrink

(* [stack_lane k] reads lane [k] of the stack of 5, 6 and 7, by an index that is
   no literal, as a kernel's are. *)
let stack_lane k =
  let lanes = List.map (Ops.int ~dtype:Dtype.Int32) [ 5; 6; 7 ] in
  Ops.v ~src:[ Shape.stack lanes; Ops.int ~dtype:Dtype.Int32 k ] Index

let storage =
  let table = Shape.param ~shape:[ Int 4 ] 0 Dtype.Int32 in
  let elements = [ (0, Array.map int [| 10; 11; 12; 13 |]) ] in
  let i = var "i" in
  let at v u = eval ~vars:[ ("i", int v) ] ~buffers:elements u in
  group "storage"
    [
      test "an index reads the element at its index" (fun () ->
          equal const (int 12) (at 2 (Ops.index table [ i ])));
      test "a load is the value it loads" (fun () ->
          equal const (int 13) (at 3 (Ops.load (Ops.index table [ i ]) [])));
      test
        "a gated load reads where its gate holds, and is its alternative \
         elsewhere" (fun () ->
          let gated =
            Ops.load (Ops.index table [ i ])
              [ Ops.int ~dtype:Dtype.Int32 (-1); Ops.O.(i < int 4) ]
          in
          equal const ~msg:"at 3" (int 13) (at 3 gated);
          equal const ~msg:"at 7, outside the storage" (int (-1)) (at 7 gated));
      test "an index at an invalid index reads zero" (fun () ->
          equal const (int 0)
            (at 7 (Ops.index table [ Shape.valid i Ops.O.(i < int 4) ])));
      test "an index outside the storage is refused" (fun () ->
          rejects (fun () -> at 4 (Ops.index table [ i ])));
      test "storage without elements is refused" (fun () ->
          rejects (fun () ->
              eval ~vars:[ ("i", int 0) ] (Ops.index table [ i ])));
      test "a lane of a vector load reads its element past the offset"
        (fun () ->
          let vector = Ops.load (vector table (Ops.int 1) 2) [] in
          equal const (int 12) (at 0 (Ops.index vector [ Ops.int 1 ])));
      test "a lane of a vector load at an invalid offset reads zero" (fun () ->
          let offset = Shape.valid i Ops.O.(i < int 2) in
          let vector = Ops.load (vector table offset 2) [] in
          equal const (int 0) (at 3 (Ops.index vector [ Ops.int 0 ])));
      test "a lane of a stack is its source" (fun () ->
          equal const (int 5) (eval (stack_lane 0));
          equal const (int 7) (eval (stack_lane 2)));
      test "a lane outside its stack is refused" (fun () ->
          rejects (fun () -> eval (stack_lane 3)));
      test "a lane of an operation of stacks is the operation of their lanes"
        (fun () ->
          let stack ns = Shape.stack (List.map (Ops.int ~dtype:Dtype.Int32) ns) in
          let sum =
            Ops.cast (Ops.add (stack [ 1; 2 ]) (stack [ 10; 20 ])) Dtype.Int64
          in
          equal const (int 22)
            (eval (Ops.v ~src:[ sum; Ops.int ~dtype:Dtype.Int32 1 ] Index)));
      test "a lane outside its vector is refused" (fun () ->
          let vector = Ops.load (vector table (Ops.int 0) 2) [] in
          rejects (fun () -> at 0 (Ops.index vector [ Ops.int 2 ])));
    ]

let kernels =
  let out = Shape.param ~shape:[ Int 16 ] 0 Dtype.Int32 in
  let r = Ops.range ~axis_type:Weak (Int 4) [ 0 ]
  and s = Ops.range ~axis_type:Weak (Int 2) [ 1 ] in
  let store ?gate index value =
    Ops.store ?gate (Ops.index out [ index ]) value
  in
  let int32 u = Ops.cast u Dtype.Int32 in
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
          equal Windtrap.int 8 (List.length (writes u)));
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
      test "a store repeated with the same value is one write" (fun () ->
          let l = Ops.range ~axis_type:Loop (Int 4) [ 0 ] in
          equal (list write)
            [ (0, 3, int 7) ]
            (writes
               (Ops.end_
                  (store (Ops.int 3) (Ops.int ~dtype:Dtype.Int32 7))
                  [ l ])));
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
          let lanes = Shape.stack [ int32 (Ops.int 7); int32 (Ops.int 8) ] in
          equal (list write)
            [ (0, 4, int 7); (0, 5, int 8) ]
            (writes (Ops.store (vector out (Ops.int 4) 2) lanes)));
      test "a store through a vector of a stack of another length is refused"
        (fun () ->
          let lanes = Shape.stack [ int32 (Ops.int 7) ] in
          rejects (fun () ->
              writes (Ops.store (vector out (Ops.int 4) 2) lanes)));
      test "a store through a vector at an invalid offset writes nothing"
        (fun () ->
          let offset = Shape.valid Ops.O.(r * int 2) Ops.O.(r < int 1) in
          let lanes = Shape.stack [ int32 r; int32 r ] in
          equal (list write)
            [ (0, 0, int 0); (0, 1, int 0) ]
            (writes (Ops.end_ (Ops.store (vector out offset 2) lanes) [ r ])));
    ]

let () =
  exit
    (run "Interpreter"
       [
         interpreter;
         casts;
         leaves;
         invalid;
         overflows;
         commits;
         reductions;
         storage;
         kernels;
       ])
