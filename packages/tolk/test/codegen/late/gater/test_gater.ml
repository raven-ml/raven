open Windtrap
open Tolk

let move u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
    (After_sources Gater.pm_move_gates_from_index)

(* The cases of an input golden are its sink's sources, in order. *)
let case file cell =
  List.nth (Ops.src (Golden.sink file)) (int_of_string (cell "src"))

(* Kernels recorded by gen/codegen/late/gater.py: padding reads through gated
   indices, on the CPU and on CUDA; a convolution with padding; a CUDA sum whose
   partial sums go through workgroup memory under a gated store. *)
let kernels = [ "pad"; "pad_value"; "conv"; "pad_cuda"; "sum_group" ]

let kernel name =
  Golden.graph (name ^ "_moved.golden") (fun () ->
      move (Golden.sink (name ^ ".golden")))

(* Generated accesses: a load or a store through an index of one to three
   indices, each a variable or a variable gated by one of two gates, the load
   possibly selected under a gate. *)
type index = Plain of int | Gated of int * int
type access = { indices : index list; store : bool; select : int option }

let g = Ops.variable ~dtype:Bool "g" (`Bool false) (`Bool true)
let h = Ops.variable ~dtype:Bool "h" (`Bool false) (`Bool true)
let gates = [| g; h; Ops.logical_not g |]

let idx n =
  Ops.variable (Printf.sprintf "i%d" n) (`Int Bigint.zero)
    (`Int (Bigint.of_int 63))

let x = Ops.variable ~dtype:Float32 "x" (`Float (-1.)) (`Float 1.)
let buf = Shape.param ~shape:[ Int 64 ] 0 Float32

let access { indices; store; select } =
  let index = function
    | Plain n -> idx n
    | Gated (gate, n) -> Ops.where gates.(gate) (idx n) Ops.invalid
  in
  let at = Ops.index buf (List.map index indices) in
  if store then Ops.store at x
  else
    let l = Ops.load at [] in
    match select with Some gate -> Ops.where gates.(gate) l x | None -> l

let index =
  let open Gen in
  frequency
    [
      (1, map (fun n -> Plain n) (int_range 0 1));
      ( 3,
        let+ gate = int_range 0 2 and+ n = int_range 0 1 in
        Gated (gate, n) );
    ]

let sink_of ~indices =
  let open Gen in
  let access =
    let+ indices = list ~size:indices index
    and+ store = bool
    and+ select = option (int_range 0 2) in
    access { indices; store; select }
  in
  map Ops.sink (list ~size:(int_range 1 3) access)

let graph_pp ppf u = Format.pp_print_string ppf (Graph.to_string u)
let accesses = Gen.with_pp graph_pp (sink_of ~indices:(Gen.int_range 1 3))

(* Accesses of one index, which the interpreter evaluates, with values of the
   gates, the indices and the stored value. *)
type binding = { g : bool; h : bool; i0 : int; i1 : int }

let evaluated =
  let open Gen in
  let binding =
    let+ g = bool
    and+ h = bool
    and+ i0 = int_range 0 63
    and+ i1 = int_range 0 63 in
    { g; h; i0; i1 }
  in
  with_pp
    (fun ppf (u, b) ->
      Format.fprintf ppf "%a@.at g=%b h=%b i0=%d i1=%d" graph_pp u b.g b.h b.i0
        b.i1)
    (pair (sink_of ~indices:(constant 1)) binding)

let write = triple int int Dtypes.value
let elements = Array.init 64 (fun i -> `Float (Float.of_int i +. 0.5))

let vars b =
  [
    ("g", `Bool b.g);
    ("h", `Bool b.h);
    ("i0", `Int (Bigint.of_int b.i0));
    ("i1", `Int (Bigint.of_int b.i1));
    ("x", `Float 0.25);
  ]

(* What the accesses under [u] compute: the value of each load, and the writes
   of the stores. *)
let computed b u =
  let vars = vars b and buffers = [ (0, elements) ] in
  let loads = List.filter (fun a -> Ops.op a <> Op.Store) (Ops.src u) in
  ( List.map (Interpreter.eval ~vars ~buffers) loads,
    Interpreter.writes ~vars ~buffers u )

(* Whether an access of [u] reads through an index whose gate fails at [b]. *)
let gate_fails b u =
  List.exists
    (fun a ->
      Ops.op a = Op.Where
      && Ops.is_invalid (Ops.nth a 2)
      && Interpreter.eval ~vars:(vars b) (Ops.nth a 0) = `Bool false)
    (Ops.toposort ~calls:Enter u)

(* A load or a store through an index whose first index is gated by an invalid
   alternative, which no target renders. *)
let gated_first_index u =
  let first_gated p =
    match Ops.src p with
    | _ :: i :: _ -> Ops.op i = Op.Where && Ops.is_invalid (Ops.nth i 2)
    | _ -> false
  in
  match (Ops.op u, Ops.src u) with
  | Op.Load, [ p ] | Op.Store, [ p; _ ] ->
      (Ops.op p = Op.Index || Ops.op p = Op.Shrink) && first_gated p
  | _ -> false

(* The value of [u] where the gate [g] is [gate], the gated index [i0] reading
   [buffers]. *)
let eval ~buffers gate u =
  Interpreter.eval
    ~vars:[ ("g", `Bool gate); ("i0", `Int Bigint.zero) ]
    ~buffers u

(* [where g (cast l w) a] of a gated load [l] of [buf], converted to [w]. *)
let selection buf w a =
  let l = Ops.load (Ops.index buf [ Ops.where g (idx 0) Ops.invalid ]) [] in
  Ops.where g (Ops.cast l w) a

(* Moving the gates computes [u]'s value where the gate fails and holds. *)
let keeps_value ~buffers u =
  List.iter
    (fun gate ->
      equal ~msg:(string_of_bool gate) Dtypes.const (eval ~buffers gate u)
        (eval ~buffers gate (move u)))
    [ false; true ]

(* Selections of a gated load of any stored type converted to any other, whose
   alternative is a constant of the conversion's type, a NaN and the infinities
   drawn often, or a float conversion of a constant of any type. A NaN or an
   infinity has no integer value, and an integer of a wide type may round to an
   infinity in a narrow float. *)
let is_non_finite = function `Float f -> not (Float.is_finite f) | _ -> false

let converted_selections =
  let open Gen in
  let non_finite =
    of_list
      [ `Float Float.nan; `Float Float.infinity; `Float Float.neg_infinity ]
  in
  let const_of dt =
    let+ v =
      if Dtype.is_int dt then Dtypes.value_of dt
      else frequency [ (1, non_finite); (3, Dtypes.value_of dt) ]
    in
    Ops.const ~dtype:dt (v :> Dtype.const)
  in
  let alternative w =
    if not (Dtype.is_float w) then const_of w
    else
      frequency
        [
          (1, const_of w);
          ( 1,
            let* s = Dtypes.stored in
            let+ c = const_of s in
            Ops.cast c w );
        ]
  in
  let selected =
    let* d = Dtypes.stored in
    let* w = Dtypes.stored in
    let+ a = alternative w in
    (d, selection (Shape.param ~shape:[ Int 64 ] 0 d) w a)
  in
  with_pp
    (fun ppf (_, u) -> Format.pp_print_string ppf (Graph.to_string u))
    selected

let gated_accesses u =
  List.filter gated_first_index (Ops.toposort ~calls:Enter u)

(* A selection of a gated load converted to another type, [where g (cast l) a],
   read where the gate fails and holds: the moved load computes the selection,
   for an alternative [a] that the load's own type holds and for one it does
   not, a subnormal, a fraction, a zero's sign, a magnitude it rounds, and a NaN
   or an infinity, which no integer holds. *)
let converted =
  let narrow = Shape.param ~shape:[ Int 64 ] 1 Float16 in
  let bytes = Shape.param ~shape:[ Int 64 ] 2 Int8 in
  let buffers =
    [
      (1, Array.make 64 (`Float 3.)); (2, Array.make 64 (`Int (Bigint.of_int 3)));
    ]
  in
  cases "keeps the value of a selection of a converted gated load of"
    ~id:(fun (buf, a) ->
      Format.asprintf "%a with %h" Dtype.pp (Ops.dtype buf) a)
    [
      (narrow, 2.);
      (narrow, 0x1p-149);
      (narrow, 0x1.0001p0);
      (narrow, -0.);
      (narrow, 1e10);
      (bytes, 2.);
      (bytes, 0.5);
      (bytes, -0.);
      (bytes, Float.infinity);
      (bytes, Float.neg_infinity);
      (bytes, Float.nan);
    ]
    (fun (buf, a) ->
      keeps_value ~buffers
        (selection buf Float32 (Ops.const ~dtype:Float32 (`Float a))))

(* The constant a selection's alternative holds, through a cast or not. *)
let rec alternative_value u =
  match Ops.op u with
  | Op.Cast -> alternative_value (Ops.nth u 0)
  | _ -> Interpreter.eval u

let pm_move_gates_from_index =
  group "pm_move_gates_from_index"
    [
      group "moves each access's gate as its golden states"
        (Golden.cases "moves.golden" (fun cell ->
             equal Uops.uop
               (case "moves_output.golden" cell)
               (move (case "moves_input.golden" cell)))
        :: List.map kernel kernels);
      prop "leaves no access through a first index gated by invalid" accesses
        (fun u ->
          cover "an access reads through a gated first index"
            (gated_accesses u <> []);
          equal (list Uops.uop) [] (gated_accesses (move u)));
      prop "keeps each load's value and each store's writes" evaluated
        (fun (u, b) ->
          cover "an access reads where its gate fails" (gate_fails b u);
          equal
            (pair (list Dtypes.const) (list write))
            (computed b u)
            (computed b (move u)));
      prop "is idempotent" accesses (Law.idempotent Uops.uop move);
      converted;
      prop
        "keeps the value of a selection of a gated load of any type converted \
         to any other"
        ~count:500 converted_selections (fun (d, u) ->
          let w = Ops.dtype u and a = Ops.nth u 2 in
          cover "the alternative is not finite"
            (is_non_finite (alternative_value a));
          cover "the load's type is narrower than the selection's"
            (Dtype.itemsize d < Dtype.itemsize w);
          let one = Dtype.truncate d (`Int Bigint.one) in
          keeps_value ~buffers:[ (0, Array.make 64 one) ] u);
    ]

let () = exit (run "Tolk.Gater" [ pm_move_gates_from_index ])
