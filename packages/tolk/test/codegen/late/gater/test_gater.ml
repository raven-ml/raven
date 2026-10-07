open Windtrap
open Tolk

let move u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
    (After_sources Gater.pm_move_gates_from_index)

(* The cases of an input golden are its sink's sources, in order. *)
let case file cell =
  List.nth (Ops.src (Golden.sink file)) (int_of_string (cell "src"))

(* Kernels compiled by tinygrad: padding reads through gated indices, on the CPU
   and on CUDA; a convolution with padding; a CUDA sum whose partial sums go
   through workgroup memory under a gated store. *)
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
let buf = Call.param ~shape:[ Int 64 ] 0 Float32

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

let accesses =
  let open Gen in
  let index =
    frequency
      [
        (1, map (fun n -> Plain n) (int_range 0 1));
        ( 3,
          let+ gate = int_range 0 2 and+ n = int_range 0 1 in
          Gated (gate, n) );
      ]
  in
  let access =
    let+ indices = list ~size:(int_range 1 3) index
    and+ store = bool
    and+ select = option (int_range 0 2) in
    access { indices; store; select }
  in
  with_pp
    (fun ppf u -> Format.pp_print_string ppf (Graph.to_string u))
    (map Ops.sink (list ~size:(int_range 1 3) access))

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
    (d, selection (Call.param ~shape:[ Int 64 ] 0 d) w a)
  in
  with_pp
    (fun ppf (_, u) -> Format.pp_print_string ppf (Graph.to_string u))
    selected

let laws =
  group "laws"
    [
      prop "no access through a first index gated by invalid remains" accesses
        (fun u ->
          is_false ~msg:"an access still reads through a gated index"
            (List.exists gated_first_index (Ops.toposort ~calls:Enter (move u))));
      prop "moving the gates is idempotent" accesses
        (Law.idempotent Uops.uop move);
      prop "a moved selection of a converted gated load keeps its value"
        ~count:500 converted_selections (fun (d, u) ->
          let one = Dtype.truncate d (`Int Bigint.one) in
          keeps_value ~buffers:[ (0, Array.make 64 one) ] u);
    ]

(* A selection of a gated load converted to another type, [where g (cast l) a],
   read where the gate fails and holds: the moved load computes the selection,
   for an alternative [a] that the load's own type holds and for one it does
   not, a subnormal, a fraction, a zero's sign, a magnitude it rounds, and a NaN
   or an infinity, which no integer holds. *)
let converted =
  let narrow = Call.param ~shape:[ Int 64 ] 1 Float16 in
  let bytes = Call.param ~shape:[ Int 64 ] 2 Int8 in
  let buffers =
    [
      (1, Array.make 64 (`Float 3.)); (2, Array.make 64 (`Int (Bigint.of_int 3)));
    ]
  in
  cases
    ~name:(fun (buf, a) -> Format.asprintf "%a %h" Dtype.pp (Ops.dtype buf) a)
    "a selection of a converted gated load keeps its value"
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

let pm_move_gates_from_index =
  group "pm_move_gates_from_index"
    (Golden.cases "moves.golden" (fun cell ->
         equal Uops.uop
           (case "moves_output.golden" cell)
           (move (case "moves_input.golden" cell)))
    :: converted :: List.map kernel kernels)

let () = exit (run "Tolk.Gater" [ pm_move_gates_from_index; laws ])
