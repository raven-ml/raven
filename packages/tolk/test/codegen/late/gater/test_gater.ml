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
let buf = Ops.param ~shape:[ Int 64 ] 0 Float32

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

let laws =
  group "laws"
    [
      prop "no access through a first index gated by invalid remains" accesses
        (fun u ->
          is_false ~msg:"an access still reads through a gated index"
            (List.exists gated_first_index (Ops.toposort ~calls:Enter (move u))));
      prop "moving the gates is idempotent" accesses
        (Law.idempotent Uops.uop move);
    ]

(* A selection of a gated load converted to another type, [where g (cast l) a],
   read where the gate fails and holds: the moved load computes the selection,
   for an alternative [a] that the load's own type holds and for one it does
   not, a subnormal, a fraction, a zero's sign and a magnitude it rounds. *)
let converted =
  let narrow = Ops.param ~shape:[ Int 64 ] 1 Float16 in
  let bytes = Ops.param ~shape:[ Int 64 ] 2 Int8 in
  let selection (buf, a) =
    let l = Ops.load (Ops.index buf [ Ops.where g (idx 0) Ops.invalid ]) [] in
    Ops.where g (Ops.cast l Float32) (Ops.const ~dtype:Float32 (`Float a))
  in
  let buffers =
    [
      (1, Array.make 64 (`Float 3.)); (2, Array.make 64 (`Int (Bigint.of_int 3)));
    ]
  in
  let eval gate u =
    Interpreter.eval
      ~vars:[ ("g", `Bool gate); ("i0", `Int Bigint.zero) ]
      ~buffers u
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
    ]
    (fun c ->
      let u = selection c in
      List.iter
        (fun gate ->
          equal ~msg:(string_of_bool gate) Dtypes.const (eval gate u)
            (eval gate (move u)))
        [ false; true ])

let pm_move_gates_from_index =
  group "pm_move_gates_from_index"
    (Golden.cases "moves.golden" (fun cell ->
         equal Uops.uop
           (case "moves_output.golden" cell)
           (move (case "moves_input.golden" cell)))
    :: converted :: List.map kernel kernels)

let () = exit (run "Tolk.Gater" [ pm_move_gates_from_index; laws ])
