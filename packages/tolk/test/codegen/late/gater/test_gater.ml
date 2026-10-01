open Windtrap
open Tolk

let move u = Ops.graph_rewrite ~ctx:() u Gater.pm_move_gates_from_index

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
            (List.exists gated_first_index (Ops.toposort (move u))));
      prop "moving the gates is idempotent" accesses
        (Law.idempotent Uops.uop move);
    ]

let pm_move_gates_from_index =
  group "pm_move_gates_from_index"
    (Golden.cases "moves.golden" (fun cell ->
         equal Uops.uop
           (case "moves_output.golden" cell)
           (move (case "moves_input.golden" cell)))
    :: List.map kernel kernels)

let () = exit (run "Tolk.Gater" [ pm_move_gates_from_index; laws ])
