open Windtrap
open Tolk

let cpu = "CPU"
let buffer slot = Ops.new_buffer ~slot cpu 4 Dtype.Float32
let slot u = match Ops.arg u with Ops.Param p -> p.slot | _ -> -1
let buffers u =
  List.filter (fun n -> Ops.op n = Buffer) (Ops.toposort ~calls:Enter u)

let renumbered ~like ~given =
  let graph slots = Ops.sink (List.map buffer slots) in
  List.map slot (buffers (Uops.numbered_like (graph like) (graph given)))

let numbered_like =
  group "numbered_like"
    [
      (* A pass that numbers storage in another order pairs one graph's slots
         with another's, so a graph's slots can name each other. *)
      test "slots swapped between the two graphs are swapped" (fun () ->
          equal (list int) [ 4; 3 ]
            (renumbered ~like:[ 4; 3 ] ~given:[ 3; 4 ]));
      test "a slot taken by a later buffer is given once" (fun () ->
          equal (list int) [ 4; 5 ]
            (renumbered ~like:[ 4; 5 ] ~given:[ 3; 4 ]));
    ]

let () = exit (run "Uops" [ numbered_like ])
