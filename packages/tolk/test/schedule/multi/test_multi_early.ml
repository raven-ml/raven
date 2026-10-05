(* Tests of Tolk.Multi under LATE_ALLREDUCE=0, which the stanza sets: an
   allreduce of a value that is not sharded is expanded where multi_pm meets
   it. *)

open Windtrap
open Tolk

let uop = Uops.uop
let multi u = Ops.graph_rewrite ~calls:Skip ~ctx:() u Multi.multi_pm
let two = Ops.Multi [ "CPU:0"; "CPU:1" ]
let whole = Ops.reshape (Ops.new_buffer ~slot:1 two 8 Float32) [ Int 2; Int 4 ]
let red = Ops.allreduce whole Add two

let recorded =
  group "multi_pm › recorded"
    (List.map
       (fun name ->
         Golden.graph (name ^ "_early.golden") (fun () ->
             multi (Golden.sink (name ^ ".golden"))))
       [ "sum_sharded_axis"; "matmul_contracted"; "explicit_allreduce" ])

let expanded =
  let memory =
    [ (1, Array.init 16 (fun j -> `Float (float_of_int (j - 5)))) ]
  in
  group "multi_pm › allreduces"
    [
      test "an allreduce of a value that is not sharded is expanded" (fun () ->
          let handled = Option.get (Allreduce.handle_allreduce red) in
          equal uop (multi handled) (multi red));
      test "each device of the expansion holds the reduction" (fun () ->
          let sum j = `Float (float_of_int (j - 5 + (j + 8 - 5))) in
          let reduction = Array.init 8 sum in
          List.iteri
            (fun k v ->
              equal
                ~msg:(Printf.sprintf "device %d" k)
                (array Dtypes.const) reduction v)
            (Tensors.eval ~buffers:memory (multi red)));
    ]

let () = exit (run "Tolk.Multi, LATE_ALLREDUCE=0" [ recorded; expanded ])
