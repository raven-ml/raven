open Windtrap
open Tolk_next

let z n = `Int (Z.of_int n)
let cpu = Ops.Single "CPU"
let write = list (triple int int Dtypes.value)
let param slot = Ops.param ~device:cpu ~shape:[ Int 4 ] slot Int32

(* [kernel f] is a kernel of storage parameters [0] and [1] that stores [f] of
   parameter [1]'s element into parameter [0]'s, over four elements. *)
let kernel f =
  let r = Ops.range ~axis_type:Loop (Int 4) [ 0 ] in
  let at slot =
    Ops.index (Ops.param ~device:cpu ~shape:[ Int 4 ] slot Int32) [ r ]
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"k" ())
    [ Ops.end_ (Ops.store (at 0) (f (at 1))) [ r ] ]

let incr = kernel (fun x -> Ops.O.(x + Ops.int ~dtype:Int32 1))
let called k dst src = Ops.after dst [ Ops.call k [ dst; src ] ]
let iota = Array.init 4 z

let writes =
  group "writes"
    [
      test "a kernel writes into its call's argument" (fun () ->
          equal write
            (List.init 4 (fun i -> (0, i, z (i + 1))))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota) ]
               (Ops.sink [ called incr (param 0) (param 1) ])));
      test "a kernel reads the state its argument names" (fun () ->
          let first = called incr (param 1) (param 2) in
          equal write
            (List.init 4 (fun i -> (0, i, z (i + 1))))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota); (2, Array.make 4 (z 10)) ]
               (Ops.sink [ called incr (param 0) (param 1); first ])
            |> List.filter (fun (s, _, _) -> s = 0)));
      test "a kernel after another reads what the other wrote" (fun () ->
          let first = called incr (param 1) (param 2) in
          equal write
            (List.init 4 (fun i -> (0, i, z 11)))
            (Kernel_graphs.writes
               ~buffers:[ (2, Array.make 4 (z 9)) ]
               (Ops.sink [ called incr (param 0) first ])
            |> List.filter (fun (s, _, _) -> s = 0)));
      test "call-local storage is scratch, apart from parameters" (fun () ->
          let scratch = Ops.alloc ~slot:0 ~device:cpu [ Int 4 ] Int32 in
          let staged = called incr scratch (param 1) in
          equal write
            (List.init 4 (fun i -> (0, i, z (i + 2))))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota) ]
               (Ops.sink [ called incr (param 0) staged ])));
      test "an argument that is not storage is refused" (fun () ->
          raises_match (Exn.invalid_arg ?substring:None) (fun () ->
              Kernel_graphs.writes
                (Ops.sink
                   [
                     Ops.after (param 0)
                       [ Ops.call incr [ param 0; Ops.exp2 (param 1) ] ];
                   ])));
    ]

let () = exit (run "Kernel_graphs" [ writes ])
