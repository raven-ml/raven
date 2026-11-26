(* The states and views of storage a schedule's calls read and write, run on
   the host and on each GPU present (Schedule): a loop's call reads each run
   of a state it writes as the trip starts, a call compiled on its own reads
   and writes views of storage at their elements, and a store through bytes
   viewed as another type writes the value's bits. Each program is scheduled
   by Schedule.create_linear_with_vars, linked and run by Tolk_engine, and
   its buffers read back. *)

open Windtrap
open Tolk
module Engine = Tolk_engine
module Buffer = Rig.Buffer

let host = Rig.host
let floats xs = Array.map (fun x -> `Float x) xs

let value =
  let close = Testable.float_rel ~rel:1e-5 ~abs:1e-6 in
  Testable.make ~pp:(Testable.pp Dtypes.value) ~equal:(fun v0 v1 ->
      match (v0, v1) with
      | `Float x, `Float y ->
          (Float.is_nan x && Float.is_nan y) || Testable.equal close x y
      | v0, v1 -> Testable.equal Dtypes.value v0 v1)

let values = array value

(* The devices the programs name: CPU is the host, and CPU:1 a software device,
   or in a GPU's tests that GPU ([on_gpu]). *)
let device name =
  match List.assoc_opt name (Run.devices ()) with
  | Some d -> d
  | None -> invalid_arg ("no device is named " ^ name)

let devices = device

(* [linear] compiled for the targets of [devices]. *)
let compile ?(devices = devices) linear =
  Realize.compile_linear ~targets:(fun n -> Run.target (devices n)) linear

let on_gpu kind name = match name with "CPU:1" -> Gpus.gpu kind | n -> device n

(* A view of [n] bytes of [b] whose first byte lies [phase] bytes past a
   16-byte boundary. *)
let view_at_phase b phase n =
  let rec first j =
    if Buffer.aligned b ~at:(j - phase) 16 then j else first (j + 1)
  in
  Buffer.view b ~first:(first 0) ~length:n

(* [n] bytes of a buffer of [d] whose first byte lies [phase] bytes past a
   16-byte boundary, holding [bytes] where given. *)
let at_phase ?bytes d phase n =
  let v = view_at_phase (Buffer.create d (n + 15)) phase n in
  Option.iter (fun src -> Buffer.copy ~src ~dst:v) bytes;
  v

(* A copy of [b] 4 bytes past a 16-byte boundary. *)
let off_phase b = at_phase ~bytes:b (Buffer.device b) 4 (Buffer.length b)

(* [linear] with the flag [f] of its loop first copied from a new buffer of [d],
   which holds the flag's first value, and that buffer. A loop's flag is storage
   the schedule makes, which no run binds. *)
let flag_from d f linear =
  let init = Ops.new_buffer d 1 Bool in
  (init, Ops.v Op.Linear ~src:(Shape.store_call f init :: Ops.src linear))

(* Three trips of a call that stores row 1 of a buffer of rows of four, plus
   one, into row [r]: the fixed read sees the state each trip starts from, so
   the third trip reads what the second wrote. *)
let reads_what_its_loop_wrote () =
  let d = "CPU" in
  let k = 4 in
  let p slot = Shape.param ~shape:[ Int k ] ~device:d slot Float32 in
  let body = Ops.sink [ Ops.store (p 0) Ops.O.(p 1 + float 1.) ] in
  let b = Ops.new_buffer d (4 * k) Float32 in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let moving =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let fixed = Shape.shrink b [ Some (Int k, Int (2 * k)) ] in
  let e = Ops.end_ (Ops.call ~precompile:true body [ moving; fixed ]) [ r ] in
  let linear, vars =
    Schedule.create_linear_with_vars
      ~cores:Cores.host (Ops.sink [ Ops.after b [ e ] ])
  in
  let given = Array.init (4 * k) Float.of_int in
  let buffer = Run.buffer (devices d) Float32 (floats given) in
  let s = Engine.link ~devices ~bound:[ (b, buffer) ] (compile linear) in
  Engine.run ~vars s [||];
  let want =
    [| 5.; 6.; 7.; 8.; 5.; 6.; 7.; 8.; 6.; 7.; 8.; 9.; 12.; 13.; 14.; 15. |]
  in
  equal values (floats want) (Run.values Float32 buffer)

(* A loop's call, reading a fixed run off its parameter's start, reads the run
   as each trip starts: rows of four, the run from element 6 at byte 24. *)
let reads_off_start_each_trip () =
  let d = "CPU" and k = 4 in
  let p slot = Shape.param ~shape:[ Int k ] ~device:d slot Float32 in
  let body = Ops.sink [ Ops.store (p 0) Ops.O.(p 1 + float 1.) ] in
  let b = Ops.new_buffer d (4 * k) Float32 in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let moving =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let fixed = Shape.shrink b [ Some (Int 6, Int 10) ] in
  let e = Ops.end_ (Ops.call ~precompile:true body [ moving; fixed ]) [ r ] in
  let linear, vars =
    Schedule.create_linear_with_vars
      ~cores:Cores.host (Ops.sink [ Ops.after b [ e ] ])
  in
  let given = Array.init (4 * k) Float.of_int in
  let buffer = Run.buffer (devices d) Float32 (floats given) in
  let s = Engine.link ~devices ~bound:[ (b, buffer) ] (compile linear) in
  Engine.run ~vars s [||];
  (* Each trip reads elements 6..9 as it starts: 6,7,8,9, then 6,7,8,9 less
     the row just stored over 6,7 by trip 1. *)
  let want =
    [| 7.; 8.; 9.; 10.; 7.; 8.; 9.; 10.; 10.; 11.; 9.; 10.; 12.; 13.; 14.; 15. |]
  in
  equal values (floats want) (Run.values Float32 buffer)

(* A back edge's call reads a fixed run as each trip starts, as a range's does:
   three trips while the flag holds, row 1 read each trip. *)
let backedge_reads_each_trip () =
  let d = "CPU" and k = 4 in
  let p slot dt = Shape.param ~shape:[ Int k ] ~device:d slot dt in
  let index = Shape.param ~addrspace:Alu 3 Int32 in
  let body =
    Ops.sink
      [
        Ops.store (p 0 Float32) Ops.O.(p 1 Float32 + float 1.);
        Ops.store
          (Shape.param ~shape:[ Int 1 ] ~device:d 2 Bool)
          (Shape.reshape Ops.O.(index + int 1 < int 3) [ Int 1 ]);
      ]
  in
  let b = Ops.new_buffer d (4 * k) Float32 in
  let f = Ops.new_buffer d 1 Bool in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let moving =
    Shape.shrink b
      [ Some (Sym Ops.O.(r * int k), Sym Ops.O.((r * int k) + int k)) ]
  in
  let fixed = Shape.shrink b [ Some (Int k, Int (2 * k)) ] in
  let call = Ops.call ~precompile:true body [ moving; fixed; f; r ] in
  let e = Ops.backedge call ~loop:r ~cond:f in
  let linear, vars =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink [ Ops.after b [ e ]; Ops.after f [ e ] ])
  in
  let f_init, linear = flag_from d f linear in
  let given = Array.init (4 * k) Float.of_int in
  let buffer = Run.buffer (devices d) Float32 (floats given) in
  let flag = Run.buffer (devices d) Bool [| `Bool true |] in
  let s =
    Engine.link ~devices ~bound:[ (b, buffer); (f_init, flag) ] (compile linear)
  in
  Engine.run ~vars s [||];
  let want =
    [| 5.; 6.; 7.; 8.; 5.; 6.; 7.; 8.; 6.; 7.; 8.; 9.; 12.; 13.; 14.; 15. |]
  in
  equal values (floats want) (Run.values Float32 buffer)

(* The writes a schedule leaves in [bufs], each given its values. *)
let leaves bufs sink =
  let linear, vars = Schedule.create_linear_with_vars ~cores:Cores.host sink in
  let run xs = Run.buffer (devices "CPU") Float32 (floats xs) in
  let bound = List.map (fun (b, xs) -> (b, run xs)) bufs in
  Engine.run ~vars (Engine.link ~devices ~bound (compile linear)) [||];
  List.map (fun (_, x) -> Run.values Float32 x) bound

let p4 slot = Shape.param ~shape:[ Int 4 ] ~device:"CPU" slot Float32
let of_ints f n = floats (Array.init n (fun i -> f i))

let row4 b r =
  Shape.shrink b
    [ Some (Sym Ops.O.(r * int 4), Sym Ops.O.((r * int 4) + int 4)) ]

(* A call given [x] to write and [x1] to read, listed over [x1]: one state at
   both, which it reads as it was before the call, as [memmove] reads. *)
let older_write_reads_before () =
  let x = Ops.new_buffer "CPU" 64 Float32 in
  let x1 = Ops.after x [ Ops.store x Ops.O.(x * float 2.) ] in
  let p slot = Shape.param ~shape:[ Int 64 ] ~device:"CPU" slot Float32 in
  let body = Ops.sink [ Ops.store (p 0) (Shape.flip (p 1) [ 0 ]) ] in
  let k = Ops.call ~precompile:true body [ x; x1 ] in
  let got =
    leaves [ (x, Array.init 64 Float.of_int) ] (Ops.sink [ Ops.after x1 [ k ] ])
  in
  equal (list values) [ of_ints (fun i -> Float.of_int (2 * (63 - i))) 64 ] got

(* A kernel writing [x] and [y], listed over [y] and over a later state of
   [x], in either listing. *)
let listed_two_storages_either_order () =
  let run order =
    let x = Ops.new_buffer "CPU" 4 Float32
    and y = Ops.new_buffer "CPU" 4 Float32 in
    let x1 = Ops.after x [ Ops.store x Ops.O.(x * float 2.) ] in
    let body =
      Ops.sink
        [ Ops.store (p4 0) Ops.O.(p4 0 + float 1.); Ops.store (p4 1) (Ops.float 7.) ]
    in
    let k = Ops.call ~precompile:true body [ x; y ] in
    let afters = [ Ops.after y [ k ]; Ops.after x1 [ k ] ] in
    leaves
      [ (x, Array.init 4 Float.of_int); (y, Array.make 4 0.) ]
      (Ops.sink (order afters))
  in
  let want =
    [ of_ints (fun i -> Float.of_int ((2 * i) + 1)) 4; floats (Array.make 4 7.) ]
  in
  equal (list values) want (run Fun.id);
  equal (list values) want (run List.rev)

(* A call reading storage it writes off its parameter's start, no [After] of
   that storage listing it: it still reads a copy at its parameter's start. *)
let off_start_read_no_after () =
  let x = Ops.new_buffer ~phase:4 "CPU" 4 Float32
  and y = Ops.new_buffer "CPU" 4 Float32 in
  let p ?phase slot =
    Shape.param ~shape:[ Int 4 ] ~device:"CPU" ?phase slot Float32
  in
  let body =
    Ops.sink [ Ops.store (p 0) (p 2); Ops.store (p ~phase:4 1) (Ops.float 9.) ]
  in
  let k = Ops.call ~precompile:true body [ y; x; x ] in
  let linear, _ =
    Schedule.create_linear_with_vars
      ~cores:Cores.host (Ops.sink [ Ops.after y [ k ] ])
  in
  let member e =
    if Shape.is_launch (Ops.nth e 0) then Shape.launch_members e else [ e ]
  in
  List.iter
    (fun c ->
      List.iteri
        (fun i a ->
          match Shape.param_start (Ops.nth c 0) i with
          | Some param ->
              satisfies ~claim:"a start its parameter takes" (pair int int)
                (Shape.fits_start ~param) (Shape.storage_phase a)
          | None -> ())
        (Ops.src_without_body c))
    (List.concat_map member (Ops.src linear));
  let out = Run.buffer host Float32 (floats (Array.make 4 0.)) in
  let given = off_phase (Run.buffer host Float32 (floats [| 1.; 2.; 3.; 4. |])) in
  Engine.run
    (Engine.link ~devices ~bound:[ (y, out); (x, given) ] (compile linear))
    [||];
  equal values (floats [| 1.; 2.; 3.; 4. |]) (Run.values Float32 out)

(* Runs of one other storage, listed in an after of [x], land as they do listed
   in that storage's own, marked or not; [x] is as it was. *)
let foreign_runs_land () =
  let check write =
    let x = Ops.new_buffer "CPU" 4 Float32
    and y = Ops.new_buffer "CPU" 4 Float32
    and z = Ops.new_buffer "CPU" 4 Float32 in
    let run a b = Shape.shrink y [ Some (Int a, Int b) ] in
    let ss =
      [ Ops.store (run 0 2) (Ops.float 7.); Ops.store (run 2 4) (Ops.float 9.) ]
    in
    let xo = write x ss in
    leaves
      [
        (x, Array.init 4 Float.of_int);
        (y, Array.make 4 0.);
        (z, Array.make 4 0.);
      ]
      (Ops.sink
         [ Ops.after z [ Ops.store z Ops.O.(xo + Ops.float 1.) ]; write y ss ])
    |> equal (list values)
         [
           of_ints Float.of_int 4;
           floats [| 7.; 7.; 9.; 9. |];
           of_ints (fun i -> Float.of_int (i + 1)) 4;
         ]
  in
  check Ops.after;
  check Ops.after_disjoint

(* A loop's fixed reads of the [After] state it writes. [b] holds [i] at [i],
   and [b1], its state after [b * 2], holds [2 i]; three trips store a run of
   [b1], plus one, into row [r]: each trip reads the run as it starts, the
   store that made [b1] and earlier trips' writes landed. *)
let op_is_loop e = Ops.op e = Op.End || Ops.op e = Op.Backedge

let doubled () =
  let b = Ops.new_buffer "CPU" 16 Float32 in
  (b, Ops.after b [ Ops.store b Ops.O.(b * float 2.) ])

let given16 = Array.init 16 Float.of_int
let run_at lo u = Shape.shrink u [ Some (Int lo, Int (lo + 4)) ]
let add1_body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 1 + float 1.) ]
let loop3 () = Ops.range ~axis_type:Loop (Int 3) [ 100 ]

let doubled_loop lo =
  let b, b1 = doubled () in
  let r = loop3 () in
  let c = Ops.call ~precompile:true add1_body [ row4 b1 r; run_at lo b1 ] in
  (b, Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ])

let leaves1 b sink = List.hd (leaves [ (b, given16) ] sink)

(* The calls the loop of [linear] runs each trip, its call and the copies made
   as a trip starts: one call, one copy of the state, and no copy of it
   before the loop, as the top level's other entry is the store. *)
let loop_trip linear =
  let trip e =
    if op_is_loop e then
      Some
        (if Ops.op (Ops.nth e 0) = Op.Linear then
           List.length (Ops.src (Ops.nth e 0))
         else 1)
    else None
  in
  (List.length (Ops.src linear), List.find_map trip (Ops.src linear))

let reads_after_run_each_trip () =
  let b, sink = doubled_loop 4 in
  equal values
    (floats
       [| 9.; 11.; 13.; 15.; 9.; 11.; 13.; 15.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (leaves1 b sink);
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host sink in
  equal (pair int (option int)) ~msg:"a store, then a loop of a copy and a call"
    (2, Some 2) (loop_trip linear)

let reads_after_run_off_start () =
  let b, sink = doubled_loop 6 in
  equal values
    (floats
       [| 13.; 15.; 17.; 19.; 13.; 15.; 17.; 19.; 18.; 20.; 17.; 19.; 24.; 26.;
          28.; 30. |])
    (leaves1 b sink)

(* Two runs of the state at other offsets of one alignment. *)
let reads_after_run_at_offsets () =
  let check lo want =
    let b, sink = doubled_loop lo in
    equal values (floats want) (leaves1 b sink)
  in
  check 4
    [| 9.; 11.; 13.; 15.; 9.; 11.; 13.; 15.; 10.; 12.; 14.; 16.; 24.; 26.;
       28.; 30. |];
  check 8
    [| 17.; 19.; 21.; 23.; 17.; 19.; 21.; 23.; 17.; 19.; 21.; 23.; 24.; 26.;
       28.; 30. |]

(* The loop's call is compiled on its own, or inlined unless [compiled]. *)
let reads_after_run_backedge () =
  let check ~compiled lo want =
  let d = "CPU" and k = 4 in
  let p slot dt = Shape.param ~shape:[ Int k ] ~device:d slot dt in
  let index = Shape.param ~addrspace:Alu 3 Int32 in
  let body =
    Ops.sink
      [
        Ops.store (p 0 Float32) Ops.O.(p 1 Float32 + float 1.);
        Ops.store
          (Shape.param ~shape:[ Int 1 ] ~device:d 2 Bool)
          (Shape.reshape Ops.O.(index + int 1 < int 3) [ Int 1 ]);
      ]
  in
  let b, b1 = doubled () in
  let f = Ops.new_buffer d 1 Bool in
  let r = loop3 () in
  let call =
    Ops.call ~precompile:compiled body [ row4 b1 r; run_at lo b1; f; r ]
  in
  let e = Ops.backedge call ~loop:r ~cond:f in
  let linear, vars =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink [ Ops.after b1 [ e ]; Ops.after f [ e ] ])
  in
  let f_init, linear = flag_from d f linear in
  let buffer = Run.buffer (devices d) Float32 (floats given16) in
  let flag = Run.buffer (devices d) Bool [| `Bool true |] in
  let s =
    Engine.link ~devices ~bound:[ (b, buffer); (f_init, flag) ] (compile linear)
  in
  Engine.run ~vars s [||];
  equal values ~msg:(if compiled then "compiled" else "inlined") (floats want)
    (Run.values Float32 buffer)
  in
  List.iter
    (fun compiled ->
      check ~compiled 4
        [| 9.; 11.; 13.; 15.; 9.; 11.; 13.; 15.; 10.; 12.; 14.; 16.; 24.; 26.;
           28.; 30. |];
      check ~compiled 6
        [| 13.; 15.; 17.; 19.; 13.; 15.; 17.; 19.; 18.; 20.; 17.; 19.; 24.;
           26.; 28.; 30. |])
    [ true; false ]

(* The state read through a join, which waits for [x1] and holds [b1]. *)
let reads_after_run_through_join () =
  let b, b1 = doubled () in
  let x = Ops.new_buffer "CPU" 4 Float32 in
  let x1 = Ops.after x [ Ops.store x (Ops.float 7.) ] in
  let j = Ops.after b1 [ x1 ] in
  let r = loop3 () in
  let c = Ops.call ~precompile:true add1_body [ row4 b1 r; run_at 4 j ] in
  let got =
    leaves
      [ (b, given16); (x, Array.make 4 0.) ]
      (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ]; x1 ])
  in
  equal values
    (floats
       [| 9.; 11.; 13.; 15.; 9.; 11.; 13.; 15.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (List.hd got);
  equal values (floats (Array.make 4 7.)) (List.nth got 1)

(* The state two stores deep: [b2] holds [2 i + 1]. *)
let reads_after_run_chain () =
  let b, b1 = doubled () in
  let b2 = Ops.after b1 [ Ops.store b1 Ops.O.(b1 + float 1.) ] in
  let r = loop3 () in
  let c = Ops.call ~precompile:true add1_body [ row4 b2 r; run_at 4 b2 ] in
  let sink = Ops.sink [ Ops.after b2 [ Ops.end_ c [ r ] ] ] in
  equal values
    (floats
       [| 10.; 12.; 14.; 16.; 10.; 12.; 14.; 16.; 11.; 13.; 15.; 17.; 25.; 27.;
          29.; 31. |])
    (leaves1 b sink);
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host sink in
  equal (pair int (option int)) ~msg:"two stores, then a loop of a copy and a call"
    (3, Some 2) (loop_trip linear)

(* Two runs of the state, read each trip. *)
let reads_two_after_runs () =
  let b, b1 = doubled () in
  let r = loop3 () in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 1 + p4 2) ] in
  let c = Ops.call ~precompile:true body [ row4 b1 r; run_at 4 b1; run_at 8 b1 ] in
  equal values
    (floats
       [| 24.; 28.; 32.; 36.; 24.; 28.; 32.; 36.; 40.; 46.; 52.; 58.; 24.; 26.;
          28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))

(* A loop naming the write by the older state, listed over the later one,
   reads the later state's run each trip. *)
let reads_after_run_named_older () =
  let b, b1 = doubled () in
  let r = loop3 () in
  let c = Ops.call ~precompile:true add1_body [ row4 b r; run_at 4 b1 ] in
  equal values
    (floats
       [| 9.; 11.; 13.; 15.; 9.; 11.; 13.; 15.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))

(* A run of an older state keeps what the graph names, stored once: 4..7 of
   [b] as given. *)
let reads_older_run () =
  let b, b1 = doubled () in
  let r = loop3 () in
  let c = Ops.call ~precompile:true add1_body [ row4 b1 r; run_at 4 b ] in
  equal values
    (floats
       [| 5.; 6.; 7.; 8.; 5.; 6.; 7.; 8.; 5.; 6.; 7.; 8.; 24.; 26.; 28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))

(* The state's run read as its own type: rows of the state, through a bitcast
   the call takes back, read each trip as [floats] in test one are. *)
let reads_after_bitcast_run () =
  let b, b1 = doubled () in
  let r = loop3 () in
  let p slot dt = Shape.param ~shape:[ Int 4 ] ~device:"CPU" slot dt in
  let body =
    Ops.sink
      [ Ops.store (p 0 Float32) Ops.O.(Ops.bitcast (p 1 Int32) Float32 + float 1.) ]
  in
  let bits = Ops.bitcast (run_at 4 b1) Int32 in
  let c = Ops.call ~precompile:true body [ row4 b1 r; bits ] in
  equal values
    (floats
       [| 9.; 11.; 13.; 15.; 9.; 11.; 13.; 15.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))


(* A store through the node a call reads: the read takes a copy, the store
   lands in the storage. *)
let store_through_read_run_lands () =
  let b = Ops.new_buffer "CPU" 8 Float32 and x = Ops.new_buffer "CPU" 4 Float32 in
  let v = Shape.shrink b [ Some (Int 2, Int 6) ] in
  let s1 = Ops.after b [ Ops.store v Ops.O.(v + float 2.) ] in
  let c = Ops.call ~precompile:true add1_body [ x; v ] in
  let got =
    leaves
      [ (b, Array.init 8 Float.of_int); (x, Array.make 4 0.) ]
      (Ops.sink [ s1; Ops.after x [ c ] ])
  in
  equal (list values)
    [ floats [| 0.; 1.; 4.; 5.; 6.; 7.; 6.; 7. |]; floats [| 3.; 4.; 5.; 6. |] ]
    got

(* The same through a reshape of the run: the view stored through hides the
   node staged. *)
let store_through_reshape_of_read_lands () =
  let b = Ops.new_buffer "CPU" 8 Float32 and x = Ops.new_buffer "CPU" 4 Float32 in
  let v = Shape.shrink b [ Some (Int 2, Int 6) ] in
  let w = Shape.reshape v [ Int 2; Int 2 ] in
  let s1 = Ops.after b [ Ops.store w Ops.O.(w + float 2.) ] in
  let c = Ops.call ~precompile:true add1_body [ x; v ] in
  let got =
    leaves
      [ (b, Array.init 8 Float.of_int); (x, Array.make 4 0.) ]
      (Ops.sink [ s1; Ops.after x [ c ] ])
  in
  equal (list values)
    [ floats [| 0.; 1.; 4.; 5.; 6.; 7.; 6.; 7. |]; floats [| 3.; 4.; 5.; 6. |] ]
    got

(* A loop reading the run a state stores through, of the state under the
   store: the run stays as it was, and the store still lands. *)
let loop_reads_store_target_older () =
  let b = Ops.new_buffer "CPU" 16 Float32 in
  let s0 = Ops.after b [ Ops.store b Ops.O.(b + float 1.) ] in
  let v = Shape.shrink s0 [ Some (Int 2, Int 6) ] in
  let s1 = Ops.after s0 [ Ops.store v Ops.O.(v + float 2.) ] in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 100 ] in
  let row =
    Shape.shrink s1
      [ Some (Sym Ops.O.((r * int 4) + int 4), Sym Ops.O.((r * int 4) + int 8)) ]
  in
  let c = Ops.call ~precompile:true add1_body [ row; v ] in
  equal values
    (floats
       [| 1.; 2.; 5.; 6.; 4.; 5.; 6.; 7.; 4.; 5.; 6.; 7.; 13.; 14.; 15.; 16. |])
    (leaves1 b (Ops.sink [ Ops.after s1 [ Ops.end_ c [ r ] ] ]))

(* A whole state's store that reads a sibling lands after it, and the state
   holds it. *)
let sibling_a_state_reads_is_part_of_it () =
  let b = Ops.new_buffer "CPU" 16 Float32 and x = Ops.new_buffer "CPU" 4 Float32 in
  let low = Shape.shrink b [ Some (Int 0, Int 8) ] in
  let high = Shape.shrink b [ Some (Int 8, Int 16) ] in
  let sa = Ops.after low [ Ops.store low (Ops.float 1.) ] in
  let sb = Ops.after b [ Ops.store high sa ] in
  let got =
    leaves
      [ (b, given16); (x, Array.make 4 0.) ]
      (Ops.sink [ sb; Ops.after x [ Ops.store x (run_at 12 sb) ] ])
  in
  equal (list values) [ floats (Array.make 16 1.); floats (Array.make 4 1.) ] got

let int_bits_of_10 = 0x41200000

let store_through_run_of_a_bitcast_state () =
  let b = Ops.new_buffer "CPU" 8 Float32 in
  let s1 = Ops.after b [ Ops.store b Ops.O.(b + float 1.) ] in
  let c = Ops.bitcast s1 Int32 in
  let run = Shape.shrink c [ Some (Int 0, Int 4) ] in
  let a = Ops.after c [ Ops.store run (Ops.int ~dtype:Int32 int_bits_of_10) ] in
  equal values
    (floats [| 10.; 10.; 10.; 10.; 5.; 6.; 7.; 8. |])
    (List.hd
       (leaves
          [ (b, Array.init 8 Float.of_int) ]
          (Ops.sink [ a ])))

let weak_value_through_a_run_of_a_bitcast () =
  let b = Ops.new_buffer "CPU" 8 Float32 in
  let c = Ops.bitcast b Int32 in
  let v = Shape.shrink c [ Some (Int 0, Int 4) ] in
  let a = Ops.after v [ Ops.store v (Ops.int int_bits_of_10) ] in
  equal values
    (floats [| 10.; 10.; 10.; 10.; 4.; 5.; 6.; 7. |])
    (List.hd (leaves [ (b, Array.init 8 Float.of_int) ] (Ops.sink [ a ])))

(* One constant stored at two states of one storage, each through a bitcast:
   the constant has no axis, and each store writes its own storage. *)
let shared_constant_stores () =
  let b0 = Ops.new_buffer "CPU" 32 Float32 and b1 = Ops.new_buffer "CPU" 32 Float32 in
  let v = Ops.bitcast b1 Int32 in
  let a = Ops.after v [ Ops.store v (Ops.int ~dtype:Int32 0x41200000) ] in
  let s = Ops.bitcast a Float32 in
  let v2 = Ops.bitcast s Int32 in
  let a2 = Ops.after v2 [ Ops.store v2 (Ops.int ~dtype:Int32 0x41200000) ] in
  let s2 = Ops.bitcast a2 Float32 in
  let r = loop3 () in
  let p slot dt = Shape.param ~shape:[ Int 4 ] ~device:"CPU" slot dt in
  let body =
    Ops.sink
      [
        Ops.store (p 0 Float32)
          Ops.O.(float 2. + Ops.bitcast (p 1 Int32) Float32);
      ]
  in
  let row =
    Shape.shrink b0
      [ Some (Sym Ops.O.((r * int 4) + int 16), Sym Ops.O.((r * int 4) + int 20)) ]
  in
  let z = Ops.new_buffer "CPU" 4 Float32 in
  let c =
    Ops.call ~precompile:true body
      [ row; Ops.bitcast (Shape.shrink b1 [ Some (Int 7, Int 11) ]) Int32 ]
  in
  let e = Ops.end_ c [ r ] in
  let got =
    leaves
      [
        (b0, Array.init 32 Float.of_int);
        (b1, Array.init 32 Float.of_int);
        (z, Array.make 4 0.);
      ]
      (Ops.sink
         [
           Ops.after b0 [ e ];
           s2;
           Ops.after z [ Ops.store z (Shape.shrink b0 [ Some (Int 25, Int 29) ]) ];
         ])
  in
  let rows = Array.concat [ Array.init 16 Float.of_int;
    [| 9.; 10.; 11.; 12.; 9.; 10.; 11.; 12.; 9.; 10.; 11.; 12. |];
    [| 28.; 29.; 30.; 31. |] ] in
  equal (list values)
    [ floats rows; floats (Array.make 32 10.); floats [| 25.; 26.; 27.; 28. |] ]
    got

(* One value stored whole into two storages of other sizes: each store writes
   all of its own storage. *)
let shared_value_stores_of_other_sizes () =
  let b = Ops.new_buffer "CPU" 16 Float32 and x = Ops.new_buffer "CPU" 32 Float32 in
  let run = Shape.shrink b [ Some (Int 0, Int 8) ] in
  let three = Ops.cast (Ops.int 3) Float32 in
  let got =
    leaves
      [ (b, given16); (x, Array.init 32 Float.of_int) ]
      (Ops.sink
         [
           Ops.after b [ Ops.store run three ];
           Ops.after x [ Ops.store x three ];
         ])
  in
  equal (list values)
    [
      floats
        [| 3.; 3.; 3.; 3.; 3.; 3.; 3.; 3.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15. |];
      floats (Array.make 32 3.);
    ]
    got

(* A state of a bitcast of storage is that bitcast of a state of the storage. *)
let bitcast_state () =
  let b = Ops.new_buffer "CPU" 32 Float32 in
  let v = Ops.bitcast b Int32 in
  let a = Ops.after v [ Ops.store v (Ops.int ~dtype:Int32 0x41200000) ] in
  (b, Ops.bitcast a Float32)

let loop_over_a_bitcast_state_writes_an_older_name () =
  let b, s = bitcast_state () in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let row =
    Shape.shrink b
      [ Some (Sym Ops.O.((r * int 4) + int 4), Sym Ops.O.((r * int 4) + int 8)) ]
  in
  let body = Ops.sink [ Ops.store (p4 1) Ops.O.(float 0. + p4 0) ] in
  let c =
    Ops.call ~precompile:true body
      [ Shape.shrink b [ Some (Int 26, Int 30) ]; row ]
  in
  let want =
    Array.concat
      [
        Array.make 4 10.;
        [| 26.; 27.; 28.; 29.; 26.; 27.; 28.; 29.; 26.; 27.; 28.; 29. |];
        Array.make 16 10.;
      ]
  in
  equal values (floats want)
    (List.hd
       (leaves [ (b, Array.init 32 Float.of_int) ]
          (Ops.sink [ Ops.after s [ Ops.end_ c [ r ] ] ])))

let call_reads_states_of_bitcasts () =
  let b1 = Ops.new_buffer "CPU" 32 Float32 and b0 = Ops.new_buffer "CPU" 32 Float32 in
  let a1 = Ops.after b1 [ Ops.store b1 (Ops.float 4.) ] in
  let v = Ops.bitcast a1 Int32 in
  let a2 = Ops.after v [ Ops.store v (Ops.int ~dtype:Int32 0x41200000) ] in
  let s = Ops.bitcast a2 Float32 in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let row = row4 b0 r in
  let body = Ops.sink [ Ops.store (p4 2) Ops.O.(float 0. + p4 0 + p4 1) ] in
  let c =
    Ops.call ~precompile:true body
      [
        Shape.shrink a1 [ Some (Int 11, Int 15) ];
        Shape.shrink s [ Some (Int 7, Int 11) ];
        row;
      ]
  in
  let z = Ops.new_buffer "CPU" 4 Float32 in
  let got =
    leaves
      [
        (b0, Array.init 32 Float.of_int);
        (b1, Array.init 32 Float.of_int);
        (z, Array.make 4 0.);
      ]
      (Ops.sink
         [
           Ops.after b0 [ Ops.end_ c [ r ] ];
           s;
           Ops.after z [ Ops.store z (Shape.shrink b1 [ Some (Int 15, Int 19) ]) ];
         ])
  in
  let rows =
    Array.concat
      [ Array.make 12 14.; Array.init 20 (fun i -> Float.of_int (i + 12)) ]
  in
  equal (list values)
    [ floats rows; floats (Array.make 32 10.); floats [| 15.; 16.; 17.; 18. |] ]
    got

(* A loop's fixed run of a later state beside another read of that state, the
   state over a bitcast and the run read each trip. *)
let later_state_over_bitcast_read_each_trip () =
  let b0 = Ops.new_buffer "CPU" 32 Float32 and b1 = Ops.new_buffer "CPU" 32 Float32 in
  let v = Shape.reshape (Shape.shrink b1 [ Some (Int 0, Int 16) ]) [ Int 4; Int 4 ] in
  let a = Ops.after v [ Ops.store v Ops.O.(v + float 2.) ] in
  let flat = Shape.reshape a [ Int 16 ] in
  let w = Ops.bitcast flat Int32 in
  let a2 = Ops.after w [ Ops.store w (Ops.int ~dtype:Int32 0x41200000) ] in
  let s2 = Ops.bitcast a2 Float32 in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let move b lo =
    Shape.shrink b
      [ Some (Sym Ops.O.((r * int 4) + int lo), Sym Ops.O.((r * int 4) + int lo + int 4)) ]
  in
  let body =
    Ops.sink
      [
        Ops.store (p4 1) Ops.O.(float 1. + p4 0);
        Ops.store (p4 4) Ops.O.(p4 2 + p4 3);
      ]
  in
  let fixed b lo = Shape.shrink b [ Some (Int lo, Int (lo + 4)) ] in
  let c =
    Ops.call ~precompile:true body
      [ fixed s2 7; move b0 12; fixed b1 20; fixed b0 28; move flat 0 ]
  in
  let e = Ops.end_ c [ r ] in
  let z = Ops.new_buffer "CPU" 4 Float32 in
  let got =
    leaves
      [
        (b0, Array.init 32 Float.of_int);
        (b1, Array.init 32 Float.of_int);
        (z, Array.make 4 0.);
      ]
      (Ops.sink
         [
           Ops.after b0 [ e ];
           Ops.after s2 [ e ];
           Ops.after z [ Ops.store z (fixed s2 3) ];
         ])
  in
  let b0_want =
    Array.concat
      [
        Array.init 12 Float.of_int;
        [| 11.; 11.; 11.; 11.; 11.; 11.; 11.; 11.; 55.; 11.; 11.; 11. |];
        Array.init 8 (fun i -> Float.of_int (i + 24));
      ]
  in
  let b1_want =
    Array.concat
      [
        [| 48.; 50.; 52.; 54.; 48.; 50.; 52.; 54.; 48.; 50.; 52.; 54. |];
        Array.make 4 10.;
        Array.init 16 (fun i -> Float.of_int (i + 16));
      ]
  in
  equal (list values)
    [ floats b0_want; floats b1_want; floats [| 10.; 10.; 10.; 10. |] ]
    got

(* The run of a state of a bitcast of a state the loop writes is read each
   trip. *)
let loop_reads_run_of_a_bitcast_state () =
  let b = Ops.new_buffer "CPU" 16 Float32 in
  let s1 = Ops.after b [ Ops.store b (Ops.float 4.) ] in
  let bits = Ops.bitcast s1 Int32 in
  let a =
    Ops.after bits [ Ops.store bits (Ops.int ~dtype:Int32 0x41200000) ]
  in
  let s = Ops.bitcast a Float32 in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 100 ] in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 1 + float 1.) ] in
  let c =
    Ops.call ~precompile:true body
      [ row4 s1 r; Shape.shrink s [ Some (Int 3, Int 7) ] ]
  in
  equal values
    (floats
       [| 11.; 11.; 11.; 11.; 12.; 11.; 11.; 11.; 10.; 10.; 10.; 10.; 10.; 10.;
          10.; 10. |])
    (leaves1 b (Ops.sink [ Ops.after s [ Ops.end_ c [ r ] ] ]))

(* Values of a state a loop writes, computed from it as each trip starts. The
   written row is row [r] of the state; [b] is 0..15 and the state doubles it. *)
let values_of_doubled ?(trips = 3) value =
  let b, b1 = doubled () in
  let r = Ops.range ~axis_type:Loop (Int trips) [ 100 ] in
  let c = Ops.call ~precompile:true add1_body [ row4 b1 r; value b1 ] in
  leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ])

let loop_values_each_trip () =
  equal values
    (floats
       [| 15.; 13.; 11.; 9.; 15.; 13.; 11.; 9.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (values_of_doubled (fun b1 -> Shape.flip (run_at 4 b1) [ 0 ]));
  equal values
    (floats
       [| 17.; 21.; 25.; 29.; 17.; 21.; 25.; 29.; 35.; 43.; 51.; 59.; 24.; 26.;
          28.; 30. |])
    (values_of_doubled (fun b1 -> Ops.O.(run_at 4 b1 * float 2.)))

(* [y] is written too; [b] is the older state beside the written one. *)
let loop_values_of_several_reads () =
  let b, b1 = doubled () in
  let y = Ops.new_buffer "CPU" 16 Float32 in
  let r = loop3 () in
  let body =
    Ops.sink
      [
        Ops.store (p4 0) Ops.O.(p4 2 + float 1.);
        Ops.store (p4 1) Ops.O.(p4 2 + float 2.);
      ]
  in
  let c =
    Ops.call ~precompile:true body
      [ row4 b1 r; row4 y r; Ops.O.(run_at 4 b1 * run_at 4 y) ]
  in
  let e = Ops.end_ c [ r ] in
  let got =
    leaves
      [ (b, given16); (y, Array.init 16 (fun i -> Float.of_int (i + 1))) ]
      (Ops.sink [ Ops.after b1 [ e ]; Ops.after y [ e ] ])
  in
  equal (list values)
    [
      floats
        [| 41.; 61.; 85.; 113.; 41.; 61.; 85.; 113.; 1723.; 3783.; 7311.;
           12883.; 24.; 26.; 28.; 30. |];
      floats
        [| 42.; 62.; 86.; 114.; 42.; 62.; 86.; 114.; 1724.; 3784.; 7312.;
           12884.; 13.; 14.; 15.; 16. |];
    ]
    got;
  equal values
    (floats
       [| 13.; 16.; 19.; 22.; 13.; 16.; 19.; 22.; 18.; 22.; 26.; 30.; 24.; 26.;
          28.; 30. |])
    (let b, b1 = doubled () in
     let r = loop3 () in
     let c =
       Ops.call ~precompile:true add1_body
         [ row4 b1 r; Ops.O.(run_at 4 b1 + run_at 4 b) ]
     in
     leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))

(* The value read twice in the body; and one leaf in two values. *)
let loop_value_shared () =
  let b, b1 = doubled () in
  let r = loop3 () in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 1 + p4 1) ] in
  let c =
    Ops.call ~precompile:true body [ row4 b1 r; Ops.O.(run_at 4 b1 * float 2.) ]
  in
  equal values
    (floats
       [| 32.; 40.; 48.; 56.; 32.; 40.; 48.; 56.; 128.; 160.; 192.; 224.; 24.;
          26.; 28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]));
  let b, b1 = doubled () in
  let r = loop3 () in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 1 + p4 2) ] in
  let leaf = run_at 4 b1 in
  let c =
    Ops.call ~precompile:true body
      [ row4 b1 r; Shape.flip leaf [ 0 ]; Ops.O.(leaf * float 2.) ]
  in
  equal values
    (floats
       [| 30.; 32.; 34.; 36.; 30.; 32.; 34.; 36.; 96.; 98.; 100.; 102.; 24.;
          26.; 28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))

let loop_value_bound_variable () =
  let v = Ops.variable ~dtype:Int32 "v" (Dtype.Value.of_int 0) (Dtype.Value.of_int 10) in
  let v3 = Shape.bind v (`Int (Bigint.of_int 3)) in
  equal values
    (floats
       [| 25.; 31.; 37.; 43.; 25.; 31.; 37.; 43.; 76.; 94.; 112.; 130.; 24.;
          26.; 28.; 30. |])
    (values_of_doubled (fun b1 -> Ops.O.(run_at 4 b1 * Ops.cast v3 Float32)))

(* A state over a flip passed as the argument: read as each trip starts. *)
let state_over_a_flip_each_trip () =
  let b, b1 = doubled () in
  let view = Shape.flip (run_at 0 b1) [ 0 ] in
  let st = Ops.after view [ Ops.store view (Ops.float 100.) ] in
  let r = loop3 () in
  let c = Ops.call ~precompile:true add1_body [ row4 b1 r; st ] in
  equal values
    (floats
       [| 101.; 101.; 101.; 101.; 102.; 102.; 102.; 102.; 102.; 102.; 102.;
          102.; 24.; 26.; 28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after st [ Ops.end_ c [ r ] ] ]))

(* A state over a view of a state over a view: the read is of the elements
   the views select, as each trip starts. *)
let state_over_nested_views_each_trip () =
  let b, b1 = doubled () in
  let inner = Shape.shrink b1 [ Some (Int 4, Int 12) ] in
  let s = Ops.after inner [ Ops.store inner (Ops.float 5.) ] in
  let h = Shape.flip (run_at 0 s) [ 0 ] in
  let st = Ops.after h [ Ops.store h (Ops.float 100.) ] in
  let r = loop3 () in
  let c = Ops.call ~precompile:true add1_body [ row4 b1 r; st ] in
  equal values
    (floats
       [| 101.; 101.; 101.; 101.; 101.; 101.; 101.; 101.; 102.; 102.; 102.;
          102.; 24.; 26.; 28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after st [ Ops.end_ c [ r ] ] ]))

(* A state over a state over a flip, through a join over other storage. *)
let state_over_a_flip_two_states_deep () =
  let run value =
    let b, b1 = doubled () in
    let x = Ops.new_buffer "CPU" 4 Float32 in
    let x1 = Ops.after x [ Ops.store x (Ops.float 7.) ] in
    let h = Shape.flip (run_at 4 b1) [ 0 ] in
    let st = Ops.after h [ Ops.store h (Ops.float 100.) ] in
    let st2 = Ops.after st [ x1 ] in
    let r = loop3 () in
    let c = Ops.call ~precompile:true add1_body [ row4 b1 r; value st2 ] in
    List.hd
      (leaves
         [ (b, given16); (x, Array.make 4 0.) ]
         (Ops.sink [ Ops.after st2 [ Ops.end_ c [ r ] ]; x1 ]))
  in
  let rest = [| 24.; 26.; 28.; 30. |] in
  equal values
    (floats
       (Array.concat
          [ Array.make 8 101.; Array.make 4 102.; rest ]))
    (run Fun.id);
  equal values
    (floats
       (Array.concat
          [ Array.make 8 201.; Array.make 4 403.; rest ]))
    (run (fun s -> Ops.O.(s * float 2.)))

let loop_value_store_body () =
  let b, b1 = doubled () in
  let r = loop3 () in
  let body = Ops.store (p4 0) Ops.O.(p4 1 + float 1.) in
  let c =
    Ops.call ~precompile:true body
      [ row4 b1 r; Shape.flip (run_at 4 b1) [ 0 ] ]
  in
  equal values
    (floats
       [| 15.; 13.; 11.; 9.; 15.; 13.; 11.; 9.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))

let loop_value_kernel_body () =
  let r_i = Ops.range ~axis_type:Weak (Int 4) [ 1 ] in
  let param slot = Shape.param ~shape:[ Int 4 ] slot Float32 in
  let at p = Ops.index p [ r_i ] in
  let kernel =
    Ops.sink ~kernel:(Ops.kernel_info ())
      [
        Ops.end_
          (Ops.store (at (param 0)) Ops.O.(at (param 1) + float 1.))
          [ r_i ];
      ]
  in
  let b, b1 = doubled () in
  let r = loop3 () in
  let c =
    Ops.call ~precompile:true kernel
      [ row4 b1 r; Shape.flip (run_at 4 b1) [ 0 ] ]
  in
  equal values
    (floats
       [| 15.; 13.; 11.; 9.; 15.; 13.; 11.; 9.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (leaves1 b (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ] ]))

let backedge_value_each_trip () =
  let b = Ops.new_buffer "CPU" 16 Float32 in
  let f = Ops.new_buffer "CPU" 1 Bool in
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let flag = Shape.param ~shape:[ Int 1 ] ~device:"CPU" 2 Bool in
  let body =
    Ops.sink
      [
        Ops.store (p4 0) Ops.O.(p4 1 + float 1.);
        Ops.store flag (Shape.reshape (Ops.const ~dtype:Bool (`Bool true)) [ Int 1 ]);
      ]
  in
  let flipped = Shape.flip (Shape.shrink b [ Some (Int 4, Int 8) ]) [ 0 ] in
  let call = Ops.call ~precompile:true body [ row4 b r; flipped; f ] in
  let e = Ops.backedge call ~loop:r ~cond:f in
  let bound = Run.buffer (devices "CPU") Float32 (floats given16) in
  let flag_buf = Run.buffer (devices "CPU") Bool [| `Bool true |] in
  let linear, vars =
    Schedule.create_linear_with_vars ~cores:Cores.host
      (Ops.sink [ Ops.after b [ e ]; Ops.after f [ e ] ])
  in
  Engine.run ~vars
    (Engine.link ~devices ~bound:[ (b, bound); (f, flag_buf) ] (compile linear))
    [||];
  (* Trip 0 reads 7 6 5 4; trip 1 row 1 becomes 8 7 6 5 at 4..7 after trip 0's
     reading, and trip 2 reads what trip 1 wrote there. *)
  equal values
    (floats
       [| 8.; 7.; 6.; 5.; 8.; 7.; 6.; 5.; 6.; 7.; 8.; 9.; 12.; 13.; 14.; 15. |])
    (Run.values Float32 bound)

let loop_contiguous_each_trip () =
  equal values
    (floats
       [| 17.; 21.; 25.; 29.; 17.; 21.; 25.; 29.; 35.; 43.; 51.; 59.; 24.; 26.;
          28.; 30. |])
    (values_of_doubled (fun b1 -> Ops.contiguous Ops.O.(run_at 4 b1 * float 2.)))

(* A state over a view a loop moves, ordered after a store of other storage:
   the view of the state the loop writes over, read each trip, and the store
   runs. *)
let moving_state_after_effects () =
  let b, b1 = doubled () in
  let x = Ops.new_buffer "CPU" 4 Float32 in
  let zs = Ops.after x [ Ops.store x (Ops.float 7.) ] in
  let r = loop3 () in
  let st = Ops.after (Shape.flip (row4 b1 r) [ 0 ]) [ zs ] in
  let c = Ops.call ~precompile:true add1_body [ row4 b1 r; st ] in
  equal (list values)
    [
      floats
        [|
          7.; 5.; 3.; 1.; 15.; 13.; 11.; 9.; 23.; 21.; 19.; 17.; 24.; 26.; 28.;
          30.;
        |];
      floats [| 7.; 7.; 7.; 7. |];
    ]
    (leaves
       [ (b, given16); (x, Array.make 4 0.) ]
       (Ops.sink [ Ops.after b1 [ Ops.end_ c [ r ] ]; zs ]))

(* A store's value ordered after a store of other storage: the value, and the
   other store runs. *)
let value_after_effects () =
  let b = Ops.new_buffer "CPU" 4 Float32 and y = Ops.new_buffer "CPU" 4 Float32 in
  let sy = Ops.after y [ Ops.store y (Ops.float 1.) ] in
  let s = Ops.after b [ Ops.store b (Ops.after (Ops.float 9.) [ sy ]) ] in
  equal (list values)
    [ floats [| 9.; 9.; 9.; 9. |]; floats [| 1.; 1.; 1.; 1. |] ]
    (leaves [ (b, Array.make 4 0.); (y, Array.make 4 0.) ] (Ops.sink [ s; sy ]))

(* A join whose same-storage dependency is its own state or one under it
   holds that state: a run or flip of it reads the state each trip. *)
let loop_reads_join_of_its_own_states () =
  let older () =
    let b = Ops.new_buffer "CPU" 16 Float32 in
    let x1 = Ops.after b [ Ops.store b Ops.O.(b + float 1.) ] in
    let x2 = Ops.after x1 [ Ops.store x1 Ops.O.(x1 * float 2.) ] in
    (b, x1, x2)
  in
  let run read =
    let b, x1, x2 = older () in
    let j = Ops.after x2 [ x1 ] in
    let r = Ops.range ~axis_type:Loop (Int 2) [ 100 ] in
    let c = Ops.call ~precompile:true add1_body [ row4 x2 r; read j ] in
    leaves1 b (Ops.sink [ Ops.after x2 [ Ops.end_ c [ r ] ] ])
  in
  let x2_tail = Array.init 8 (fun i -> 2. *. Float.of_int (i + 9)) in
  equal values
    (floats (Array.concat [ [| 3.; 5.; 7.; 9.; 4.; 6.; 8.; 10. |]; x2_tail ]))
    (run (fun j -> Shape.shrink j [ Some (Int 0, Int 4) ]));
  equal values
    (floats (Array.concat [ [| 9.; 7.; 5.; 3.; 4.; 6.; 8.; 10. |]; x2_tail ]))
    (run (fun j -> Shape.flip (Shape.shrink j [ Some (Int 0, Int 4) ]) [ 0 ]));
  equal values
    (floats
       [| 15.; 13.; 11.; 9.; 15.; 13.; 11.; 9.; 10.; 12.; 14.; 16.; 24.; 26.;
          28.; 30. |])
    (values_of_doubled (fun b1 ->
         Shape.flip (run_at 4 (Ops.after b1 [ b1 ])) [ 0 ]))

let loop_reads_older_than_listing () =
  let b = Ops.new_buffer "CPU" 16 Float32 in
  let x1 = Ops.after b [ Ops.store b Ops.O.(b + float 1.) ] in
  let x2 = Ops.after x1 [ Ops.store x1 Ops.O.(x1 * float 2.) ] in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 100 ] in
  let c =
    Ops.call ~precompile:true add1_body
      [ row4 x1 r; Shape.shrink x1 [ Some (Int 4, Int 8) ] ]
  in
  equal values
    (floats
       [| 6.; 7.; 8.; 9.; 6.; 7.; 8.; 9.; 18.; 20.; 22.; 24.; 26.; 28.; 30.;
          32. |])
    (leaves1 b (Ops.sink [ Ops.after x2 [ Ops.end_ c [ r ] ] ]))

let loop_flip_of_written_storage () =
  let b = Ops.new_buffer "CPU" 16 Float32 in
  let r = loop3 () in
  let flipped = Shape.flip (Shape.shrink b [ Some (Int 0, Int 4) ]) [ 0 ] in
  let c = Ops.call ~precompile:true add1_body [ row4 b r; flipped ] in
  (* Trip 0 reads 3 2 1 0 and writes row 0: 4 3 2 1. Trip 1 reads 1 2 3 4
     flipped from row 0 as written: 1 2 3 4 + 1 at row 1. Trip 2 the same. *)
  equal values
    (floats
       [| 4.; 3.; 2.; 1.; 2.; 3.; 4.; 5.; 2.; 3.; 4.; 5.; 12.; 13.; 14.; 15. |])
    (leaves1 b (Ops.sink [ Ops.after b [ Ops.end_ c [ r ] ] ]))

(* A loop in the body of a call of [b], over the body's parameter [p]: four
   trips store [p]'s elements 4 to 7, as [read] gives them, plus one, into row
   [r]. Each trip reads them as it starts: trip 1 overwrites them, and trips 2
   and 3 read what trip 1 wrote. The loop's call is compiled on its own, or
   inlined unless [compiled]. *)
let loop_in_body_reads ~precompile ~compiled read =
  let p = Shape.param ~shape:[ Int 16 ] ~device:"CPU" 0 Float32 in
  let r = Ops.range ~axis_type:Loop (Int 4) [ 100 ] in
  let c =
    Ops.call ~precompile:compiled add1_body [ row4 p r; read (run_at 4 p) ]
  in
  let body = Ops.sink [ Ops.after p [ Ops.end_ c [ r ] ] ] in
  let b = Ops.new_buffer "CPU" 16 Float32 in
  leaves1 b (Ops.sink [ Ops.after b [ Ops.call ~precompile body [ b ] ] ])

let loop_in_body_reads_each_trip () =
  let check read want =
    List.iter
      (fun (msg, precompile, compiled) ->
        equal values ~msg (floats want)
          (loop_in_body_reads ~precompile ~compiled read))
      [
        ("precompiled", true, true);
        ("inlined", false, true);
        ("precompiled, its loop's call inlined", true, false);
        ("inlined, its loop's call inlined", false, false);
      ]
  in
  check Fun.id
    [| 5.; 6.; 7.; 8.; 5.; 6.; 7.; 8.; 6.; 7.; 8.; 9.; 6.; 7.; 8.; 9. |];
  check
    (fun v -> Shape.flip v [ 0 ])
    [| 8.; 7.; 6.; 5.; 8.; 7.; 6.; 5.; 6.; 7.; 8.; 9.; 6.; 7.; 8.; 9. |]

(* The elements [Tensors], the reference interpreter, leaves in [b], given
   [given16], once [sink] runs. *)
let tensors1 b sink =
  let slot = (Ops.param_arg_of b).slot in
  let after = Array.map (fun x -> `Float x) given16 in
  List.iter
    (fun (s, i, v) -> if s = slot then after.(i) <- v)
    (Tensors.writes ~buffers:[ (slot, Array.copy after) ] sink);
  after

(* Nested loops of a call over a buffer of 0 to 15: trip [(o, i)] stores into
   row [2 o + i] the elements 4 to 7, reversed, plus one, as the trip starts.
   The outer end's range varies slowest, at top level and in a body,
   precompiled or inlined, the loop's call compiled on its own or inlined. *)
let nested_loops_of_a_call () =
  let o = Ops.range ~axis_type:Loop (Int 2) [ 100 ]
  and i = Ops.range ~axis_type:Loop (Int 2) [ 101 ] in
  let loop ~compiled outer inner u =
    let row = row4 u Ops.O.((o * int 2) + i) in
    let c =
      Ops.call ~precompile:compiled add1_body
        [ row; Shape.flip (run_at 4 u) [ 0 ] ]
    in
    Ops.end_ (Ops.end_ c [ inner ]) [ outer ]
  in
  let p = Shape.param ~shape:[ Int 16 ] ~device:"CPU" 0 Float32 in
  let check outer inner want =
    List.iter (fun compiled ->
    let loop = loop ~compiled in
    let top b = Ops.sink [ Ops.after b [ loop outer inner b ] ] in
    let in_body precompile b =
      let body = Ops.sink [ Ops.after p [ loop outer inner p ] ] in
      Ops.sink [ Ops.after b [ Ops.call ~precompile body [ b ] ] ]
    in
    List.iter
      (fun (msg, sink) ->
        let msg = if compiled then msg else msg ^ ", its call inlined" in
        let b = Ops.new_buffer "CPU" 16 Float32 in
        let sink = sink b in
        equal values ~msg:("Tensors, " ^ msg) (floats want) (tensors1 b sink);
        equal values ~msg (floats want) (leaves1 b sink))
      [
        ("at top level", top);
        ("in a precompiled body", in_body true);
        ("in an inlined body", in_body false);
      ])
    [ true; false ]
  in
  (* Rows 0, 1, 2, 3 in turn: rows 2 and 3 read row 1 as written. *)
  check o i
    [| 8.; 7.; 6.; 5.; 8.; 7.; 6.; 5.; 6.; 7.; 8.; 9.; 6.; 7.; 8.; 9. |];
  (* Rows 0, 2, 1, 3: row 2 reads row 1 as given. *)
  check i o
    [| 8.; 7.; 6.; 5.; 8.; 7.; 6.; 5.; 8.; 7.; 6.; 5.; 6.; 7.; 8.; 9. |]

(* A loop on a state, reading the run of the first state its store goes
   through: the read holds the run as it was, the store still lands. *)
let loop_reads_older_plain_run () =
  let b = Ops.new_buffer "CPU" 16 Float32 in
  let v = Shape.shrink b [ Some (Int 2, Int 6) ] in
  let s1 = Ops.after b [ Ops.store v Ops.O.(v + float 2.) ] in
  let r = Ops.range ~axis_type:Loop (Int 2) [ 100 ] in
  let row =
    Shape.shrink s1
      [ Some (Sym Ops.O.((r * int 4) + int 4), Sym Ops.O.((r * int 4) + int 8)) ]
  in
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 1 + float 1.) ] in
  let c = Ops.call ~precompile:true body [ row; v ] in
  let got =
    List.hd
      (leaves
         [ (b, Array.init 16 Float.of_int) ]
         (Ops.sink [ Ops.after s1 [ Ops.end_ c [ r ] ] ]))
  in
  equal values
    (floats
       [| 0.; 1.; 4.; 5.; 3.; 4.; 5.; 6.; 3.; 4.; 5.; 6.; 12.; 13.; 14.; 15. |])
    got

(* Two loops in sequence each read a fixed run: the second reads what the first
   left, and each run is read as its loop's trip starts. *)
let chained_loops_read_runs () =
  let body = Ops.sink [ Ops.store (p4 0) Ops.O.(p4 1 + float 1.) ] in
  let run u lo = Shape.shrink u [ Some (Ops.Int lo, Ops.Int (lo + 4)) ] in
  let loop n = Ops.range ~axis_type:Loop (Int 2) [ n ] in
  let call args = Ops.call ~precompile:true body args in
  let given = Array.init 16 Float.of_int in
  let states () =
    let b = Ops.new_buffer "CPU" 16 Float32 in
    let b1 = Ops.after b [ Ops.store b Ops.O.(b * float 2.) ] in
    let r1 = loop 100 and r2 = loop 101 in
    let k1 = Ops.end_ (call [ row4 b1 r1; run b1 8 ]) [ r1 ] in
    let s1 = Ops.after b1 [ k1 ] in
    let k2 = Ops.end_ (call [ row4 s1 r2; run s1 4 ]) [ r2 ] in
    List.hd (leaves [ (b, given) ] (Ops.sink [ Ops.after s1 [ k2 ] ]))
  in
  equal values ~msg:"runs of states"
    (floats
       [| 18.; 20.; 22.; 24.; 18.; 20.; 22.; 24.; 16.; 18.; 20.; 22.; 24.; 26.;
          28.; 30. |])
    (states ());
  let plain () =
    let b = Ops.new_buffer "CPU" 16 Float32 in
    let r1 = loop 100 and r2 = loop 101 in
    let k1 = Ops.end_ (call [ row4 b r1; run b 8 ]) [ r1 ] in
    let s1 = Ops.after b [ k1 ] in
    let k2 = Ops.end_ (call [ row4 s1 r2; run b 8 ]) [ r2 ] in
    List.hd (leaves [ (b, given) ] (Ops.sink [ Ops.after s1 [ k2 ] ]))
  in
  equal values ~msg:"runs of storage"
    (floats
       [| 9.; 10.; 11.; 12.; 9.; 10.; 11.; 12.; 8.; 9.; 10.; 11.; 12.; 13.;
          14.; 15. |])
    (plain ())

(* A value of storage written before the loop reads no trip's writes, and a read
   of a state older than the loop's write reads it as it was: both stay stored
   once. *)
let loop_reads_earlier_states () =
  let r = Ops.range ~axis_type:Loop (Int 3) [ 100 ] in
  let body = Ops.sink [ Ops.store (p4 0) (p4 1) ] in
  let first4 u = Shape.shrink u [ Some (Int 0, Int 4) ] in
  let given = Array.init 16 Float.of_int in
  let x = Ops.new_buffer "CPU" 16 Float32
  and y = Ops.new_buffer "CPU" 4 Float32 in
  let y1 = Ops.after y [ Ops.store y (first4 x) ] in
  let c = Ops.call ~precompile:true body [ row4 x r; Ops.O.(y1 + float 1.) ] in
  let got =
    leaves
      [ (x, given); (y, Array.make 4 0.) ]
      (Ops.sink [ Ops.after x [ Ops.end_ c [ r ] ]; y1 ])
  in
  equal values ~msg:"storage written before the loop"
    (floats
       [| 1.; 2.; 3.; 4.; 1.; 2.; 3.; 4.; 1.; 2.; 3.; 4.; 12.; 13.; 14.; 15. |])
    (List.hd got);
  let x = Ops.new_buffer "CPU" 16 Float32 in
  let x1 = Ops.after x [ Ops.store x Ops.O.(x * float 2.) ] in
  let flipped = Shape.flip (first4 x) [ 0 ] in
  let c = Ops.call ~precompile:true body [ row4 x1 r; flipped ] in
  let got = leaves [ (x, given) ] (Ops.sink [ Ops.after x1 [ Ops.end_ c [ r ] ] ]) in
  equal values ~msg:"an older state"
    (floats
       [| 3.; 2.; 1.; 0.; 3.; 2.; 1.; 0.; 3.; 2.; 1.; 0.; 24.; 26.; 28.; 30. |])
    (List.hd got)

(* The law of a loop's reads: a loop of three trips whose call stores, into a
   row of each buffer moving 16 bytes a trip, the sum of runs that move with the
   trip or stay fixed, reads each run as the trip starts, whatever storage the
   loop has written. Fixed runs start at any element, off 16 bytes too. The
   buffers live on [d] of [devices].

   Each drawn program is a kernel of its own, about 30 ms of Clang on fudge:
   the phase of each run's constant start, and the program's flips and calls,
   make its kernel. A start that a variable placed would take another path,
   so the law draws constants. A case draws one to eight programs, each on
   buffers and a loop range of its own, and runs them as one schedule, whose
   distinct kernels compile in parallel. Programs that store one value into
   one shape may share a kernel; a case of one program runs alone. *)

type loop_step = {
  target : int;  (* The buffer written, at [lo + 4 t]. *)
  lo : int;
  reads : (int * int * int) list;  (* A read's buffer, start and row stride. *)
}

let gen_loop_program r =
  let m = 2 + Random.State.int r 2 in
  let sizes = Array.init m (fun _ -> if Random.State.bool r then 16 else 24) in
  let elems n = (n - 12) / 4 in
  let step j =
    let n = sizes.(j) in
    let lo = 4 * Random.State.int r (elems n + 1) in
    let read () =
      let b = Random.State.int r m in
      let nb = sizes.(b) in
      if Random.State.bool r then
        (* A fixed run, at any element. *)
        (b, Random.State.int r (nb - 3), 0)
      else (b, 4 * Random.State.int r (elems nb + 1), 4)
    in
    let reads = List.init (1 + Random.State.int r 2) (fun _ -> read ()) in
    { target = j; lo; reads }
  in
  (sizes, List.init m step)

let no_flips st = List.map (fun _ -> false) st.reads

(* The loop of [steps] over [storage], buffers or a body's parameters, each
   read reversed where [flipped] says: a value of its run, no view. Its call
   is compiled on its own, or inlined unless [compiled]. *)
let loop_over ?(flipped = no_flips) ?(compiled = true) ?(axis = 100) d storage
    steps =
  let r = Ops.range ~axis_type:Loop (Int 3) [ axis ] in
  let view b lo stride =
    Shape.shrink storage.(b)
      [
        (if stride = 0 then Some (Ops.Int lo, Ops.Int (lo + 4))
         else
           let start = Ops.O.((r * int stride) + int lo) in
           Some (Ops.Sym start, Ops.Sym Ops.O.(start + int 4)));
      ]
  in
  let slot = ref 0 and args = ref [] in
  let param v =
    let s = !slot in
    incr slot;
    args := v :: !args;
    Shape.param ~shape:[ Int 4 ] ~device:d s Float32
  in
  let stores =
    List.map
      (fun st ->
        let read (b, o, rs) flip =
          let v = view b o rs in
          param (if flip then Shape.flip v [ 0 ] else v)
        in
        let reads = List.map2 read st.reads (flipped st) in
        let target = param (view st.target st.lo 4) in
        let v = List.fold_left Ops.add (List.hd reads) (List.tl reads) in
        Ops.store target v)
      steps
  in
  let call = Ops.call ~precompile:compiled (Ops.sink stores) (List.rev !args) in
  Ops.end_ call [ r ]

let afters storage e =
  Ops.sink (List.map (fun b -> Ops.after b [ e ]) (Array.to_list storage))

let loop_of_program ~compiled ~axis d sizes steps =
  let bufs = Array.map (fun n -> Ops.new_buffer d n Float32) sizes in
  (bufs, afters bufs (loop_over ~compiled ~axis d bufs steps))

(* The loop in the body of a call of the buffers, precompiled or inlined: the
   loop's storage is the body's parameters. *)
let loop_in_body ~precompile ~compiled ~flipped ~axis d sizes steps =
  let params =
    Array.mapi
      (fun j n -> Shape.param ~shape:[ Int n ] ~device:d j Float32)
      sizes
  in
  let body =
    afters params (loop_over ~flipped ~compiled ~axis d params steps)
  in
  let bufs = Array.map (fun n -> Ops.new_buffer d n Float32) sizes in
  (bufs, afters bufs (Ops.call ~precompile body (Array.to_list bufs)))

(* What the loop leaves: each trip reads the state the trip starts from, and
   each step writes its own buffer. *)
let loop_spec ?(flipped = no_flips) steps init =
  let cur = Array.map Array.copy init in
  for t = 0 to 2 do
    let start = Array.map Array.copy cur in
    List.iter
      (fun st ->
        for i = 0 to 3 do
          let sum =
            List.fold_left2
              (fun acc (b, o, rs) flip ->
                let at = if flip then 3 - i else i in
                acc +. start.(b).(o + (t * rs) + at))
              0. st.reads (flipped st)
          in
          cur.(st.target).(st.lo + (t * 4) + i) <- sum
        done)
      steps
  done;
  cur

(* The same law over a state: each buffer holds its [After] state, [b * 2],
   when the loop starts, and a read names that state or the buffer as given. *)
let loop_of_states ~axis d sizes steps older =
  let bufs = Array.map (fun n -> Ops.new_buffer d n Float32) sizes in
  let states =
    Array.map (fun b -> Ops.after b [ Ops.store b Ops.O.(b * float 2.) ]) bufs
  in
  let r = Ops.range ~axis_type:Loop (Int 3) [ axis ] in
  let view u lo stride =
    Shape.shrink u
      [
        (if stride = 0 then Some (Ops.Int lo, Ops.Int (lo + 4))
         else
           let start = Ops.O.((r * int stride) + int lo) in
           Some (Ops.Sym start, Ops.Sym Ops.O.(start + int 4)));
      ]
  in
  let slot = ref 0 and args = ref [] in
  let param v =
    let s = !slot in
    incr slot;
    args := v :: !args;
    Shape.param ~shape:[ Int 4 ] ~device:d s Float32
  in
  let stores =
    List.map
      (fun st ->
        let reads =
          List.mapi
            (fun i (b, o, rs) ->
              let u = if List.nth (older st) i then bufs.(b) else states.(b) in
              param (view u o rs))
            st.reads
        in
        let target = param (view states.(st.target) st.lo 4) in
        let v = List.fold_left Ops.add (List.hd reads) (List.tl reads) in
        Ops.store target v)
      steps
  in
  let call = Ops.call ~precompile:true (Ops.sink stores) (List.rev !args) in
  let e = Ops.end_ call [ r ] in
  let afters =
    List.map (fun b -> Ops.after b [ e ]) (Array.to_list states)
  in
  (bufs, Ops.sink afters)

let loop_states_spec steps older init =
  let cur = Array.map (Array.map (fun x -> 2. *. x)) init in
  for t = 0 to 2 do
    let start = Array.map Array.copy cur in
    List.iter
      (fun st ->
        for i = 0 to 3 do
          let sum =
            List.fold_left2
              (fun acc (b, o, rs) older ->
                let held = if older then init else start in
                acc +. held.(b).(o + (t * rs) + i))
              0. st.reads (older st)
          in
          cur.(st.target).(st.lo + (t * 4) + i) <- sum
        done)
      steps
  done;
  cur

(* A drawn program: its sink, the buffers bound to its storage, and what each
   holds once it ran, read as the type given, with the name of its
   program. *)
type drawn = {
  sink : Ops.t;
  bound : (Ops.t * Buffer.t) list;
  wants : (string * Dtype.t * Dtype.value array * Buffer.t) list;
}

(* The programs [programs] of a case, run as one schedule, and each of their
   buffers checked against what its program wants. *)
let run_drawn ?(devices = devices) programs =
  let sink = Ops.sink (List.concat_map (fun p -> Ops.src p.sink) programs) in
  let linear, vars = Schedule.create_linear_with_vars ~cores:Cores.host sink in
  let bound = List.concat_map (fun p -> p.bound) programs in
  Engine.run ~vars (Engine.link ~devices ~bound (compile ~devices linear)) [||];
  List.iter
    (fun p ->
      List.iter
        (fun (msg, dt, want, b) -> equal ~msg values want (Run.values dt b))
        p.wants)
    programs

(* The seeds of a case's programs, one to eight. *)
let program_seeds =
  Gen.list ~size:(Gen.int_range 1 8) (Gen.int_range 1 1_000_000)

let initial_loop_values sizes =
  Array.mapi
    (fun j n ->
      Array.init n (fun i -> Float.of_int (((j * 7) + (i * 13) + 3) mod 11)))
    sizes

(* The loop program [k] of seed [seed]: [sink] over [bufs], bound to new
   buffers of [d] holding [init], which hold [want] once it ran. *)
let loop_program ~devices d ~k ~seed sink bufs init want =
  let buffers =
    Array.map (fun xs -> Run.buffer (devices d) Float32 (floats xs)) init
  in
  let msg j = Printf.sprintf "program %d (seed %d), buffer %d" k seed j in
  {
    sink;
    bound = Array.to_list (Array.map2 (fun b x -> (b, x)) bufs buffers);
    wants =
      List.init (Array.length bufs) (fun j ->
          (msg j, Dtype.Float32, floats want.(j), buffers.(j)));
  }

(* Whether a step reads a fixed run off 16 bytes. *)
let reads_off_16 s =
  List.exists (fun (_, o, rs) -> rs = 0 && o mod 4 <> 0) s.reads

let loop_states_law ?(devices = devices) d =
  let program k seed =
    let r = Random.State.make [| seed |] in
    let sizes, steps = gen_loop_program r in
    let flags =
      List.map
        (fun st -> List.map (fun _ -> Random.State.int r 4 = 0) st.reads)
        steps
    in
    let tbl = List.combine steps flags in
    let older st = List.assq st tbl in
    let init = initial_loop_values sizes in
    let reads_own s =
      List.exists2 (fun (b, _, _) o -> b = s.target && not o) s.reads (older s)
    in
    cover "a fixed run read off 16 bytes" (List.exists reads_off_16 steps);
    cover "a run of the state the step writes" (List.exists reads_own steps);
    cover "a read of the older state"
      (List.exists (fun st -> List.exists Fun.id (older st)) steps);
    let bufs, sink = loop_of_states ~axis:(100 + k) d sizes steps older in
    loop_program ~devices d ~k ~seed sink bufs init
      (loop_states_spec steps older init)
  in
  prop ~count:20 "a loop reads each run of an After state as its trip starts"
    program_seeds (fun seeds -> run_drawn ~devices (List.mapi program seeds))

let loop_reads_law ?(devices = devices) d =
  let program k seed =
    let r = Random.State.make [| seed |] in
    let sizes, steps = gen_loop_program r in
    let init = initial_loop_values sizes in
    let reads_own s = List.exists (fun (b, _, _) -> b = s.target) s.reads in
    cover "a fixed run read off 16 bytes" (List.exists reads_off_16 steps);
    cover "a run read of a buffer the step writes"
      (List.exists reads_own steps);
    let compiled = Random.State.bool r in
    cover "an inlined call in the loop" (not compiled);
    let bufs, sink =
      loop_of_program ~compiled ~axis:(100 + k) d sizes steps
    in
    loop_program ~devices d ~k ~seed sink bufs init (loop_spec steps init)
  in
  prop ~count:16 "a loop reads each run as its trip starts" program_seeds
    (fun seeds -> run_drawn ~devices (List.mapi program seeds))

(* The law of a loop's reads in a body: the loop of [loop_reads_law] runs in
   the body of a call, precompiled or inlined, over the body's parameters, and
   a read may be the reverse of its run, a value the call computes. Each trip
   reads each run and value as it starts, and Tensors, the reference
   interpreter, agrees. *)
let loop_in_body_law ?(devices = devices) d =
  let program k seed =
    let r = Random.State.make [| seed |] in
    let sizes, steps = gen_loop_program r in
    let flips =
      List.map
        (fun st -> List.map (fun _ -> Random.State.bool r) st.reads)
        steps
    in
    let flipped st = List.assq st (List.combine steps flips) in
    let precompile = Random.State.bool r in
    let init = initial_loop_values sizes in
    let reads_own flip s =
      List.exists2
        (fun (b, _, rs) f -> b = s.target && f = flip && (flip || rs = 0))
        s.reads (flipped s)
    in
    cover "a precompiled body" precompile;
    cover "an inlined body" (not precompile);
    cover "a fixed run of the parameter the step writes"
      (List.exists (reads_own false) steps);
    cover "a reversed run of the parameter the step writes"
      (List.exists (reads_own true) steps);
    let compiled = Random.State.bool r in
    cover "an inlined call in the loop" (not compiled);
    let bufs, sink =
      loop_in_body ~precompile ~compiled ~flipped ~axis:(100 + k) d sizes steps
    in
    let want = loop_spec ~flipped steps init in
    let slot b = (Ops.param_arg_of b).slot in
    let given = Array.map (fun xs -> Array.map (fun x -> `Float x) xs) init in
    let writes =
      Tensors.writes
        ~buffers:
          (Array.to_list (Array.map2 (fun b x -> (slot b, x)) bufs given))
        sink
    in
    Array.iteri
      (fun j b ->
        let after = Array.copy given.(j) in
        List.iter (fun (s, i, v) -> if s = slot b then after.(i) <- v) writes;
        let msg = Printf.sprintf "program %d (seed %d), Tensors" k seed in
        equal values ~msg (floats want.(j)) after)
      bufs;
    loop_program ~devices d ~k ~seed sink bufs init want
  in
  prop ~count:18 "a loop in a body reads its parameters as each trip starts"
    program_seeds (fun seeds -> run_drawn ~devices (List.mapi program seeds))

(* The law of stores through byte views: a store into float32 storage, or
   into a state of it, through its bytes viewed as another type, where the
   bytes are whole elements of the storage, writes the value's bits into those
   elements and leaves the others. The target is the bytes read back as
   float32, or of the other type itself; one row or rows of two; from a
   constant or a bound start; its value a constant or a run of the state it
   stores into. A case draws one to eight programs, run as one schedule, as
   the loop laws' cases do. *)

type byte_store = {
  of_state : bool;  (* The storage after a store of [x + 1]. *)
  via : Dtype.t;  (* The type its bytes are viewed as. *)
  first : int;  (* The first element of the storage the store covers. *)
  count : int;  (* The elements it covers. *)
  typed : bool;  (* The target is of [via], not read back as float32. *)
  rows : bool;
  bound : bool;
  reads : int option;  (* The first element of the run of the state stored. *)
}

let pp_byte_store ppf s =
  Format.fprintf ppf
    "%s through %a, elements %d..%d%s%s%s, %s"
    (if s.of_state then "state" else "storage")
    Dtype.pp s.via s.first (s.first + s.count - 1)
    (if s.typed then ", typed" else "")
    (if s.rows then ", rows" else "")
    (if s.bound then ", bound start" else "")
    (match s.reads with
    | Some r -> Printf.sprintf "reading from %d" r
    | None -> "a constant")

let byte_elements = 16

let gen_byte_store r =
  let via = List.nth Dtype.[ Uint8; Float16; Int64 ] (Random.State.int r 3) in
  let size = Dtype.itemsize via in
  (* An int64 spans two elements: it starts and ends on even ones. *)
  let step = if size = 8 then 2 else 1 in
  let count = step * (1 + Random.State.int r (8 / step)) in
  let first =
    step * Random.State.int r (((byte_elements - count) / step) + 1)
  in
  let typed = Random.State.bool r in
  let target_numel = if typed then count * 4 / size else count in
  {
    of_state = Random.State.bool r;
    via;
    first;
    count;
    typed;
    rows = target_numel mod 2 = 0 && Random.State.bool r;
    bound = size < 8 && Random.State.bool r;
    reads =
      (if typed || Random.State.bool r then None
       else Some (Random.State.int r (byte_elements - count + 1)));
  }

(* The bits each element of a typed constant store gives the elements it
   covers, from the first. *)
let typed_store = function
  | Dtype.Uint8 -> (Ops.const ~dtype:Uint8 (`Int (Bigint.of_int 7)), fun _ -> 0x07070707l)
  | Float16 -> (Ops.const ~dtype:Float16 (`Float 1.5), fun _ -> 0x3E003E00l)
  | Int64 ->
      ( Ops.const ~dtype:Int64 (`Int (Bigint.of_int 3)),
        fun j -> if j mod 2 = 0 then 3l else 0l )
  | dt -> invalid_arg (Format.asprintf "no typed store of %a" Dtype.pp dt)

(* The program of the store [s] on [d], named [name] in its check, its bound
   start the variable [start]. *)
let stores_through_bytes ?(devices = devices) ~name ~start d s =
  let n = byte_elements in
  let x = Ops.new_buffer d n Float32 in
  let st =
    if s.of_state then Ops.after x [ Ops.store x Ops.O.(x + float 1.) ] else x
  in
  let size = Dtype.itemsize s.via in
  let k = s.count * 4 / size in
  let first, last =
    if not s.bound then (Ops.Int (s.first * 4 / size), Ops.Int ((s.first * 4 / size) + k))
    else
      let v =
        Shape.bind
          (Ops.variable start (`Int Bigint.zero) (`Int (Bigint.of_int n)))
          (`Int (Bigint.of_int s.first))
      in
      let per = 4 / size in
      let at = Ops.O.(v * int per) in
      (Ops.Sym at, Ops.Sym Ops.O.(at + int k))
  in
  let viewed = Shape.shrink (Ops.bitcast st s.via) [ Some (first, last) ] in
  let m = if s.typed then k else s.count in
  let shape = if s.rows then [ Ops.Int (m / 2); Int 2 ] else [ Ops.Int m ] in
  let target =
    Shape.reshape (if s.typed then viewed else Ops.bitcast viewed Float32) shape
  in
  let broadcast c =
    Shape.expand (Shape.reshape c (List.map (fun _ -> Ops.Int 1) shape)) shape
  in
  let before i = Float.of_int i +. if s.of_state then 1. else 0. in
  let value, covered =
    match s.reads with
    | Some r ->
        let run = Shape.shrink st [ Some (Int r, Int (r + s.count)) ] in
        ( Shape.reshape Ops.O.((run * float 2.) + float 1.) shape,
          fun j -> Int32.bits_of_float ((before (r + j) *. 2.) +. 1.) )
    | None when s.typed ->
        let c, bits = typed_store s.via in
        (broadcast c, bits)
    | None ->
        ( broadcast (Ops.cast (Ops.float 2.5) Float32),
          fun _ -> Int32.bits_of_float 2.5 )
  in
  let sink = Ops.sink [ Ops.after st [ Ops.store target value ] ] in
  let buffer =
    Run.buffer (devices d) Float32 (floats (Array.init n Float.of_int))
  in
  let want =
    Array.init n (fun i ->
        if i >= s.first && i < s.first + s.count then covered (i - s.first)
        else Int32.bits_of_float (before i))
  in
  let msg = Format.asprintf "%s: %a" name pp_byte_store s in
  let want = Array.map (fun b -> `Int (Bigint.of_int32 b)) want in
  { sink; bound = [ (x, buffer) ]; wants = [ (msg, Int32, want, buffer) ] }

let byte_views ?devices d =
  let program k seed =
    let s = gen_byte_store (Random.State.make [| seed |]) in
    cover "a state" s.of_state;
    cover "storage" (not s.of_state);
    cover "a value reading its target" (s.reads <> None);
    cover "a typed target" s.typed;
    cover "rows" s.rows;
    cover "a bound start" s.bound;
    cover "through int64" (s.via = Int64);
    cover "through float16" (s.via = Float16);
    cover "through uint8" (s.via = Uint8);
    let name = Printf.sprintf "program %d (seed %d)" k seed in
    let start = Printf.sprintf "first%d" k in
    stores_through_bytes ?devices ~name ~start d s
  in
  prop ~count:25
    "a store through whole elements of bytes viewed as another type writes \
     the value's bits into them"
    program_seeds
    (fun seeds -> run_drawn ?devices (List.mapi program seeds))

(* The law of stores through runs of bytes that cut elements: a store into
   float32 storage, or into a state of it, through a run of its bytes viewed
   as a narrower type, from a constant start that may start or end inside an
   element, writes the value's bytes into the run and leaves every other byte,
   as [Tensors] computes. Its value is a constant, or the run reversed, which
   reads the bytes it overwrites. *)
let stores_through_cut_bytes ?(devices = devices) ~name d r =
  let n = byte_elements in
  let via = if Random.State.bool r then Dtype.Uint8 else Float16 in
  let per = 4 / Dtype.itemsize via in
  let count = 1 + Random.State.int r (2 * per) in
  let first = Random.State.int r ((n * per) - count + 1) in
  let of_state = Random.State.bool r and reverse = Random.State.bool r in
  let x = Ops.new_buffer d n Float32 in
  let st =
    if of_state then Ops.after x [ Ops.store x Ops.O.(x + float 1.) ] else x
  in
  let target =
    Shape.shrink (Ops.bitcast st via) [ Some (Int first, Int (first + count)) ]
  in
  let value =
    if reverse then Shape.flip target [ 0 ]
    else
      Ops.const ~dtype:via (`Float 1.5)
  in
  let sink = Ops.sink [ Ops.after st [ Ops.store target value ] ] in
  let given = Array.init n Float.of_int in
  let slot = (Ops.param_arg_of x).slot in
  let want = Array.map (fun v -> `Float v) given in
  List.iter
    (fun (s, i, v) -> if s = slot then want.(i) <- v)
    (Tensors.writes ~buffers:[ (slot, Array.copy want) ] sink);
  let bits = function
    | `Float v -> `Int (Bigint.of_int32 (Int32.bits_of_float v))
    | v -> v
  in
  let buffer = Run.buffer (devices d) Float32 (floats given) in
  let msg =
    Format.asprintf "%s: %s through %a, elements %d..%d, %s" name
      (if of_state then "state" else "storage")
      Dtype.pp via first (first + count - 1)
      (if reverse then "reversed" else "a constant")
  in
  ( (first mod per <> 0, (first + count) mod per <> 0),
    {
      sink;
      bound = [ (x, buffer) ];
      wants = [ (msg, Int32, Array.map bits want, buffer) ];
    } )

let cut_byte_views ?devices d =
  let program k seed =
    let name = Printf.sprintf "program %d (seed %d)" k seed in
    let (cuts_start, cuts_end), program =
      stores_through_cut_bytes ?devices ~name d (Random.State.make [| seed |])
    in
    cover "a start inside an element" cuts_start;
    cover "an end inside an element" cuts_end;
    cover "whole elements" ((not cuts_start) && not cuts_end);
    program
  in
  (* Seed 19 draws whole elements, 2 float16s from the 22nd, which 16 % of
     the programs draw. *)
  prop ~count:25 ~examples:[ [ 19 ] ]
    "a store through a run of bytes that cuts elements writes the value's \
     bytes into it, the others kept"
    program_seeds
    (fun seeds -> run_drawn ?devices (List.mapi program seeds))

(* A store through a bitcast of a view of float32 storage that is no run, its
   elements reversed or a column of rows, writes the value's bits through the
   view, as [Tensors] computes: the bitcast moves to the value. *)
let stores_through_bitcast_views () =
  let x = Ops.new_buffer "CPU" 16 Float32 in
  let y = Ops.new_buffer "CPU" 16 Float32 in
  let columns u =
    Shape.shrink (Shape.reshape u [ Int 4; Int 4 ]) [ None; Some (Int 0, Int 2) ]
  in
  let given = Array.init 16 Float.of_int
  and ys = Array.init 16 (fun k -> 100. +. Float.of_int k) in
  let bits = function
    | `Float v -> `Int (Bigint.of_int32 (Int32.bits_of_float v))
    | v -> v
  in
  List.iter
    (fun (msg, view) ->
      let sink =
        Ops.sink
          [
            Ops.after x
              [ Ops.store (Ops.bitcast (view x) Uint8) (Ops.bitcast (view y) Uint8) ];
          ]
      in
      let slot b = (Ops.param_arg_of b).slot in
      let want = Array.map (fun v -> `Float v) given in
      List.iter
        (fun (s, i, v) -> if s = slot x then want.(i) <- v)
        (Tensors.writes
           ~buffers:
             [ (slot x, Array.copy want); (slot y, Array.map (fun v -> `Float v) ys) ]
           sink);
      let linear, vars = Schedule.create_linear_with_vars
                           ~cores:Cores.host sink in
      let bx = Run.buffer (device "CPU") Float32 (floats given)
      and by = Run.buffer (device "CPU") Float32 (floats ys) in
      Engine.run ~vars
        (Engine.link ~devices ~bound:[ (x, bx); (y, by) ] (compile linear))
        [||];
      equal values ~msg (Array.map bits want) (Run.values Int32 bx))
    [ ("reversed", fun u -> Shape.flip u [ 0 ]); ("columns", columns) ]

(* Calls of views

   A call compiled on its own takes flat storage at each argument: a kernel
   given as code, or a precompiled call. [a] holds 8 floats and [y] 1 to 4;
   each call copies four floats from one argument into another, by default
   from its second into its first. *)

let a_values = Array.init 8 Float.of_int
let y_values = [| 1.; 2.; 3.; 4. |]

let copy_kernel ~dst ~src ps =
  let i = Ops.range ~axis_type:Weak (Int 4) [ 0 ] in
  let at k = Ops.index (List.nth ps k) [ i ] in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"copy4" ())
    [ Ops.end_ (Ops.store (at dst) (at src)) [ i ] ]

let precompiled_copy ?(dst = 0) ?(src = 1) args =
  let p slot = Shape.param ~shape:[ Int 4 ] ~device:"CPU" slot Float32 in
  let ps = List.mapi (fun i _ -> p i) args in
  Ops.call ~precompile:true
    (Ops.sink [ Ops.store (List.nth ps dst) (List.nth ps src) ])
    args

let kernel_copy ?(dst = 0) ?(src = 1) args =
  let placeholders = List.mapi (fun i u -> Shape.placeholder_like u i) args in
  Ops.call (copy_kernel ~dst ~src placeholders) args

type copy = ?dst:int -> ?src:int -> Ops.t list -> Ops.t

let copies : (string * copy) list =
  [ ("a precompiled call", precompiled_copy); ("a kernel", kernel_copy) ]

(* The values of [into] after a run of [sink], [into] holding [init] before
   it and [from] holding [values], placed by [at] (default as it is). *)
let run_copy ?(at = Fun.id) ~into ~init ~from values sink =
  let linear, _ = Schedule.create_linear_with_vars ~cores:Cores.host sink in
  let out = Run.buffer host Float32 (floats init) in
  let bound =
    [ (into, out); (from, at (Run.buffer host Float32 (floats values))) ]
  in
  Engine.run (Engine.link ~devices ~bound (compile linear)) [||];
  Run.values Float32 out

let at_offset a = Shape.shrink a [ Some (Ops.Int 4, Ops.Int 8) ]

(* Elements 1 to 4: 4 bytes past the 16-byte boundary a parameter takes. *)
let off_boundary a = Shape.shrink a [ Some (Ops.Int 1, Ops.Int 5) ]
let first_four a = Shape.shrink a [ Some (Ops.Int 0, Ops.Int 4) ]
let dims = List.map (fun n -> Ops.Int n)

let every_other a =
  let pairs = Shape.reshape a (dims [ 4; 2 ]) in
  let firsts = Shape.shrink pairs [ None; Some (Ops.Int 0, Ops.Int 1) ] in
  Shape.reshape firsts [ Int 4 ]

let transposed a =
  let t = Shape.permute (Shape.reshape a (dims [ 2; 4 ])) [ 1; 0 ] in
  first_four (Shape.reshape t [ Int 8 ])

let first_row a =
  let m = Shape.reshape a (dims [ 2; 4 ]) in
  Shape.shrink m [ Some (Ops.Int 0, Ops.Int 1); None ]

(* [n] trips of [call] in a loop over rows of four of [ys] and [xs], each of
   [size] floats (default [4 * n]), the call's arguments [args ys xs r] on
   the trip [r]. *)
let looped ?size ~n call args =
  let size = Option.value size ~default:(4 * n) in
  let ys = Ops.new_buffer "CPU" size Float32
  and xs = Ops.new_buffer "CPU" size Float32 in
  let r = Ops.range ~axis_type:Loop (Int n) [ 100 ] in
  let e = Ops.end_ (call (args ys xs r)) [ r ] in
  (ys, xs, Ops.sink [ Ops.after ys [ e ] ])

let row r b =
  let at = Ops.O.(r * int 4) in
  Shape.shrink b [ Some (Sym at, Sym Ops.O.(at + int 4)) ]

(* The states a call reads, each trip of its loop, through views of them. *)
let states =
  [
      loop_reads_law "CPU";
      loop_states_law "CPU";
      loop_in_body_law "CPU";
      test
        "a call given an older state to write and a later to read reads it \
         before the call"
        older_write_reads_before;
      test
        "a kernel listed over two storages names the later state in either \
         listing"
        listed_two_storages_either_order;
      test
        "a call reads storage it writes off its parameter's start, no after \
         listing the storage"
        off_start_read_no_after;
      test "a loop reads earlier states as they were" loop_reads_earlier_states;
      test "runs of one other storage land as listed in an after"
        foreign_runs_land;
      test "a loop's value of a state it writes is computed each trip"
        loop_values_each_trip;
      test "a loop reads a value of several states each trip"
        loop_values_of_several_reads;
      test "a loop's value read twice in the body, or sharing a leaf"
        loop_value_shared;
      test "a loop's value through a bound variable" loop_value_bound_variable;
      test "a state over a view that is no run is read each trip"
        state_over_a_flip_each_trip;
      test "a state over a view of a state over a view is read each trip"
        state_over_nested_views_each_trip;
      test "a state over a flip, two states deep, is read each trip"
        state_over_a_flip_two_states_deep;
      test "a store body takes a value computed each trip" loop_value_store_body;
      test "a kernel body takes a value computed each trip"
        loop_value_kernel_body;
      test "a back edge's value is computed each trip" backedge_value_each_trip;
      test "a contiguous value of a written read is read each trip"
        loop_contiguous_each_trip;
      test "a state over a view a loop moves, ordered after effects, is read \
            each trip"
        moving_state_after_effects;
      test "a value ordered after effects is the value" value_after_effects;
      test "a loop's fixed read of an After state it writes is read each trip"
        reads_after_run_each_trip;
      test "an After state's fixed run off its parameter's start"
        reads_after_run_off_start;
      test "an After state's fixed runs at other offsets"
        reads_after_run_at_offsets;
      test "an After state's fixed run under a back edge"
        reads_after_run_backedge;
      test "an After state's fixed run through a join"
        reads_after_run_through_join;
      test "a fixed run of the state two stores deep" reads_after_run_chain;
      test "two fixed runs of an After state" reads_two_after_runs;
      test "a loop named by an older state, a run of the later"
        reads_after_run_named_older;
      test "a bitcast run of an After state read each trip"
        reads_after_bitcast_run;
      test "two loops in sequence read their runs" chained_loops_read_runs;
      test "a loop in a body reads its parameter's run each trip"
        loop_in_body_reads_each_trip;
      test "nested loops of a call run the outer range slowest"
        nested_loops_of_a_call;
      test "a loop's older run of storage leaves the state before it"
        loop_reads_older_plain_run;
      test "a loop reads a join of its own states each trip"
        loop_reads_join_of_its_own_states;
      test "a loop reading a state older than its listing reads it as it was"
        loop_reads_older_than_listing;
      test "a store through a run a call reads lands" store_through_read_run_lands;
      test "a store through a reshape of a run a call reads lands"
        store_through_reshape_of_read_lands;
      test "a loop reading a run of the state under a store through it keeps \
            the store"
        loop_reads_store_target_older;
      test "a whole state holds the sibling store its own store reads"
        sibling_a_state_reads_is_part_of_it;
      test "a store through a run of a bitcast of a state"
        store_through_run_of_a_bitcast_state;
      test "a weak value through a run of a bitcast"
        weak_value_through_a_run_of_a_bitcast;
      test "one constant stored at two states through bitcasts" shared_constant_stores;
      test "one value stored whole into storages of other sizes"
        shared_value_stores_of_other_sizes;
      test "a loop over a bitcast state writes an older name of its storage"
        loop_over_a_bitcast_state_writes_an_older_name;
      test "a call reads states of a bitcast and of its storage"
        call_reads_states_of_bitcasts;
      test "a later state over a bitcast is read each trip"
        later_state_over_bitcast_read_each_trip;
      test "a loop reads a run of a state of a bitcast each trip"
        loop_reads_run_of_a_bitcast_state;
      test "an older state's run stays as it was" reads_older_run;
      test "a call's fixed read sees what its loop wrote"
        reads_what_its_loop_wrote;
      test "a loop's fixed read off its parameter's start is read each trip"
        reads_off_start_each_trip;
      test "a back edge's fixed read sees what its loop wrote"
        backedge_reads_each_trip;
      test "a loop reads a flip of its written storage each trip"
        loop_flip_of_written_storage;
  ]

(* A call compiled on its own reads and writes views of storage. *)
let views =
  let reads (call : copy) view expected () =
    let a = Ops.new_buffer "CPU" 8 Float32
    and d = Ops.new_buffer "CPU" 4 Float32 in
    let sink = Ops.sink [ Ops.after d [ call [ d; view a ] ] ] in
    run_copy ~into:d ~init:(Array.make 4 0.) ~from:a a_values sink
    |> equal values (floats expected)
  in
  let writes ~msg (call : copy) view expected =
    let a = Ops.new_buffer "CPU" 8 Float32
    and y = Ops.new_buffer "CPU" 4 Float32 in
    let sink = Ops.sink [ Ops.after a [ call [ view a; y ] ] ] in
    run_copy ~into:a ~init:a_values ~from:y y_values sink
    |> equal ~msg values (floats expected)
  in
  [
      test "a call reads a view of storage at its elements" (fun () ->
          List.iter
            (fun (_, call) ->
              reads call at_offset [| 4.; 5.; 6.; 7. |] ();
              reads call off_boundary [| 1.; 2.; 3.; 4. |] ();
              reads call every_other [| 0.; 2.; 4.; 6. |] ();
              reads call transposed [| 0.; 4.; 1.; 5. |] ())
            copies);
      test "a call reads storage off its parameter's start through a copy"
        (fun () ->
          List.iter
            (fun (msg, (call : copy)) ->
              let y = Ops.new_buffer ~phase:4 "CPU" 4 Float32
              and d = Ops.new_buffer "CPU" 4 Float32 in
              let sink = Ops.sink [ Ops.after d [ call [ d; y ] ] ] in
              run_copy ~at:off_phase ~into:d ~init:(Array.make 4 0.) ~from:y
                y_values sink
              |> equal ~msg values (floats y_values))
            copies);
      test "a call reads a state of a view at its elements" (fun () ->
          List.iter
            (fun (msg, (call : copy)) ->
              let a = Ops.new_buffer "CPU" 8 Float32
              and d = Ops.new_buffer "CPU" 4 Float32 in
              let stored = Ops.store (at_offset a) (first_four a) in
              let state = Ops.after (at_offset a) [ stored ] in
              let sink = Ops.sink [ Ops.after d [ call [ d; state ] ] ] in
              run_copy ~into:d ~init:(Array.make 4 0.) ~from:a a_values sink
              |> equal ~msg values (floats [| 0.; 1.; 2.; 3. |]))
            copies);
      test "a call writes through a run of storage from its first element"
        (fun () ->
          let written = [| 1.; 2.; 3.; 4.; 4.; 5.; 6.; 7. |] in
          List.iter
            (fun (call, make) ->
              writes ~msg:(call ^ ", its first four") make first_four written;
              writes ~msg:(call ^ ", its first row") make first_row written)
            copies);
      test "a call writes through a run of storage wherever it starts"
        (fun () ->
          (* Twelve floats, written at the run of four from element [lo]:
             the first, the middle and the last, each on a 16-byte
             boundary. *)
          List.iter
            (fun (call, (make : copy)) ->
              List.iter
                (fun lo ->
                  let a = Ops.new_buffer "CPU" 12 Float32
                  and y = Ops.new_buffer "CPU" 4 Float32 in
                  let run =
                    Shape.shrink a [ Some (Ops.Int lo, Ops.Int (lo + 4)) ]
                  in
                  let sink = Ops.sink [ Ops.after a [ make [ run; y ] ] ] in
                  let init = Array.init 12 Float.of_int in
                  let expected =
                    Array.mapi
                      (fun i x ->
                        if i >= lo && i < lo + 4 then y_values.(i - lo) else x)
                      init
                  in
                  run_copy ~into:a ~init ~from:y y_values sink
                  |> equal
                       ~msg:(Printf.sprintf "%s, from element %d" call lo)
                       values (floats expected))
                [ 0; 4; 8 ])
            copies);
      test "a call writes through a run of a run of storage" (fun () ->
          (* The elements 4 to 8 of the run from element 4 of twelve: the run
             from element 8, a run's run. *)
          List.iter
            (fun (call, (make : copy)) ->
              let a = Ops.new_buffer "CPU" 12 Float32
              and y = Ops.new_buffer "CPU" 4 Float32 in
              let outer = Shape.shrink a [ Some (Ops.Int 4, Ops.Int 12) ] in
              let run = Shape.shrink outer [ Some (Ops.Int 4, Ops.Int 8) ] in
              let sink = Ops.sink [ Ops.after a [ make [ run; y ] ] ] in
              let init = Array.init 12 Float.of_int in
              let expected =
                Array.mapi
                  (fun i x -> if i >= 8 then y_values.(i - 8) else x)
                  init
              in
              run_copy ~into:a ~init ~from:y y_values sink
              |> equal ~msg:call values (floats expected))
            copies);
      test "a call writes through a row of a reshaped run of storage"
        (fun () ->
          (* The first row of elements [lo] to [lo + 8] of twelve, seen as
             two rows of four: a run of the storage from element [lo],
             through a reshape, which no shrink of a shrink folds. *)
          List.iter
            (fun (call, (make : copy)) ->
              List.iter
                (fun lo ->
                  let a = Ops.new_buffer "CPU" 12 Float32
                  and y = Ops.new_buffer "CPU" 4 Float32 in
                  let inner =
                    Shape.shrink a [ Some (Ops.Int lo, Ops.Int (lo + 8)) ]
                  in
                  let run =
                    Shape.shrink
                      (Shape.reshape inner (dims [ 2; 4 ]))
                      [ Some (Ops.Int 0, Ops.Int 1); None ]
                  in
                  let sink = Ops.sink [ Ops.after a [ make [ run; y ] ] ] in
                  let init = Array.init 12 Float.of_int in
                  let expected =
                    Array.mapi
                      (fun i x ->
                        if i >= lo && i < lo + 4 then y_values.(i - lo) else x)
                      init
                  in
                  run_copy ~into:a ~init ~from:y y_values sink
                  |> equal
                       ~msg:(Printf.sprintf "%s, from element %d" call lo)
                       values (floats expected))
                [ 0; 4 ])
            copies);
      test "a second kernel writes through the first's state of a run"
        (fun () ->
          let a = Ops.new_buffer "CPU" 8 Float32
          and y = Ops.new_buffer "CPU" 4 Float32 in
          let copy = copy_kernel ~dst:0 ~src:1 in
          let first = Shape.custom_kernel [ first_four a; y ] copy in
          let second =
            Shape.custom_kernel [ List.hd first; at_offset a ] copy
          in
          let sink = Ops.sink [ Ops.after a [ List.hd second ] ] in
          run_copy ~into:a ~init:a_values ~from:y y_values sink
          |> equal values (floats [| 4.; 5.; 6.; 7.; 4.; 5.; 6.; 7. |]));
      test "a call in a loop writes through its row of storage on each trip"
        (fun () ->
          List.iter
            (fun (msg, (call : copy)) ->
              let xs_values = Array.init 12 Float.of_int in
              let ys, xs, sink =
                looped ~n:3 (fun args -> call args) (fun ys xs r ->
                    [ row r ys; row r xs ])
              in
              run_copy ~into:ys ~init:(Array.make 12 0.) ~from:xs xs_values sink
              |> equal ~msg values (floats xs_values))
            copies);
  ]

let byte_stores =
  [
    byte_views "CPU";
    cut_byte_views "CPU";
    test
      "a store through a bitcast of a view that is no run writes the value's \
       bits through the view"
      stores_through_bitcast_views;
  ]

(* The laws on the GPU [kind], at the device the programs name CPU:1. A GPU's
   tests skip without one. *)
let on_a_gpu kind =
  let devices = on_gpu kind in
  [
    loop_reads_law ~devices "CPU:1";
    loop_states_law ~devices "CPU:1";
    byte_views ~devices "CPU:1";
    cut_byte_views ~devices "CPU:1";
  ]
