open Windtrap
open Tolk_next
module B = Nx_device.Buffer

(* The Metal device, or a skip where there is none. *)
let metal =
  lazy (match Nx_metal_device.get 0 with Ok d -> Some d | Error _ -> None)

let metal () =
  match Lazy.force metal with
  | Some d -> d
  | None -> skip ~reason:"no Metal device" ()

let devices () =
  Tolk_next_engine.device [ ("CPU", Nx_device.host); ("METAL", metal ()) ]

let nx name = (devices () name).device

(* Buffers *)

let floats = array float_exact

let host_view b =
  if Nx_device.equal (B.device b) Nx_device.host then b
  else Result.get_ok (B.borrow Nx_device.host b)

let floats_of b =
  let a = B.bigarray Bigarray.float32 (host_view b) in
  Array.init (Bigarray.Array1.dim a) (fun i -> a.{i})

let new_floats name xs =
  let b = B.create (nx name) Float32 (Array.length xs) in
  let a = B.bigarray Bigarray.float32 (host_view b) in
  Array.iteri (fun i x -> a.{i} <- x) xs;
  b

(* Kernels *)

let param ?(n = 4) device slot = Ops.param ~shape:[ Int n ] ~device slot Float32

(* The kernel that stores [x + c] of each element [x] of slot 1 into slot 0. *)
let adds_kernel ?(n = 4) ?(c = 1.) device =
  let out = param ~n device 0 and inp = param ~n device 1 in
  let i = Ops.range (Int n) [ 0 ] in
  let x = Ops.load (Ops.index inp [ i ]) [] in
  let st =
    Ops.store (Ops.index out [ i ]) (Ops.add x (Ops.float ~dtype:Float32 c))
  in
  Ops.sink ~kernel:(Ops.kernel_info ~name:"k" ()) [ Ops.end_ st [ i ] ]

let adds ?n ?c out inp =
  Ops.call (adds_kernel ?n ?c (Option.get (Ops.device out))) [ out; inp ]

let storage ?(n = 4) device = Ops.new_buffer (Single device) n Float32
let linear calls = Ops.v Linear ~src:calls

let compile ?profile calls =
  let devices = devices () in
  Hcq2.compile_linear ?profile
    ~devices:(fun n -> (devices n).compiler)
    (linear calls)

let run_calls ?profile ?(vars = []) ?(slots = [||]) ~bound calls =
  let s =
    Tolk_next_engine.link ~devices:(devices ()) ~bound (compile ?profile calls)
  in
  Tolk_next_engine.run ~vars s slots;
  Nx_device.synchronize (metal ());
  s

let chain n = List.init (n + 1) (fun _ -> storage "METAL")

let chained bufs =
  List.init
    (List.length bufs - 1)
    (fun k -> adds (List.nth bufs (k + 1)) (List.nth bufs k))

let slot u = match Ops.arg u with Param p -> p.slot | _ -> -1

(* A range of [n] trips around a kernel that adds one to each window of four
   floats. *)
let ranged n =
  let r = Ops.range (Int n) [ 7 ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let src = storage ~n:(4 * n) "METAL" and dst = storage ~n:(4 * n) "METAL" in
  (src, dst, Ops.end_ (adds (window dst) (window src)) [ r ])

let windows_bound n src dst =
  [
    (src, [ new_floats "METAL" (Array.init (4 * n) float_of_int) ]);
    (dst, [ new_floats "METAL" (Array.make (4 * n) 0.) ]);
  ]

let windows n () =
  let src, dst, e = ranged n in
  let bound = windows_bound n src dst in
  ignore (run_calls ~bound [ e ]);
  equal floats
    (Array.init (4 * n) (fun i -> float_of_int (i + 1)))
    (floats_of (List.hd (List.assq dst bound)))

(* Phase (D54) *)

(* The floor of a float16 [3; 4] of strides [1; 3] over the twelve halves of a
   buffer whose first lies [offset] bytes into its memory, as a kernel that
   takes it to lie there: [offset] bytes past a 16-byte boundary. Code
   generation reads the twelve halves as three vectors of four, whose addresses
   the phase keeps aligned. *)
let floor_of_strided offset =
  let phase = offset in
  let half ?phase slot =
    Ops.param ~shape:[ Int 12 ] ~device:(Single "METAL") ?phase slot Float16
  in
  let out = half 0 and inp = half ~phase 1 in
  let i = Ops.range (Int 3) [ 0 ] and j = Ops.range (Int 4) [ 1 ] in
  let x = Ops.cast (Ops.index inp [ Ops.O.((j * int 3) + i) ]) Float32 in
  let y = Ops.cast (Ops.floor x) Float16 in
  let kernel =
    Ops.sink
      ~kernel:(Ops.kernel_info ~name:"floor" ())
      [
        Ops.end_
          (Ops.store (Ops.index out [ Ops.O.((i * int 4) + j) ]) y)
          [ i; j ];
      ]
  in
  let memory = B.create (nx "METAL") Float16 20 in
  let halves b = B.bigarray Bigarray.float16 (host_view b) in
  let m = halves memory in
  for k = 0 to 19 do
    m.{k} <- 100.5
  done;
  let input = B.view memory ~offset Float16 12 in
  let v = halves input in
  for k = 0 to 11 do
    v.{k} <- float_of_int k +. 0.5
  done;
  let result = B.create (nx "METAL") Float16 12 in
  let o = Ops.new_buffer (Single "METAL") 12 Float16
  and x = Ops.new_buffer ~phase (Single "METAL") 12 Float16 in
  ignore
    (run_calls
       ~bound:[ (o, [ result ]); (x, [ input ]) ]
       [ Ops.call kernel [ o; x ] ]);
  let r = halves result in
  Array.init 12 (fun k -> r.{k})

(* The floor of each element the view reaches, in the view's order. *)
let floors = Array.init 12 (fun k -> float_of_int ((k / 4) + (3 * (k mod 4))))

let phases =
  [
    slow
      "a float16 buffer 2 or 6 bytes into its memory is read where it lies \
       with its phase" (fun () ->
        List.iter
          (fun offset ->
            equal ~msg:(string_of_int offset) floats floors
              (floor_of_strided offset))
          [ 2; 6 ]);
  ]

let execution =
  group "execution"
    [
      slow "a chain of kernels computes what the interpreter says" (fun () ->
          let b = chain 3 in
          let calls = chained b in
          let input = [| 1.; 2.; 3.; 4. |] in
          let bound = List.map (fun u -> (u, [ new_floats "METAL" input ])) b in
          ignore (run_calls ~bound calls);
          let interpreted =
            Kernel_graphs.linear_writes
              ~buffers:
                [ (slot (List.hd b), Array.map (fun x -> `Float x) input) ]
              (linear calls)
          in
          List.iter
            (fun out ->
              let expected =
                List.filter_map
                  (fun (s, _, v) ->
                    match v with
                    | `Float x when s = slot out -> Some x
                    | _ -> None)
                  interpreted
              in
              equal floats (Array.of_list expected)
                (floats_of (List.hd (List.assq out bound))))
            (List.tl b));
      slow "a copy to the host and back runs on the host, between batches"
        (fun () ->
          let a0 = storage "METAL" and a = storage "METAL" in
          let h = storage "CPU" and h2 = storage "CPU" in
          let b = storage "METAL" and b2 = storage "METAL" in
          let bound =
            List.map
              (fun u ->
                let d = List.hd (Option.to_list (Ops.device u)) in
                let name =
                  match d with Single n -> n | Multi ns -> List.hd ns
                in
                (u, [ new_floats name [| 1.; 2.; 3.; 4. |] ]))
              [ a0; a; h; h2; b; b2 ]
          in
          let calls =
            [
              adds a a0;
              Ops.store_call h a;
              adds ~c:3. h2 h;
              Ops.store_call b h2;
              adds b2 b;
            ]
          in
          ignore (run_calls ~bound calls);
          equal floats [| 6.; 7.; 8.; 9. |]
            (floats_of (List.hd (List.assq b2 bound))));
      slow "copies out, in and out again leave the host the bytes copied in"
        (fun () ->
          let vram = storage ~n:1024 "METAL" in
          let host = storage ~n:1024 "CPU" and fresh = storage ~n:1024 "CPU" in
          let data = Array.init 1024 float_of_int in
          let bound =
            [
              (vram, [ new_floats "METAL" (Array.make 1024 0.) ]);
              (host, [ new_floats "CPU" (Array.make 1024 0.) ]);
              (fresh, [ new_floats "CPU" data ]);
            ]
          in
          let copyout = Ops.store_call host vram in
          ignore
            (run_calls ~bound [ copyout; Ops.store_call vram fresh; copyout ]);
          equal floats data (floats_of (List.hd (List.assq host bound))));
      slow
        "a run waits for its batch's previous run before it rewrites the \
         batch's arguments" (fun () ->
          let n = 1 lsl 20 in
          let s =
            Tolk_next_engine.link ~devices:(devices ())
              (compile
                 [
                   adds ~n ~c:10.
                     (param ~n (Single "METAL") 1)
                     (param ~n (Single "METAL") 0);
                 ])
          in
          let runs =
            List.map
              (fun x ->
                ( x,
                  new_floats "METAL" (Array.make n x),
                  new_floats "METAL" (Array.make n 0.) ))
              [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. ]
          in
          List.iter
            (fun (_, src, dst) -> Tolk_next_engine.run s [| [ src ]; [ dst ] |])
            runs;
          Nx_device.synchronize (metal ());
          List.iter
            (fun (x, _, dst) ->
              let got = floats_of dst in
              equal float_exact ~msg:(string_of_float x) (x +. 10.) got.(0);
              equal float_exact ~msg:(string_of_float x) (x +. 10.) got.(n - 1))
            runs);
      slow "a profile records a span of each kernel on the device, in order"
        (fun () ->
          let b = chain 2 in
          let bound =
            List.map (fun u -> (u, [ new_floats "METAL" (Array.make 4 0.) ])) b
          in
          let p = Nx_device.Profile.start () in
          let events =
            Fun.protect
              ~finally:(fun () ->
                if Nx_device.Profile.enabled () then
                  ignore (Nx_device.Profile.stop p))
              (fun () ->
                ignore (run_calls ~profile:true ~bound (chained b));
                Nx_device.Profile.stop p)
          in
          let spans =
            List.filter_map
              (function
                | Nx_device.Profile.Span { device; name; start; stop; _ }
                  when Nx_device.equal device (metal ()) ->
                    Some (name, start, stop)
                | _ -> None)
              events
          in
          equal (list string) [ "k"; "k" ] (List.map (fun (n, _, _) -> n) spans);
          List.iter
            (fun (_, start, stop) -> at_least int ~than:start stop)
            spans;
          match spans with
          | [ (_, _, first_stop); (_, second_start, _) ] ->
              at_least int ~than:first_stop second_start
          | _ -> ());
      slow "a profiled batch run twice keeps the second run's spans" (fun () ->
          let b = chain 1 in
          let bound =
            List.map (fun u -> (u, [ new_floats "METAL" (Array.make 4 0.) ])) b
          in
          let s =
            Tolk_next_engine.link ~devices:(devices ()) ~bound
              (compile ~profile:true (chained b))
          in
          let p = Nx_device.Profile.start () in
          let events =
            Fun.protect
              ~finally:(fun () ->
                if Nx_device.Profile.enabled () then
                  ignore (Nx_device.Profile.stop p))
              (fun () ->
                Tolk_next_engine.run s [||];
                Tolk_next_engine.run s [||];
                Nx_device.synchronize (metal ());
                Nx_device.Profile.stop p)
          in
          let spans =
            List.filter
              (function
                | Nx_device.Profile.Span { device; _ } ->
                    Nx_device.equal device (metal ())
                | _ -> false)
              events
          in
          equal int 1 (List.length spans));
      slow "a launch size that reads a variable is set on each run" (fun () ->
          let v =
            Ops.variable ~dtype:Int32 "v" (`Int Z.one) (`Int (Z.of_int 4))
          in
          let out = param (Single "METAL") 0 in
          let i = Ops.range (Sym v) [ 0 ] in
          let kernel =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"up_to_v" ())
              [
                Ops.end_
                  (Ops.store (Ops.index out [ i ])
                     (Ops.float ~dtype:Float32 9.))
                  [ i ];
              ]
          in
          let o = storage "METAL" in
          let bound = [ (o, [ new_floats "METAL" (Array.make 4 0.) ]) ] in
          let s =
            run_calls ~vars:[ ("v", 3) ] ~bound [ Ops.call kernel [ o ] ]
          in
          equal floats [| 9.; 9.; 9.; 0. |]
            (floats_of (List.hd (List.assq o bound)));
          Tolk_next_engine.run ~vars:[ ("v", 4) ] s [||];
          Nx_device.synchronize (metal ());
          equal floats [| 9.; 9.; 9.; 9. |]
            (floats_of (List.hd (List.assq o bound))));
      slow "each trip of a range sets the launch size that reads a variable"
        (fun () ->
          let v =
            Ops.variable ~dtype:Int32 "v" (`Int Z.one) (`Int (Z.of_int 4))
          in
          let out = param (Single "METAL") 0 in
          let i = Ops.range (Sym v) [ 0 ] in
          let kernel =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"up_to_v" ())
              [
                Ops.end_
                  (Ops.store (Ops.index out [ i ])
                     (Ops.float ~dtype:Float32 9.))
                  [ i ];
              ]
          in
          let r = Ops.range (Int 3) [ 7 ] in
          let o = storage ~n:12 "METAL" in
          let start = Ops.mul r (Ops.int 4) in
          let window =
            Ops.shrink o [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
          in
          let bound = [ (o, [ new_floats "METAL" (Array.make 12 0.) ]) ] in
          ignore
            (run_calls
               ~vars:[ ("v", 3) ]
               ~bound
               [ Ops.end_ (Ops.call kernel [ window ]) [ r ] ]);
          equal floats
            (Array.init 12 (fun k -> if k mod 4 < 3 then 9. else 0.))
            (floats_of (List.hd (List.assq o bound))));
      slow "a kernel of 33 buffers runs from its arguments' buffer" (fun () ->
          let n = 32 in
          let out = param (Single "METAL") 0 in
          let ins = List.init n (fun k -> param (Single "METAL") (k + 1)) in
          let i = Ops.range (Int 4) [ 0 ] in
          let sum =
            List.fold_left
              (fun acc p -> Ops.add acc (Ops.load (Ops.index p [ i ]) []))
              (Ops.float ~dtype:Float32 0.)
              ins
          in
          let kernel =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"sum33" ())
              [ Ops.end_ (Ops.store (Ops.index out [ i ]) sum) [ i ] ]
          in
          let o = storage "METAL"
          and xs = List.init n (fun _ -> storage "METAL") in
          let bound =
            (o, [ new_floats "METAL" (Array.make 4 0.) ])
            :: List.mapi
                 (fun k x ->
                   (x, [ new_floats "METAL" (Array.make 4 (float_of_int k)) ]))
                 xs
          in
          ignore (run_calls ~bound [ Ops.call kernel (o :: xs) ]);
          equal floats
            (Array.make 4 (float_of_int (n * (n - 1) / 2)))
            (floats_of (List.hd (List.assq o bound))));
      slow "each trip of a range runs its kernel on its own window" (windows 3);
      slow "a range of 20000 trips runs from one indirect command buffer"
        (windows 20000);
      slow "a profiled range records a span of each trip's kernel" (fun () ->
          let src, dst, e = ranged 3 in
          let p = Nx_device.Profile.start () in
          let events =
            Fun.protect
              ~finally:(fun () ->
                if Nx_device.Profile.enabled () then
                  ignore (Nx_device.Profile.stop p))
              (fun () ->
                ignore
                  (run_calls ~profile:true ~bound:(windows_bound 3 src dst)
                     [ e ]);
                Nx_device.Profile.stop p)
          in
          let spans =
            List.filter
              (function
                | Nx_device.Profile.Span { device; _ } ->
                    Nx_device.equal device (metal ())
                | _ -> false)
              events
          in
          equal int 3 (List.length spans));
    ]

let () =
  exit
    (run "Tolk_next.Ops_metal (execution)"
       [ execution; group "phase (D54)" phases ])
