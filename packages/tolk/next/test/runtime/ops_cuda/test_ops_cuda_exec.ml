open Windtrap
open Tolk_next
module B = Nx_device.Buffer

(* The CUDA devices, and a skip where there are too few. *)
let gpus = lazy (List.init (Nx_cuda_device.count ()) Nx_cuda_device.v)

let cuda () =
  match Lazy.force gpus with
  | d :: _ -> d
  | [] -> skip ~reason:"no CUDA device" ()

let devices () =
  let named =
    List.mapi
      (fun i d -> ((if i = 0 then "CUDA" else Printf.sprintf "CUDA:%d" i), d))
      (Lazy.force gpus)
  in
  ignore (cuda ());
  Tolk_next_engine.device (("CPU", Nx_device.host) :: named)

let nx name = (devices () name).device

(* Buffers *)

let floats = array float_exact

(* A GPU's memory is not the host's: values cross by copies. *)
let floats_of b =
  let a = Bigarray.(Array1.create float32 c_layout (B.nbytes b / 4)) in
  B.copy ~src:b ~dst:(B.of_bigarray a);
  Array.init (Bigarray.Array1.dim a) (fun i -> a.{i})

let new_floats name xs =
  let b = B.create (nx name) Float32 (Array.length xs) in
  B.copy
    ~src:(B.of_bigarray (Bigarray.(Array1.of_array float32 c_layout) xs))
    ~dst:b;
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
  Nx_device.synchronize (cuda ());
  s

let chain n = List.init (n + 1) (fun _ -> storage "CUDA")

let chained bufs =
  List.init
    (List.length bufs - 1)
    (fun k -> adds (List.nth bufs (k + 1)) (List.nth bufs k))

let slot u = match Ops.arg u with Param p -> p.slot | _ -> -1

let execution =
  group "execution"
    [
      slow "a chain of kernels computes what the interpreter says" (fun () ->
          let b = chain 3 in
          let calls = chained b in
          let input = [| 1.; 2.; 3.; 4. |] in
          let bound = List.map (fun u -> (u, [ new_floats "CUDA" input ])) b in
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
          let a0 = storage "CUDA" and a = storage "CUDA" in
          let h = storage "CPU" and h2 = storage "CPU" in
          let b = storage "CUDA" and b2 = storage "CUDA" in
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
          let vram = storage ~n:1024 "CUDA" in
          let host = storage ~n:1024 "CPU" and fresh = storage ~n:1024 "CPU" in
          let data = Array.init 1024 float_of_int in
          let bound =
            [
              (vram, [ new_floats "CUDA" (Array.make 1024 0.) ]);
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
                     (param ~n (Single "CUDA") 1)
                     (param ~n (Single "CUDA") 0);
                 ])
          in
          let runs =
            List.map
              (fun x ->
                ( x,
                  new_floats "CUDA" (Array.make n x),
                  new_floats "CUDA" (Array.make n 0.) ))
              [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. ]
          in
          List.iter
            (fun (_, src, dst) -> Tolk_next_engine.run s [| [ src ]; [ dst ] |])
            runs;
          Nx_device.synchronize (cuda ());
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
            List.map (fun u -> (u, [ new_floats "CUDA" (Array.make 4 0.) ])) b
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
                  when Nx_device.equal device (cuda ()) ->
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
            List.map (fun u -> (u, [ new_floats "CUDA" (Array.make 4 0.) ])) b
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
                Nx_device.synchronize (cuda ());
                Nx_device.Profile.stop p)
          in
          let spans =
            List.filter
              (function
                | Nx_device.Profile.Span { device; _ } ->
                    Nx_device.equal device (cuda ())
                | _ -> false)
              events
          in
          equal int 1 (List.length spans));
      slow "a launch size that reads a variable is set on each run" (fun () ->
          let v =
            Ops.variable ~dtype:Int32 "v" (`Int Bigint.one)
              (`Int (Bigint.of_int 4))
          in
          let out = param (Single "CUDA") 0 in
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
          let o = storage "CUDA" in
          let bound = [ (o, [ new_floats "CUDA" (Array.make 4 0.) ]) ] in
          let s =
            run_calls ~vars:[ ("v", 3) ] ~bound [ Ops.call kernel [ o ] ]
          in
          equal floats [| 9.; 9.; 9.; 0. |]
            (floats_of (List.hd (List.assq o bound)));
          Tolk_next_engine.run ~vars:[ ("v", 4) ] s [||];
          Nx_device.synchronize (cuda ());
          equal floats [| 9.; 9.; 9.; 9. |]
            (floats_of (List.hd (List.assq o bound))));
      slow "a kernel of 33 buffers runs from its arguments' buffer" (fun () ->
          let n = 32 in
          let out = param (Single "CUDA") 0 in
          let ins = List.init n (fun k -> param (Single "CUDA") (k + 1)) in
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
          let o = storage "CUDA"
          and xs = List.init n (fun _ -> storage "CUDA") in
          let bound =
            (o, [ new_floats "CUDA" (Array.make 4 0.) ])
            :: List.mapi
                 (fun k x ->
                   (x, [ new_floats "CUDA" (Array.make 4 (float_of_int k)) ]))
                 xs
          in
          ignore (run_calls ~bound [ Ops.call kernel (o :: xs) ]);
          equal floats
            (Array.make 4 (float_of_int (n * (n - 1) / 2)))
            (floats_of (List.hd (List.assq o bound))));
      slow "a copy between two GPUs goes through the host's staging memory"
        (fun () ->
          (match Lazy.force gpus with
          | _ :: _ :: _ -> ()
          | _ -> skip ~reason:"fewer than two CUDA devices" ());
          let a = storage "CUDA" and b = storage "CUDA:1" in
          let bound =
            [
              (a, [ new_floats "CUDA" [| 1.; 2.; 3.; 4. |] ]);
              (b, [ new_floats "CUDA:1" (Array.make 4 0.) ]);
            ]
          in
          ignore (run_calls ~bound [ Ops.store_call b a ]);
          Nx_device.synchronize (nx "CUDA:1");
          equal floats [| 1.; 2.; 3.; 4. |]
            (floats_of (List.hd (List.assq b bound))));
      slow "each trip of a range runs its kernel on its own window" (fun () ->
          let r = Ops.range (Int 3) [ 7 ] in
          let window u =
            let start = Ops.mul r (Ops.int 4) in
            Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
          in
          let src = storage ~n:12 "CUDA" and dst = storage ~n:12 "CUDA" in
          let bound =
            [
              (src, [ new_floats "CUDA" (Array.init 12 float_of_int) ]);
              (dst, [ new_floats "CUDA" (Array.make 12 0.) ]);
            ]
          in
          ignore
            (run_calls ~bound
               [ Ops.end_ (adds (window dst) (window src)) [ r ] ]);
          equal floats
            (Array.init 12 (fun i -> float_of_int (i + 1)))
            (floats_of (List.hd (List.assq dst bound))));
    ]

let () = exit (run "Tolk_next.Ops_cuda (execution)" [ execution ])
