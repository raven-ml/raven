(* Batches of NV's queues run through the engine on the first NVIDIA GPU: each
   test skips where there is none. *)

open Windtrap
open Tolk
module B = Nx_device.Buffer

(* The first NVIDIA GPU through the kernel driver, or a skip where there is
   none. *)
let nv =
  lazy
    (match Nx_nv_device.get ~interface:Kernel 0 with
    | Ok d -> Some d
    | Error _ -> None)

let nv () =
  match Lazy.force nv with
  | Some d -> d
  | None -> skip ~reason:"no NVIDIA GPU" ()

let devices () = Tolk_engine.device [ ("CPU", Nx_device.host); ("NV", nv ()) ]
let nx name = (devices () name).device

(* Buffers, written and read through the host *)

let floats = array float_exact

let floats_of b =
  let host = B.create Nx_device.host Float32 (B.length b) in
  B.copy ~src:b ~dst:host;
  let a = B.bigarray Bigarray.float32 host in
  Array.init (Bigarray.Array1.dim a) (fun i -> a.{i})

let new_floats name xs =
  let host = B.create Nx_device.host Float32 (Array.length xs) in
  let a = B.bigarray Bigarray.float32 host in
  Array.iteri (fun i x -> a.{i} <- x) xs;
  if name = "CPU" then host
  else
    let b = B.create (nx name) Float32 (Array.length xs) in
    B.copy ~src:host ~dst:b;
    b

(* Kernels *)

(* The call of the kernel storing [x + 1] of each element [x] of [inp] into
   [out], four floats each. *)
let adds out inp =
  let device = Option.get (Ops.device out) in
  let param slot = Ops.param ~shape:[ Int 4 ] ~device slot Float32 in
  let i = Ops.range (Int 4) [ 0 ] in
  let x = Ops.load (Ops.index (param 1) [ i ]) [] in
  let st =
    Ops.store
      (Ops.index (param 0) [ i ])
      (Ops.add x (Ops.float ~dtype:Float32 1.))
  in
  Ops.call
    (Ops.sink ~kernel:(Ops.kernel_info ~name:"k" ()) [ Ops.end_ st [ i ] ])
    [ out; inp ]

let storage ?(n = 4) device = Ops.new_buffer (Single device) n Float32
let linear calls = Ops.v Linear ~src:calls

let compile ?(profile = Hcq2.Unstamped) calls =
  let devices = devices () in
  Hcq2.compile_linear ~profile
    ~devices:(fun n -> (devices n).compiler)
    (linear calls)

let link ?profile ~bound calls =
  Tolk_engine.link ~devices:(devices ()) ~bound (compile ?profile calls)

let run_calls ?profile ~bound calls =
  let s = link ?profile ~bound calls in
  Tolk_engine.run s [||];
  Nx_device.synchronize (nv ());
  s

let chain n = List.init (n + 1) (fun _ -> storage "NV")

let chained bufs =
  List.init
    (List.length bufs - 1)
    (fun k -> adds (List.nth bufs (k + 1)) (List.nth bufs k))

let bound_to xs = List.map (fun u -> (u, [ new_floats "NV" xs ]))

(* A range of [n] trips around [k] kernels adding one, each trip on its own
   windows of four floats: the first kernel reads [src], each writes a buffer of
   its own, which the next reads. *)
let ranged ~k n =
  let r = Ops.range (Int n) [ 7 ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Ops.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let bufs = List.init (k + 1) (fun _ -> storage ~n:(4 * n) "NV") in
  let calls =
    List.init k (fun i ->
        adds (window (List.nth bufs (i + 1))) (window (List.nth bufs i)))
  in
  (bufs, Ops.end_ (linear calls) [ r ])

let spans events =
  List.filter_map
    (function
      | Nx_device.Profile.Span { device; name; start; stop; _ }
        when Nx_device.equal device (nv ()) ->
          Some (name, start, stop)
      | _ -> None)
    events

let profiled f =
  let p = Nx_device.Profile.start () in
  Fun.protect
    ~finally:(fun () ->
      if Nx_device.Profile.enabled () then ignore (Nx_device.Profile.stop p))
    (fun () ->
      f ();
      Nx_device.Profile.stop p)

(* Each trip of a range of [n] trips around [k] kernels adds [k] to its own
   window. *)
let trips ~k n () =
  let bufs, e = ranged ~k n in
  let bound =
    List.mapi
      (fun i u ->
        let xs =
          if i = 0 then Array.init (4 * n) float_of_int
          else Array.make (4 * n) 0.
        in
        (u, [ new_floats "NV" xs ]))
      bufs
  in
  ignore (run_calls ~bound [ e ]);
  equal floats
    (Array.init (4 * n) (fun i -> float_of_int (i + k)))
    (floats_of (List.hd (List.assq (List.nth bufs k) bound)))

(* A linked batch keeps what it launches: its programs are dropped at link time,
   and their code must outlive the collections before its run. *)
let held_by_batch () =
  let src = storage "NV" and dst = storage "NV" in
  let x = new_floats "NV" [| 1.; 2.; 3.; 4. |]
  and y = new_floats "NV" (Array.make 4 0.) in
  let s =
    Tolk_engine.link ~devices:(devices ())
      ~bound:[ (src, [ x ]); (dst, [ y ]) ]
      (compile [ adds dst src ])
  in
  for _ = 1 to 4 do
    Gc.full_major ();
    Nx_device.synchronize (nv ())
  done;
  Tolk_engine.run s [||];
  Nx_device.synchronize (nv ());
  equal floats [| 2.; 3.; 4.; 5. |] (floats_of y)

let execution =
  group "execution"
    [
      test "a linked batch runs after collections, its programs held by it"
        held_by_batch;
      slow "a chain of kernels adds one per kernel" (fun () ->
          let b = chain 3 in
          let bound = bound_to [| 1.; 2.; 3.; 4. |] b in
          ignore (run_calls ~bound (chained b));
          equal floats [| 4.; 5.; 6.; 7. |]
            (floats_of (List.hd (List.assq (List.nth b 3) bound))));
      slow "copies from the host and back run on the copy engine, in the batch"
        (fun () ->
          let h = storage "CPU" and a = storage "NV" and a' = storage "NV" in
          let h' = storage "CPU" in
          let calls = [ Ops.store_call a h; adds a' a; Ops.store_call h' a' ] in
          let bound =
            [
              (h, [ new_floats "CPU" [| 1.; 2.; 3.; 4. |] ]);
              (a, [ new_floats "NV" (Array.make 4 0.) ]);
              (a', [ new_floats "NV" (Array.make 4 0.) ]);
              (h', [ new_floats "CPU" (Array.make 4 0.) ]);
            ]
          in
          ignore (run_calls ~bound calls);
          equal floats [| 2.; 3.; 4.; 5. |]
            (floats_of (List.hd (List.assq h' bound))));
      slow "eight runs of one batch without synchronizing each add one"
        (fun () ->
          let b = chain 1 in
          let bound = bound_to [| 0.; 0.; 0.; 0. |] b in
          let s = link ~bound (chained b @ chained (List.rev b)) in
          for _ = 1 to 8 do
            Tolk_engine.run s [||]
          done;
          Nx_device.synchronize (nv ());
          equal floats [| 16.; 16.; 16.; 16. |]
            (floats_of (List.hd (List.assq (List.hd b) bound))));
      slow "a profile records a span of each kernel on the device, in order"
        (fun () ->
          let b = chain 2 in
          let bound = bound_to (Array.make 4 0.) b in
          let spans =
            spans
              (profiled (fun () ->
                   ignore (run_calls ~profile:Stamped ~bound (chained b))))
          in
          equal (list string) [ "k"; "k" ] (List.map (fun (n, _, _) -> n) spans);
          List.iter
            (fun (_, start, stop) -> at_least int ~than:start stop)
            spans;
          match spans with
          | [ (_, _, first_stop); (_, second_start, _) ] ->
              at_least int ~than:first_stop second_start
          | _ -> ());
      slow "each trip of a range runs its kernel on its own window"
        (trips ~k:1 5);
      slow
        "each trip of a range of two kernels runs them on the trip's own \
         windows, chained"
        (trips ~k:2 3);
    ]

let () = exit (run "Tolk.Ops_nv execution" [ execution ])
