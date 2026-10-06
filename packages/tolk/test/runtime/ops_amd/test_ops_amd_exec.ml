open Windtrap
open Tolk
module B = Nx_device.Buffer

(* The first AMD GPU through the kernel driver, or a skip where there is
   none. *)
let amd =
  lazy
    (match Nx_amd_device.get ~interface:Kernel 0 with
    | Ok d -> Some d
    | Error _ -> None)

let amd () =
  match Lazy.force amd with Some d -> d | None -> skip ~reason:"no AMD GPU" ()

let devices () = Tolk_engine.device [ ("CPU", Nx_device.host); ("AMD", amd ()) ]
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
   [out], [n] floats each, four by default. *)
let adds ?(n = 4) out inp =
  let device = Option.get (Ops.device out) in
  let param slot = Call.param ~shape:[ Int n ] ~device slot Float32 in
  let i = Ops.range (Int n) [ 0 ] in
  let x = Ops.load (Ops.index (param 1) [ i ]) [] in
  let st =
    Ops.store
      (Ops.index (param 0) [ i ])
      (Ops.add x (Ops.float ~dtype:Float32 1.))
  in
  Ops.call
    (Ops.sink ~kernel:(Ops.kernel_info ~name:"k" ()) [ Ops.end_ st [ i ] ])
    [ out; inp ]

(* The call of the kernel storing into each of the [rows] floats of [out] the
   sum over [reps] passes of its column of [src], [rows] floats wide and [cols]
   long, each element plus the pass's index: every pass reads [src] again. *)
let sums ~rows ~cols ~reps out src =
  let device = Option.get (Ops.device out) in
  let out_p = Call.param ~shape:[ Int rows ] ~device 0 Float32 in
  let src_p = Call.param ~shape:[ Int (rows * cols) ] ~device 1 Float32 in
  let g = Ops.range (Int rows) [ 0 ] in
  let rep = Ops.range ~axis_type:Reduce (Int reps) [ 1 ] in
  let i = Ops.range ~axis_type:Reduce (Int cols) [ 2 ] in
  let x = Ops.load (Ops.index src_p [ Ops.O.((i * Ops.int rows) + g) ]) [] in
  let sum = Ops.reduce (Ops.add x (Ops.cast rep Float32)) Op.Add [ rep; i ] in
  Ops.call
    (Ops.sink
       ~kernel:(Ops.kernel_info ~name:"sums" ~opts_to_apply:[] ())
       [ Ops.end_ (Ops.store (Ops.index out_p [ g ]) sum) [ g ] ])
    [ out; src ]

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
  Nx_device.synchronize (amd ());
  s

let chain n = List.init (n + 1) (fun _ -> storage "AMD")

let chained bufs =
  List.init
    (List.length bufs - 1)
    (fun k -> adds (List.nth bufs (k + 1)) (List.nth bufs k))

let bound_to xs = List.map (fun u -> (u, [ new_floats "AMD" xs ]))

(* A range of [n] trips around a kernel adding one, each trip on its own window
   of four floats of [src] and [dst]. *)
let ranged n =
  let r = Ops.range (Int n) [ 7 ] in
  let window u =
    let start = Ops.mul r (Ops.int 4) in
    Shape.shrink u [ Some (Sym start, Sym (Ops.add start (Ops.int 4))) ]
  in
  let src = storage ~n:(4 * n) "AMD" and dst = storage ~n:(4 * n) "AMD" in
  (src, dst, Ops.end_ (adds (window dst) (window src)) [ r ])

let spans events =
  List.filter_map
    (function
      | Nx_device.Profile.Span { device; name; start; stop; _ }
        when Nx_device.equal device (amd ()) ->
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

(* The events of a profile around a chain of three kernels; a GPU whose stable
   power state another process holds skips. *)
let profile_chain ?counters ?trace () =
  let b = chain 3 in
  let bound = bound_to (Array.make 4 0.) b in
  let p = Nx_device.Profile.start ?counters ?trace () in
  Fun.protect
    ~finally:(fun () ->
      if Nx_device.Profile.enabled () then ignore (Nx_device.Profile.stop p))
    (fun () ->
      match run_calls ~bound (chained b) with
      | exception Failure why
        when String.ends_with ~suffix:"which another process holds" why ->
          skip ~reason:why ()
      | _ -> Nx_device.Profile.stop p)

(* How much later than a kernel's span the host may see the kernel end: its
   launch and its wake. *)
let late_ms = 20

let execution =
  group "execution"
    [
      slow "a chain of kernels adds one per kernel" (fun () ->
          let b = chain 3 in
          let bound = bound_to [| 1.; 2.; 3.; 4. |] b in
          ignore (run_calls ~bound (chained b));
          equal floats [| 4.; 5.; 6.; 7. |]
            (floats_of (List.hd (List.assq (List.nth b 3) bound))));
      slow "copies from the host and back run on the copy engine, in the batch"
        (fun () ->
          let h = storage "CPU" and a = storage "AMD" and a' = storage "AMD" in
          let h' = storage "CPU" in
          let calls = [ Call.store_call a h; adds a' a; Call.store_call h' a' ] in
          let bound =
            [
              (h, [ new_floats "CPU" [| 1.; 2.; 3.; 4. |] ]);
              (a, [ new_floats "AMD" (Array.make 4 0.) ]);
              (a', [ new_floats "AMD" (Array.make 4 0.) ]);
              (h', [ new_floats "CPU" (Array.make 4 0.) ]);
            ]
          in
          ignore (run_calls ~bound calls);
          equal floats [| 2.; 3.; 4.; 5. |]
            (floats_of (List.hd (List.assq h' bound))));
      slow "a batch's copy engine reads what the host wrote to mapped memory"
        (fun () ->
          let a = storage "AMD" and h = storage "CPU" in
          let m = B.create ~memory:Mapped (amd ()) Float32 4 in
          let written =
            match B.borrow Nx_device.host m with
            | Ok v -> B.bigarray Bigarray.float32 v
            | Error why -> failwith why
          in
          let read = new_floats "CPU" (Array.make 4 0.) in
          let s =
            link ~bound:[ (a, [ m ]); (h, [ read ]) ] [ Call.store_call h a ]
          in
          for round = 1 to 8 do
            let xs = Array.init 4 (fun i -> float_of_int ((10 * round) + i)) in
            Array.iteri (fun i x -> written.{i} <- x) xs;
            Tolk_engine.run s [||];
            Nx_device.synchronize (amd ());
            equal floats
              ~msg:(Printf.sprintf "round %d" round)
              xs (floats_of read)
          done);
      slow "eight runs of one batch without synchronizing each add one"
        (fun () ->
          let b = chain 1 in
          let bound = bound_to [| 0.; 0.; 0.; 0. |] b in
          let s = link ~bound (chained b @ chained (List.rev b)) in
          for _ = 1 to 8 do
            Tolk_engine.run s [||]
          done;
          Nx_device.synchronize (amd ());
          equal floats [| 16.; 16.; 16.; 16. |]
            (floats_of (List.hd (List.assq (List.hd b) bound))));
      slow "a thousand runs back to back while the host allocates" (fun () ->
          (* Each run's batch starts with a memory barrier while the host
             submits the next and allocates and frees device memory, as a
             model's calls do: a GFX12 compute queue hung within a few hundred
             runs while its barrier flushed the host data path. *)
          let n = 1 lsl 20 in
          let src = storage ~n "AMD" and dst = storage ~n "AMD" in
          let bound =
            [
              (src, [ new_floats "AMD" (Array.make n 1.) ]);
              (dst, [ new_floats "AMD" (Array.make n 0.) ]);
            ]
          in
          let s = link ~bound [ adds ~n dst src ] in
          for i = 1 to 1000 do
            Tolk_engine.run s [||];
            ignore (B.create (amd ()) Float32 n);
            if i mod 20 = 0 then Gc.full_major ()
          done;
          Nx_device.synchronize (amd ());
          equal floats (Array.make n 2.)
            (floats_of (List.hd (List.assq dst bound))));
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
      slow "the host sees a long kernel end as its span ends" (fun () ->
          (* Runs of 300 to 600 ms, of two lengths, long enough for the host to
             sleep while it waits. The GPU's interrupt wakes it at once; a host
             that woke only every 200 ms would see most of them end late. *)
          let rows = 256 and cols = 65536 in
          let out = storage ~n:rows "AMD"
          and src = storage ~n:(rows * cols) "AMD" in
          let bound =
            [
              (out, [ B.create (amd ()) Float32 rows ]);
              (src, [ B.create (amd ()) Float32 (rows * cols) ]);
            ]
          in
          let run reps =
            let s =
              link ~profile:Stamped ~bound [ sums ~rows ~cols ~reps out src ]
            in
            for run = 1 to 3 do
              let seen = ref 0 in
              let events =
                profiled (fun () ->
                    let submitted = Nx_device.Profile.now () in
                    Tolk_engine.run s [||];
                    Nx_device.synchronize (amd ());
                    seen := Nx_device.Profile.now () - submitted)
              in
              match spans events with
              | [ (_, start, stop) ] ->
                  at_least int
                    ~msg:
                      (Printf.sprintf "%d passes, run %d: span, in ns" reps run)
                    ~than:(!seen - (late_ms * 1_000_000))
                    (stop - start)
              | spans -> equal int ~msg:"one span" 1 (List.length spans)
            done
          in
          run 256;
          run 320);
      slow "a profile that counts has each kernel's run count, in order"
        (fun () ->
          let events =
            profile_chain ~counters:[ "GRBM_GUI_ACTIVE"; "SQ_BUSY_CYCLES" ] ()
          in
          let counted =
            List.filter_map
              (function
                | Nx_device.Profile.Counters c
                  when Nx_device.equal c.device (amd ()) ->
                    Some (c.name, c.counters)
                | _ -> None)
              events
          in
          equal (list string) [ "k"; "k"; "k" ] (List.map fst counted);
          List.iter
            (fun (_, counters) ->
              equal (list string) ~msg:"the counters asked for"
                [ "GRBM_GUI_ACTIVE"; "SQ_BUSY_CYCLES" ]
                (List.map fst counters);
              is_true ~msg:"the GPU was busy"
                (Array.fold_left ( + ) 0 (List.assoc "GRBM_GUI_ACTIVE" counters)
                > 0))
            counted);
      slow "a profile that traces has each kernel's run traced by every engine"
        (fun () ->
          let events = profile_chain ~trace:true () in
          let a = Option.get (Nx_amd_device.of_device (amd ())) in
          let props = Nx_amd_device.props a in
          let engines = props.shader_engines * props.xccs in
          let traces =
            List.filter_map
              (function
                | Nx_device.Profile.Trace t
                  when Nx_device.equal t.device (amd ()) ->
                    Some (t.name, t.part, String.length t.data)
                | _ -> None)
              events
          in
          equal
            (list (pair string int))
            ~msg:"every engine of every run"
            (List.concat_map
               (fun _ -> List.init engines (fun se -> ("k", se)))
               [ 1; 2; 3 ])
            (List.map (fun (name, se, _) -> (name, se)) traces);
          List.iter
            (fun (_, se, n) ->
              is_true ~msg:(Printf.sprintf "engine %d wrote" se) (n > 0))
            traces;
          let waves =
            List.filter
              (function
                | Nx_device.Profile.Span s ->
                    Nx_device.equal s.device (amd ())
                    && String.starts_with ~prefix:"SE " s.lane
                | _ -> false)
              events
          in
          match props.target with
          | 9, _, _ ->
              equal int ~msg:"no realtime markers on GFX9" 0 (List.length waves)
          | _ ->
              is_true ~msg:"the waves are spans" (waves <> []);
              List.iter
                (function
                  | Nx_device.Profile.Span s ->
                      equal string ~msg:"named after the kernel" "k" s.name;
                      is_true ~msg:"in order" (s.start <= s.stop)
                  | _ -> ())
                waves);
      slow "a batch runs only under the profile request it was encoded for"
        (fun () ->
          let b = chain 1 in
          let bound = bound_to (Array.make 4 0.) b in
          let traced = Nx_device.Profile.start ~trace:true () in
          let s =
            Fun.protect
              ~finally:(fun () -> ignore (Nx_device.Profile.stop traced))
              (fun () ->
                match link ~bound (chained b) with
                | exception Failure why
                  when String.ends_with ~suffix:"which another process holds"
                         why ->
                    skip ~reason:why ()
                | s -> s)
          in
          let refused () =
            raises_match
              (Exn.invalid_arg
                 ~substring:"encoded for traces, and the profile asks for")
              (fun () -> Tolk_engine.run s [||])
          in
          refused ();
          let counting =
            Nx_device.Profile.start ~counters:[ "GRBM_GUI_ACTIVE" ] ()
          in
          Fun.protect
            ~finally:(fun () -> ignore (Nx_device.Profile.stop counting))
            refused);
      slow "each trip of a range runs its kernel on its own window" (fun () ->
          let n = 5 in
          let src, dst, e = ranged n in
          let bound =
            [
              (src, [ new_floats "AMD" (Array.init (4 * n) float_of_int) ]);
              (dst, [ new_floats "AMD" (Array.make (4 * n) 0.) ]);
            ]
          in
          ignore (run_calls ~bound [ e ]);
          equal floats
            (Array.init (4 * n) (fun i -> float_of_int (i + 1)))
            (floats_of (List.hd (List.assq dst bound))));
    ]

let () = exit (run "Tolk.Ops_amd execution" [ execution ])
