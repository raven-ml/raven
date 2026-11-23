(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Programs of rig.program loaded and run on an agent's devices: they leave the
   bytes they leave on this machine's. *)

open Windtrap
open Remote_job
open Remote_program
module G = Rig_program
module P = Rig_support.Polled

(* Four runs' ints [(groups, trips)]. The last's trips are negative: no
   trip. *)
let frames = [ (8, 1); (3, 0); (5, 4); (3, -1) ]

(* The inputs' bytes after each of [frames], by the description's own
   account: memory 0 keeps what each run's fill leaves. *)
let model () =
  let memory = Array.make 8 0 in
  List.map
    (fun (groups, trips) ->
      for i = 0 to groups - 1 do
        memory.(i) <- 100 + (10 * trips) + i
      done;
      (words 8 (fun i -> memory.(i)), le64 (max 0 trips)))
    frames

(* [frames]' runs of the description on [ds], their inputs' bytes after
   each. *)
let runs ds =
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load
         (description (archs (with_host ds)))
         (Iarray.of_array (with_host ds)))
  in
  List.map
    (fun (groups, trips) ->
      let x = far_of_string ds.(1) (String.make size '\000') in
      let c = far_of_string ds.(0) (le64 0) in
      ignore
        (G.run p
           {
             inputs = [| x; c |];
             ints = [| groups; trips |];
           });
      (read x, read c))
    frames

let here () =
  let open_ i =
    fst (P.open_ (Printf.sprintf "program-here:%d-%d" i (Unix.getpid ())))
  in
  [| open_ 0; open_ 1 |]

(* A program loaded here and on an agent leaves equal bytes. *)
let same_bytes () =
  let local = runs (here ()) in
  equal ~msg:"here, by the description's own account"
    (list (pair string string))
    (model ()) local;
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let far = runs (Array.of_list (polled h)) in
  equal ~msg:"on the agent" (list (pair string string)) local far

(* A run's answer is the host's point, reached once the run's work there is
   done. *)
let host_point () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let ds = Array.of_list (polled h) in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load
         (description (archs (with_host ds)))
         (Iarray.of_array (with_host ds)))
  in
  let x = Rig.Buffer.create ds.(1) size and c = Rig.Buffer.create ds.(0) 8 in
  let ends = G.run p { inputs = [| x; c |]; ints = [| 1; 0 |] } in
  equal ~msg:"the host's point" (list string)
    [ Rig.name h ]
    (List.map
       (fun p -> Option.fold ~none:"" ~some:Rig.name (Rig.Point.device p))
       (Array.to_list ends));
  Rig.Point.wait ends.(0)

(* Times *)

(* [unlooped] on [h]'s Polled devices and [h], and a frame of it whose run
   leaves [expected] in its inputs. *)
let timed_on h =
  let ds = with_host (Array.of_list (polled h)) in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load (unlooped (archs ds)) (Iarray.of_array ds))
  in
  let frame () =
    let x = far_of_string ds.(1) (String.make size '\000') in
    let c = far_of_string ds.(0) (le64 0) in
    { G.inputs = [| x; c |]; ints = [| 8; 0 |] }
  in
  (p, frame)

let inputs_of (f : G.frame) = (read f.inputs.(0), read f.inputs.(1))

(* A time on an agent has each step's duration there, and its run leaves the
   bytes a run leaves. A program with a loop raises before any work. *)
let times () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let p, frame = timed_on h in
  let f = frame () in
  let t0 = Rig.Profile.now () in
  let times = G.time p f in
  window_law times (Rig.Profile.now () - t0);
  equal ~msg:"the inputs" (pair string string) expected (inputs_of f);
  let ds = with_host (Array.of_list (polled h)) in
  let looped =
    require_ok ~pp:Format.pp_print_string
      (G.load (description (archs ds)) (Iarray.of_array ds))
  in
  let before = Rig.submitted h in
  raises_match (Exn.invalid_arg ~sub:"Rig_program.time: a program with a loop")
    (fun () -> G.time looped (frame ()));
  equal ~msg:"no work for a loop" int before (Rig.submitted h)

(* A time on an agent's device that times no work has [-1] for a step of
   parts, as the agent read it, and [0] for a submission of no parts. *)
let untimed () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let d =
    match Rig_remote.devices h "UNTIMED" with
    | Ok ds -> List.hd ds
    | Error why -> failf "devices: %s" why
  in
  let t =
    {
      G.devices = [| Rig.arch d |];
      memory = [||];
      images =
        [| { G.device = 0; binary = "functions" } |];
      inputs = [| { G.device = 0; bytes = size } |];
      ints = 0;
      steps =
        [|
          G.Submit
            {
              device = 0;
              slots = [| G.Input 0 |];
              parts =
                [|
                  launch ~image:0 ~groups:(fixed 8) "fill"
                    (le64 0 ^ le64 100) [| ref_ Write 0 0 |];
                |];
            };
          G.Submit { device = 0; slots = [||]; parts = [||] };
        |];
    }
  in
  let p = require_ok ~pp:Format.pp_print_string (G.load t [| d |]) in
  let x = far_of_string d (String.make size '\000') in
  equal ~msg:"its times" (array int) [| -1; 0 |]
    (G.time p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the fill" string (words 8 (fun i -> 100 + i)) (read x)

(* A time of a program of no [Submit] step on an agent has only [0]s, and its
   load made no memory there, where a load of one with a [Submit] step makes
   8 bytes per step for its times. *)
let no_submit () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let d = List.hd (polled h) in
  let move =
    G.Move
      {
        src = G.Input 0;
        src_offset = fixed 0;
        dst = G.Input 1;
        dst_offset = fixed 0;
        length = 8;
      }
  in
  let program steps =
    {
      G.devices = [| Rig.arch d |];
      memory = [||];
      images = [||];
      inputs = [| { G.device = 0; bytes = 8 }; { G.device = 0; bytes = 8 } |];
      ints = 0;
      steps;
    }
  in
  let made steps =
    let before = Rig.allocated h in
    let p =
      require_ok ~pp:Format.pp_print_string (G.load (program steps) [| d |])
    in
    (p, Rig.allocated h - before)
  in
  let p, bytes = made [| move |] in
  equal ~msg:"memory made at load" int 0 bytes;
  let y = far_of_string d (le64 0) in
  equal ~msg:"its times" (array int) [| 0 |]
    (G.time p { inputs = [| far_of_string d (le64 7); y |]; ints = [||] });
  equal ~msg:"the move's bytes" string (le64 7) (read y);
  let none =
    G.Submit { device = 0; slots = [||]; parts = [||] }
  in
  let _, bytes = made [| move; none |] in
  equal ~msg:"memory made at load, with a Submit step" int
    (Rig.allocation h 16) bytes

(* A binary that is no program is the agent's refusal. *)
let refused_binary () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  match Rig.Image.load h "no program" with
  | Ok _ -> fail "the agent loaded no program"
  | Error why -> contains ~sub:"not a program" why

(* Rails *)

let host_record h =
  match Rig.capability h Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host r) -> r
  | _ -> failf "%s has no host record" (Rig.name h)

(* The 64-bit word at byte [at] of [a], read with acquire order. *)
let word_at (a : Rig_remote_abi.area) at = Int64.to_int (Rig_support.load64 a at)

(* On [h]: the run's int 0 moved into the outbound area of rail [id], then a
   send of the transfer of count 1. *)
let rail_sender h id =
  {
    G.devices = [| Rig.arch h |];
    memory = [| G.Rail { rail = id; area = Outbound } |];
    images = [||];
    inputs = [||];
    ints = 1;
    steps =
      [|
        G.Move
          {
            src = G.Ints;
            src_offset = fixed 0;
            dst = G.Memory { memory = 0; offset = 0; length = 8 };
            dst_offset = fixed 0;
            length = 8;
          };
        G.Send { rail = id; count = fixed 1 };
      |];
  }

let named_rail h id =
  {
    G.devices = [| Rig.arch h |];
    memory = [| G.Rail { rail = id; area = Outbound } |];
    images = [||];
    inputs = [||];
    ints = 0;
    steps = [||];
  }

let rail_is_dropped h id =
  let why = require_error (G.load (named_rail h id) [| h |]) in
  contains ~msg:"the load's refusal" ~sub:"no rail" why

(* An agent's program writes a word into its end of a TCP rail and sends it;
   the word arrives at this process's end, where a program here reads it. *)
let rail_end_to_end () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let r =
    require_ok ~pp:Format.pp_print_string
      ((host_record h).rail None
         ~send:[| { Rig_remote_abi.src = 0; dst = 0; length = 8 } |]
         ~receive:[||])
  in
  let local = require_some r.local in
  let fill = rail_sender h r.id in
  raises_match ~msg:"rails for another machine" Exn.invalid_arg (fun () ->
      G.load ~rails:(fun _ -> None) fill [| h |]);
  let far = require_ok ~pp:Format.pp_print_string (G.load fill [| h |]) in
  let pt =
    (G.run far
       { inputs = [||]; ints = [| 42 |] }).(0)
  in
  Rig.Point.wait pt;
  (* [arrived] is the word at byte 256 of the counts. *)
  until ~what:"the transfer's arrival" (fun () -> word_at local.counts 256 >= 1);
  let read_inbound =
    {
      G.devices = [| Rig.arch Rig.host |];
      memory = [| G.Rail { rail = r.id; area = Inbound } |];
      images = [||];
      inputs = [| { G.device = 0; bytes = 8 } |];
      ints = 0;
      steps =
        [|
          G.Move
            {
              src = G.Memory { memory = 0; offset = 0; length = 8 };
              src_offset = fixed 0;
              dst = G.Input 0;
              dst_offset = fixed 0;
              length = 8;
            };
        |];
    }
  in
  let rails id = if id = r.id then Some local else None in
  let here =
    require_ok ~pp:Format.pp_print_string
      (G.load ~rails read_inbound [| Rig.host |])
  in
  let x = Rig.Buffer.create Rig.host 8 in
  ignore
    (G.run here
       { inputs = [| x |]; ints = [||] });
  equal ~msg:"the word, here" string (le64 42) (read_host x);
  ignore (Sys.opaque_identity (r, far, here))

(* Only the agent knows a rail area's size. It accepts a view of the whole
   area, and refuses one past it. *)
let rail_view () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let d = List.hd (polled h) in
  let r =
    require_ok ~pp:Format.pp_print_string
      ((host_record h).rail None
         ~send:[| { Rig_remote_abi.src = 0; dst = 0; length = 8 } |]
         ~receive:[||])
  in
  let load length =
    let t =
      {
        (named_rail d r.id) with
        memory =
          [| G.Rail { rail = r.id; area = Counts } |];
        inputs = [| { G.device = 0; bytes = 8 } |];
        steps =
          [|
            G.Move
              {
                src = G.Memory { memory = 0; offset = 0; length };
                src_offset = fixed 0;
                dst = G.Input 0;
                dst_offset = fixed 0;
                length = 8;
              };
          |];
      }
    in
    G.load t [| d |]
  in
  is_ok ~msg:"the whole counts area" ~pp:Format.pp_print_string (load 384);
  contains ~msg:"past the counts area" ~sub:"outside"
    (require_error (load 385));
  ignore (Sys.opaque_identity r)

(* A reachable rail still names both agents' ends after collection. *)
let retained_rail () =
  with_job ~n:2 @@ fun j _ ->
  let hs = Array.of_list (Rig_remote.hosts j) in
  let[@inline never] make () =
    require_ok ~pp:Format.pp_print_string
      ((host_record hs.(0)).rail
         (Some (host_record hs.(1)))
         ~send:[| { Rig_remote_abi.src = 0; dst = 0; length = 8 } |]
         ~receive:[||])
  in
  let r = make () in
  Gc.full_major ();
  Array.iter
    (fun h ->
      is_ok ~pp:Format.pp_print_string (G.load (named_rail h r.id) [| h |]))
    hs;
  ignore (Sys.opaque_identity r)

(* Keeping only the local end keeps the remote end available for new loads. *)
let retained_local_end () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let[@inline never] make () =
    let r =
      require_ok ~pp:Format.pp_print_string
        ((host_record h).rail None
           ~send:[| { Rig_remote_abi.src = 0; dst = 0; length = 8 } |]
           ~receive:[||])
    in
    (r.id, require_some r.local)
  in
  let id, local = make () in
  Gc.full_major ();
  is_ok ~pp:Format.pp_print_string (G.load (named_rail h id) [| h |]);
  ignore (Sys.opaque_identity local)

(* Dropping the controller's rail prevents further loads naming it. Programs
   already loaded on both agents keep their ends and can still transfer. *)
let loaded_rail_ends () =
  with_job ~n:2 @@ fun j _ ->
  let hs = Array.of_list (Rig_remote.hosts j) in
  let sender = hs.(0) and receiver = hs.(1) in
  let[@inline never] load () =
    let rail =
      require_ok ~pp:Format.pp_print_string
        ((host_record sender).rail
           (Some (host_record receiver))
           ~send:[| { Rig_remote_abi.src = 0; dst = 0; length = 8 } |]
           ~receive:[||])
    in
    let send =
      require_ok ~pp:Format.pp_print_string
        (G.load (rail_sender sender rail.id) [| sender |])
    in
    let read_end =
      {
        G.devices = [| Rig.arch receiver |];
        memory =
          [|
            G.Rail { rail = rail.id; area = Counts };
            G.Rail { rail = rail.id; area = Inbound };
          |];
        images = [| host_image 0 "arrival" |];
        inputs = [| { G.device = 0; bytes = 8 }; { G.device = 0; bytes = 8 } |];
        ints = 0;
        steps =
          [|
            (* test/fixtures' [arrival] stores [arrived], read with acquire
               order, in input 0 and, once the transfer arrived, its bytes in
               input 1. *)
            G.Submit
              {
                device = 0;
                slots =
                  [|
                    G.Input 0;
                    G.Memory { memory = 0; offset = 256; length = 8 };
                    G.Input 1;
                    G.Memory { memory = 1; offset = 0; length = 8 };
                  |];
                parts =
                  [|
                    launch ~image:0 ~groups:(fixed 1) "arrival"
                      (String.make 32 '\000')
                      [|
                        ref_ Write 0 0;
                        ref_ Read 8 1;
                        ref_ Write 16 2;
                        ref_ Read 24 3;
                      |];
                  |];
              };
          |];
      }
    in
    let receive =
      require_ok ~pp:Format.pp_print_string (G.load read_end [| receiver |])
    in
    (rail.id, send, receive)
  in
  let id, send, receive = load () in
  Gc.full_major ();
  rail_is_dropped sender id;
  rail_is_dropped receiver id;
  Array.iter
    (fun h ->
      let opened = require_ok (Rig_remote.devices h "COLLECT") in
      equal ~msg:"collection opens no device" int 0 (List.length opened))
    hs;
  Array.iter Rig.Point.wait
    (G.run send
       { inputs = [||]; ints = [| 42 |] });
  let arrived = Rig.Buffer.create receiver 8 in
  let bytes = Rig.Buffer.create receiver 8 in
  until ~what:"the retained rail's transfer" (fun () ->
      Array.iter Rig.Point.wait
        (G.run receive
           {
             inputs = [| arrived; bytes |];
             ints = [||];
           });
      String.get_int64_le (read arrived) 0 >= 1L);
  equal ~msg:"the received word" string (le64 42) (read bytes)

(* An end that nothing reaches has no reader: a transfer that arrives at it is
   discarded, and the job goes on. Here the agent's program keeps its end and
   sends after this process's end was collected. Once the agent reads its [sent]
   count at 1, the transfer came here before the read's answer. *)
let collected_end () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let[@inline never] load () =
    let rail =
      require_ok ~pp:Format.pp_print_string
        ((host_record h).rail None
           ~send:[| { Rig_remote_abi.src = 0; dst = 0; length = 8 } |]
           ~receive:[||])
    in
    let send =
      require_ok ~pp:Format.pp_print_string
        (G.load (rail_sender h rail.id) [| h |])
    in
    let read_sent =
      {
        G.devices = [| Rig.arch h |];
        memory = [| G.Rail { rail = rail.id; area = Counts } |];
        images = [||];
        inputs = [| { G.device = 0; bytes = 8 } |];
        ints = 0;
        steps =
          [|
            G.Move
              {
                src = G.Memory { memory = 0; offset = 128; length = 8 };
                src_offset = fixed 0;
                dst = G.Input 0;
                dst_offset = fixed 0;
                length = 8;
              };
          |];
      }
    in
    let sent =
      require_ok ~pp:Format.pp_print_string (G.load read_sent [| h |])
    in
    (rail.id, send, sent)
  in
  let id, send, sent = load () in
  Gc.full_major ();
  rail_is_dropped h id;
  Array.iter Rig.Point.wait
    (G.run send
       { inputs = [||]; ints = [| 42 |] });
  let count = Rig.Buffer.create h 8 in
  until ~what:"the transfer's send" (fun () ->
      let points =
        G.run sent
          {
            inputs = [| count |];
            ints = [||];
          }
      in
      Array.iter Rig.Point.wait points;
      String.get_int64_le (read count) 0 >= 1L);
  equal ~msg:"the job's failure" (option string) None (Rig_remote.failed j);
  ignore (Sys.opaque_identity send)

(* A program whose memory the agent's device cannot hold is the load's [Error],
   and the agent goes on: a program that fits loads after it. *)
let too_large () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let ds = with_host (Array.of_list (polled h)) in
  let t = description (archs ds) in
  let huge =
    {
      t with
      memory =
        [|
          G.Alloc
            {
              device = 0;
              kind = Rig.Buffer.Device;
              bytes = 1 lsl 50;
              init = "";
            };
        |];
    }
  in
  let why = require_error (G.load huge (Iarray.of_array ds)) in
  contains ~msg:"names the memory" ~sub:"no memory" why;
  is_ok ~pp:Format.pp_print_string ~msg:"a program that fits, after"
    (G.load t (Iarray.of_array ds))

(* A run raises at the call on a frame the agent would refuse, and the job goes
   on: here input 0, a host buffer, is named by a Polled device that maps no
   host memory, so it cannot borrow it. *)
let refused_frame () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let d =
    match Rig_remote.devices h "MAPLESS" with
    | Ok [ d ] -> d
    | Ok _ -> fail "MAPLESS is one device"
    | Error why -> failf "devices: %s" why
  in
  let t =
    {
      G.devices = archs [| h; d |];
      memory = [||];
      images = [| { G.device = 1; binary = "functions" } |];
      inputs = [| { G.device = 0; bytes = 8 } |];
      ints = 0;
      steps =
        [|
          G.Submit
            {
              device = 1;
              slots = [| G.Input 0 |];
              parts =
                [|
                  launch ~image:0 ~groups:(fixed 1) "copy"
                    (le64 0 ^ le64 0 ^ le64 0)
                    [| ref_ G.Write 0 0; ref_ G.Write 8 0 |];
                |];
            };
        |];
    }
  in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load t (Iarray.of_array [| h; d |]))
  in
  let x = Rig.Buffer.create h 8 in
  raises_match Exn.invalid_arg (fun () ->
      G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the job's failure" (option string) None (Rig_remote.failed j)

(* A run raises at the call on an input the agent's host does not address that
   a launch on that host names, and the job goes on. *)
let unaddressed_input () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let d =
    match Rig_remote.devices h "HIDDEN" with
    | Ok [ d ] -> d
    | Ok _ -> fail "HIDDEN is one device"
    | Error why -> failf "devices: %s" why
  in
  let t =
    {
      G.devices = archs (with_host [| d |]);
      memory = [||];
      images = [| host_image 1 "affine" |];
      inputs = [| { G.device = 0; bytes = 8 } |];
      ints = 0;
      steps =
        [|
          G.Submit
            {
              device = 1;
              slots = [| G.Input 0 |];
              parts =
                [|
                  launch ~image:0 ~groups:(fixed 1) "affine"
                    (le64 0 ^ le64 0 ^ le64 1 ^ le64 1 ^ le64 1)
                    [| ref_ Write 0 0; ref_ Read 8 0 |];
                |];
            };
        |];
    }
  in
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load t (Iarray.of_array (with_host [| d |])))
  in
  let x = Rig.Buffer.create d 8 in
  raises_match Exn.invalid_arg (fun () ->
      G.run p { inputs = [| x |]; ints = [||] });
  equal ~msg:"the job's failure" (option string) None (Rig_remote.failed j)

(* A run raises at the call on threads per group, read from the frame's ints,
   past the most of its launch's function, which the agent stated at load, and
   the job goes on: Polled's main has at most 1024 threads a group, and x by
   y past it is refused whatever z is, as every submit refuses it. A loop's
   trip word changes as the run goes, so the call checks no launch that reads
   it, here in a loop of no trips. *)
let too_many_threads () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let d = List.hd (polled h) in
  let main threads =
    G.Submit
      {
        device = 0;
        slots = [||];
        parts =
          [|
            G.Launch
              {
                image = 0;
                name = "main";
                label = "main";
                params = "";
                scalars = [||];
                refs = [||];
                groups = (fixed 1, fixed 1, fixed 1);
                threads;
                shared = fixed 0;
              };
          |];
      }
  in
  let t =
    {
      G.devices = archs [| d |];
      memory = [||];
      images =
        [| { G.device = 0; binary = "functions" } |];
      inputs = [||];
      ints = 3;
      steps =
        [|
          main (int_at 0, fixed 2, int_at 2);
          G.Loop
            {
              trips = fixed 0;
              trip = Some 1;
              flag = None;
              body = [| main (int_at 1, fixed 1, fixed 1) |];
            };
        |];
    }
  in
  let p =
    require_ok ~pp:Format.pp_print_string (G.load t (Iarray.of_array [| d |]))
  in
  equal ~msg:"what the agent stated" (array (option int))
    [| Some 1024; Some 1024 |] (G.threads p);
  raises ~msg:"513x2x1 threads"
    (Invalid_argument
       "Rig_program.run: main runs at most 1024 threads per group; the frame \
        sets 513x2x1") (fun () ->
      G.run p
        { inputs = [||]; ints = [| 513; 4096; 1 |] });
  raises ~msg:"513x2x0 threads, past the most in x and y alone"
    (Invalid_argument
       "Rig_program.run: main runs at most 1024 threads per group; the frame \
        sets 513x2x0") (fun () ->
      G.run p
        { inputs = [||]; ints = [| 513; 4096; 0 |] });
  Array.iter Rig.Point.wait
    (G.run p
       { inputs = [||]; ints = [| 512; 0; 1 |] });
  equal ~msg:"the job's failure" (option string) None (Rig_remote.failed j)

let () =
  Watchdog.start ();
  exit
    (run "rig_remote.program"
       [
         test ~timeout:60.
           "a program leaves the same bytes on an agent's devices as here"
           same_bytes;
         test ~timeout:60. "a run on an agent is a point of its machine's host"
           host_point;
         test ~timeout:60.
           "a time on an agent has each step's duration there, and refuses a \
            loop"
           times;
         test ~timeout:60.
           "a time on an agent's device that times nothing has -1 for a step"
           untimed;
         test ~timeout:60.
           "a time of no Submit step on an agent has 0s, its load no memory \
            for them"
           no_submit;
         test ~timeout:60. "an agent refuses a binary that is no program"
           refused_binary;
         test ~timeout:60.
           "an agent answers a program its devices cannot hold, and goes on"
           too_large;
         test ~timeout:60.
           "a program's send carries a word on a rail, which a program reads \
            at its other end"
           rail_end_to_end;
         test ~timeout:60. "an agent checks a rail view against its area"
           rail_view;
         test ~timeout:60. "a reachable rail keeps both agent ends"
           retained_rail;
         test ~timeout:60. "a reachable local end keeps the agent end"
           retained_local_end;
         test ~timeout:60.
           "loaded programs keep their ends after the controller drops a rail"
           loaded_rail_ends;
         test ~timeout:60.
           "a transfer to a collected end is discarded, and the job goes on"
           collected_end;
         test ~timeout:60.
           "a run raises at the call on a frame the agent would refuse"
           refused_frame;
         test ~timeout:60.
           "a run raises at the call on an input its host step cannot address"
           unaddressed_input;
         test ~timeout:60.
           "a run raises at the call on more threads than a launch's function \
            has"
           too_many_threads;
       ])
