(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Programs of rig.program loaded and run on an agent's devices: they leave the
   bytes they leave on this machine's. *)

open Windtrap
open Remote_job
module G = Rig_program
module P = Rig_support.Polled

let le64 v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int v);
  Bytes.to_string b

let words n f = String.concat "" (List.init n (fun i -> le64 (f i)))
let size = 64
let archs ds = Iarray.of_array (Array.map Rig.arch ds)

let polled h =
  match Rig_remote.devices h "POLLED" with
  | Ok ds -> ds
  | Error why -> failf "devices: %s" why

let affine =
  {
    G.obj = Rig_host_support.fixture ~dir:"../host/fixtures" "affine";
    entry = "affine";
  }

let launch ?(holes : _ iarray = [||]) ~image ~groups kernel bytes refs =
  {
    G.queue = "COMPUTE:0";
    after = [||];
    work =
      G.Launch
        {
          image;
          kernel;
          params = { G.bytes; holes };
          refs;
          groups = (groups, G.Fixed 1, G.Fixed 1);
          threads = (G.Fixed 1, G.Fixed 1, G.Fixed 1);
          shared = G.Fixed 0;
        };
  }

let ref_ at slot = { Rig.Submission.at; slot }

(* Device 0 fills [Int 0] words of memory 0 from 100, memory of two copies that
   starts as zeros; device 1 copies all of it into input 0 through its borrow;
   host code adds one to input 1's first word [Int 1] times. *)
let description archs =
  {
    G.devices = archs;
    memory =
      [|
        G.Alloc
          {
            device = 0;
            kind = Rig.Buffer.Device;
            bytes = size;
            init = { bytes = String.make size '\000'; holes = [||] };
            copies = Two;
          };
      |];
    images =
      [|
        { G.device = 0; binary = { bytes = "functions"; holes = [||] } };
        { G.device = 1; binary = { bytes = "functions"; holes = [||] } };
      |];
    code = [| affine |];
    inputs = [| { G.device = 1; bytes = size }; { G.device = 0; bytes = 8 } |];
    ints = 2;
    steps =
      [|
        G.Submit
          {
            device = 0;
            buffers =
              [|
                ( G.Memory { memory = 0; offset = 0; length = size },
                  Rig.Buffer.Read_write );
              |];
            fixed = [||];
            parts =
              [|
                launch ~image:0 ~groups:(G.Int 0) "fill"
                  (le64 0 ^ le64 100)
                  [| ref_ 0 0 |];
              |];
          };
        G.Submit
          {
            device = 1;
            buffers =
              [|
                ( G.Memory { memory = 0; offset = 0; length = size },
                  Rig.Buffer.Read );
                (G.Input 0, Rig.Buffer.Read_write);
              |];
            fixed = [||];
            parts =
              [|
                launch ~image:1 ~groups:(G.Fixed 1) "copy"
                  (le64 0 ^ le64 0 ^ le64 size)
                  [| ref_ 0 0; ref_ 8 1 |];
              |];
          };
        G.Loop
          {
            trips = G.Int 1;
            trip = None;
            flag = None;
            body =
              [|
                G.Host
                  {
                    code = 0;
                    buffers =
                      [|
                        (G.Input 1, Rig.Buffer.Read_write);
                        (G.Input 1, Rig.Buffer.Read);
                      |];
                    values = [| G.Fixed 1; G.Fixed 1; G.Fixed 1 |];
                    split = None;
                  };
              |];
          };
      |];
  }

(* Three runs of the description on [ds], with ints [(groups, trips)], their
   inputs' bytes after each. *)
let runs ds =
  let p =
    require_ok ~pp:Format.pp_print_string
      (G.load (description (archs ds)) (Iarray.of_array ds))
  in
  List.map
    (fun (groups, trips) ->
      let x = far_of_string ds.(1) (String.make size '\000') in
      let c = far_of_string ds.(0) (le64 0) in
      ignore (G.run p { inputs = [| x; c |]; ints = [| groups; trips |] });
      (read x, read c))
    [ (8, 1); (3, 0); (5, 4) ]

let here () =
  let open_ i =
    fst (P.open_ (Printf.sprintf "program-here:%d-%d" i (Unix.getpid ())))
  in
  [| open_ 0; open_ 1 |]

(* RFC 0031's Law 11: a share loaded here and on an agent leaves equal bytes. *)
let same_bytes () =
  let local = runs (here ()) in
  equal ~msg:"here, by the description's own account"
    (list (pair string string))
    [
      (words 8 (fun i -> 100 + i), le64 1);
      (* Run 1 fills the other copy. *)
      (words 3 (fun i -> 100 + i) ^ String.make (size - 24) '\000', le64 0);
      (* Run 2 fills run 0's copy, whose last words run 0 left. *)
      (words 8 (fun i -> 100 + i), le64 4);
    ]
    local;
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
      (G.load (description (archs ds)) (Iarray.of_array ds))
  in
  let x = Rig.Buffer.create ds.(1) size and c = Rig.Buffer.create ds.(0) 8 in
  let points = G.run p { inputs = [| x; c |]; ints = [| 1; 0 |] } in
  equal ~msg:"the host's point" (list string)
    [ Rig.name h ]
    (List.map (fun p -> Rig.name (Rig.Point.device p)) (Array.to_list points));
  Rig.Point.wait points.(0)

(* A binary that is no program is the agent's refusal. *)
let refused_binary () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  match Rig.Image.load h "no program" with
  | Ok _ -> fail "the agent loaded no program"
  | Error why -> contains ~sub:"not a program" why

(* Rails *)

(* test/program's fixture [ready]: stores its value 2 at its buffer 0, then
   calls the function at its value 0 with its values 1 and 3. *)
let ready =
  {
    G.obj = Rig_host_support.fixture ~dir:"../program/fixtures" "ready";
    entry = "ready";
  }

let host_record h =
  match Rig.capability h Rig_remote_abi.key with
  | Some (Rig_remote_abi.Host r) -> r
  | _ -> failf "%s has no host record" (Rig.name h)

(* The 64-bit word at byte [at] of [a]. *)
let word_at (a : Rig_remote_abi.area) at =
  Int64.to_int
    (String.get_int64_le
       (String.init 8 (fun i -> Bigarray.Array1.get a (at + i)))
       0)

(* An agent's program writes a word into its end of a TCP rail and calls the
   rail's ready function, as host code after a GPU's work does; the word arrives
   at this process's end, where a program here reads it. *)
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
  let fill =
    {
      G.devices = [| Rig.arch h |];
      memory = [| G.Rail { rail = r.id; area = Outbound } |];
      images = [||];
      code = [| ready |];
      inputs = [||];
      ints = 1;
      steps =
        [|
          G.Host
            {
              code = 0;
              buffers =
                [|
                  ( G.Memory { memory = 0; offset = 0; length = 8 },
                    Rig.Buffer.Read_write );
                |];
              values =
                [|
                  G.Leaf (Ready r.id);
                  G.Leaf (Ready_arg r.id);
                  G.Int 0;
                  G.Fixed 1;
                |];
              split = None;
            };
        |];
    }
  in
  raises_match ~msg:"rails for another machine" Exn.invalid_arg (fun () ->
      G.load ~rails:(fun _ -> None) fill [| h |]);
  let far = require_ok ~pp:Format.pp_print_string (G.load fill [| h |]) in
  let pt = (G.run far { inputs = [||]; ints = [| 42 |] }).(0) in
  Rig.Point.wait pt;
  (* [arrived] is the word at byte 256 of the counts. *)
  until ~what:"the transfer's arrival" (fun () -> word_at local.counts 256 >= 1);
  let read_inbound =
    {
      G.devices = [| Rig.arch Rig.host |];
      memory = [| G.Rail { rail = r.id; area = Inbound } |];
      images = [||];
      code = [||];
      inputs = [| { G.device = 0; bytes = 8 } |];
      ints = 0;
      steps =
        [|
          G.Move
            {
              src = G.Memory { memory = 0; offset = 0; length = 8 };
              dst = G.Input 0;
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
  ignore (G.run here { inputs = [| x |]; ints = [||] });
  equal ~msg:"the word, here" string (le64 42) (read_host x);
  r.release ()

(* A program whose memory the agent's device cannot hold is the load's [Error],
   and the agent goes on: a program that fits loads after it. *)
let too_large () =
  with_job @@ fun j _ ->
  let h = List.hd (Rig_remote.hosts j) in
  let ds = Array.of_list (polled h) in
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
              init = { bytes = ""; holes = [||] };
              copies = One;
            };
        |];
    }
  in
  let why = require_error (G.load huge (Iarray.of_array ds)) in
  contains ~msg:"names the memory" ~sub:"no memory" why;
  is_ok ~pp:Format.pp_print_string ~msg:"a program that fits, after"
    (G.load t (Iarray.of_array ds))

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
         test ~timeout:60. "an agent refuses a binary that is no program"
           refused_binary;
         test ~timeout:60.
           "an agent answers a program its devices cannot hold, and goes on"
           too_large;
         test ~timeout:60.
           "a program's host code sends a word on a rail, which a program \
            reads at its other end"
           rail_end_to_end;
       ])
