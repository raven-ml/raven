(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Runs of programs, each row beside the floor that bounds it: the same
   submissions made by hand and submitted with Rig.submit. A row's distance to
   its floor is rig.program's share of a run. The step row times what a compiled
   decode step's host does: its own devices' submissions and the runs it hands
   to another machine, through a loopback [rig agent]. *)

module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module G = Rig_program

let strf = Printf.sprintf

(* Each row runs the queues every [drain] runs, so they stay short. *)
let drain = 64

let le64 v =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 (Int64.of_int v);
  Bytes.to_string b

let opened = ref 0

let polled () =
  incr opened;
  P.open_ (strf "bench-program:%d" !opened)

(* Runs on Polled *)

(* [steps] launches of Polled's [fill], one group each, alternating between two
   devices, each writing its device's input. *)
let fills steps =
  let a, pa = polled () and b, pb = polled () in
  let ds = [| a; b |] in
  let launch d =
    {
      G.queue = "COMPUTE:0";
      after = [||];
      work =
        G.Launch
          {
            image = d;
            kernel = "fill";
            params = { bytes = le64 0 ^ le64 7; holes = [||] };
            refs = [| { Sub.at = 0; slot = 0 } |];
            groups = (G.Fixed 1, G.Fixed 1, G.Fixed 1);
            threads = (G.Fixed 1, G.Fixed 1, G.Fixed 1);
            shared = G.Fixed 0;
          };
    }
  in
  let step i =
    let d = i land 1 in
    G.Submit
      {
        device = d;
        parts = [| launch d |];
        buffers = [| (G.Input d, B.Read_write) |];
        fixed = [||];
      }
  in
  let t =
    {
      G.devices = Iarray.of_array (Array.map Rig.arch ds);
      memory = [||];
      images =
        Iarray.init 2 (fun d ->
            { G.device = d; binary = { bytes = "functions"; holes = [||] } });
      code = [||];
      inputs = Iarray.init 2 (fun d -> { G.device = d; bytes = 8 });
      ints = 0;
      steps = Iarray.init steps step;
    }
  in
  (ds, [| pa; pb |], t)

type run = { ps : P.t array; go : unit -> unit; mutable n : int }

let drained r =
  r.n <- r.n + 1;
  if r.n mod drain = 0 then Array.iter (fun p -> ignore (P.run p)) r.ps

let described steps () =
  let ds, ps, t = fills steps in
  let p = Result.get_ok (G.load t (Iarray.of_array ds)) in
  let frame =
    { G.inputs = Array.map (fun d -> B.create d 8) ds; ints = [||] }
  in
  { ps; go = (fun () -> ignore (G.run p frame)); n = 0 }

(* The same submissions, made by hand: one per step, each run's block stored
   once. *)
let by_hand steps () =
  let ds, ps, _ = fills steps in
  let images =
    Array.map (fun d -> Result.get_ok (Rig.Image.load d "functions")) ds
  in
  let xs = Array.map (fun d -> [| B.create d 8 |]) ds in
  let sub i =
    let d = i land 1 in
    let work =
      Sub.Launch
        {
          image = images.(d);
          kernel = "fill";
          params = 16;
          refs = [| { Sub.at = 0; slot = 0 } |];
        }
    in
    let s =
      Sub.make ~access:[| B.Read_write |] ds.(d)
        [| { Sub.queue = "COMPUTE:0"; after = [||]; work } |]
    in
    let run = Sub.Run.make () and b = Sub.block s 0 in
    Sub.Run.groups run b 1 1 1;
    Sub.Run.threads run b 1 1 1;
    Sub.Run.int64 run b 8 7;
    (s, run, xs.(d))
  in
  let subs = Array.init steps sub in
  let go () =
    Array.iter
      (fun (s, run, buffers) -> ignore (Rig.submit s ~run ~buffers ~waits:[||]))
      subs
  in
  { ps; go; n = 0 }

let polled_rows =
  let row name setup =
    Thumper.bench_with_setup ~setup name (fun r ->
        r.go ();
        drained r)
  in
  Thumper.group "polled"
    [
      row "launch-1" (described 1);
      row "floor-launch-1" (by_hand 1);
      row "launches-64" (described 64);
      row "floor-launches-64" (by_hand 64);
    ]

(* A decode step on two machines *)

let key = String.make 32 'k'

(* [rig agent], built in dev/rig/bin, serving at a port it chooses: its process,
   its standard input, which ends it once closed, and its port. *)
let agent () =
  let exe =
    Filename.concat (Filename.dirname Sys.executable_name) "../../bin/main.exe"
  in
  let key_r, key_w = Unix.pipe ~cloexec:true () in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process exe
      [| exe; "agent"; "127.0.0.1:0" |]
      key_r out_w Unix.stderr
  in
  Unix.close key_r;
  Unix.close out_w;
  let line = key ^ "\n" in
  ignore (Unix.write_substring key_w line 0 (String.length line));
  let ic = Unix.in_channel_of_descr out_r in
  let rec port () =
    match In_channel.input_line ic with
    | None -> None
    | Some l when String.starts_with ~prefix:"listening " l ->
        let i = String.rindex l ':' in
        int_of_string_opt (String.sub l (i + 1) (String.length l - i - 1))
    | Some _ -> port ()
  in
  let port = port () in
  close_in ic;
  match port with
  | Some port -> (pid, key_w, port)
  | None ->
      Unix.close key_w;
      ignore (Unix.waitpid [] pid);
      failwith "rig agent: no port"

(* The submissions a decode step makes on the controller's own devices, and the
   runs it hands each other machine: 4 KB of words each. *)
let local = 1880
let shares = 2
let ints = 500

type step = {
  job : Rig_remote.t;
  pid : int;
  input : Unix.file_descr;
  host : Rig.t;
  share : G.loaded;
  frame : G.frame;
  subs : (Sub.t * Sub.Run.t) array;
  lock : Mutex.t;
  mutable last : int list; (* The host's values of the last two steps. *)
}

let stepping () =
  let pid, input, port = agent () in
  match
    Rig_remote.connect
      ~key:(Result.get_ok (Rig_remote.key key))
      [ ("127.0.0.1", port) ]
  with
  | Error why ->
      Unix.close input;
      ignore (Unix.waitpid [] pid);
      failwith why
  | Ok job ->
      let host = List.hd (Rig_remote.hosts job) in
      let nothing =
        {
          G.devices = [| Rig.arch host |];
          memory = [||];
          images = [||];
          code = [||];
          inputs = [||];
          ints;
          steps = [||];
        }
      in
      let share = Result.get_ok (G.load nothing [| host |]) in
      let d = Result.get_ok (Rig.memory_device (strf "bench-step:%d" pid)) in
      let arg = B.create Rig.host 8 in
      let fill =
        {
          Sub.queue = "COMPUTE:0";
          after = [||];
          work =
            Sub.Fill
              {
                fill = Rig_support.bump;
                arg;
                ring_units = 0;
                segment_bytes = 0;
              };
        }
      in
      let sub () = (Sub.make d [| fill |], Sub.Run.make ()) in
      {
        job;
        pid;
        input;
        host;
        share;
        frame = { G.inputs = [||]; ints = Array.make ints 1 };
        subs = Array.init local (fun _ -> sub ());
        lock = Mutex.create ();
        last = [];
      }

let stepped s =
  Rig_remote.close s.job;
  Unix.close s.input;
  ignore (Unix.waitpid [] s.pid)

(* One step: back-pressure on the step two before, then, under the hand-over
   lock, the other machine's runs and this machine's submissions. The clock
   stops when the last submit returns. *)
let step s =
  (match s.last with [ _; v ] -> Rig.wait s.host v | _ -> ());
  Mutex.lock s.lock;
  let v = ref 0 in
  for _ = 1 to shares do
    v := Rig.Point.value (G.run s.share s.frame).(0)
  done;
  Array.iter
    (fun (sub, run) -> ignore (Rig.submit sub ~run ~buffers:[||] ~waits:[||]))
    s.subs;
  Mutex.unlock s.lock;
  s.last <- (match s.last with [] -> [ !v ] | x :: _ -> [ !v; x ])

let step_rows =
  Thumper.group "step"
    [
      Thumper.bench_with_setup ~setup:stepping ~teardown:stepped "remote-1900"
        step;
    ]

(* Runs whose work takes an agent's device a while *)

(* test/program's fixture [spin]: a fill that counts to the word its argument
   points at. *)
let spin_object () =
  let machine = match Rig.arch Rig.host with "arm64" -> "aarch64" | m -> m in
  let file =
    Filename.concat
      (Filename.dirname Sys.executable_name)
      (strf "../../test/program/fixtures/spin_%s.o" machine)
  in
  In_channel.with_open_bin file In_channel.input_all

(* Counts that take the M1 Max's fill about 100 us. *)
let iterations = 100_000
let spins = 16

type spinning = {
  sjob : Rig_remote.t;
  spid : int;
  sprogram : G.loaded;
}

(* A program of one step on [d]: the [spin] fill. *)
let spinner d =
  let t =
    {
      G.devices = [| Rig.arch d |];
      memory =
        [|
          G.Alloc
            {
              device = 0;
              kind = B.Device;
              bytes = 8;
              init = { bytes = le64 iterations; holes = [||] };
              copies = G.One;
            };
        |];
      images = [||];
      code = [| { G.obj = spin_object (); entry = "spin" } |];
      inputs = [||];
      ints = 0;
      steps =
        [|
          G.Submit
            {
              device = 0;
              parts =
                [|
                  {
                    G.queue = "COMPUTE:0";
                    after = [||];
                    work =
                      G.Fill
                        {
                          fill = G.Code 0;
                          arg = { memory = 0; offset = 0; length = 8 };
                          ring_units = 0;
                          segment_bytes = 0;
                        };
                  };
                |];
              buffers = [||];
              fixed = [||];
            };
        |];
    }
  in
  Result.get_ok (G.load t [| d |])

(* bench_agent.exe, whose Polled device runs its own queue, and the spinner on
   it. *)
let spinning () =
  let exe =
    Filename.concat (Filename.dirname Sys.executable_name) "bench_agent.exe"
  in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let spid = Unix.create_process exe [| exe |] Unix.stdin out_w Unix.stderr in
  Unix.close out_w;
  let ic = Unix.in_channel_of_descr out_r in
  let port = int_of_string (input_line ic) in
  close_in ic;
  let sjob =
    Result.get_ok
      (Rig_remote.connect
         ~key:(Result.get_ok (Rig_remote.key key))
         [ ("127.0.0.1", port) ])
  in
  let host = List.hd (Rig_remote.hosts sjob) in
  let d = List.hd (Result.get_ok (Rig_remote.devices host "POLLED")) in
  { sjob; spid; sprogram = spinner d }

let spun s =
  Rig_remote.close s.sjob;
  ignore (Unix.waitpid [] s.spid)

let no_frame = { G.inputs = [||]; ints = [||] }

(* [spins] runs handed over at once, until the controller sees the last one's
   value: each run's fill, plus what the agent does between them that its device
   does not overlap. *)
let spins_of p =
  let pt = ref (G.run p no_frame).(0) in
  for _ = 2 to spins do
    pt := (G.run p no_frame).(0)
  done;
  Rig.Point.wait !pt

(* The floor: the same runs on a Polled device of this process that runs its own
   queue, which no agent sits in front of. *)
let spinning_here () =
  incr opened;
  let d, _ = P.open_ ~runs:`Itself (strf "bench-spin:%d" !opened) in
  spinner d

let spin_rows =
  Thumper.group "agent"
    [
      Thumper.bench_with_setup ~setup:spinning ~teardown:spun
        (strf "spins-%d" spins) (fun s -> spins_of s.sprogram);
      Thumper.bench_with_setup ~setup:spinning_here
        (strf "floor-spins-%d" spins)
        spins_of;
    ]

let () = exit @@ Thumper.run "rig_program" [ polled_rows; step_rows; spin_rows ]
