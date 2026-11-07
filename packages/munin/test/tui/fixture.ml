(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The stores the screens are drawn from.

   Every log is written from the format's text (Munin.Store_format) with fixed
   times and ids, so that a screen is the same text from one run of the tests to
   the next. A run that reads running or crashed needs a writer that took its
   lock: a child of the test support program takes it ([write_run]). No run
   directory is renamed, since Windows refuses to rename a directory that holds
   a file another process has open without sharing its deletion, as a writer
   holds its log.

   The main store is a sweep of six runs of a ResNet on CIFAR-10 over the
   learning rate, one running, one failed, one whose loss went NaN, and two
   older runs of an MLP on MNIST, one of which crashed. Values come from closed
   forms plus a hashed noise, so that they are the same on every system. *)

open Munin

let now = 1793535480.0 (* 2026-11-01T12:18:00Z *)

(* Noise *)

(* [noise seed i] is in [-1, 1], from splitmix64 of [seed] and [i]: integer
   arithmetic, the same on every system. *)
let noise seed i =
  let open Int64 in
  let z = ref (add (mul (of_int seed) 0x9E3779B97F4A7C15L) (of_int i)) in
  z := mul (logxor !z (shift_right_logical !z 30)) 0xBF58476D1CE4E5B9L;
  z := mul (logxor !z (shift_right_logical !z 27)) 0x94D049BB133111EBL;
  z := logxor !z (shift_right_logical !z 31);
  let u = to_float (shift_right_logical !z 11) /. 9007199254740992. in
  (2. *. u) -. 1.

(* JSON *)

let num x =
  if Float.is_nan x then {|{"float":"NaN"}|}
  else if x = Float.infinity then {|{"float":"Infinity"}|}
  else if x = Float.neg_infinity then {|{"float":"-Infinity"}|}
  else Munin_float_text.Float_print.exact_float x

(* A JSON string (RFC 8259, section 7): a quote, a backslash and a control
   character escaped, every other byte as it is, UTF-8 included. *)
let str s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b {|\"|}
      | '\\' -> Buffer.add_string b {|\\|}
      | c when Char.code c < 0x20 -> Printf.bprintf b "\\u%04x" (Char.code c)
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let obj members = "{" ^ String.concat "," members ^ "}"
let member k v = str k ^ ":" ^ v

let value : Value.t -> string = function
  | `Bool b -> string_of_bool b
  | `Int n -> string_of_int n
  | `Float x -> num x
  | `String s -> str s

let values kvs = obj (List.map (fun (k, v) -> member k v) kvs)

(* A run of the sweep: its learning rate sets how low its losses go, and a run
   diverges from [nan_from] on. *)
type sweep = {
  seed : int;
  lr : float;
  warmup : int;
  floor : float; (* the train loss it tends to *)
  val_floor : float; (* the validation loss at its best *)
  overfit : float; (* how much the validation loss rises after 11,000 *)
  rate : float; (* steps per second *)
  nan_from : int option;
}

(* Runs *)

type run = {
  id : string;
  experiment : string;
  name : string;
  steps : int option;
  params : (string * Value.t) list;
  command : string list;
  host : string;
  pid : int;
  commit : string;
  dirty : bool;
  tags : string list;
  notes : string option;
  started : float;
  records : string list; (* every record after the start, in order *)
  updated : float; (* the log's modification time *)
  status : Run.status; (* the status it reads with *)
  sweep : sweep option; (* the curves a sweep run logs *)
}

let provenance r =
  obj
    [
      member "command" ("[" ^ String.concat "," (List.map str r.command) ^ "]");
      member "cwd" (str "/home/ada/src/resnet");
      member "hostname" (str r.host);
      member "pid" (string_of_int r.pid);
      member "ocaml" (str "5.4.0");
      member "raven" (str "1.0.0~dev");
      member "os" (str "Linux 6.12.1");
      member "arch" (str "x86_64");
      member "cpu" (str "AMD EPYC 7543 32-Core Processor");
      member "git"
        (obj
           [
             member "commit" (str r.commit);
             member "dirty" (string_of_bool r.dirty);
           ]);
      member "seeds" (obj [ member "init" "7" ]);
      member "env" (obj [ member "CUDA_VISIBLE_DEVICES" (str "0") ]);
    ]

let start_record r =
  obj
    ([
       member "type" (str "start");
       member "time" (num r.started);
       member "format" "1";
       member "experiment" (str r.experiment);
       member "name" (str r.name);
     ]
    @ (match r.steps with
      | Some n -> [ member "steps" (string_of_int n) ]
      | None -> [])
    @ [
        member "params"
          (values (List.map (fun (k, v) -> (k, value v)) r.params));
        member "provenance" (provenance r);
      ])

let metric_record ~time ?goal ?step_metric key =
  obj
    ([
       member "type" (str "metric");
       member "time" (num time);
       member "key" (str key);
     ]
    @ (match goal with
      | Some `Minimize -> [ member "goal" (str "minimize") ]
      | Some `Maximize -> [ member "goal" (str "maximize") ]
      | None -> [])
    @
    match step_metric with
    | Some m -> [ member "step_metric" (str m) ]
    | None -> [])

let samples_record kind ~time ~step kvs =
  obj
    [
      member "type" (str kind);
      member "time" (num time);
      member "step" (string_of_int step);
      member "values" (values (List.map (fun (k, x) -> (k, num x)) kvs));
    ]

let end_record ~time status =
  obj
    ([ member "type" (str "end"); member "time" (num time) ]
    @
    match status with
    | `Finished -> [ member "status" (str "finished") ]
    | `Killed -> [ member "status" (str "killed") ]
    | `Failed e -> [ member "status" (str "failed"); member "error" (str e) ])

let use_record ~time name version =
  obj
    [
      member "type" (str "use");
      member "time" (num time);
      member "name" (str name);
      member "version" (string_of_int version);
    ]

(* Training curves *)

let steps_per_epoch = 2000
let total = 20000
let decay s = exp (-.Float.of_int s /. 2500.)

let train_loss w s =
  match w.nan_from with
  | Some n when s >= n -> Float.nan
  | _ ->
      w.floor
      +. ((2.30 -. w.floor) *. decay s)
      +. (0.04 *. (0.3 +. decay s) *. noise w.seed s)

let val_loss w s =
  match w.nan_from with
  | Some n when s >= n -> Float.nan
  | _ ->
      w.val_floor
      +. ((2.0 -. w.val_floor) *. decay s)
      +. (w.overfit *. Float.max 0. (Float.of_int (s - 11000)) /. 9000.)
      +. (0.004 *. noise (w.seed + 1) s)

let lr_at w s =
  if s < w.warmup then w.lr *. Float.of_int (s + 1) /. Float.of_int w.warmup
  else
    let p = Float.of_int (s - w.warmup) /. Float.of_int (total - w.warmup) in
    w.lr *. 0.5 *. (1. +. cos (Float.pi *. p))

let finite_or x v = if Float.is_nan x then Float.nan else v

let train_values w s =
  let l = train_loss w s in
  [
    ("epoch", Float.of_int s /. Float.of_int steps_per_epoch);
    ("train/loss", l);
    ( "train/acc",
      finite_or l
        (Float.min 0.999
           (1.02 -. (0.24 *. l) +. (0.004 *. noise (w.seed + 2) s))) );
    ( "train/grad_norm",
      finite_or l (1.0 +. (3.0 *. decay s) +. (0.25 *. noise (w.seed + 3) s)) );
    ("train/lr", lr_at w s);
    ("train/samples_s", 2180. +. (45. *. noise (w.seed + 4) s));
    ( "train/data_ms",
      4.6
      +. (0.5 *. noise (w.seed + 5) s)
      +. if noise (w.seed + 6) s > 0.93 then 9. else 0. );
  ]

let val_values w s =
  let l = val_loss w s in
  [
    ("val/loss", l);
    ( "val/acc",
      finite_or l
        (0.955 -. (0.5 *. (l -. 0.38)) +. (0.002 *. noise (w.seed + 7) s)) );
    ("val/top5", finite_or l (0.9985 -. (0.06 *. (l -. 0.38))));
    ( "val/ece",
      finite_or l
        (0.022 +. (0.012 *. (l -. 0.38)) +. (0.0006 *. noise (w.seed + 8) s)) );
  ]

let declarations ~time =
  [
    metric_record ~time "epoch";
    metric_record ~time ~goal:`Minimize "train/loss";
    metric_record ~time ~goal:`Maximize "train/acc";
    metric_record ~time "train/grad_norm";
    metric_record ~time "train/lr";
    metric_record ~time "train/samples_s";
    metric_record ~time "train/data_ms";
    metric_record ~time ~goal:`Minimize ~step_metric:"epoch" "val/loss";
    metric_record ~time ~goal:`Maximize ~step_metric:"epoch" "val/acc";
    metric_record ~time ~goal:`Maximize ~step_metric:"epoch" "val/top5";
    metric_record ~time ~goal:`Minimize ~step_metric:"epoch" "val/ece";
    metric_record ~time ~goal:`Maximize "test/acc";
  ]

let gib = 1073741824.
let bytes x = Float.round (x *. gib)

(* The readings of the machine and the GPU at [t], [busy] the share of the time
   the run computes, every 15 s. *)
let reading_values w ~i ~busy =
  let n k = noise (w.seed + 20 + k) i in
  (* Every 16th reading falls in an evaluation, when the GPU waits on data. *)
  let busy_gpu =
    if busy && i mod 16 = 0 then 41. else if busy then 98. else 0.
  in
  let busy_cpu =
    if busy && i mod 16 = 0 then 38. else if busy then 71. else 4.
  in
  [
    ("sys/cpu_busy_pct", busy_cpu +. (3. *. n 0));
    ("sys/load_per_core", (if busy then 1.4 else 0.1) +. (0.08 *. n 1));
    ("sys/mem_used_bytes", bytes ((if busy then 33.0 else 9.2) +. (0.2 *. n 2)));
    ("sys/mem_total_bytes", bytes 62.7);
    ("sys/swap_used_bytes", 0.);
    ("sys/disk_read_bytes_per_s", Float.round (12e6 +. (2e6 *. n 3)));
    ("sys/disk_write_bytes_per_s", Float.round (3.1e6 +. (0.5e6 *. n 4)));
    ("sys/disk_busy_pct", 6. +. n 5);
    ("sys/interval_s", 15.);
    ("sys/proc_cpu_cores", (if busy then 9.6 else 0.2) +. (0.3 *. n 6));
    ("sys/proc_rss_bytes", bytes (if busy then 18.2 else 2.1));
    ("sys/proc_interval_s", 15.);
    ("dev/CUDA:0/allocated_bytes", bytes (if busy then 20.9 else 0.4));
    ("dev/CUDA:0/budget_bytes", bytes 22.0);
    ("dev/CUDA:0/memory_bytes", bytes 24.0);
    ("dev/CUDA:0/used_bytes", bytes (if busy then 21.3 else 0.8));
    ("dev/CUDA:0/busy_pct", busy_gpu +. if busy then n 7 else 0.);
    ("dev/CUDA:0/temperature_c", (if busy then 71. else 38.) +. n 8);
    ("dev/CUDA:0/power_w", (if busy then 280. else 31.) +. (6. *. n 9));
  ]

(* [log w ~start ~upto ~ended] is the records of a sweep run that logs every 50
   steps up to [upto], evaluates every 500, samples every 15 s, and [ended] says
   how it ended. Steps take [1 / rate] s, after 22 s of setup. *)
let sweep_records w ~start ~upto ~ended =
  let time_of s = start +. 22. +. (Float.of_int s /. w.rate) in
  let last = time_of upto in
  let train =
    List.init
      ((upto / 50) + 1)
      (fun k ->
        let s = k * 50 in
        let vs = train_values w s in
        let vs = if s mod 500 = 0 then vs @ val_values w s else vs in
        (time_of s, samples_record "metrics" ~time:(time_of s) ~step:s vs))
  in
  let readings =
    let n = Float.to_int ((last -. start -. 5.) /. 15.) + 1 in
    List.init n (fun i ->
        let t = start +. 5. +. (15. *. Float.of_int i) in
        let step =
          max 0
            (min upto (Float.to_int ((t -. start -. 22.) *. w.rate) / 50 * 50))
        in
        ( t,
          samples_record "reading" ~time:t ~step
            (reading_values w ~i ~busy:(t > start +. 22.)) ))
  in
  let merged =
    List.stable_sort (fun (a, _) (b, _) -> Float.compare a b) (train @ readings)
  in
  let head =
    declarations ~time:(start +. 0.5)
    @ [ use_record ~time:(start +. 1.) "cifar10" 3 ]
  in
  let tail, updated =
    match ended with
    | None -> ([], last)
    | Some (status, at_end) ->
        let t = last +. 30. in
        let test =
          if at_end then
            [
              samples_record "metrics" ~time:t ~step:upto
                [ ("test/acc", 0.955 -. (0.5 *. (val_loss w upto -. 0.38))) ];
            ]
          else []
        in
        (test @ [ end_record ~time:(t +. 1.) status ], t +. 1.)
  in
  (head @ List.map snd merged @ tail, updated)

(* The sweep *)

let sweep_run ~id ~name ~started ~pid ~commit ~dirty ~tags ?notes w ~upto ~ended
    ?(status = `Silent) () =
  let records, updated = sweep_records w ~start:started ~upto ~ended in
  let status =
    match ended with Some (e, _) -> (e :> Run.status) | None -> status
  in
  {
    id;
    experiment = "cifar-resnet";
    name;
    steps = Some total;
    params =
      [
        ("lr", `Float w.lr);
        ("warmup", `Int w.warmup);
        ("batch", `Int 256);
        ("epochs", `Int 10);
        ("wd", `Float 0.05);
        ("opt", `String "adamw");
      ];
    command =
      [
        "./train.exe";
        "--lr";
        Printf.sprintf "%g" w.lr;
        "--warmup";
        string_of_int w.warmup;
      ];
    host = "gpu-01";
    pid;
    commit;
    dirty;
    tags;
    notes;
    started;
    records;
    updated;
    status;
    sweep = Some w;
  }

let commit = "3c1d9e04b7a2f5e8c6d0a1b2c3d4e5f6a7b8c9d0"
let commit_dirty = "5eaf82b19d3c4e7f8a0b1c2d3e4f5a6b7c8d9e0f"

let tidy_otter =
  sweep_run ~id:"20261101T104102Z-7f3a0c21" ~name:"tidy-otter"
    ~started:1793529662.0 ~pid:48211 ~commit:commit_dirty ~dirty:true
    ~tags:[ "sweep" ] ~notes:"cosine schedule, no label smoothing"
    {
      seed = 11;
      lr = 3e-4;
      warmup = 500;
      floor = 0.26;
      val_floor = 0.3970;
      overfit = 0.004;
      rate = 2.14;
      nan_from = None;
    }
    ~upto:12350 ~ended:None ~status:`Running ()

let amber_fox =
  sweep_run ~id:"20261101T070205Z-a1f3c9e2" ~name:"amber-fox"
    ~started:1793516525.0 ~pid:47102 ~commit ~dirty:false ~tags:[ "sweep" ]
    {
      seed = 23;
      lr = 1e-3;
      warmup = 500;
      floor = 0.22;
      val_floor = 0.3818;
      overfit = 0.006;
      rate = 2.20;
      nan_from = None;
    }
    ~upto:total
    ~ended:(Some (`Finished, true))
    ()

let brisk_wren =
  sweep_run ~id:"20261101T061040Z-0b9d4e17" ~name:"brisk-wren"
    ~started:1793513440.0 ~pid:46990 ~commit ~dirty:false ~tags:[ "sweep" ]
    {
      seed = 37;
      lr = 1e-4;
      warmup = 2000;
      floor = 0.34;
      val_floor = 0.4092;
      overfit = 0.0;
      rate = 2.21;
      nan_from = None;
    }
    ~upto:total
    ~ended:(Some (`Finished, true))
    ()

let calm_lynx =
  sweep_run ~id:"20261101T045531Z-c4e81a90" ~name:"calm-lynx"
    ~started:1793508931.0 ~pid:46815 ~commit ~dirty:false ~tags:[ "sweep" ]
    {
      seed = 41;
      lr = 3e-3;
      warmup = 2000;
      floor = 0.24;
      val_floor = 0.3888;
      overfit = 0.012;
      rate = 2.18;
      nan_from = None;
    }
    ~upto:total
    ~ended:(Some (`Finished, true))
    ()

let dusty_mole =
  sweep_run ~id:"20261101T034002Z-5d27b3f8" ~name:"dusty-mole"
    ~started:1793504402.0 ~pid:46622 ~commit ~dirty:false ~tags:[ "sweep" ]
    {
      seed = 53;
      lr = 1e-2;
      warmup = 500;
      floor = 0.30;
      val_floor = 0.4560;
      overfit = 0.0;
      rate = 2.20;
      nan_from = Some 4210;
    }
    ~upto:total
    ~ended:(Some (`Finished, true))
    ()

let eager_hawk =
  sweep_run ~id:"20261101T030344Z-e9a04c63" ~name:"eager-hawk"
    ~started:1793502224.0 ~pid:46501 ~commit ~dirty:false ~tags:[ "sweep" ]
    {
      seed = 67;
      lr = 3e-2;
      warmup = 500;
      floor = 0.9;
      val_floor = 1.0;
      overfit = 0.0;
      rate = 2.10;
      nan_from = None;
    }
    ~upto:1130
    ~ended:(Some (`Failed "Rig.Lost (CUDA:0, \"out of memory\")", false))
    ()

(* The MLP runs log their loss every 50 steps and evaluate every epoch of 469
   steps, without readings. *)
let mlp ~id ~name ~started ~pid ~hidden ~seed ~floor ~upto ~ended =
  let time_of s = started +. 3. +. (Float.of_int s /. 9.5) in
  let loss s =
    floor
    +. ((2.30 -. floor) *. exp (-.Float.of_int s /. 400.))
    +. (0.01 *. noise seed s)
  in
  let declared = started +. 0.5 in
  let train =
    List.init
      ((upto / 50) + 1)
      (fun k ->
        let s = k * 50 in
        ( time_of s,
          samples_record "metrics" ~time:(time_of s) ~step:s
            [ ("train/loss", loss s) ] ))
  in
  let evals =
    List.init (upto / 469) (fun e ->
        let s = (e + 1) * 469 in
        let v = loss s +. 0.012 in
        ( time_of s +. 0.2,
          samples_record "metrics"
            ~time:(time_of s +. 0.2)
            ~step:s
            [ ("val/loss", v); ("val/acc", 0.99 -. (0.3 *. v)) ] ))
  in
  let records =
    [
      metric_record ~time:declared ~goal:`Minimize "train/loss";
      metric_record ~time:declared ~goal:`Minimize "val/loss";
      metric_record ~time:declared ~goal:`Maximize "val/acc";
    ]
    @ List.map snd
        (List.stable_sort
           (fun (a, _) (b, _) -> Float.compare a b)
           (train @ evals))
  in
  let records, updated =
    match ended with
    | Some status ->
        ( records @ [ end_record ~time:(time_of upto +. 1.) status ],
          time_of upto +. 1. )
    | None -> (records, time_of upto)
  in
  {
    id;
    experiment = "mnist-mlp";
    name;
    steps = Some 4690;
    params =
      [
        ("lr", `Float 0.1);
        ("batch", `Int 64);
        ("hidden", `Int hidden);
        ("opt", `String "sgd");
      ];
    command = [ "./mlp.exe"; "--hidden"; string_of_int hidden ];
    host = "gpu-01";
    pid;
    commit;
    dirty = false;
    tags = [];
    notes = None;
    started;
    records;
    updated;
    status = (match ended with Some e -> (e :> Run.status) | None -> `Crashed);
    sweep = None;
  }

let mlp_wide =
  mlp ~id:"20261031T162010Z-44c0de1a" ~name:"mlp-wide" ~started:1793463610.0
    ~pid:41007 ~hidden:1024 ~seed:79 ~floor:0.052 ~upto:4690
    ~ended:(Some `Finished)

let mlp_base =
  mlp ~id:"20261031T140551Z-9e5b2f04" ~name:"mlp-base" ~started:1793455551.0
    ~pid:40888 ~hidden:256 ~seed:83 ~floor:0.061 ~upto:2900 ~ended:None

let sweep =
  [ tidy_otter; amber_fox; brisk_wren; calm_lynx; dusty_mole; eager_hawk ]

(* Runs in the states the main store lacks *)

let base =
  {
    seed = 91;
    lr = 3e-4;
    warmup = 500;
    floor = 0.26;
    val_floor = 0.40;
    overfit = 0.0;
    rate = 2.2;
    nan_from = None;
  }

(* A run whose writer holds its lock and has logged no sample. *)
let warm_start =
  let started = now -. 12. in
  {
    (sweep_run ~id:"20261101T121748Z-1b2c3d4e" ~name:"warm-start" ~started
       ~pid:48977 ~commit ~dirty:false ~tags:[] base ~upto:0 ~ended:None
       ~status:`Running ())
    with
    records = declarations ~time:(started +. 0.5);
    updated = started +. 0.5;
  }

(* Two runs whose lock is not probed, since they have no writer file: one that
   wrote 3 s ago and one that last wrote 2h14m ago. *)
let unprobed ~id ~name ~quiet =
  let started = now -. quiet -. 22. -. (3150. /. base.rate) in
  sweep_run ~id ~name ~started ~pid:5120 ~commit ~dirty:false ~tags:[] base
    ~upto:3150 ~ended:None ()

let far_node =
  unprobed ~id:"20261101T115340Z-2c3d4e5f" ~name:"far-node" ~quiet:3.

let lost_node =
  unprobed ~id:"20261101T093947Z-3d4e5f60" ~name:"lost-node" ~quiet:8040.

let quiet_heron =
  sweep_run ~id:"20261101T030700Z-4e5f6071" ~name:"quiet-heron"
    ~started:1793502420.0 ~pid:46588 ~commit ~dirty:false ~tags:[] base
    ~upto:8200
    ~ended:(Some (`Killed, false))
    ()

let states = [ warm_start; far_node; lost_node; quiet_heron ]
let runs = sweep @ [ mlp_wide; mlp_base ]

(* Writing *)

let mkdir path = if not (Sys.file_exists path) then Sys.mkdir path 0o755
let log_text r = String.concat "\n" (start_record r :: r.records) ^ "\n"

let run_dir store id =
  Filename.concat (Filename.concat (Store.root store) "runs") id

let log_path store id = Filename.concat (run_dir store id) "events.jsonl"

let write_file ?(flags = [ Unix.O_CREAT; Unix.O_EXCL ]) path text =
  let fd = Unix.openfile path (Unix.O_WRONLY :: flags) 0o644 in
  let n = Unix.write_substring fd text 0 (String.length text) in
  Unix.close fd;
  assert (n = String.length text)

(* Writers that a child holds, so that their runs read running: its pid and the
   pipe whose end lets it exit. *)
let held = ref []

(* [release ()] lets every child that holds a run's lock exit, and waits for
   it. *)
let release () =
  List.iter
    (fun (pid, w) ->
      Unix.close w;
      ignore (Unix.waitpid [] pid))
    !held;
  held := []

let () = at_exit release

(* [take ~child store id] has a child resume the run [id] and hold its lock, and
   is the child's pid and the pipe on its standard input. *)
let take ~child store id =
  let in_r, in_w = Unix.pipe ~cloexec:true () in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let pid =
    Unix.create_process child
      [| child; "take"; Store.root store; id |]
      in_r out_w Unix.stderr
  in
  Unix.close in_r;
  Unix.close out_w;
  let ic = Unix.in_channel_of_descr out_r in
  let line = In_channel.input_line ic in
  close_in ic;
  if line <> Some "taken" then failwith ("Fixture: no child took " ^ id);
  (pid, in_w)

(* A run that reads running or crashed needs a writer that took its lock under
   this machine's identities. The fixture writes the run's start, a child
   resumes it and takes the lock, which also writes the run's [writer], and the
   fixture then writes the run's log over the child's, through a descriptor of
   its own. The lock is the child's open file's, which that write keeps. A
   running run's child holds the lock until this process exits, and a crashed
   run's child exits at once. *)
let write_run ~child store r =
  mkdir (Filename.concat (Store.root store) "runs");
  mkdir (run_dir store r.id);
  let log = log_path store r.id in
  (match r.status with
  | `Running | `Crashed ->
      write_file log (start_record r ^ "\n");
      let pid, w = take ~child store r.id in
      write_file ~flags:[ Unix.O_TRUNC ] log (log_text r);
      if r.status = `Running then held := (pid, w) :: !held
      else begin
        Unix.close w;
        ignore (Unix.waitpid [] pid)
      end
  | `Silent | `Finished | `Failed _ | `Killed -> write_file log (log_text r));
  if r.tags <> [] then Store.add_tags store r.id r.tags;
  Option.iter (fun n -> Store.set_notes store r.id (Some n)) r.notes;
  Unix.utimes log r.updated r.updated

let status_name : Run.status -> string = function
  | `Running -> "running"
  | `Silent -> "silent"
  | `Crashed -> "crashed"
  | `Finished -> "finished"
  | `Failed e -> "failed: " ^ e
  | `Killed -> "killed"

(* A run reads crashed only where its writer recorded the boot it ran under
   (Munin.Store_format, status), which no writer does on Windows. [boots ~child
   dir] tells, from the [writer] file a child writes as it takes a run in a
   store under [dir]. *)
let boots =
  let known = ref None in
  fun ~child dir ->
    match !known with
    | Some b -> b
    | None ->
        let root = Filename.concat dir "boots" in
        Sys.mkdir root 0o755;
        let store = Store.open_ ~root () in
        let r = { quiet_heron with id = "20261101T000000Z-b0075000" } in
        mkdir (Filename.concat root "runs");
        mkdir (run_dir store r.id);
        write_file (log_path store r.id) (start_record r ^ "\n");
        let pid, w = take ~child store r.id in
        Unix.close w;
        ignore (Unix.waitpid [] pid);
        let writer =
          In_channel.with_open_bin
            (Filename.concat (run_dir store r.id) "writer")
            In_channel.input_all
        in
        let b =
          let n = String.length writer in
          let rec has i =
            i + 6 <= n && (String.sub writer i 6 = {|"boot"|} || has (i + 1))
          in
          has 0
        in
        known := Some b;
        b

(* [make ~child root runs] writes a store of [runs] at [root], a directory that
   does not exist yet, and is the store, without the runs that read crashed
   where no writer records its boot ([boots]). It fails, naming the run, if a
   run reads back with another status than it was written for. *)
let make ~child root runs =
  Sys.mkdir root 0o755;
  let store = Store.open_ ~root () in
  let runs =
    if boots ~child (Filename.dirname root) then runs
    else List.filter (fun r -> r.status <> `Crashed) runs
  in
  List.iter (write_run ~child store) runs;
  List.iter
    (fun r ->
      let got = Option.map Run.status (Store.run store r.id) in
      if got <> Some r.status then
        Printf.ksprintf failwith "Fixture: %s reads %s, not %s" r.name
          (match got with Some s -> status_name s | None -> "missing")
          (status_name r.status))
    runs;
  store

(* [feed store r ~step ~time] appends to [r]'s log the record [r] logs at
   [step], as its program would, and gives the log the modification time
   [time]. *)
let feed store r ~step ~time =
  let w = Option.get r.sweep in
  let vs = train_values w step in
  let vs = if step mod 500 = 0 then vs @ val_values w step else vs in
  let t = r.started +. 22. +. (Float.of_int step /. w.rate) in
  let record = samples_record "metrics" ~time:t ~step vs in
  let log = log_path store r.id in
  write_file ~flags:[ Unix.O_APPEND ] log (record ^ "\n");
  Unix.utimes log time time
