(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module A = Rig_amd
module S = Rig_amd_support
module Abi = Rig_amd_abi
module Pm4 = Abi.Pm4

let strf = Printf.sprintf

let contains_sub s sub =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

let read_fixture name =
  In_channel.with_open_bin ("fixtures/" ^ name) In_channel.input_all

let words p =
  let s = Abi.Packet.encode Int64.of_int p in
  Array.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let host_buffer s =
  let b = Bigarray.(Array1.create char c_layout (String.length s)) in
  String.iteri (fun i c -> b.{i} <- c) s;
  Rig.Buffer.of_bigarray b

let get b =
  let n = Rig.Buffer.length b in
  let a = Bigarray.(Array1.create char c_layout n) in
  Rig.Buffer.copy ~src:b ~dst:(Rig.Buffer.of_bigarray a);
  String.init n (fun i -> a.{i})

let copy ~dst src =
  { Rig.Submission.queue = "COPY:0"; after = [||]; work = Copy { src; dst } }

(* A dispatch of [name] of the loaded object [p], one workgroup of 64, which
   reads no arguments. *)
let dispatch g co p name =
  let gpu = (A.capability g).gpu in
  let k = Option.get (Abi.Code_object.kernel co name) in
  let base = Option.get (Rig.Image.entry p name) - k.descriptor in
  S.words_part ~queue:"COMPUTE:0"
    (words
       (Pm4.run gpu
          (Pm4.dispatch gpu k ~program:(base + k.entry) ~scratch:0 ~args:0
             ~packet:0 ~threads:(64, 1, 1) ~groups:(1, 1, 1) ())))

let page = 4096

let lost what f =
  match f () with
  | _ -> failf "%s raised no Lost" what
  | exception Rig.Lost (_, why) -> why

(* After the fault, in a process of its own: the kernel driver schedules no
   queue of a process whose address space of the GPU faulted, so the GPU's use
   after a fault is another process's. The child runs a copy of 64 bytes through
   rig and exits 0 once it holds them, 1 if they differ, 2 if its value is not
   reached within 10 s. The parent holds the GPU lock. *)
let after_env = "RIG_AMD_FAULT_AFTER"

let after () =
  let made = ref None in
  let make () =
    Result.map
      (fun g ->
        made := Some g;
        g)
      (Rig_amd_amdgpu.open_ 0)
  in
  let c =
    match Rig.open_ (module A) ~name:"AMD:after" make with
    | Ok c -> c
    | Error why ->
        prerr_endline why;
        exit 1
  in
  let g = Option.get !made in
  let src = Rig.Buffer.create ~memory:Pinned c 64 in
  let dst = Rig.Buffer.create c 64
  and back = Rig.Buffer.create ~memory:Pinned c 64 in
  Rig.Buffer.copy ~src:(host_buffer (String.make 64 'y')) ~dst:src;
  let s =
    Rig.Submission.make ~reads:0 ~writes:0 c
      [| copy ~dst src; copy ~dst:back dst |]
  in
  let v = Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||]) in
  let rec reached n =
    A.signaled g >= v
    || n > 0
       && begin
         A.sleep g ~seen:(A.signaled g) ~still_ms:500;
         reached (n - 1)
       end
  in
  if not (reached 20) then exit 2;
  exit (if get back = String.make 64 'y' then 0 else 1)

(* A fault no sleep read, in a process of its own: the child runs [wild] through
   rig, lets 500 ms of CPU time pass without reading the device, and opens GPU 0
   again. It exits 0 if the open answers Error naming the fault, 1 otherwise. *)
let unread_env = "RIG_AMD_FAULT_UNREAD"

let unread () =
  let made = ref None in
  let make () =
    Result.map
      (fun g ->
        made := Some g;
        g)
      (Rig_amd_amdgpu.open_ 0)
  in
  let c =
    match Rig.open_ (module A) ~name:"AMD:unread" make with
    | Ok c -> c
    | Error why ->
        prerr_endline why;
        exit 1
  in
  let g = Option.get !made in
  let bin = read_fixture "kernels_gfx1201.hsaco" in
  let co = Result.get_ok (Abi.Code_object.of_string bin) in
  let p =
    match Rig.Image.load c bin with
    | Ok p -> p
    | Error why ->
        prerr_endline why;
        exit 1
  in
  ignore
    (Rig.submit
       (Rig.Submission.make ~reads:0 ~writes:0 c [| dispatch g co p "wild" |])
       ~reads:[||] ~writes:[||] ~waits:[||]);
  let t0 = Sys.time () in
  while Sys.time () -. t0 < 0.5 do
    ()
  done;
  match Rig_amd_amdgpu.open_ 0 with
  | Ok _ ->
      prerr_endline "an open after an unread fault answered a device";
      exit 1
  | Error why ->
      prerr_endline why;
      exit (if contains_sub why "faulted in this process" then 0 else 1)

(* Value 1 stores to address 0, which no GPU maps; value 2, queued behind it,
   would copy into a watched buffer. The kernel driver reports the fault: the
   device is lost, every wait and use answers at once with the report, nothing
   the device was given after the fault runs, an open of the GPU in a faulted
   process answers Error naming the fault, whether or not a sleep read it, and a
   new process runs work on the GPU. *)
let faults () =
  let g = S.gpu () in
  let d = S.rig g in
  let bin = read_fixture "kernels_gfx1201.hsaco" in
  let co = Result.get_ok (Abi.Code_object.of_string bin) in
  let p =
    match Rig.Image.load d bin with Ok p -> p | Error why -> fail why
  in
  let zeros = String.make 64 '\000' in
  (* The watched bytes are the host's, on a page of their own the device maps,
     so the host reads them whatever became of the device. *)
  let whole = Bigarray.(Array1.create char c_layout (2 * page)) in
  Bigarray.Array1.fill whole '\000';
  let a = Rig.Buffer.address (Rig.Buffer.of_bigarray whole) in
  let first = (page - (a land (page - 1))) land (page - 1) in
  let host = Bigarray.Array1.sub whole first 64 in
  let watched =
    Option.get (Rig.Buffer.borrow d (Rig.Buffer.of_bigarray host))
  in
  let src = Rig.Buffer.create ~memory:Pinned d 64 in
  Rig.Buffer.copy ~src:(host_buffer (String.make 64 'x')) ~dst:src;
  let v1 = S.submit g [| dispatch g co p "wild" |] in
  let v2 = S.submit g [| copy ~dst:watched src |] in
  let why =
    lost "the wait for the value after the fault" (fun () -> Rig.wait d v2)
  in
  contains ~msg:"the report" ~sub:"memory fault at" why;
  contains ~msg:"the wait for the fault's own value" ~sub:"memory fault at"
    (lost "the wait for the fault's value" (fun () -> Rig.wait d v1));
  contains ~msg:"a submit after the fault" ~sub:"memory fault at"
    (lost "a submit after the fault" (fun () -> S.submit g [||]));
  S.still ~msg:"the watched bytes (sampled)" string zeros
    (fun () -> String.init 64 (fun i -> host.{i}))
    ~ms:200;
  (match Rig_amd_amdgpu.open_ 0 with
  | Ok _ -> fail "an open of the GPU in the process its fault lost"
  | Error why ->
      contains ~msg:"an open in this process"
        ~sub:"GPU 0 faulted in this process" why;
      contains ~msg:"an open in this process, the fault" ~sub:"memory fault at"
        why);
  equal int ~msg:"an open after a fault no sleep read, in a new process" 0
    (Sys.command
       (strf "%s=1 %s" unread_env (Filename.quote Sys.executable_name)));
  equal int ~msg:"a new process's copy" 0
    (Sys.command
       (strf "%s=1 %s" after_env (Filename.quote Sys.executable_name)))

let () =
  if Sys.getenv_opt after_env = Some "1" then after ();
  if Sys.getenv_opt unread_env = Some "1" then unread ();
  S.hold_gpu ();
  exit
    (run "rig_amd fault"
       [
         group ~timeout:60. "fault"
           [ test "a fault of the work is the kernel driver's report" faults ];
       ])
