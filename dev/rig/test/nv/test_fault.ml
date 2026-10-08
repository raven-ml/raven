(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A fault on the GPU, through rig. A kernel stores to address 0, which no page
   table maps: the RM stops the channel and reports the fault. The device is
   lost with the report, every later use of it raises Lost, the process's files
   and mappings come back once its memory is collected, and a fresh process
   opens the GPU and launches the kernel correctly. Each runs in a forked child,
   one fault per run; the suite forks before any call to NVIDIA's driver. *)

open Windtrap
module N = Rig_nv
module P = Rig_nv_nvidia
module S = Rig_nv_support

let strf = Printf.sprintf
let timeout = 60.

(* The work: [double_index] over [n] 32-bit words at [out]. *)
let n = 1000
let lines path = In_channel.with_open_text path In_channel.input_lines
let open_files () = Array.length (Sys.readdir "/proc/self/fd")
let mappings () = List.length (lines "/proc/self/maps")

(* [in_child f] is [f ()] in a forked child, its result sent back through a
   pipe. [f] calls no verb: a failure in the child would end the child's copy of
   the run. *)
let in_child (f : unit -> 'a) : ('a, string) result =
  let r, w = Unix.pipe ~cloexec:true () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let result = try Ok (f ()) with e -> Error (Printexc.to_string e) in
      let oc = Unix.out_channel_of_descr w in
      Marshal.to_channel oc result [];
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let result = Marshal.from_channel ic in
      close_in ic;
      ignore (Unix.waitpid [] pid);
      result

let names = ref 0

(* GPU 0, opened through rig under a name of its own. *)
let open_gpu () =
  let g = Result.get_ok (P.open_ 0) in
  incr names;
  match
    Rig.open_ (module N) ~name:(strf "NV:fault%d" !names) (fun () -> Ok g)
  with
  | Ok d -> { S.d; g }
  | Error why -> failwith why

(* Collects and drains twice: a stopped device's word goes back at the second
   drain, once every domain passed a minor collection since the first. *)
let collect () =
  for _ = 1 to 2 do
    Gc.full_major ();
    Gc.full_major ();
    ignore (Rig.Buffer.create Rig.host 8)
  done

(* Launches [double_index] on [t] over [out], an address of [t]'s. *)
let double t out =
  let k = S.kernels t in
  let l = S.launches t.g in
  Fun.protect
    ~finally:(fun () -> S.free_launches l)
    (fun () ->
      S.run t [| S.words (S.launch l k "double_index" ~blocks:4 [ out; n ]) |])

(* Opens the GPU and launches the kernel over pinned memory: whether its words
   are the kernel's. *)
let launch_correctly () =
  let t = open_gpu () in
  let out = Rig.Buffer.create ~memory:Pinned t.d (4 * n) in
  double t (Rig.Buffer.address out);
  let h = Rig.Buffer.create Rig.host (4 * n) in
  Rig.Buffer.copy ~src:out ~dst:h;
  let a = Rig.Buffer.bigarray Bigarray.int32 h in
  let ok =
    List.for_all (fun i -> a.{i} = Int32.of_int (2 * i)) (List.init n Fun.id)
  in
  S.close t;
  ok

let lost d = function
  | Rig.Lost (d', why) when Rig.equal d d' -> Some why
  | _ -> None

let outcome d f =
  match f () with
  | _ -> "returned"
  | exception e -> (
      match lost d e with
      | Some why -> "Lost: " ^ why
      | None -> "raised " ^ Printexc.to_string e)

type report = {
  run : string;
  lost : string option;
  later : (string * string) list;
  files : int * int;
  maps : int * int;
}

(* After a launch on a GPU that opened and stopped once, which made what the
   process keeps for good: a fault, what it answers, and the process's files and
   mappings once the device's memory is collected. *)
let fault () =
  ignore (launch_correctly ());
  collect ();
  let files = open_files () and maps = mappings () in
  let t = open_gpu () in
  let d = t.d in
  let run = outcome d (fun () -> double t 0) in
  let later =
    [
      ("Buffer.create", outcome d (fun () -> Rig.Buffer.create d 64));
      ( "Image.load",
        outcome d (fun () ->
            Rig.Image.load d (S.fixture "kernels_sm89.cubin")) );
      ( "Submission.make",
        outcome d (fun () -> Rig.Submission.make ~reads:0 ~writes:0 d [||]) );
      ("wait", outcome d (fun () -> Rig.wait d (Rig.submitted d)));
    ]
  in
  let lost = Rig.lost d in
  collect ();
  {
    run;
    lost;
    later;
    files = (files, open_files ());
    maps = (maps, mappings ());
  }

let need_gpu () =
  if P.count () = 0 then skip ~reason:"the machine has no NVIDIA GPU" ();
  S.hold_gpu ()

(* The lines of a report that start with [prefix]. *)
let starting prefix why =
  String.split_on_char '\n' why |> List.filter (String.starts_with ~prefix)

(* A channel error's line up to its name: "channel error 31 (NAME)". *)
let named l =
  match String.index_opt l ')' with Some i -> String.sub l 0 (i + 1) | None -> l

(* An MMU fault's line as its address and its access, the first and last of its
   fields. *)
let address_access l =
  match String.split_on_char '|' l |> List.map String.trim with
  | first :: (_ :: _ as rest) -> (first, List.nth rest (List.length rest - 1))
  | _ -> (l, "")

(* The device is lost with the RM's report, which every later use raises again:
   the channel group's error once, by number and name, then the MMU's fault at
   address 0 on a write. The files and mappings the device took come back, its
   timeline word's included. *)
let test_fault () =
  need_gpu ();
  let r = require_ok (in_child fault) in
  let why = require_some ~msg:("lost, after the run " ^ r.run) r.lost in
  equal ~msg:"the report's channel errors" (list string)
    [ "channel error 31 (FIFO_ERROR_MMU_ERR_FLT)" ]
    (List.map named (starting "channel error " why));
  equal ~msg:"the report's MMU faults" (list (pair string string))
    [ ("MMU fault: 0x0", "VIRT_WRITE") ]
    (List.map address_access (starting "MMU fault: " why));
  equal ~msg:"the run" string ("Lost: " ^ why) r.run;
  List.iter
    (fun (use, got) ->
      equal ~msg:("a later " ^ use) string ("Lost: " ^ why) got)
    r.later;
  equal ~msg:"files" int (fst r.files) (snd r.files);
  equal ~msg:"mappings" int (fst r.maps) (snd r.maps)

let test_fresh () =
  need_gpu ();
  equal bool true (require_ok (in_child launch_correctly))

let () =
  S.hold_gpu ();
  exit
    (run "rig_nv fault"
       [
         group ~timeout "a kernel storing to address 0"
           [
             test
               "loses the device with the RM's report, which every later use \
                raises, and gives its files back"
               test_fault;
             test "leaves a fresh process to open the GPU and launch correctly"
               test_fresh;
           ];
       ])
