(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A fault on the GPU, through rig. A kernel stores to address 0, which no page
   table maps: the RM stops the channel and reports the fault. The device is
   lost with the report, every later use of it raises Lost, the process's files
   and mappings come back once its memory is collected, but the page of its
   timeline word, which the device keeps for the process, and a fresh process
   opens the GPU and launches the kernel correctly. Each runs in a forked child,
   one fault per run; the suite forks before any call to NVIDIA's driver. *)

open Windtrap
module N = Rig_nv
module S = Rig_nv_support

let timeout = 60.

(* The work: [double_index] over [n] 32-bit words at [out]. *)
let n = 1000
let open_files () = Array.length (Sys.readdir "/proc/self/fd")

(* [in_child f] is [f ()] in a forked child, its result sent back through a
   pipe. [f] calls no verb: a failure in the child would end the child's copy of
   the run. *)
let in_child (f : unit -> 'a) : ('a, string) result =
  let r, w = Unix.pipe ~cloexec:true () in
  Rig_support.needs_fork ();
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

(* Collects, then frees [d]'s cache, which first drains [d]: a lost device's
   memory collected after its stop returns so. *)
let collect d =
  Gc.full_major ();
  Rig.free_cache d

(* Launches [double_index] on [t] over [out], an address of [t]'s. *)
let double t out =
  let k = S.kernels t in
  let l = S.launches t.g in
  Fun.protect
    ~finally:(fun () -> S.free_launches l)
    (fun () ->
      let words = Rig.Buffer.create Rig.host 8 in
      let part = Rig.Submission.words words in
      let fixed =
        Option.fold ~none:[]
          ~some:(fun b -> [ (b, Rig.Read_write) ])
          out
      in
      let s = Rig.Submission.make ~fixed t.d [| part |] in
      let address = Option.fold ~none:0 ~some:(Rig.Submission.address s) out in
      let ws = S.launch l k "double_index" ~blocks:4 [ address; n ] in
      Rig.Buffer.write words Bigarray.int32 (fun a ->
          Array.iteri (fun i w -> a.{i} <- Int32.of_int w) ws);
      let run = Rig.Submission.Run.make s in
      Rig.Point.wait (Rig.submit run ~buffers:[||]))

(* Opens the GPU and launches the kernel over pinned memory: whether its words
   are the kernel's. *)
(* Launches on a device it then closes: whether the launch wrote what it should,
   and the device. *)
let launched () =
  let t = S.open_ () in
  let out = Rig.Buffer.create ~memory:Pinned t.d (4 * n) in
  double t (Some out);
  let h = Rig.Buffer.create Rig.host (4 * n) in
  Rig.Buffer.copy ~src:out ~dst:h;
  let ok =
    Rig.Buffer.read h Bigarray.int32 (fun a ->
        List.for_all
          (fun i -> a.{i} = Int32.of_int (2 * i))
          (List.init n Fun.id))
  in
  S.close t;
  (ok, t.d)

let launch_correctly () = fst (launched ())

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
  maps : string list * string list;
}

(* After a launch on a GPU that opened and stopped once, which made what the
   process keeps for good: a fault, what it answers, and the process's files and
   mappings once the device's memory is collected. *)
let fault () =
  collect (snd (launched ()));
  let files = open_files () and maps = Rig_nv_nvidia_support.mappings () in
  let t = S.open_ () in
  let d = t.d in
  let run = outcome d (fun () -> double t None) in
  let later =
    [
      ("Buffer.create", outcome d (fun () -> Rig.Buffer.create d 64));
      ( "Image.load",
        outcome d (fun () -> Rig.Image.load d (S.fixture "kernels_sm89.cubin"))
      );
      ("Submission.make", outcome d (fun () -> Rig.Submission.make d [||]));
      ( "submit",
        outcome d (fun () ->
            let empty = Rig.Submission.make d [||] in
            Rig.submit (Rig.Submission.Run.make empty) ~buffers:[||]) );
    ]
  in
  let lost = Rig.lost d in
  collect d;
  {
    run;
    lost;
    later;
    files = (files, open_files ());
    maps = (maps, Rig_nv_nvidia_support.mappings ());
  }

(* These laws are NVIDIA's kernel driver's: its RM stops a faulted channel,
   and the files and mappings they count are its own. *)
let need_gpu () =
  match Rig_nv_support.path () with
  | Some Kernel -> ()
  | Some Pci ->
      skip
        ~reason:
          "the faults' laws are NVIDIA's kernel driver's; GPU 0 is over PCI"
        ()
  | None -> skip ~reason:"the machine has no NVIDIA GPU" ()

(* The lines of a report that start with [prefix]. *)
let starting prefix why =
  String.split_on_char '\n' why |> List.filter (String.starts_with ~prefix)

(* A channel error's line up to its name: "channel error 31 (NAME)". *)
let named l =
  match String.index_opt l ')' with
  | Some i -> String.sub l 0 (i + 1)
  | None -> l

(* An MMU fault's line as its address and its access, the first and last of its
   fields. *)
let address_access l =
  match String.split_on_char '|' l |> List.map String.trim with
  | first :: (_ :: _ as rest) -> (first, List.nth rest (List.length rest - 1))
  | _ -> (l, "")

(* The bytes and the permissions of a line of [/proc/self/maps]. *)
let bytes_of line = Scanf.sscanf line "%x-%x" (fun a b -> b - a)
let perms line = Scanf.sscanf line "%x-%x %s" (fun _ _ p -> p)

(* The device is lost with the RM's report, which every later use raises again:
   the channel group's error once, by number and name, then the MMU's fault at
   address 0 on a write. The files the device took come back, and its
   mappings: its timeline word lies in rig's array. *)
let test_fault () =
  need_gpu ();
  let r = require_ok (in_child fault) in
  let why = require_some ~msg:("lost, after the run " ^ r.run) r.lost in
  equal ~msg:"the report's channel errors" (list string)
    [ "channel error 31 (FIFO_ERROR_MMU_ERR_FLT)" ]
    (List.map named (starting "channel error " why));
  equal ~msg:"the report's MMU faults"
    (list (pair string string))
    [ ("MMU fault: 0x0", "VIRT_WRITE") ]
    (List.map address_access (starting "MMU fault: " why));
  equal ~msg:"the run" string ("Lost: " ^ why) r.run;
  List.iter
    (fun (use, got) ->
      equal ~msg:("a later " ^ use) string ("Lost: " ^ why) got)
    r.later;
  equal ~msg:"files" int (fst r.files) (snd r.files);
  let before, after = r.maps in
  let kept =
    List.filter (fun l -> perms l <> "---p" && not (List.mem l before)) after
  in
  equal ~msg:"the bytes of each GPU mapping the device keeps" (list int) []
    (List.map bytes_of kept)

let test_fresh () =
  need_gpu ();
  equal bool true (require_ok (in_child launch_correctly))

let () =
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
