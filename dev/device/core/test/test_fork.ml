(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A forked child of a process that opened devices. Its own suite: a process
   that ran a domain cannot fork. *)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled

let timeout = 60.
let empty d = Sub.make ~reads:0 ~writes:0 ~waits:0 d [||]
let raises_lost f = match f () with _ -> false | exception C.Lost _ -> true

let status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "stopped by signal %d" n

(* Runs [child] in a forked child and is what it returned: the child writes its
   answer to a pipe and leaves without running this process's exit. *)
let in_child child =
  let r, w = Unix.pipe () in
  match Unix.fork () with
  | 0 ->
      Unix.close r;
      let lines = try child () with e -> [ Printexc.to_string e ] in
      let oc = Unix.out_channel_of_descr w in
      List.iter (fun l -> output_string oc (l ^ "\n")) lines;
      close_out oc;
      Unix._exit 0
  | pid ->
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let lines = In_channel.input_lines ic in
      close_in ic;
      let _, status = Unix.waitpid [] pid in
      (lines, status)

(* The child sees the device lost, calls no driver function and reads no word:
   memory dropped in it stays, even once the word reads every submitted value,
   which would free it in a process that read it. *)
let test_child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let d, p = P.open_ "fork:child" in
  let b = ref (Some (B.create d 64)) in
  let v = C.Point.value (C.submit (empty d)) in
  let calls = P.log p in
  let lines, ended =
    in_child (fun () ->
        let lost = C.lost d in
        let submit = raises_lost (fun () -> C.submit (empty d)) in
        let wait = raises_lost (fun () -> C.wait d v) in
        b := None;
        Gc.full_major ();
        ignore (B.create C.host 8);
        P.set_word p v;
        Gc.full_major ();
        ignore (B.create C.host 8);
        [
          Option.value ~default:"not lost" lost;
          Printf.sprintf "submit raises Lost: %b" submit;
          Printf.sprintf "wait raises Lost: %b" wait;
          Printf.sprintf "driver calls since the fork: %s"
            (String.concat " "
               (List.filteri (fun i _ -> i >= List.length calls) (P.log p)));
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string)
    [
      "forked";
      "submit raises Lost: true";
      "wait raises Lost: true";
      "driver calls since the fork: ";
    ]
    lines;
  equal (option string) None (C.lost d);
  C.wait d v;
  ignore (Sys.opaque_identity !b)

(* An io device whose memory is host bytes, whose state its library holds. *)
module Store = struct
  type t = unit

  type region =
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int

  let alloc () n =
    Some (Bigarray.Array1.create Bigarray.char Bigarray.c_layout n)

  let free () _ = ()
  let read () _ ~at:_ ~dst:_ ~len:_ = ()
  let write () _ ~at:_ ~src:_ ~len:_ = ()
  let pages () r = Some r
  let prefetch () _ ~at:_ ~len:_ = ()
  let stop () = ()
end

(* An io device's state is its library's, which decides what a fork does to it:
   the core leaves it usable in the child, where a driver's device is lost. *)
let test_io_child () =
  if Sys.win32 then skip ~reason:"Windows has no fork" ();
  let io =
    require_ok ~pp:Format.pp_print_string
      (C.open_io (module Store) ~name:"fork:io" (fun () -> Ok ()))
  in
  let d, _ = P.open_ "fork:beside-io" in
  let lines, ended =
    in_child (fun () ->
        [
          Option.value ~default:"not lost" (C.lost io);
          Printf.sprintf "bytes made: %d" (B.length (B.create io 8));
          Option.value ~default:"not lost" (C.lost d);
        ])
  in
  equal string "exited 0" (status ended);
  equal (list string) [ "not lost"; "bytes made: 8"; "forked" ] lines

let tests =
  [
    group ~timeout "fork"
      [
        test "a forked child's devices are lost for good" test_child;
        test "a forked child's io devices stay its library's" test_io_child;
      ];
  ]

let () = exit (run "device_core.fork" tests)
