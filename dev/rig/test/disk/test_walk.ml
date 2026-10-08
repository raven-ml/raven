(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Failure walks of the disk. A child process of the suite runs an operation
   under a limit of its own: of a file's size, [n] bytes at the edges of the
   file's pages, so that the write that would grow a file past [n] fails; or of
   open files, [n] descriptors left, so that an open that needs one more fails.
   It runs each four times. After each failure the outcome is the one
   rig_disk.mli states, the disk is not lost, the descriptors and the C heap are
   back where they were, and the operation runs again once the limit is
   lifted. *)

open Windtrap
module B = Rig.Buffer
module Support = Rig_disk_support

let strf = Printf.sprintf
let timeout = 60.
let reps = 4

(* The least growth of the C heap over [reps] runs a failure may leave: the
   allocator's rounding. A leak shows in every run, while the allocator's own
   caches move single runs either way. *)
let heap_slack = 128

(* The file the operations reach, and the limits walked: bytes at the edges of
   its pages and its size, where nothing fails, and descriptors left. *)
let size = (4 * 4096) + 1
let pattern = String.init size (fun i -> Char.chr (65 + (i mod 26)))

type limit = Bytes of int | Descriptors of int

let limits =
  List.map (fun n -> Bytes n) [ 0; 1; 4095; 4096; 4097; size - 1; size ]
  @ [ Descriptors 0; Descriptors 1 ]

let limit_args = function
  | Bytes n -> [ "bytes"; string_of_int n ]
  | Descriptors n -> [ "descriptors"; string_of_int n ]

let of_args kind n =
  let n = int_of_string n in
  if kind = "bytes" then Bytes n else Descriptors n

(* Files *)

(* The suite and its children write their files beside this executable, under
   _build, and remove them as they end. *)
let exe =
  let e = Sys.executable_name in
  if Filename.is_relative e then Filename.concat (Sys.getcwd ()) e else e

let dir = Filename.concat (Filename.dirname exe) "walk-files"
let names = ref 0

let new_path () =
  incr names;
  Filename.concat dir (strf "f%d" !names)

let clear () =
  if Sys.file_exists dir then
    Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir)
  else Sys.mkdir dir 0o755

let host_of_string s =
  let b = B.create Rig.host (String.length s) in
  String.iteri (Bigarray.Array1.set (B.bigarray Bigarray.char b)) s;
  b

let read b =
  let h = B.create Rig.host (B.length b) in
  B.copy ~src:b ~dst:h;
  let a = B.bigarray Bigarray.char h in
  String.init (Bigarray.Array1.dim a) (Bigarray.Array1.get a)

(* Limits *)

let checked = function 0 -> () | e -> failwith (strf "setrlimit: errno %d" e)

(* Leaves [n] descriptors to open: lowers the limit of open files to 16 above
   the highest open one, and takes all but [n] of those left. Answers the
   function that gives them back and restores the limit. *)
let leave n =
  let before = Support.open_files () in
  let highest =
    Array.fold_left
      (fun m fd -> Option.fold ~none:m ~some:(max m) (int_of_string_opt fd))
      0 (Sys.readdir "/dev/fd")
  in
  checked (Support.set_open_files (highest + 1 + 16));
  let rec take acc =
    match Unix.openfile "/dev/null" [ O_RDONLY; O_CLOEXEC ] 0 with
    | fd -> take (fd :: acc)
    | exception Unix.Unix_error ((EMFILE | ENFILE), _, _) -> acc
  in
  let taken = take [] in
  List.iteri (fun i fd -> if i < n then Unix.close fd) taken;
  fun () ->
    List.iteri (fun i fd -> if i >= n then Unix.close fd) taken;
    checked (Support.set_open_files before)

(* Puts [lim] in place, and answers the function that lifts it. *)
let impose = function
  | Bytes n ->
      checked (Support.set_file_size n);
      fun () -> checked (Support.set_file_size max_int)
  | Descriptors n -> leave n

(* [under lim f] is [f ()] under [lim]. *)
let under lim f =
  let lift = impose lim in
  Fun.protect ~finally:lift f

(* Operations *)

type op = Create | Open | Copy

let op_name = function Create -> "create" | Open -> "open" | Copy -> "copy"
let of_op = function "create" -> Create | "open" -> Open | _ -> Copy

let opened path = function
  | Ok _ -> "opened"
  | Error why when String.starts_with ~prefix:path why ->
      "refused naming the path"
  | Error why -> "refused: " ^ why

(* [answer op path lim] is what [op] on the file at [path] answered under [lim].
   An open and a copy reach a file made before the limit. *)
let answer op path lim =
  match op with
  | Create -> opened path (under lim (fun () -> Rig_disk.create_file path size))
  | Open ->
      Out_channel.with_open_bin path (fun oc -> output_string oc pattern);
      opened path (under lim (fun () -> Rig_disk.of_file path))
  | Copy -> (
      let file = Result.get_ok (Rig_disk.create_file path size) in
      let src = host_of_string pattern in
      match under lim (fun () -> B.copy ~src ~dst:file) with
      | () -> "copied"
      | exception Sys_error m when String.starts_with ~prefix:path m ->
          "Sys_error naming the file")

(* What rig_disk.mli says [op] answers under [lim], and whether it leaves its
   path. *)
let expected op lim =
  let failed =
    match (op, lim) with
    | Open, Bytes _ | Copy, Descriptors _ -> false
    | _, Bytes n -> n < size
    | _, Descriptors n -> n = 0
  in
  match op with
  | Create when failed -> [ "refused naming the path"; "removed" ]
  | Open when failed -> [ "refused naming the path"; "left" ]
  | Create | Open -> [ "opened"; "left" ]
  | Copy when failed -> [ "Sys_error naming the file"; "left" ]
  | Copy -> [ "copied"; "left" ]

(* Walking *)

(* Collects what the operation dropped, drains the disk, which closes the
   descriptors of collected files, and returns the host's cache. *)
let census () =
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create Rig_disk.device 0);
  Rig.free_cache Rig.host;
  (Support.heap_bytes (), Support.descriptors ())

(* Runs [op] under [lim]: what it answered, whether it left its path, and
   whether the descriptors came back; then runs it again with no limit, which
   must write the file. Answers those and the C heap's growth. *)
let attempt op lim =
  let path = new_path () in
  let heap, fds = census () in
  let got = answer op path lim in
  let left = if Sys.file_exists path then "left" else "removed" in
  let heap', fds' = census () in
  if Sys.file_exists path then Sys.remove path;
  let again =
    match Rig_disk.create_file path size with
    | Error why -> "again: " ^ why
    | Ok file ->
        B.copy ~src:(host_of_string pattern) ~dst:file;
        if read file = pattern then "again: the file's bytes"
        else "again: other bytes"
  in
  let fds =
    match (fds, fds') with
    | Some a, Some b when a <> b -> strf "descriptors: %+d" (b - a)
    | _ -> "descriptors: as before"
  in
  let growth = match (heap, heap') with Some h, Some h' -> h' - h | _ -> 0 in
  ([ got; left; fds; again ], growth)

(* The child: runs [op] under [lim] once uncounted, then [reps] times, and
   prints what the runs answered, the disk's loss, and the C heap's growth if it
   grew in every run. *)
let child op lim =
  clear ();
  ignore (attempt op lim);
  let runs = List.init reps (fun _ -> attempt op lim) in
  List.iter print_endline (List.sort_uniq compare (List.concat_map fst runs));
  let lost = Option.value ~default:"not lost" (Rig.lost Rig_disk.device) in
  print_endline ("the disk: " ^ lost);
  let least = List.fold_left (fun m (_, g) -> Int.min m g) max_int runs in
  if least >= heap_slack then
    Printf.printf "C heap growth: %s\n"
      (String.concat ", " (List.map (fun (_, g) -> string_of_int g) runs))

(* The suite *)

let walked op lim () =
  if Sys.win32 then skip ~reason:"no limits of a process" ();
  let args = Array.of_list (exe :: "walk" :: op_name op :: limit_args lim) in
  let ic = Unix.open_process_args_in exe args in
  let out = In_channel.input_all ic in
  (match Unix.close_process_in ic with
  | WEXITED 0 -> ()
  | _ -> failf "the child failed: %s" out);
  let lines =
    expected op lim @ [ "descriptors: as before"; "again: the file's bytes" ]
  in
  equal (list string)
    (List.sort_uniq compare lines @ [ "the disk: not lost" ])
    (String.split_on_char '\n' (String.trim out))

let walk op =
  group ~timeout (op_name op)
    (List.map
       (fun lim -> test (String.concat " " (limit_args lim)) (walked op lim))
       limits)

let () =
  match Array.to_list Sys.argv with
  | [ _; "walk"; op; kind; n ] ->
      child (of_op op) (of_args kind n);
      clear ();
      Sys.rmdir dir
  | _ ->
      exit
        (run "rig_disk.walk"
           [
             group "an operation under a limit answers as rig_disk.mli states"
               [ walk Create; walk Open; walk Copy ];
           ])
