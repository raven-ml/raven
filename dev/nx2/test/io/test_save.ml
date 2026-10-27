(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Saving and errors. Every save takes a value of any set, replaces what its
   path names with a new file and leaves the value's memory to its next
   donation; a file that cannot be read or written raises [Failure] starting
   with its path, and a value a save cannot read raises [Invalid_argument]
   naming the save. *)

open Windtrap
open Io_support

(* The six saves, each of a uint8 value of [shape]: text holds a matrix, every
   other format the image [shape] makes. *)
type save = {
  name : string;
  shape : int array;
  save : 'd. string -> 'd Nx.uint8_t -> unit;
}

let image = [| 2; 3; 3 |]

let saves =
  let entry x = Archive.of_list [ ("x", Nx.P x) ] in
  [
    { name = "npy"; shape = image; save = (fun p x -> Nx_io.save_npy p x) };
    {
      name = "npz";
      shape = image;
      save = (fun p x -> Nx_io.save_npz p (entry x));
    };
    {
      name = "safetensors";
      shape = image;
      save = (fun p x -> Nx_io.save_safetensors p (entry x));
    };
    {
      name = "text";
      shape = [| 2; 9 |];
      save = (fun p x -> Nx_io.save_txt p x);
    };
    { name = "png"; shape = image; save = (fun p x -> Nx_io.save_png p x) };
    { name = "jpeg"; shape = image; save = (fun p x -> Nx_io.save_jpeg p x) };
  ]

let name s = s.name
let numel s = Array.fold_left ( * ) 1 s

(* The host value a save writes: its elements are their C-order positions. *)
let host s = Nx.create Nx.uint8 s.shape (Array.init (numel s.shape) Fun.id)

(* The bytes [s] writes for [x]. *)
let written s x = read (saved "" s.save x)

(* The loaders, each discarding what it loads. *)
let loaders =
  [
    ("npy", fun p -> ignore (Nx_io.load_npy p));
    ("npz", fun p -> ignore (Nx_io.load_npz p));
    ("safetensors", fun p -> ignore (Nx_io.load_safetensors p));
    ("gguf", fun p -> ignore (Nx_io.load_gguf p));
    ("text", fun p -> ignore (Nx_io.load_txt Nx.float32 p));
    ("image", fun p -> ignore (Nx_io.load_image p));
  ]

(* Saving *)

let saving =
  group "saving"
    [
      cases ~name "a save leaves the file alone in its directory" saves
        (fun s ->
          let dir = temp_dir () in
          let path = Filename.concat dir "x" in
          s.save path (host s);
          s.save path (host s);
          equal (array string) [| "x" |] (Sys.readdir dir));
      cases ~name "a save replaces the file its path names" saves (fun s ->
          let path = file "old" in
          s.save path (host s);
          equal string (written s (host s)) (read path));
      cases ~name "a value of another set or of every set saves" saves (fun s ->
          let module M = (val Nx.devices [ Nx_support.memory 0 ]) in
          let expected = written s (host s) in
          equal ~msg:"another set" string expected
            (written s (Nx.place M.on (host s)));
          let every =
            Nx.reshape s.shape (Nx.arange Nx.uint8 0 (numel s.shape) 1)
          in
          equal ~msg:"every set" string expected (written s every));
      (let device = Nx_support.memory 3 in
       let module Mem = (val Nx.devices [ device ]) in
       (* Values in memory of their own: one made from OCaml data, and one
          computed on a memory device, since a placement onto it from the host
          shares the host's memory. *)
       let n = 1 lsl 16 in
       let on_host () = Nx.create Nx.float32 [| n |] (Array.make n 1.) in
       let on_mem () =
         let x = Nx.place Mem.on (on_host ()) in
         Nx.add x (Nx.zeros_like x)
       in
       let donated d make save =
         let saved_then_donated x =
           save x;
           Nx.neg (Nx.donate x)
         in
         let y, grew = grown d make saved_then_donated in
         less ~msg:"no memory of the result's size" int ~than:(4 * n) grew;
         equal (array float_exact) [| -1. |]
           (Nx.to_array (Nx.slice [ Nx.R (0, 1) ] y))
       in
       let npy x = Nx_io.save_npy (temp_file ()) x in
       let npz x =
         Nx_io.save_npz (temp_file ()) (Archive.of_list [ ("x", Nx.P x) ])
       in
       let safetensors x =
         Nx_io.save_safetensors (temp_file ())
           (Archive.of_list [ ("x", Nx.P x) ])
       in
       cases ~name:fst
         "a save leaves its value's memory to the value's next donation"
         [
           ("npy on the host", fun () -> donated Rig.host on_host npy);
           ("npz on the host", fun () -> donated Rig.host on_host npz);
           ( "safetensors on the host",
             fun () -> donated Rig.host on_host safetensors );
           ("npy on a memory device", fun () -> donated device on_mem npy);
         ]
         (fun (_, check) -> check ()));
    ]

(* Errors *)

(* How a call in a child process ended. *)
exception Raised_in_child of string

(* [in_child ~seconds f] runs [f] in a forked child, then raises as it did:
   [Failure] with its message, any other exception as [Raised_in_child] with its
   text. Fails if the child still runs after [seconds]: a load blocked inside a
   system call would hang the suite, since no timeout interrupts one. *)
let in_child ~seconds f =
  let r, w = Unix.pipe ~cloexec:true () in
  match Unix.fork () with
  | 0 ->
      let outcome =
        match f () with
        | () -> "R"
        | exception Failure m -> "F" ^ m
        | exception e -> "E" ^ Printexc.to_string e
      in
      ignore (Unix.write_substring w outcome 0 (String.length outcome));
      Unix._exit 0
  | child -> (
      Unix.close w;
      let ic = Unix.in_channel_of_descr r in
      let deadline = Unix.gettimeofday () +. seconds in
      let rec wait () =
        match Unix.waitpid [ Unix.WNOHANG ] child with
        | 0, _ when Unix.gettimeofday () < deadline ->
            Unix.sleepf 0.01;
            wait ()
        | 0, _ ->
            Unix.kill child Sys.sigkill;
            ignore (Unix.waitpid [] child);
            close_in ic;
            failf "the call still ran after %g s" seconds
        | _ -> ()
      in
      wait ();
      let outcome = In_channel.input_all ic in
      close_in ic;
      let text = String.sub outcome 1 (String.length outcome - 1) in
      match outcome.[0] with
      | 'R' -> ()
      | 'F' -> failwith text
      | _ -> raise (Raised_in_child text))

let errors =
  group "errors"
    [
      cases ~name:fst "a file that cannot be opened fails, naming it" loaders
        (fun (_, load) ->
          let path = Filename.concat (temp_dir ()) "absent" in
          fails ~path (fun () -> load path));
      cases ~name:fst
        "a path that names no regular file, a directory or a FIFO, fails, \
         naming it"
        loaders (fun (_, load) ->
          let dir = temp_dir () in
          fails ~path:dir (fun () -> load dir);
          if Sys.win32 then skip ~reason:"no FIFO on Windows" ();
          (* A FIFO with no writer: a reader that opens it waits for one, so the
             load runs where the test can stop it. *)
          let fifo = Filename.concat (temp_dir ()) "fifo" in
          Unix.mkfifo fifo 0o600;
          fails ~path:fifo (fun () ->
              in_child ~seconds:10. (fun () -> load fifo)));
      cases ~name "a save into a missing directory fails, naming the path" saves
        (fun s ->
          let path = Filename.concat (temp_dir ()) "missing/x" in
          fails ~path (fun () -> s.save path (host s)));
      test "a save of a dead value names the save" (fun () ->
          let x = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
          ignore (Nx.neg (Nx.donate x));
          invalid ~by:"Nx_io.save_npy" (fun () ->
              Nx_io.save_npy (temp_file ()) x));
    ]

let () = exit (run "nx.io saving" [ saving; errors ])
