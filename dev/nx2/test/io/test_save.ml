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
  by : string;  (** The function, as its refusals name it. *)
  shape : int array;
  save : 'd. string -> 'd Nx.uint8_t -> unit;
}

let image = [| 2; 3; 3 |]

let saves =
  let entry x = Archive.of_list [ ("x", Nx.P x) ] in
  [
    {
      name = "npy";
      by = "Nx_io.save_npy";
      shape = image;
      save = (fun p x -> Nx_io.save_npy p x);
    };
    {
      name = "npz";
      by = "Nx_io.save_npz";
      shape = image;
      save = (fun p x -> Nx_io.save_npz p (entry x));
    };
    {
      name = "safetensors";
      by = "Nx_io.save_safetensors";
      shape = image;
      save = (fun p x -> Nx_io.save_safetensors p (entry x));
    };
    {
      name = "text";
      by = "Nx_io.save_txt";
      shape = [| 2; 9 |];
      save = (fun p x -> Nx_io.save_txt p x);
    };
    {
      name = "png";
      by = "Nx_io.save_png";
      shape = image;
      save = (fun p x -> Nx_io.save_png p x);
    };
    {
      name = "jpeg";
      by = "Nx_io.save_jpeg";
      shape = image;
      save = (fun p x -> Nx_io.save_jpeg p x);
    };
  ]

let name s = s.name
let numel s = Array.fold_left ( * ) 1 s

(* The host value a save writes: its elements are their C-order positions. *)
let host s = Nx.create Nx.uint8 s.shape (Array.init (numel s.shape) Fun.id)

(* The bytes [s] writes for [x]. *)
let written s x = read (saved "" s.save x)

(* [dead s] is a value of [s]'s shape whose donation has ended it. *)
let dead s =
  let x = host s in
  ignore (Nx.neg (Nx.donate x));
  x

(* A permission's bits, printed in octal. *)
let permissions =
  Testable.make ~equal:Int.equal ~pp:(fun ppf p -> Format.fprintf ppf "0o%o" p)

(* Traced values *)

type ('v, 's, 'd) Nx.Prim.payload += Mark : ('v, 's, 'd) Nx.Prim.payload

(* [traced f] is [f trace] inside an interpretation, [trace x] a value of [x]'s
   form that the interpretation traces. *)
let traced f =
  let mark i ~by op =
    Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Mark) op
  in
  Nx.Prim.interpret ~name:"test.io" Values mark (fun i ->
      f (fun x -> Nx.Prim.traced i (Nx.Prim.form x) Mark))

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
      cases ~name "a value of another set or a host creation saves" saves
        (fun s ->
          let module M = (val Nx.devices [ Nx_support.memory 0 ]) in
          let expected = written s (host s) in
          equal ~msg:"another set" string expected
            (written s (Nx.place M.on (host s)));
          let made =
            Nx.reshape s.shape (Nx.arange Nx.uint8 0 (numel s.shape) 1)
          in
          equal ~msg:"a host creation" string expected (written s made));
      (let entries live dead =
         Archive.of_list [ ("a", Nx.P live); ("b", Nx.P dead) ]
       in
       let second save s path = save path (entries (host s) (dead s)) in
       let npz = List.nth saves 1 and safetensors = List.nth saves 2 in
       cases ~name:fst
         "a failed save leaves the file it was to replace as it was, and \
          nothing beside it"
         (List.map (fun s -> (s.name, (s.by, fun p -> s.save p (dead s)))) saves
         @ [
             ("npz, its second entry dead", (npz.by, second Nx_io.save_npz npz));
             ( "safetensors, its second entry dead",
               (safetensors.by, second Nx_io.save_safetensors safetensors) );
           ])
         (fun (_, (by, save)) ->
           let dir = temp_dir () in
           let path = Filename.concat dir "old" in
           write path "old";
           invalid ~by (fun () -> save path);
           equal ~msg:"the file" string "old" (read path);
           equal ~msg:"the directory" (array string) [| "old" |]
             (Sys.readdir dir)));
      cases ~name
        "a refused rename keeps the new file, and the Failure names it" saves
        (fun s ->
          let dir = temp_dir () in
          let path = Filename.concat dir "x" in
          (* A directory that is not empty, which no rename replaces. *)
          Sys.mkdir path 0o755;
          write (Filename.concat path "occupant") "";
          let message =
            match s.save path (host s) with
            | () -> fail "the rename over a directory was not refused"
            | exception Failure m -> m
          in
          let kept =
            List.filter (( <> ) "x") (Array.to_list (Sys.readdir dir))
          in
          equal ~msg:"files beside the path" int 1 (List.length kept);
          let kept = Filename.concat dir (List.hd kept) in
          starts_with ~affix:(path ^ ": ") message;
          contains ~sub:kept message;
          equal ~msg:"the kept file" string (written s (host s)) (read kept));
      cases ~name
        "a new file has the permissions 0o666 less the umask, whatever the \
         file it replaces had"
        saves (fun s ->
          if Sys.win32 then skip ~reason:"no umask on Windows" ();
          let dir = temp_dir () in
          let fresh = Filename.concat dir "fresh"
          and old = Filename.concat dir "old" in
          write old "old";
          Unix.chmod old 0o644;
          let umask = Unix.umask 0o277 in
          Fun.protect
            ~finally:(fun () -> ignore (Unix.umask umask))
            (fun () ->
              s.save fresh (host s);
              s.save old (host s));
          let perm path = (Unix.stat path).st_perm in
          equal ~msg:"a new path" permissions 0o400 (perm fresh);
          equal ~msg:"a replaced file" permissions 0o400 (perm old);
          equal ~msg:"the file" string (written s (host s)) (read old));
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

(* Loaded values *)

(* A file's contents: [n] int32 elements counting up from [start]. A value of
   2^14 elements spans pages. *)
let contents =
  Gen.with_pp
    (fun ppf (n, start) -> Format.fprintf ppf "(%d, %d)" n start)
    (Gen.pair (Gen.of_list [ 1; 5; 1 lsl 14 ]) (Gen.int_range 0 9))

let value_of (n, start) =
  Nx.init Nx.int32 [| n |] (fun i -> Int32.of_int (start + i.(0)))

(* How a file ended: a save put a new file at its path, or a remove deleted
   it. *)
type ending = Saved_over | Removed

(* A directory holding one npy file, at most, modelled as the file's contents
   and number, files numbered in the order they were saved, and how each file
   that is gone ended. *)
type dir_model = {
  mutable file : ((int * int) * int) option;
  ended : (int, ending) Hashtbl.t;
  mutable last : int;
}

(* A value loaded from a file, modelled as the contents it was loaded with and
   the file it was loaded from. *)
type held_model = { loaded : int * int; dir : dir_model; file : int }

let dir = abstract "dir"
let held = abstract "held"
let path d = Filename.concat d "x"
let holds (d : dir_model) = Option.is_some d.file

(* Windows can neither delete nor replace a file whose pages are mapped until
   its values are collected, so there the program never does. *)
let replaceable d = not (Sys.win32 && holds d)

(* [ends d e] records that [d]'s file ended by [e]. *)
let ends (d : dir_model) e =
  Option.iter (fun (_, file) -> Hashtbl.replace d.ended file e) d.file

(* How a held value's file ended, when the value is read or saved. Neither
   ending happens on Windows. *)
let fate (h : held_model) =
  if not Sys.win32 then begin
    let ending = Hashtbl.find_opt h.dir.ended h.file in
    cover "its file saved over" (ending = Some Saved_over);
    cover "its file removed" (ending = Some Removed)
  end

let commands =
  let save d c =
    ends d Saved_over;
    d.last <- d.last + 1;
    d.file <- Some (c, d.last)
  in
  [
    command "dir"
      (Gen.unit @-> makes dir)
      (fun () -> { file = None; ended = Hashtbl.create 8; last = 0 })
      (fun () -> temp_dir ());
    command "save"
      ~pre:(fun d _ -> replaceable d)
      (dir ^-> contents @-> returns unit)
      save
      (fun d c -> Nx_io.save_npy (path d) (value_of c));
    command "load" ~pre:holds
      (dir ^-> makes held)
      (fun d ->
        let loaded, file = Option.get d.file in
        { loaded; dir = d; file })
      (fun d ->
        let x = Nx_io.load_npy (path d) in
        equal ~msg:"the loaded value" memory Over_file (memory_of x);
        Nx.unpack Nx.int32 x);
    command "save_held"
      ~pre:(fun _ d -> replaceable d)
      (held ^-> dir ^-> returns unit)
      (fun h d ->
        fate h;
        if not Sys.win32 then
          cover "to its own file"
            (match d.file with
            | Some (_, file) -> h.dir == d && file = h.file
            | None -> false);
        save d h.loaded)
      (fun x d -> Nx_io.save_npy (path d) x);
    command "read"
      (held ^-> returns packed)
      (fun h ->
        fate h;
        Nx.P (value_of h.loaded))
      (fun x -> Nx.P x);
    command "collect" (Gen.unit @-> returns unit) Gc.full_major Gc.full_major;
  ]
  @
  if Sys.win32 then []
  else
    [
      command "remove" ~pre:holds
        (dir ^-> returns unit)
        (fun d ->
          ends d Removed;
          d.file <- None)
        (fun d -> Sys.remove (path d));
    ]

let loaded =
  group "loaded values"
    [
      (* Programs of 60 calls read a value whose file was removed in about one
         case in five, so every run reaches each label. *)
      stateful ~steps:60
        "a value loaded over its file's pages keeps its value after a save to \
         its file's path or a delete of the file, and saves again"
        commands;
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
      cases ~name:fst
        "a traced value, which has no bytes, is refused, naming the function"
        (List.map (fun s -> (s.name, (s.by, s.shape, s.save))) saves
        @ [
            ( "encode_png",
              ("Nx_io.encode_png", image, fun _ x -> ignore (Nx_io.encode_png x))
            );
          ])
        (fun (_, (by, shape, save)) ->
          let x = Nx.zeros Nx.uint8 shape in
          traced @@ fun trace ->
          invalid ~by (fun () -> save (temp_file ()) (trace x)));
    ]

let () = exit (run "nx.io saving" [ saving; loaded; errors ])
