(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Files on the disk, written under this test's directory in _build and removed
   as the tests end. The suite runs itself as a child process where a test needs
   a process of its own: one with a low limit of open files, or one that counts
   the descriptors it inherited. *)

open Windtrap
module B = Rig.Buffer

let strf = Printf.sprintf
let timeout = 60.
let disk = Rig_disk.device
let host = Rig.host

(* The disk keeps at most this many descriptors open while no copy uses them. *)
let max_descriptors = 64

(* The bytes of one of the disk's io_uring requests on Linux, whose edges large
   copies cross. *)
let segment = 2 lsl 20

external set_open_files : int -> int = "rig_disk_test_set_open_files"
external drop_pages : string -> int = "rig_disk_test_drop_pages"

let contains s sub =
  let n = String.length s and k = String.length sub in
  let rec from i = i + k <= n && (String.sub s i k = sub || from (i + 1)) in
  from 0

(* A test that can block in C runs under this: the process ends, naming [what],
   if [f] has not returned after 10 s. *)
let watchdog what f =
  let finished = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        let until = Unix.gettimeofday () +. 10. in
        while (not (Atomic.get finished)) && Unix.gettimeofday () < until do
          Unix.sleepf 0.01
        done;
        if not (Atomic.get finished) then begin
          prerr_endline ("watchdog: " ^ what ^ " did not return in 10 s");
          Unix._exit 2
        end)
  in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set finished true;
      Domain.join d)
    f

(* [once f] is a function that answers [f ()], made by its first call from any
   domain. [Lazy] would raise when two domains force it at once. *)
let once f =
  let made = ref None and lock = Mutex.create () in
  fun () ->
    Mutex.protect lock @@ fun () ->
    match !made with
    | Some v -> v
    | None ->
        let v = f () in
        made := Some v;
        v

(* Bytes *)

(* Bytes print by their length, start and digest past a line. *)
let octets =
  let pp ppf s =
    let n = String.length s in
    if n <= 48 then Format.fprintf ppf "%S" s
    else
      Format.fprintf ppf "%d bytes from %S, md5 %s" n (String.sub s 0 32)
        (Digest.to_hex (Digest.string s))
  in
  Testable.make ~pp ~equal:String.equal

(* [same expected got] fails at the first byte where [got] differs from
   [expected], showing 16 bytes of each from there. *)
let same ?msg expected got =
  let n = Int.min (String.length expected) (String.length got) in
  let rec first i =
    if i < n && expected.[i] = got.[i] then first (i + 1) else i
  in
  let i = first 0 in
  if i < n || String.length expected <> String.length got then
    let from s = String.sub s i (Int.min 16 (String.length s - i)) in
    failf "%sbyte %d of %d bytes (%d got) differs: expected %S, got %S"
      (Option.fold ~none:"" ~some:(fun m -> m ^ ": ") msg)
      i (String.length expected) (String.length got) (from expected) (from got)

(* [pattern seed n] is [n] capital letters that differ from one seed to the next
   and repeat at no power of two. Writes are lower case letters, and a file that
   replaced another holds '#'. *)
let pattern seed n =
  String.init n (fun i -> Char.chr (65 + ((i + (i / 251) + (seed * 7)) mod 26)))

let zeros n = String.make n '\000'

(* [host_of_string ~first s] is a host buffer holding [s], [first] bytes into
   its memory. *)
let host_of_string ?(first = 0) s =
  let n = String.length s in
  let b = B.view (B.create host (first + n)) ~first ~length:n in
  let a = B.bigarray Bigarray.char b in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set a i c) s;
  b

let string_of_host b =
  let a = B.bigarray Bigarray.char b in
  String.init (Bigarray.Array1.dim a) (Bigarray.Array1.unsafe_get a)

(* [read b] is [b]'s bytes, copied into host memory. *)
let read b =
  let h = B.create host (B.length b) in
  B.copy ~src:b ~dst:h;
  string_of_host h

let write b s = B.copy ~src:(host_of_string s) ~dst:b

(* A device whose memory is the host's, which maps any host memory. *)
let memory_device = lazy (require_ok (Rig.memory_device "DISK-TEST"))

(* [borrowed d b] is the bytes of [d]'s borrow of [b], if [d] maps it. *)
let borrowed d b =
  Option.map
    (fun p -> if Rig.equal d host then string_of_host p else read p)
    (B.borrow d b)

let poke b i c = Bigarray.Array1.set (B.bigarray Bigarray.char b) i c

(* Files *)

let dir = "files"
let names = Atomic.make 0

let clear dir =
  if Sys.file_exists dir then
    Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir)
  else Sys.mkdir dir 0o755

let new_path () =
  Filename.concat dir (strf "f%d" (Atomic.fetch_and_add names 1))

let contents path = In_channel.with_open_bin path In_channel.input_all
let remove path = try Sys.remove path with Sys_error _ -> ()
let removing paths f = Fun.protect ~finally:(fun () -> List.iter remove paths) f

(* The inodes of the files this suite made: on macOS, all /dev/fd tells of a
   descriptor's file ([open_on_made]). *)
let made = Hashtbl.create 1024
let made_lock = Mutex.create ()
let inode path = (Unix.stat path).st_ino

let register path =
  let ino = inode path in
  Mutex.protect made_lock (fun () -> Hashtbl.replace made ino ())

let make_file ?(path = new_path ()) s =
  Out_channel.with_open_bin path (fun oc -> output_string oc s);
  register path;
  path

let create path n =
  let b = require_ok ~pp:Format.pp_print_string (Rig_disk.create_file path n) in
  register path;
  b

let of_file path = require_ok ~pp:Format.pp_print_string (Rig_disk.of_file path)

(* [open_where ours] is the number of this process's descriptors whose /dev/fd
   entry [ours] holds. *)
let open_where ours =
  Array.fold_left
    (fun n fd -> if ours (Filename.concat "/dev/fd" fd) then n + 1 else n)
    0 (Sys.readdir "/dev/fd")

let on_inode ino entry =
  match Unix.stat entry with
  | { st_kind = S_REG; st_ino; _ } -> st_ino = ino
  | _ -> false
  | exception Unix.Unix_error _ -> false

(* [open_on_made ()] is the number of this process's descriptors open on files
   this suite made. Linux's /dev/fd entries are links to the files; macOS's
   answer only the file's inode, which APFS never gives another file, while ext4
   does at once. *)
let open_on_made () =
  let files = Filename.concat (Sys.getcwd ()) dir ^ Filename.dir_sep in
  open_where (fun entry ->
      match Unix.readlink entry with
      | target -> String.starts_with ~prefix:files target
      | exception Unix.Unix_error (EINVAL, _, _) -> (
          match Unix.stat entry with
          | { st_kind = S_REG; st_ino; _ } ->
              Mutex.protect made_lock (fun () -> Hashtbl.mem made st_ino)
          | _ -> false
          | exception Unix.Unix_error _ -> false)
      | exception Unix.Unix_error _ -> false)

(* [after_tick path] returns once a file written now gets a later change time
   than [path]'s: the clock of file times, coarse on Linux, has moved on. *)
let after_tick path =
  let changed p = (Unix.stat p).st_ctime in
  let t = changed path and probe = new_path () in
  let rec wait k =
    Out_channel.with_open_bin probe ignore;
    if changed probe <= t then
      if k = 0 then failf "file times did not move past %s's" path
      else begin
        Unix.sleepf 0.001;
        wait (k - 1)
      end
  in
  Fun.protect ~finally:(fun () -> remove probe) (fun () -> wait 10_000)

let needs_dev_fd () =
  if Sys.win32 then skip ~reason:"no /dev/fd to list descriptors" ()

(* Other files, read after a file to make the disk use more descriptors than it
   keeps. They stay reachable, so none closes by collection. *)
let others_count = max_descriptors + 6

let others =
  once @@ fun () ->
  Array.init others_count (fun i ->
      let s = pattern i 1 in
      (of_file (make_file s), s))

(* [use_others k] reads the first [k] other files: the ones whose bytes
   differ. *)
let use_others k =
  let o = others () in
  List.filter
    (fun i ->
      let b, s = o.(i) in
      read b <> s)
    (List.init k Fun.id)

let use_all_others () =
  equal ~msg:"other files that read wrong" (list int) []
    (use_others others_count)

(* [replace path how] puts another file of the same size at [path], all '#':
   renamed over it, or made after its file was removed. *)
type how = Rename | Recreate

(* The ways [replace] replaces a file. *)
let hows = [ Rename; Recreate ]

let pp_how ppf h =
  Format.pp_print_string ppf
    (match h with Rename -> "rename" | Recreate -> "recreate")

let replace path how =
  let n = (Unix.stat path).st_size in
  match how with
  | Rename -> Sys.rename (make_file (String.make n '#')) path
  | Recreate ->
      Sys.remove path;
      ignore (make_file ~path (String.make n '#'))

let names_file = "Sys_error naming the file"

(* [naming path m] is what a [Sys_error m] from [path]'s buffer says. *)
let naming path m = if contains m path then names_file else "Sys_error: " ^ m

(* [child args] is the trimmed output of this suite run as [args]. *)
let child args =
  let exe = Sys.executable_name in
  let ic = Unix.open_process_args_in exe (Array.of_list (exe :: args)) in
  let out = In_channel.input_all ic in
  match Unix.close_process_in ic with
  | WEXITED 0 -> String.trim out
  | _ -> failf "the child %s failed: %s" (String.concat " " args) out

(* Generators *)

let page_edges = [ 0; 1; 2; 4095; 4096; 4097; 16383; 16384; 16385 ]
let segment_edges = [ segment - 1; segment; segment + 1 ]
let ints = Gen.of_list ~pp:Format.pp_print_int

(* A size or an offset, at the edges of pages and of the disk's requests. *)
let extent =
  Gen.frequency
    [
      (4, ints (page_edges @ [ 65535; 65536; 65537 ]));
      (1, ints segment_edges);
      (3, Gen.int_range 0 70_000);
    ]

(* A length: an extent, or no bytes or one byte, drawn often enough (about one
   case in six each) that the 100 cases of a property reach both whatever its
   seed. An extent alone draws each one case in 24, which 100 cases miss once in
   about 70 seeds. *)
let lengths = Gen.frequency [ (1, ints [ 0; 1 ]); (3, extent) ]

(* [clamp size (a, l)] is the range of [l] bytes from byte [a] that fits a file
   of [size] bytes, [max_int] reaching its end. *)
let clamp size (a, l) =
  let at = min a size in
  (at, min l (size - at))

let small_extent =
  Gen.frequency [ (2, ints (page_edges @ [ max_int ])); (1, Gen.nat) ]

(* The disk *)

let device_facts () =
  equal
    (quad string string int (pair bool bool))
    ("DISK", "", max_int, (false, false))
    ( Rig.name disk,
      Rig.arch disk,
      Rig.budget disk,
      (Rig.computes disk, Rig.reaches disk host) )

let the_disk =
  group ~timeout "the disk"
    [
      test
        "DISK has no arch and a budget of max_int, and neither computes nor \
         reaches the host's memory"
        device_facts;
      test "Buffer.create on DISK is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"") (fun () ->
              B.create disk 16));
      test "a file's buffer is DISK's" (fun () ->
          let path = make_file "abc" in
          removing [ path ] @@ fun () ->
          equal bool true (Rig.equal disk (B.device (of_file path))));
    ]

(* Copies *)

type source = From_host of int | From_device | From_file

let pp_source ppf = function
  | From_host first -> Format.fprintf ppf "host+%d" first
  | From_device -> Format.pp_print_string ppf "device"
  | From_file -> Format.pp_print_string ppf "file"

let sources =
  Gen.of_list ~pp:pp_source
    [ From_host 0; From_host 1; From_host 7; From_device; From_file ]

(* A copy into the [len] bytes at [at] of a new file of [at + len + tail]. *)
let copy_in =
  Gen.(
    quad nat (pair extent lengths) (ints [ 0; 1; 3; 4096 ]) sources
    |> with_pp (fun ppf (seed, (at, len), tail, src) ->
        Format.fprintf ppf "seed %d, %d bytes at %d, %d after, from %a" seed len
          at tail pp_source src))

let test_copy_in (seed, (at, len), tail, src) =
  let bytes = pattern seed len in
  let path = new_path () and source = new_path () in
  removing [ path; source ] @@ fun () ->
  let file = create path (at + len + tail) in
  let src =
    match src with
    | From_host first -> host_of_string ~first bytes
    | From_device ->
        let d = B.create (Lazy.force memory_device) len in
        write d bytes;
        d
    | From_file -> of_file (make_file ~path:source bytes)
  in
  let window = B.view file ~first:at ~length:len in
  B.copy ~src ~dst:window;
  cover "no bytes" (len = 0);
  cover "one byte" (len = 1);
  cover "across a page" ((at mod 4096) + len > 4096);
  cover "a request's bytes or more" (len >= segment);
  cover "to the file's end" (tail = 0 && len > 0);
  same ~msg:"the file" (zeros at ^ bytes ^ zeros tail) (contents path);
  same ~msg:"read back" bytes (read window);
  same ~msg:"opened again" bytes
    (read (B.view (of_file path) ~first:at ~length:len))

(* A file read by views of views, at any byte, from its end too. *)
let viewed_size = (2 * segment) + 16384 + 5

let viewed =
  lazy
    (let s = pattern 3 viewed_size in
     let path = make_file s in
     (path, s, of_file path))

(* [placed (from_end, (a, l))] is the range of [l] bytes from byte [a] of the
   file read by views, [a] counted from its end with [from_end]. *)
let placed (from_end, (a, l)) =
  let n = viewed_size in
  clamp n ((if from_end then n - min a n else a), l)

type into = To_host of int | To_device

let pp_into ppf = function
  | To_host first -> Format.fprintf ppf "host+%d" first
  | To_device -> Format.pp_print_string ppf "device"

(* The range copied, its first byte counted from the file's end or not, and the
   bytes before and after it in the view it is a view of. *)
let views =
  Gen.(
    quad bool (pair extent lengths)
      (pair (ints [ 0; 1; 4096 ]) (ints [ 0; 1; 4096 ]))
      (Gen.of_list ~pp:pp_into [ To_host 0; To_host 3; To_device ]))

let test_copy_out (from_end, range, (before, after), into) =
  let _, s, file = Lazy.force viewed in
  let n = viewed_size in
  let first, length = placed (from_end, range) in
  let outer = Int.max 0 (first - before) in
  let outer_end = Int.min n (first + length + after) in
  let v =
    B.view
      (B.view file ~first:outer ~length:(outer_end - outer))
      ~first:(first - outer) ~length
  in
  cover "no bytes" (length = 0);
  cover "the last byte" (first + length = n && length > 0);
  cover "across a request" (first / segment <> (first + length - 1) / segment);
  let got =
    match into with
    | To_host first ->
        let h = B.view (B.create host (first + length)) ~first ~length in
        B.copy ~src:v ~dst:h;
        string_of_host h
    | To_device ->
        let d = B.create (Lazy.force memory_device) length in
        B.copy ~src:v ~dst:d;
        read d
  in
  same (String.sub s first length) got

let test_large_copy () =
  let n = (40 lsl 20) + 1 and at = 4097 in
  let bytes = pattern 5 n in
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file = create path (at + n + 3) in
  let window = B.view file ~first:at ~length:n in
  write window bytes;
  same ~msg:"the file" (zeros at ^ bytes ^ zeros 3) (contents path);
  same ~msg:"read back" bytes (read window)

let test_cold_read () =
  let s = pattern 6 ((3 * segment) + 5) in
  let path = make_file s in
  removing [ path ] @@ fun () ->
  match drop_pages path with
  | -1 -> skip ~reason:"only Linux drops a file's pages without privileges" ()
  | 0 ->
      let b = of_file path in
      watchdog "a copy of a file whose pages were dropped" (fun () ->
          same s (read b))
  | code -> failf "drop_pages: errno %d" code

let test_copy_into_opened () =
  let path = make_file "x" in
  removing [ path ] @@ fun () ->
  let dst = of_file path in
  raises_match (Exn.invalid_arg ~substring:"Rig.Buffer.copy: ") (fun () ->
      B.copy ~src:(host_of_string "y") ~dst);
  equal octets ~msg:"the file" "x" (contents path)

(* 20,000 copies of 64 bytes into a file, each from a fresh host buffer that
   nothing else holds: a collection during a copy, such as one an allocation of
   the disk's own runs, must not free its source. *)
let test_source_held () =
  let k = 64 and count = 20_000 in
  let s = pattern 7 (k * count) in
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file = create path (k * count) in
  for i = 0 to count - 1 do
    B.copy
      ~src:(host_of_string (String.sub s (i * k) k))
      ~dst:(B.view file ~first:(i * k) ~length:k)
  done;
  same s (contents path)

let copies =
  group ~timeout "copies"
    [
      prop
        "a copy into a file and back moves every byte, at any offset and \
         length, from the host, a device or another file"
        copy_in test_copy_in;
      prop
        "a copy from a view of a view of a file reads what the system's reads \
         read, into the host or a device"
        views test_copy_out;
      test "a copy of 40 MiB at an odd offset moves every byte, both ways"
        test_large_copy;
      test "a copy reads a file whose pages the system dropped" test_cold_read;
      test
        "a copy into a file writes its source's bytes, which nothing else holds"
        test_source_held;
      test "a copy into a file opened for reading is refused"
        test_copy_into_opened;
    ]

(* Borrows *)

type kind = Opened | Created

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with Opened -> "opened" | Created -> "created")

let kinds = Gen.of_list ~pp:pp_kind [ Opened; Created ]

(* The bytes of the file read by views, in a file [create_file] made. *)
let created_viewed =
  lazy
    (let _, s, _ = Lazy.force viewed in
     let path = new_path () in
     let b = create path viewed_size in
     write b s;
     b)

let test_borrow (kind, on_host, range) =
  let _, s, opened = Lazy.force viewed in
  let file =
    match kind with Opened -> opened | Created -> Lazy.force created_viewed
  in
  let first, length = placed range in
  let d = if on_host then host else Lazy.force memory_device in
  cover "no bytes" (length = 0);
  cover "the last byte" (first + length = viewed_size && length > 0);
  equal (option octets)
    (Some (String.sub s first length))
    (borrowed d (B.view file ~first ~length))

let test_copy_on_write () =
  let page = 16384 in
  let s = pattern 1 (3 * page) in
  let path = make_file s in
  removing [ path ] @@ fun () ->
  let file = of_file path in
  let window = B.view file ~first:page ~length:page in
  let on_host = require_some (B.borrow host window) in
  poke on_host 0 'z';
  let written = "z" ^ String.sub s (page + 1) (page - 1) in
  equal octets ~msg:"the host's borrow" written (string_of_host on_host);
  equal (option octets) ~msg:"a device's borrow" (Some written)
    (borrowed (Lazy.force memory_device) window);
  equal octets ~msg:"a copy" (String.sub s page page) (read window);
  equal octets ~msg:"the file" s (contents path)

let test_two_opens () =
  let s = pattern 2 100 in
  let path = make_file s in
  removing [ path ] @@ fun () ->
  let one = of_file path and two = of_file path in
  equal (pair bool bool) ~msg:"overlaps, whole and in part" (false, false)
    ( B.overlaps one two,
      B.overlaps
        (B.view one ~first:10 ~length:20)
        (B.view two ~first:0 ~length:50) );
  poke (require_some (B.borrow host one)) 10 'z';
  equal (option octets) ~msg:"the other's borrow" (Some s) (borrowed host two)

let test_mapping_kept () =
  let s = pattern 4 5000 in
  let path = make_file s in
  removing [ path ] @@ fun () ->
  let file = of_file path in
  poke (require_some (B.borrow host file)) 7 'z';
  Gc.full_major ();
  Gc.full_major ();
  equal (option char) ~msg:"a later borrow" (Some 'z')
    (Option.map (fun s -> s.[7]) (borrowed host file))

let test_prefetch_unopened () =
  let s = pattern 5 (3 * 16384) in
  let path = make_file s in
  let file = of_file path in
  ignore (require_some (B.borrow host file));
  use_all_others ();
  Sys.remove path;
  let first = 16383 and length = 16386 in
  equal (option octets)
    (Some (String.sub s first length))
    (borrowed (Lazy.force memory_device) (B.view file ~first ~length))

let test_empty_borrows () =
  let path = new_path () and empty = make_file "" in
  removing [ path; empty ] @@ fun () ->
  let created = create path 0 and opened = of_file empty in
  equal
    (pair (option octets) (option octets))
    (Some "", Some "")
    (borrowed host created, borrowed host opened)

let borrows =
  group ~timeout "borrows"
    [
      prop
        "a borrow of a file's bytes, by the host or a device that maps host \
         memory, holds what a copy reads"
        Gen.(triple kinds bool (pair bool (pair extent lengths)))
        test_borrow;
      test
        "a write through a borrow of an opened file is the process's: borrows \
         see it, copies and the file do not"
        test_copy_on_write;
      test
        "two opens of one file are two memories: they do not overlap, and a \
         write through one's borrow is not seen through the other's"
        test_two_opens;
      test
        "an opened file's pages stay mapped, with the process's writes, while \
         its buffer is reachable"
        test_mapping_kept;
      test
        "a device borrows a file whose path names nothing any more, and its \
         bytes are the file's"
        test_prefetch_unopened;
      test "a file of no bytes borrows as no bytes" test_empty_borrows;
    ]

(* Opening and creating *)

let test_length_at_open () =
  let path = make_file "0123456789" in
  removing [ path ] @@ fun () ->
  let file = of_file path in
  Out_channel.with_open_gen [ Open_append; Open_binary ] 0o644 path (fun oc ->
      output_string oc "abc");
  equal int 10 (B.length file)

(* [refused r path] checks that [r] is an error that starts with [path]. *)
let refused ?msg r path =
  let why =
    require_error ?msg
      ~pp:(fun ppf _ -> Format.pp_print_string ppf "a buffer")
      r
  in
  if not (String.starts_with ~prefix:path why) then
    failf "%S does not start with the path %S" why path

type unopenable = Missing | Directory | Fifo

let pp_unopenable ppf u =
  Format.pp_print_string ppf
    (match u with
    | Missing -> "a missing file"
    | Directory -> "a directory"
    | Fifo -> "a FIFO")

let test_of_file_refused u =
  let path =
    match u with
    | Missing -> new_path ()
    | Directory -> dir
    | Fifo ->
        if Sys.win32 then skip ~reason:"no FIFO" ();
        let p = new_path () in
        Unix.mkfifo p 0o600;
        p
  in
  removing (if u = Fifo then [ path ] else []) @@ fun () ->
  watchdog "of_file" (fun () -> refused (Rig_disk.of_file path) path)

let test_nul_path () =
  let path = Filename.concat dir "a\000b" in
  List.iter
    (fun (what, r) -> refused ~msg:what r path)
    [
      ("of_file", Rig_disk.of_file path);
      ("create_file", Rig_disk.create_file path 1);
    ]

type taken = File | Link_to_file | Dangling_link | A_directory | No_directory

let pp_taken ppf t =
  Format.pp_print_string ppf
    (match t with
    | File -> "a file"
    | Link_to_file -> "a link to a file"
    | Dangling_link -> "a link to nothing"
    | A_directory -> "a directory"
    | No_directory -> "a missing directory")

(* [test_create_refused t] makes [t] at a path, asks create_file to create
   there, and checks that nothing changed: the file's bytes, the link's target,
   the absence of a dangling link's target. *)
let test_create_refused t =
  let target = new_path () in
  let path =
    match t with
    | File -> make_file ~path:target "old"
    | Link_to_file ->
        ignore (make_file ~path:target "old");
        let p = new_path () in
        Unix.symlink (Filename.basename target) p;
        p
    | Dangling_link ->
        let p = new_path () in
        Unix.symlink (Filename.basename target) p;
        p
    | A_directory -> dir
    | No_directory -> Filename.concat (new_path ()) "f"
  in
  if Sys.win32 && (t = Link_to_file || t = Dangling_link) then
    skip ~reason:"no symbolic links" ();
  removing [ target; path ] @@ fun () ->
  refused (Rig_disk.create_file path 4) path;
  match t with
  | File | Link_to_file -> equal octets "old" (contents target)
  | Dangling_link ->
      equal bool ~msg:"the link's target exists" false (Sys.file_exists target)
  | A_directory -> equal bool ~msg:"a directory" true (Sys.is_directory path)
  | No_directory -> ()

let test_negative_size n =
  let path = new_path () in
  raises_match (Exn.invalid_arg ~substring:"Rig_disk.create_file: ") (fun () ->
      Rig_disk.create_file path n);
  equal bool ~msg:"a file at the path" false (Sys.file_exists path)

let test_umask mask =
  if Sys.win32 then skip ~reason:"no umask" ();
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let old = Unix.umask mask in
  Fun.protect
    ~finally:(fun () -> ignore (Unix.umask old))
    (fun () -> ignore (create path 1));
  equal int (0o666 land lnot mask) ((Unix.stat path).st_perm land 0o777)

let test_create_empty () =
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file = create path 0 in
  equal (pair int string) (0, "") (B.length file, contents path)

let opening =
  group ~timeout "opening and creating"
    [
      test
        "an opened file's length is the file's size when opened, whatever it \
         grows to"
        test_length_at_open;
      cases
        ~name:(fun u -> Format.asprintf "%a" pp_unopenable u)
        "of_file refuses a path that names no regular file, naming it, without \
         waiting"
        [ Missing; Directory; Fifo ]
        test_of_file_refused;
      test "a path holding a NUL byte is refused, naming it" test_nul_path;
      cases
        ~name:(fun t -> Format.asprintf "%a" pp_taken t)
        "create_file refuses a path that names something or cannot hold a \
         file, naming it and changing nothing"
        [ File; Link_to_file; Dangling_link; A_directory; No_directory ]
        test_create_refused;
      cases ~name:string_of_int
        "create_file refuses a negative size, creating nothing" [ -1; min_int ]
        test_negative_size;
      cases ~name:(strf "0o%03o")
        "a created file's permissions are 0o666 less the umask"
        [ 0o022; 0o077; 0o002 ] test_umask;
      test "a created file of no bytes is an empty file at its path"
        test_create_empty;
    ]

(* Descriptors *)

type op = Copy | Barrier

let pp_op ppf o =
  Format.pp_print_string ppf
    (match o with Copy -> "copy" | Barrier -> "barrier")

let replaced_case =
  Gen.(
    quad kinds
      (Gen.of_list ~pp:pp_how hows)
      (frequency
         [
           (2, ints [ 0; 1; 63; 64; 65; others_count ]);
           (1, int_range 0 others_count);
         ])
      (Gen.of_list ~pp:pp_op [ Copy; Barrier ]))

(* A file is used, then [k] other files, then something replaces its path, then
   the file is copied from or ordered. *)
let test_replaced (kind, how, k, op) =
  let s = pattern k 64 in
  let path = new_path () in
  removing [ path ] @@ fun () ->
  ignore (others ());
  let file =
    match kind with
    | Opened -> of_file (make_file ~path s)
    | Created ->
        let b = create path 64 in
        write b s;
        b
  in
  equal octets ~msg:"before" s (read file);
  equal (list int) ~msg:"other files that read wrong" [] (use_others k);
  replace path how;
  cover "after more files than the disk keeps descriptors of"
    (k >= max_descriptors);
  cover "renamed over" (how = Rename);
  cover "recreated" (how = Recreate);
  match op with
  | Copy -> (
      match read file with
      | got -> equal octets ~msg:"its own bytes" s got
      | exception Sys_error m -> equal string names_file (naming path m))
  | Barrier -> (
      match Rig_disk.barrier file with
      | () -> ()
      | exception Sys_error m -> equal string names_file (naming path m))

let test_own_writes () =
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file = create path 8 in
  write file "abcdefgh";
  use_all_others ();
  write (B.view file ~first:2 ~length:3) "xyz";
  use_all_others ();
  equal octets ~msg:"read" "abxyzfgh" (read file);
  equal octets ~msg:"its path" "abxyzfgh" (contents path)

let test_borrow_writes () =
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file = create path 8 in
  let pages = require_some (B.borrow host file) in
  after_tick path;
  poke pages 2 'x';
  use_all_others ();
  poke pages 3 'y';
  Rig_disk.barrier file;
  let expected = "\000\000xy\000\000\000\000" in
  equal octets ~msg:"read" expected (read file);
  equal octets ~msg:"its path" expected (contents path)

let test_mapped_barrier () =
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file = create path 8 in
  write file "abcdefgh";
  ignore (require_some (B.borrow host file));
  Rig_disk.barrier file;
  use_all_others ();
  equal octets "abcdefgh" (read file)

(* 20 times: a file is read, other files close its descriptor, its file is
   removed and a file of the same size made at its path, and the file is read
   again. *)
let test_recreated () =
  let wrong =
    List.init 20 (fun i ->
        let s = pattern i 64 in
        let path = make_file s in
        removing [ path ] @@ fun () ->
        let file = of_file path in
        ignore (read file);
        use_all_others ();
        replace path Recreate;
        match read file with
        | got when got = s -> []
        | got -> [ strf "%d: %S" i got ]
        | exception Sys_error m when contains m path -> [])
  in
  equal (list string) [] (List.concat wrong)

let test_bound () =
  needs_dev_fd ();
  let files =
    List.init 150 (fun i ->
        let s = pattern i (1 + i) in
        (of_file (make_file s), s))
  in
  let wrong () =
    List.concat
      (List.mapi (fun i (b, s) -> if read b = s then [] else [ i ]) files)
  in
  equal (list int) ~msg:"files that read wrong" [] (wrong ());
  at_most int ~msg:"descriptors" ~than:max_descriptors (open_on_made ());
  equal (list int) ~msg:"files that read wrong, again" [] (wrong ());
  at_most int ~msg:"descriptors, again" ~than:max_descriptors (open_on_made ())

let test_inherited () =
  needs_dev_fd ();
  let path = make_file "x" in
  removing [ path ] @@ fun () ->
  let file = of_file path in
  equal octets "x" (read file);
  let ino = inode path in
  at_least int ~msg:"descriptors here" ~than:1 (open_where (on_inode ino));
  equal string ~msg:"descriptors in a child" "0"
    (child [ "open-on"; string_of_int ino ]);
  ignore (Sys.opaque_identity file)

let test_too_many () =
  if Sys.win32 then skip ~reason:"no limit of open files" ();
  equal text "ninth: f8\nfirst: f0" (child [ "open-files" ])

let test_truncated kind =
  let s = pattern 2 10 in
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file =
    match kind with
    | Opened -> of_file (make_file ~path s)
    | Created ->
        let b = create path 10 in
        write b s;
        b
  in
  equal octets ~msg:"before" s (read file);
  Unix.truncate path 4;
  raises_match (Exn.sys_error ~substring:path) (fun () ->
      read (B.view file ~first:2 ~length:5));
  equal (option string) ~msg:"the disk lost" None (Rig.lost disk);
  let other = make_file "other" in
  removing [ other ] @@ fun () ->
  equal octets ~msg:"another file" "other" (read (of_file other))

let descriptors =
  group ~timeout "descriptors"
    [
      prop
        "a file whose path names another file since reads its own bytes or \
         raises Sys_error naming it, as its barrier does"
        replaced_case test_replaced;
      test
        "a created file's own copies keep it its own, through closed \
         descriptors"
        test_own_writes;
      test
        "a created file's writes through its borrow keep it its own, through \
         closed descriptors"
        test_borrow_writes;
      test
        "a created file whose pages are borrowed keeps it its own across a \
         barrier, through closed descriptors"
        test_mapped_barrier;
      test
        "a file recreated at its path after its descriptor closed is never \
         read as the new file"
        test_recreated;
      test
        "the disk keeps at most 64 descriptors open, and each of 150 files \
         reads its own bytes"
        test_bound;
      test "a process the program starts inherits no descriptor of the disk"
        test_inherited;
      test
        "an open refused for too many files closes the disk's descriptors and \
         tries once more"
        test_too_many;
      cases
        ~name:(fun k -> Format.asprintf "%a" pp_kind k)
        "a copy past the end of a file truncated since raises Sys_error naming \
         it, and the disk goes on"
        [ Opened; Created ] test_truncated;
    ]

(* Barriers *)

let test_barrier_dead () =
  let path = new_path () in
  removing [ path ] @@ fun () ->
  let file = create path 4 in
  ignore
    (Rig.Claim.with_ ~read:[ file ] ~donate:[] (fun c ->
         Rig.Claim.consume c ~why:"gone" file));
  raises_match (Exn.invalid_arg ~substring:"") (fun () -> Rig_disk.barrier file)

let barriers =
  group ~timeout "barriers"
    [
      test "barrier returns on a file opened for reading" (fun () ->
          let path = make_file "abc" in
          removing [ path ] @@ fun () -> Rig_disk.barrier (of_file path));
      cases ~name:fst "barrier refuses a buffer of another device"
        [
          ("the host", fun () -> B.create host 4);
          ("a memory device", fun () -> B.create (Lazy.force memory_device) 4);
        ]
        (fun (_, b) ->
          let b = b () in
          raises_match (Exn.invalid_arg ~substring:"Rig_disk.barrier: ")
            (fun () -> Rig_disk.barrier b));
      test "barrier refuses a dead buffer" test_barrier_dead;
    ]

(* A model of files *)

(* A file as the model knows it: the bytes its copies read, the bytes its pages
   show, whether another file replaced its path, and whether more other files
   than the disk keeps descriptors of were read since it was made. A created
   file's pages are its bytes. *)
type model = {
  kind : kind;
  bytes : Bytes.t;
  pages : Bytes.t;
  mutable replaced : bool;
  mutable after_others : bool;
}

type file = { path : string; buf : B.t }

let model kind bytes =
  let pages = match kind with Opened -> Bytes.copy bytes | Created -> bytes in
  { kind; bytes; pages; replaced = false; after_others = false }

let file = abstract "f" ~release:(fun f -> remove f.path)
let sizes = Gen.frequency [ (1, ints page_edges); (1, Gen.int_range 0 5000) ]
let ranges = Gen.pair small_extent small_extent
let letters = Gen.char_range 'a' 'z'

(* [judged m outcome accept] accepts [Ok (Ok v)] as [accept v], and an error
   naming the file once [m]'s path was replaced. *)
let judged m outcome accept =
  match outcome with
  | Ok (Ok v) -> accept v
  | Ok (Error e) when m.replaced && e = names_file -> ()
  | Ok (Error e) -> fail e
  | Error e -> raise e

let sys_result f x =
  match x () with v -> Ok v | exception Sys_error m -> Error (naming f.path m)

let model_commands =
  [
    command "of_file"
      (Gen.pair Gen.nat sizes @-> makes file)
      (fun (seed, n) -> model Opened (Bytes.of_string (pattern seed n)))
      (fun (seed, n) ->
        let path = make_file (pattern seed n) in
        { path; buf = of_file path });
    command "create_file"
      (sizes @-> makes file)
      (fun n -> model Created (Bytes.make n '\000'))
      (fun n ->
        let path = new_path () in
        { path; buf = create path n });
    command "copy into"
      ~pre:(fun m r _ ->
        m.kind = Created || snd (clamp (Bytes.length m.bytes) r) > 0)
      (file ^-> ranges @-> letters @-> judges (result unit string))
      (fun m r c outcome ->
        let at, len = clamp (Bytes.length m.bytes) r in
        match (m.kind, outcome) with
        | Opened, Error (Invalid_argument _) -> ()
        | Opened, _ -> fail "a copy into an opened file was not refused"
        | Created, _ -> judged m outcome (fun () -> Bytes.fill m.bytes at len c))
      (fun f r c ->
        let at, len = clamp (B.length f.buf) r in
        sys_result f (fun () ->
            write (B.view f.buf ~first:at ~length:len) (String.make len c)));
    command "copy from"
      (file ^-> ranges @-> judges (result octets string))
      (fun m r outcome ->
        let at, len = clamp (Bytes.length m.bytes) r in
        cover "a copy from a replaced file" m.replaced;
        cover "a copy after more other files than the disk keeps descriptors of"
          m.after_others;
        judged m outcome (equal octets (Bytes.sub_string m.bytes at len)))
      (fun f r ->
        let at, len = clamp (B.length f.buf) r in
        sys_result f (fun () -> read (B.view f.buf ~first:at ~length:len)));
    command "borrow"
      (file ^-> ranges @-> judges (option octets))
      (fun m r outcome ->
        let at, len = clamp (Bytes.length m.bytes) r in
        match outcome with
        | Ok (Some s) -> equal octets (Bytes.sub_string m.pages at len) s
        | Ok None when m.replaced -> ()
        | Ok None -> fail "the host's borrow of a file is None"
        | Error e -> raise e)
      (fun f r ->
        let at, len = clamp (B.length f.buf) r in
        borrowed host (B.view f.buf ~first:at ~length:len));
    command "write through a borrow"
      ~pre:(fun m _ _ -> Bytes.length m.bytes > 0)
      (file ^-> small_extent @-> letters @-> judges (option unit))
      (fun m a c outcome ->
        match outcome with
        | Ok (Some ()) -> Bytes.set m.pages (min a (Bytes.length m.pages - 1)) c
        | Ok None when m.replaced -> ()
        | Ok None -> fail "the host's borrow of a file is None"
        | Error e -> raise e)
      (fun f a c ->
        Option.map
          (fun p -> poke p (min a (B.length f.buf - 1)) c)
          (B.borrow host f.buf));
    command "read its path"
      ~pre:(fun m -> not m.replaced)
      (file ^-> returns octets)
      (fun m -> Bytes.to_string m.bytes)
      (fun f -> contents f.path);
    command "barrier"
      (file ^-> judges (result unit string))
      (fun m outcome -> judged m outcome Fun.id)
      (fun f -> sys_result f (fun () -> Rig_disk.barrier f.buf));
    command "replace its path"
      ~pre:(fun m _ -> not m.replaced)
      (file ^-> Gen.of_list ~pp:pp_how hows @-> returns unit)
      (fun m _ -> m.replaced <- true)
      (fun f how -> replace f.path how);
    command "read other files"
      (file
      ^-> ints [ max_descriptors - 1; others_count ]
      @-> returns (list int))
      (fun m k ->
        if k >= max_descriptors then m.after_others <- true;
        [])
      (fun _ k -> use_others k);
    command "count descriptors"
      (Gen.unit @-> judges int)
      (fun () outcome ->
        match outcome with
        | Ok n -> at_most int ~than:max_descriptors n
        | Error e -> raise e)
      (fun () -> if Sys.win32 then 0 else open_on_made ());
  ]

let models =
  group ~timeout "a model"
    [
      stateful ~count:200
        "a file's copies, borrows and barriers follow its bytes and pages, \
         whatever replaces its path, within 64 descriptors"
        model_commands;
    ]

(* Two domains *)

(* Files whose bytes copies never change: opened ones, and created ones whose
   copies write the bytes they hold. *)
type shelf = {
  opened : (string * string) array;
  written : (file * string) array;
}

let shelf =
  once @@ fun () ->
  let file i n =
    let s = pattern i n in
    (make_file s, s)
  in
  let sizes = [ 0; 1; 4096; 16385; segment + 7 ] in
  let opened = Array.of_list (List.mapi file sizes) in
  let written =
    Array.of_list
      (List.mapi
         (fun i n ->
           let s = pattern (i + 10) n in
           let path = new_path () in
           let buf = create path n in
           write buf s;
           ({ path; buf }, s))
         sizes)
  in
  { opened; written }

let shelved = Gen.int_range 0 4

let big_ranges =
  let at =
    Gen.frequency
      [ (2, ints (page_edges @ segment_edges @ [ max_int ])); (1, Gen.nat) ]
  in
  Gen.pair at at

let sub s r =
  let at, len = clamp (String.length s) r in
  String.sub s at len

(* [racing outcome accept] accepts [Ok (Ok v)] as [accept v] and fails on any
   error. *)
let racing outcome accept =
  match outcome with
  | Ok (Ok v) -> accept v
  | Ok (Error e) -> fail e
  | Error e -> raise e

let domain_commands =
  let opened i = snd (shelf ()).opened.(i)
  and written i = (shelf ()).written.(i) in
  [
    command "copy from an opened file"
      (shelved @-> big_ranges @-> returns octets)
      (fun i r -> sub (opened i) r)
      (fun i r ->
        let path, _ = (shelf ()).opened.(i) in
        let b = of_file path in
        let at, len = clamp (B.length b) r in
        read (B.view b ~first:at ~length:len));
    command "copy a created file's bytes into it"
      (shelved @-> big_ranges @-> judges (result unit string))
      (fun _ _ outcome -> racing outcome Fun.id)
      (fun i r ->
        let f, s = written i in
        let at, len = clamp (B.length f.buf) r in
        sys_result f (fun () ->
            write (B.view f.buf ~first:at ~length:len) (String.sub s at len)));
    command "copy from a created file"
      (shelved @-> big_ranges @-> judges (result octets string))
      (fun i r outcome ->
        racing outcome (equal octets (sub (snd (written i)) r)))
      (fun i r ->
        let f, _ = written i in
        let at, len = clamp (B.length f.buf) r in
        sys_result f (fun () -> read (B.view f.buf ~first:at ~length:len)));
    command "read other files"
      (Gen.unit @-> returns (list int))
      (fun () -> [])
      (fun () -> use_others others_count);
  ]

let domains =
  group ~timeout "domains"
    [
      stateful ~domains:2 ~count:10 ~steps:4
        "copies from two domains at once read and write each file's own bytes, \
         through closed and reopened descriptors"
        domain_commands;
    ]

(* Children *)

(* Prints how many of this process's descriptors are open on the file [ino]. *)
let open_on_child ino = print_int (open_where (on_inode (int_of_string ino)))

(* Lowers this process's limit of open files to 16 more than it uses, opens 8
   files on the disk, takes every descriptor left, then opens a ninth file and
   reads the first again. *)
let open_files_child () =
  let dir = "files-limit" in
  clear dir;
  let paths =
    Array.init 9 (fun i ->
        make_file ~path:(Filename.concat dir (strf "f%d" i)) (strf "f%d" i))
  in
  let highest =
    Array.fold_left
      (fun m fd -> Option.fold ~none:m ~some:(max m) (int_of_string_opt fd))
      0 (Sys.readdir "/dev/fd")
  in
  (match set_open_files (highest + 1 + 16) with
  | 0 -> ()
  | e -> failwith (strf "setrlimit: errno %d" e));
  let rec take acc =
    match Unix.openfile "/dev/null" [ O_RDONLY; O_CLOEXEC ] 0 with
    | fd -> take (fd :: acc)
    | exception Unix.Unix_error ((EMFILE | ENFILE), _, _) -> acc
  in
  let said = function Ok b -> read b | Error why -> "Error " ^ why in
  let opened = Array.init 8 (fun i -> Rig_disk.of_file paths.(i)) in
  Array.iter (fun b -> ignore (said b)) opened;
  let taken = take [] in
  Printf.printf "ninth: %s\n" (said (Rig_disk.of_file paths.(8)));
  Printf.printf "first: %s\n" (said opened.(0));
  List.iter Unix.close taken

let () =
  match Array.to_list Sys.argv with
  | [ _; "open-on"; ino ] -> open_on_child ino
  | [ _; "open-files" ] -> open_files_child ()
  | _ ->
      clear dir;
      let code =
        run "rig_disk"
          [
            the_disk;
            copies;
            borrows;
            opening;
            descriptors;
            barriers;
            models;
            domains;
          ]
      in
      clear dir;
      exit code
