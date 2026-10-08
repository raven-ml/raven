(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Copies, borrows, opens and barriers of files on the disk, each row beside the
   floor that bounds it: the system calls the disk makes, called from C on a
   descriptor the floor holds. A row's distance to its floor is the disk's and
   the core's share.

   The files live under this bench's directory in _build, cleared at its start
   and removed at its end. A file is in the system's cache after a row's first
   run, so the rows time the way between the cache and memory, where the disk's
   own costs show, and reach the storage device only in a barrier. *)

module B = Rig.Buffer

external open_ : string -> bool -> int = "rig_disk_bench_open"
external close : int -> unit = "rig_disk_bench_close"

external pread :
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "rig_disk_bench_read_byte" "rig_disk_bench_read"

external pwrite :
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "rig_disk_bench_write_byte" "rig_disk_bench_write"

external sync : int -> int = "rig_disk_bench_sync"
external map : int -> int -> int -> int = "rig_disk_bench_map"
external evict : int -> int = "rig_disk_bench_evict"
external evicts : unit -> bool = "rig_disk_bench_evicts"

let strf = Printf.sprintf
let kib = 1024
let mib = 1024 * kib
let gib = 1024 * mib

(* Files *)

let dir = "files"

let clear () =
  if Sys.file_exists dir then
    Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir)

(* Thumper measures each row in a child that leaves without running [at_exit],
   so only the bench's own process removes the files. *)
let () =
  clear ();
  if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
  at_exit (fun () ->
      clear ();
      Sys.rmdir dir)

let path name = Filename.concat dir name

(* [data name n] is the path of a file of [n] bytes, none zero, written once:
   rows that follow find it written. *)
let data name n =
  let p = path name in
  if not (Sys.file_exists p) then begin
    let chunk =
      String.init (Int.min n mib) (fun i -> Char.chr (1 + (i mod 255)))
    in
    Out_channel.with_open_bin (p ^ ".part") (fun oc ->
        let left = ref n in
        while !left > 0 do
          let k = Int.min !left mib in
          Out_channel.output_substring oc chunk 0 k;
          left := !left - k
        done);
    Sys.rename (p ^ ".part") p
  end;
  p

(* [fresh name] is the path [name], naming nothing. *)
let fresh name =
  let p = path name in
  if Sys.file_exists p then Sys.remove p;
  p

let ok = function Ok v -> v | Error why -> failwith why
let host n = B.create Rig.host n

let size n =
  if n >= gib then strf "%dG" (n / gib)
  else if n >= mib then strf "%dM" (n / mib)
  else strf "%dK" (n / kib)

(* The floors' descriptors, memory and results *)

let opened p ~writable =
  let h = open_ p writable in
  if h < 0 then failwith (strf "%s: open failed: %d" p (-h));
  h

(* Host memory of [n] bytes and its address. The floor keeps the pair, so the
   buffer keeps the memory. *)
let memory n =
  let b = host n in
  (b, B.address b)

let moved what n k =
  if k <> n then failwith (strf "%s: %d of %d bytes" what k n)

let row name setup f = Thumper.bench_with_setup ~setup name f

(* Copies *)

let sizes = [ 4 * kib; mib; 64 * mib; gib ]

let read_rows =
  let reading n () =
    (ok (Rig_disk.of_file (data (strf "read-%s" (size n)) n)), host n)
  in
  let floor n () =
    let h = opened (data (strf "read-%s" (size n)) n) ~writable:false in
    (h, memory n)
  in
  Thumper.group "read"
    (List.concat_map
       (fun n ->
         [
           row (size n) (reading n) (fun (file, h) -> B.copy ~src:file ~dst:h);
           row
             (strf "floor-%s" (size n))
             (floor n)
             (fun (h, (_, a)) -> moved "pread" n (pread h 0 a n));
         ])
       sizes)

let write_rows =
  let writing n () =
    (host n, ok (Rig_disk.create_file (fresh (strf "write-%s" (size n))) n))
  in
  let floor n () =
    let h = opened (data (strf "write-floor-%s" (size n)) n) ~writable:true in
    (h, memory n)
  in
  Thumper.group "write"
    (List.concat_map
       (fun n ->
         [
           row (size n) (writing n) (fun (h, file) -> B.copy ~src:h ~dst:file);
           row
             (strf "floor-%s" (size n))
             (floor n)
             (fun (h, (_, a)) -> moved "pwrite" n (pwrite h 0 a n));
         ])
       sizes)

(* Cold reads: each run first drops the file's pages from the system's cache, so
   the read reaches the storage device; the floor drops them too. Only on Linux,
   where a process can drop one file's pages. *)

let cold_rows =
  let n = 64 * mib in
  let cold () =
    let p = data "cold-64M" n in
    let h = opened p ~writable:true in
    moved "sync" 0 (sync h);
    (p, h)
  in
  let dropped h = moved "evict" 0 (evict h) in
  Thumper.group "read"
    (if not (evicts ()) then []
     else
       [
         row "cold-64M"
           (fun () ->
             let p, h = cold () in
             (h, ok (Rig_disk.of_file p), host n))
           (fun (h, file, dst) ->
             dropped h;
             B.copy ~src:file ~dst);
         row "floor-cold-64M"
           (fun () -> (snd (cold ()), memory n))
           (fun (h, (_, a)) ->
             dropped h;
             moved "pread" n (pread h 0 a n));
       ])

(* Borrows: of_file and the first borrow on the host, which maps the file, and
   the same with a read of every byte through the mapping, beside a copy of them
   ([read/64M]). macOS maps a file in about a microsecond while another mapping
   of it lives, and in hundreds once none does. A mapping lives until the
   collection after the next drain of the disk, so each run starts with a
   collection and maps the next of four files in turn, which no earlier run's
   mapping holds; the floors do the same. *)

let borrowed = 64 * mib

type turn = { files : string array; mutable k : int }

let next t =
  Gc.full_major ();
  t.k <- (t.k + 1) land 3;
  t.files.(t.k)

let borrow_rows =
  let turn () =
    {
      files = Array.init 4 (fun i -> data (strf "borrow-%d" i) borrowed);
      k = 0;
    }
  in
  let pages t =
    Option.get (B.borrow Rig.host (ok (Rig_disk.of_file (next t))))
  in
  let into () = (turn (), B.bigarray Bigarray.char (host borrowed)) in
  let floor dst t =
    let h = opened (next t) ~writable:false in
    moved "map" 0 (map h borrowed dst);
    close h
  in
  (* Cold, on Linux: each run drops its file's pages first, so every page the
     read touches faults in from the storage device, beside [read/cold-64M], a
     copy of the same bytes. A page another mapping holds stays cached, so each
     run maps a file no earlier run's mapping holds, as above. *)
  let cold () =
    let t = turn () in
    let drop =
      Array.map
        (fun p ->
          let h = opened p ~writable:true in
          moved "sync" 0 (sync h);
          h)
        t.files
    in
    (t, drop)
  in
  let dropped (t, drop) =
    let p = next t in
    moved "evict" 0 (evict drop.(t.k));
    p
  in
  Thumper.group "borrow"
    ([
       row "first-64M" turn pages;
       row "first-read-64M" into (fun (t, dst) ->
           Bigarray.Array1.blit (B.bigarray Bigarray.char (pages t)) dst);
       row "floor-first-64M" turn (floor 0);
       row "floor-first-read-64M"
         (fun () -> (turn (), memory borrowed))
         (fun (t, (_, a)) -> floor a t);
     ]
    @
    if not (evicts ()) then []
    else
      [
        row "cold-read-64M"
          (fun () -> (cold (), B.bigarray Bigarray.char (host borrowed)))
          (fun (c, dst) ->
            let file = ok (Rig_disk.of_file (dropped c)) in
            let pages = Option.get (B.borrow Rig.host file) in
            Bigarray.Array1.blit (B.bigarray Bigarray.char pages) dst);
        row "floor-cold-read-64M"
          (fun () -> (cold (), memory borrowed))
          (fun (c, (_, a)) ->
            let h = opened (dropped c) ~writable:false in
            moved "map" 0 (map h borrowed a);
            close h);
      ])

(* Opens: one file, and more files than the disk keeps descriptors for, each
   open evicting the least recently used. *)

let many = 128

let open_rows =
  let small i = data (strf "small-%d" i) (4 * kib) in
  let smalls () = Array.init many small in
  let open_floor p = close (opened p ~writable:false) in
  Thumper.group "open"
    [
      row "file" (fun () -> small 0) (fun p -> ok (Rig_disk.of_file p));
      row "many-128" smalls (Array.map (fun p -> ok (Rig_disk.of_file p)));
      row "floor-file" (fun () -> small 0) open_floor;
      row "floor-many-128" smalls (fun ps ->
          Array.iter close (Array.map (opened ~writable:false) ps));
    ]

(* Barriers: a write of 1 MiB ordered before later changes, beside the system's
   call. *)

let barrier_rows =
  let n = mib in
  Thumper.group "barrier"
    [
      row "1M"
        (fun () -> (host n, ok (Rig_disk.create_file (fresh "barrier") n)))
        (fun (h, file) ->
          B.copy ~src:h ~dst:file;
          Rig_disk.barrier file);
      row "floor-1M"
        (fun () -> (opened (data "barrier-floor" n) ~writable:true, memory n))
        (fun (h, (_, a)) ->
          moved "pwrite" n (pwrite h 0 a n);
          moved "sync" 0 (sync h));
    ]

(* Parallel copies: a 4 KiB read while another domain reads 1 GiB without pause,
   the 1 GiB read while another domain reads 4 KiB without pause, and two
   domains reading 64 MiB of their own files at once. A lock held across
   transfers would show in the first as a wait for the large read, or in the
   second as the small reads taking the lock from it. *)

(* Another domain running [f] until the row ends. *)
let beside f =
  let stop = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        let f = f () in
        while not (Atomic.get stop) do
          f ()
        done)
  in
  (stop, d)

let stop (stop, d) =
  Atomic.set stop true;
  Domain.join d

(* Another domain running [f] once each time [ask] asks, until [stop]. The asks
   are numbered from 1; [finished] is the last one done, [-1] stops. *)
type partner = {
  asked : int Atomic.t;
  finished : int Atomic.t;
  d : unit Domain.t;
}

let partner f =
  let asked = Atomic.make 0 and finished = Atomic.make 0 in
  let d =
    Domain.spawn (fun () ->
        let f = f () in
        let next = ref 1 in
        while Atomic.get asked >= 0 do
          if Atomic.get asked >= !next then begin
            f ();
            Atomic.set finished !next;
            incr next
          end
          else Domain.cpu_relax ()
        done)
  in
  { asked; finished; d }

(* [together p f] runs [f] here while [p] runs its own once. *)
let together p f =
  let k = Atomic.fetch_and_add p.asked 1 + 1 in
  f ();
  while Atomic.get p.finished < k do
    Domain.cpu_relax ()
  done

let part p =
  Atomic.set p.asked (-1);
  Domain.join p.d

let parallel_rows =
  let small () = data "read-4K" (4 * kib) and large () = data "read-1G" gib in
  let disk_read p n () =
    let file = ok (Rig_disk.of_file p) and h = host n in
    fun () -> B.copy ~src:file ~dst:h
  in
  let floor_read p n () =
    let h = opened p ~writable:false and m = memory n in
    fun () -> moved "pread" n (pread h 0 (snd m) n)
  in
  let half = 64 * mib in
  let halves () = (data "read-64M" half, data "read-64M-b" half) in
  let next_to name read (p, n) (p', n') =
    Thumper.bench_with_setup name
      ~setup:(fun () ->
        let rival = beside (read (p' ()) n') in
        (read (p ()) n (), rival))
      ~teardown:(fun (_, rival) -> stop rival)
      (fun (f, _) -> f ())
  in
  let small = (small, 4 * kib) and large = (large, gib) in
  let two name read =
    Thumper.bench_with_setup name
      ~setup:(fun () ->
        let a, b = halves () in
        (read a half (), partner (read b half)))
      ~teardown:(fun (_, p) -> part p)
      (fun (f, p) -> together p f)
  in
  Thumper.group "parallel"
    [
      next_to "read-4K-beside-1G" disk_read small large;
      next_to "read-1G-beside-4K" disk_read large small;
      two "read-64M-two" disk_read;
      next_to "floor-read-4K-beside-1G" floor_read small large;
      next_to "floor-read-1G-beside-4K" floor_read large small;
      two "floor-read-64M-two" floor_read;
    ]

let config = Thumper.Config.(default |> deadline 60.)

let () =
  exit
  @@ Thumper.run ~config "rig_disk"
       [
         read_rows;
         cold_rows;
         write_rows;
         borrow_rows;
         open_rows;
         barrier_rows;
         parallel_rows;
       ]
