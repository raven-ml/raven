(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NumPy formats. A value saved as npy, and an archive saved as npz, loads as
   itself, bit for bit, over every dtype npy holds, views and the values that
   break codecs; files numpy and other zip writers wrote load as they meant; an
   aligned little-endian array loads over the file's pages and any other into
   memory of its own; a file cut short or corrupted raises [Failure] naming it,
   and never anything else. *)

open Windtrap
open Io_support

let save_npy path (Nx.P x) = Nx_io.save_npy path x

(* A file the format describes in the header [dict], as numpy writes it. *)
let npy_file dict data =
  let used = 10 + String.length dict + 1 in
  let dict = dict ^ String.make ((64 - (used mod 64)) mod 64) ' ' ^ "\n" in
  "\x93NUMPY\001\000"
  ^ String.init 2 (fun i ->
      Char.chr ((String.length dict lsr (8 * i)) land 0xff))
  ^ dict ^ data

(* A float32 vector of 2^17 + 3 elements from a drawn start: its data spans many
   pages and ends inside one, where every other drawn value fits in the file's
   first page. *)
let long = (1 lsl 17) + 3

let long_case =
  Gen.map
    (fun start ->
      Case
        ( D.Float32,
          [| long |],
          Array.init long (fun i -> Float.of_int ((start + i) mod 4096) /. 8.),
          Plain ))
    (Gen.int_range 0 4095)

(* Values of every dtype npy holds, and one in nine a long vector. The law
   draws 200 cases, so that it misses the long vector with probability
   6 10^-11, and each kind of value, 1/7 of the others, with probability
   below 10^-10. *)
let npy_cases =
  Gen.with_pp
    (fun ppf (Case (dt, s, _, view)) ->
      Format.fprintf ppf "%a %a %a" D.pp dt pp_shape s pp_view view)
    (Gen.frequency [ (8, case npy_dtypes); (1, long_case) ])

let npy =
  let loads name expected () =
    equal packed expected (Nx_io.load_npy (fixture name))
  in
  group "npy"
    [
      prop ~count:200 "a value saved as npy loads as itself, bit for bit"
        npy_cases
        (fun (Case (_, s, _, _) as c) ->
          covers c;
          cover "a value of 2^17 + 3 elements" (s = [| long |]);
          Law.round_trip packed string
            (fun x -> saved ".npy" save_npy x)
            Nx_io.load_npy (value c));
      cases "files numpy wrote load" ~id:fst
        [
          ( "fortran.npy",
            loads "fortran.npy"
              (Nx.P (Nx.reshape [| 2; 3 |] (Nx.arange_f Nx.float32 1. 7. 1.)))
          );
          ( "big_endian.npy",
            loads "big_endian.npy"
              (Nx.P (Nx.create Nx.int16 [| 3 |] [| 1; 256; -2 |])) );
          ( "version2.npy",
            loads "version2.npy" (Nx.P (Nx.arange Nx.int32 7 10 1)) );
          ( "version3.npy",
            loads "version3.npy"
              (Nx.P (Nx.create Nx.bool [| 3 |] [| true; false; true |])) );
        ]
        (fun (_, check) -> check ());
      cases
        "an aligned little-endian file loads over its pages, and a big-endian \
         one into memory of its own"
        ~id:fst
        [
          ("C order", ("version2.npy", Over_file));
          ("Fortran order", ("fortran.npy", Over_file));
          ("big-endian", ("big_endian.npy", Own_memory));
        ]
        (fun (_, (name, expected)) ->
          equal memory expected (memory_of (Nx_io.load_npy (fixture name))));
      test "a complex big-endian element keeps its real part first" (fun () ->
          let data =
            "\x3f\xf0\000\000\000\000\000\000\xc0\000\000\000\000\000\000\000"
          in
          let path =
            file
              (npy_file
                 "{'descr': '>c16', 'fortran_order': False, 'shape': (1,), }"
                 data)
          in
          equal packed
            (Nx.P (Nx.create Nx.complex128 [| 1 |] [| { re = 1.; im = -2. } |]))
            (Nx_io.load_npy path));
      cases "a file that is no npy file fails, naming it" ~id:fst
        [
          ("a text file", fun () -> "hello, nx\n");
          ("an npz file", fun () -> read (fixture "savez.npz"));
        ]
        (fun (_, contents) ->
          let path = file (contents ()) in
          fails ~path (fun () -> Nx_io.load_npy path));
      test "a dtype nx lacks fails, naming the file" (fun () ->
          let path = fixture "unicode.npy" in
          fails ~path (fun () -> Nx_io.load_npy path));
      test "a byte past the data, or every proper prefix, fails" (fun () ->
          let whole =
            read (saved ".npy" save_npy (Nx.P (Nx.arange Nx.int16 0 5 1)))
          in
          let path = file (whole ^ "\000") in
          fails ~path (fun () -> Nx_io.load_npy path);
          prefixes_fail Nx_io.load_npy whole);
      test "any header byte flipped loads or fails" (fun () ->
          let whole =
            read (saved ".npy" save_npy (Nx.P (Nx.arange Nx.int16 0 5 1)))
          in
          flips_load_or_fail Nx_io.load_npy ~upto:128 whole);
      cases "a malformed header fails, naming the file" ~id:fst
        [
          ( "a field twice",
            "{'descr': '<i2', 'descr': '<i2', 'fortran_order': False, 'shape': \
             (1,), }" );
          ( "an unknown field",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (1,), 'x': 1, }"
          );
          ("no shape", "{'descr': '<i2', 'fortran_order': False, }");
          ( "a negative extent",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (-1,), }" );
          ( "a byte-order mark on a wide dtype",
            "{'descr': '|i2', 'fortran_order': False, 'shape': (1,), }" );
          ( "33 axes",
            "{'descr': '<i2', 'fortran_order': False, 'shape': ("
            ^ String.concat ", " (List.init 33 (fun _ -> "1"))
            ^ "), }" );
          ( "an extent past 2^53",
            "{'descr': '<i2', 'fortran_order': False, 'shape': (0, \
             9007199254740993), }" );
        ]
        (fun (_, dict) ->
          let path = file (npy_file dict "\000\000") in
          fails ~path (fun () -> Nx_io.load_npy path));
      cases "a dtype npy lacks is refused, naming the save"
        ~id:(fun (D.Any dt) -> D.name dt)
        D.
          [
            Any Bfloat16;
            Any Float8_e4m3fn;
            Any Float8_e5m2;
            Any Float4_e2m1fn;
            Any Int4;
            Any Uint4;
            Any Bit;
          ]
        (fun (D.Any dt) ->
          invalid ~by:"Nx_io.save_npy" (fun () ->
              Nx_io.save_npy (temp_file ()) (Nx.zeros dt [| 2 |])));
      test "a bit value's refusal names the cast to bool" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Nx.cast Nx.bool") (fun () ->
              Nx_io.save_npy (temp_file ()) (Nx.zeros Nx.bit [| 8 |])));
    ]

(* npy from blocks *)

(* Blocks of one dtype and cell, each [k] rows of drawn elements, flipped
   along axis 0 or not. *)
type blocks =
  | Blocks : ('v, 's) D.t * int array * ('v array * int * bool) list -> blocks

let blocks =
  let open Gen in
  (let* (D.Any dt) = of_list npy_dtypes in
   let* cell = array ~size:(int_range 0 2) (int_range 0 3) in
   let numel = Array.fold_left ( * ) 1 cell in
   let* count =
     frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 5) ]
   in
   let+ bs =
     list ~size:(constant count)
       (let* k = int_range 0 4 in
        let* vs = array ~size:(constant (k * numel)) (element dt) in
        let+ flipped = bool in
        (vs, k, flipped))
   in
   Blocks (dt, cell, bs))
  |> with_pp (fun ppf (Blocks (dt, cell, bs)) ->
      Format.fprintf ppf "%a, cell %a, extents [%s]" D.pp dt pp_shape cell
        (String.concat "; " (List.map (fun (_, k, _) -> string_of_int k) bs)))

let block dt cell (vs, k, flipped) =
  let x = Nx.create dt (Array.append [| k |] cell) vs in
  if flipped && k > 0 then Nx.flip ~axes:[ 0 ] x else x

let joined dt cell xs =
  match xs with
  | [] -> Nx.P (Nx.zeros dt (Array.append [| 0 |] cell))
  | xs -> Nx.P (Nx.concatenate ~axis:0 xs)

let by = "Nx_io.save_npy_concat"

let concat =
  group "npy from blocks"
    [
      prop "blocks saved as one npy load as their concatenation, mapped"
        blocks (fun (Blocks (dt, cell, bs)) ->
          cover "no block" (bs = []);
          cover "one block" (List.length bs = 1);
          cover "several blocks" (List.length bs > 1);
          cover "a block of no row" (List.exists (fun (_, k, _) -> k = 0) bs);
          cover "a cell of no element" (Array.mem 0 cell);
          let xs = List.map (block dt cell) bs in
          let path = temp_file ~suffix:".npy" () in
          Nx_io.save_npy_concat path dt cell (List.to_seq xs);
          let loaded = Nx_io.load_npy path in
          equal packed (joined dt cell xs) loaded;
          equal memory Over_file (memory_of loaded));
      test "mapped blocks join into one file" (fun () ->
          let shard i =
            let rows = Nx.arange Nx.int32 (6 * i) (6 * (i + 1)) 1 in
            saved ".npy" save_npy (Nx.P (Nx.reshape [| 3; 2 |] rows))
          in
          let shards =
            List.init 3 (fun i -> Nx.unpack Nx.int32 (Nx_io.load_npy (shard i)))
          in
          let path = temp_file ~suffix:".npy" () in
          Nx_io.save_npy_concat path Nx.int32 [| 2 |] (List.to_seq shards);
          equal packed
            (Nx.P (Nx.reshape [| 9; 2 |] (Nx.arange Nx.int32 0 18 1)))
            (Nx_io.load_npy path));
      (* The file beside the path holds the header's width and every block
         before the one being forced, so each block is written before the
         next is forced. *)
      test "a block is forced once the one before it is written" (fun () ->
          let dir = temp_dir () in
          let path = Filename.concat dir "x.npy" in
          let written = ref [] in
          let rec watched i () =
            if i = 4 then Seq.Nil
            else begin
              let size =
                match Array.to_list (Sys.readdir dir) with
                | [ t ] -> (Unix.stat (Filename.concat dir t)).Unix.st_size
                | _ -> -1
              in
              written := size :: !written;
              Seq.Cons (Nx.full Nx.uint8 [| 1; 64 |] i, watched (i + 1))
            end
          in
          Nx_io.save_npy_concat path Nx.uint8 [| 64 |] (watched 0);
          let header = (Unix.stat path).Unix.st_size - (4 * 64) in
          equal (list int)
            (List.init 4 (fun i -> header + (64 * i)))
            (List.rev !written));
      test "a block of another cell is refused, and the path is left as it was"
        (fun () ->
          let dir = temp_dir () in
          let path = Filename.concat dir "old" in
          write path "old";
          let zeros s = Nx.zeros Nx.float32 s in
          let xs = [ zeros [| 2; 3 |]; zeros [| 2; 4 |] ] in
          invalid ~by (fun () ->
              Nx_io.save_npy_concat path Nx.float32 [| 3 |] (List.to_seq xs));
          equal ~msg:"the file" string "old" (read path);
          equal ~msg:"the directory" (array string) [| "old" |]
            (Sys.readdir dir));
      test "what forcing the blocks raises passes on, and the path is left as \
            it was"
        (fun () ->
          let dir = temp_dir () in
          let path = Filename.concat dir "old" in
          write path "old";
          let blocks () =
            Seq.Cons (Nx.zeros Nx.int8 [| 1 |], fun () -> failwith "no block")
          in
          raises (Failure "no block") (fun () ->
              Nx_io.save_npy_concat path Nx.int8 [||] blocks);
          equal ~msg:"the file" string "old" (read path);
          equal ~msg:"the directory" (array string) [| "old" |]
            (Sys.readdir dir));
      cases "a dtype or a cell npy cannot hold is refused" ~id:fst
        [
          ("bfloat16", fun () ->
              Nx_io.save_npy_concat (temp_file ()) Nx.bfloat16 [||] Seq.empty);
          ("a negative extent", fun () ->
              Nx_io.save_npy_concat (temp_file ()) Nx.int8 [| -1 |] Seq.empty);
          ("32 axes", fun () ->
              Nx_io.save_npy_concat (temp_file ()) Nx.int8 (Array.make 32 1)
                Seq.empty);
        ]
        (fun (_, save) -> invalid ~by save);
    ]

(* npy windows *)

(* An array of [n] rows of a cell, and windows of its rows. *)
type windows =
  | Windows : ('v, 's) D.t * int array * 'v array * (int * int) list -> windows

let windows =
  let open Gen in
  (let* (D.Any dt) = of_list npy_dtypes in
   let* cell = array ~size:(int_range 0 2) (int_range 0 3) in
   let* n = frequency [ (1, constant 0); (4, int_range 1 9) ] in
   let shape = Array.append [| n |] cell in
   let* vs =
     array ~size:(constant (Array.fold_left ( * ) 1 shape)) (element dt)
   in
   let+ ws =
     list
       ~size:(frequency [ (1, constant 0); (5, int_range 1 6) ])
       (let* r = int_range 0 n in
        let+ m = frequency [ (1, constant 0); (4, int_range 0 (n - r)) ] in
        (r, m))
   in
   Windows (dt, shape, vs, ws))
  |> with_pp (fun ppf (Windows (dt, s, _, ws)) ->
      Format.fprintf ppf "%a %a, windows [%s]" D.pp dt pp_shape s
        (String.concat "; "
           (List.map (fun (r, m) -> Printf.sprintf "%d, %d" r m) ws)))

let rows_of path ws =
  Nx_io.load_npy_rows path
    ~row:(Array.of_list (List.map fst ws))
    ~rows:(Array.of_list (List.map snd ws))

(* The windows [ws] of [load_npy path], sliced and joined. *)
let sliced path ws =
  let (Nx.P x) = Nx_io.load_npy path in
  let s = Nx.shape x in
  let part (r, m) = Nx.slice [ Nx.R (r, r + m) ] x in
  match ws with
  | [] ->
      s.(0) <- 0;
      Nx.P (Nx.zeros (Nx.dtype x) s)
  | ws -> Nx.P (Nx.concatenate ~axis:0 (List.map part ws))

let overlap ws =
  List.exists
    (fun (r, m) ->
      List.exists
        (fun (r', m') -> (r, m) != (r', m') && r < r' + m' && r' < r + m)
        ws)
    ws

let decreasing ws =
  let rec go = function
    | (r, _) :: ((r', _) :: _ as rest) -> r' < r || go rest
    | _ -> false
  in
  go ws

(* A file of [pages] pages of float32 rows of one page each, its data on the
   second page: the header padded to 4096 bytes. *)
let paged pages =
  let dict =
    Printf.sprintf
      "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, 1024), }" pages
  in
  let used = 10 + String.length dict + 1 in
  let dict = dict ^ String.make (4096 - used) ' ' ^ "\n" in
  "\x93NUMPY\001\000"
  ^ String.init 2 (fun i ->
      Char.chr ((String.length dict lsr (8 * i)) land 0xff))
  ^ dict
  ^ pattern (pages * 4096) 3

let rows =
  group "npy windows"
    [
      prop
        "windows of a saved value load as the value's rows, sliced and \
         joined, in memory of its own"
        windows (fun (Windows (dt, s, vs, ws)) ->
          cover "no window" (ws = []);
          cover "a window of no row" (List.exists (fun (_, m) -> m = 0) ws);
          cover "overlapping windows" (overlap ws);
          cover "windows out of order" (decreasing ws);
          cover "a cell of no element" (Array.exists (( = ) 0) s);
          cover "a vector" (Array.length s = 1);
          let path = saved ".npy" save_npy (Nx.P (Nx.create dt s vs)) in
          let loaded = rows_of path ws in
          equal packed (sliced path ws) loaded;
          equal memory Own_memory (memory_of loaded));
      cases "files numpy wrote load as their rows" ~id:fst
        [
          ("big-endian", ("big_endian.npy", [ (2, 1); (0, 2); (1, 0) ]));
          ("version 2.0", ("version2.npy", [ (1, 2); (0, 3) ]));
          ("version 3.0", ("version3.npy", [ (2, 1); (2, 1) ]));
        ]
        (fun (_, (name, ws)) ->
          let path = fixture name in
          equal packed (sliced path ws) (rows_of path ws));
      test "a vector in Fortran order loads as its rows" (fun () ->
          let path =
            file
              (npy_file
                 "{'descr': '<i2', 'fortran_order': True, 'shape': (3,), }"
                 "\001\000\002\000\003\000")
          in
          equal packed
            (Nx.P (Nx.create Nx.int16 [| 3 |] [| 3; 1; 2 |]))
            (rows_of path [ (2, 1); (0, 2) ]));
      cases "an array whose rows a window cannot hold fails, naming the file"
        ~id:fst
        [
          ("Fortran order of two axes", fun () -> fixture "fortran.npy");
          ( "no axis",
            fun () ->
              file
                (npy_file
                   "{'descr': '<i2', 'fortran_order': False, 'shape': (), }"
                   "\001\000") );
          ("no file", fun () -> Filename.concat (temp_dir ()) "x.npy");
        ]
        (fun (_, path) ->
          let path = path () in
          fails ~path (fun () -> rows_of path []));
      test "every proper prefix fails" (fun () ->
          let whole =
            read (saved ".npy" save_npy (Nx.P (Nx.arange Nx.int16 0 5 1)))
          in
          prefixes_fail (fun path -> rows_of path [ (0, 1) ]) whole);
      (let path =
         lazy (saved ".npy" save_npy (Nx.P (Nx.arange Nx.int16 0 5 1)))
       in
       cases "a window outside the array is refused" ~id:fst
         [
           ("a negative row", ([| -1 |], [| 1 |]));
           ("a negative count", ([| 0 |], [| -1 |]));
           ("rows past the last", ([| 3 |], [| 3 |]));
           ("a window of no row past the end", ([| 6 |], [| 0 |]));
           ("counts of another length", ([| 0; 1 |], [| 1 |]));
         ]
         (fun (_, (row, rows)) ->
           invalid ~by:"Nx_io.load_npy_rows" (fun () ->
               Nx_io.load_npy_rows (Lazy.force path) ~row ~rows)));
      test "windows of more than 2^53 rows of no element are refused"
        (fun () ->
          let path =
            file
              (npy_file
                 "{'descr': '<i2', 'fortran_order': False, 'shape': \
                  (9007199254740992, 0), }"
                 "")
          in
          let n = 1 lsl 53 in
          raises_match
            (Exn.invalid_arg
               ~sub:"Nx_io.load_npy_rows: the windows hold more than 2^53 rows")
            (fun () ->
              Nx_io.load_npy_rows path ~row:[| 0; 0 |] ~rows:[| n; n |]));
      test "a window of no row may start at the end" (fun () ->
          let path =
            saved ".npy" save_npy (Nx.P (Nx.arange Nx.int16 0 5 1))
          in
          equal packed
            (Nx.P (Nx.zeros Nx.int16 [| 0 |]))
            (rows_of path [ (5, 0) ]));
      (* Scattered pages of a file the system does not hold: the call reads
         those pages from storage and no other beyond what its header's read
         takes, which a call of no window measures. The file lies in the
         working directory, on the build tree's disk, as a temporary
         directory may be memory, which reads nothing from storage. The count
         assumes the system reads a 4 KiB read of a file as one page: a file
         system of larger blocks reads more and fails the test. *)
      test "windows of a file not in memory read their pages from storage"
        (fun () ->
          if not (Sys.file_exists "/proc/self/io") then
            skip ~reason:"the system does not count a process's reads" ();
          let pages = 2048 in
          let path =
            Filename.temp_file ~temp_dir:(Sys.getcwd ()) "nx-io-paged" ".npy"
          in
          Fun.protect ~finally:(fun () -> Sys.remove path) @@ fun () ->
          write path (paged pages);
          let cold ws =
            (match drop_pages path with
            | 0 -> ()
            | -1 ->
                skip ~reason:"only Linux drops a file's pages unprivileged" ()
            | code -> failf "drop_pages: errno %d" code);
            let before = storage_reads () in
            let x = rows_of path ws in
            (x, storage_reads () - before)
          in
          let _, header = cold [] in
          (* Rows 29 apart from 1 MiB on, and a window of 3. *)
          let ws =
            List.init 60 (fun j -> (256 + (29 * j), 1)) @ [ (2000, 3) ]
          in
          let x, read = cold ws in
          equal packed (sliced path ws) x;
          if read = 0 then
            skip ~reason:"the file system reads nothing from storage" ();
          equal ~msg:"bytes read from storage" int
            (4096 * (60 + 3))
            (read - header));
    ]

(* npz *)

let save_npz path l = Nx_io.save_npz path (Archive.of_list l)

(* [xorshift n] is gen/generate.py's noise: the low byte of each of [n] states
   of Marsaglia's xorshift32 from 2463534242. *)
let xorshift n =
  let mask = 0xffff_ffff and x = ref 2463534242 in
  Array.init n (fun _ ->
      x := !x lxor ((!x lsl 13) land mask);
      x := !x lxor (!x lsr 17);
      x := !x lxor ((!x lsl 5) land mask);
      !x land 0xff)

(* [without_zip64_end s] is the archive [s] without its ZIP64 end record and
   locator, as Python's zipfile, and so numpy, writes an archive of 65535
   entries: the classic end record's 16-bit count is then the count itself. *)
let without_zip64_end s =
  let eocd = String.length s - 22 in
  if String.sub s (eocd - 20) 4 <> "PK\006\007" then s
  else
    let record = Int64.to_int (String.get_int64_le s (eocd - 12)) in
    String.sub s 0 record ^ String.sub s eocd 22

let npz =
  let one = Nx.P (Nx.zeros Nx.int8 [| 1 |]) in
  group "npz"
    [
      prop "an archive saved as npz loads as itself"
        (Gen.with_pp pp_entries (archives npy_dtypes))
        (fun entries ->
          List.iter (fun (_, c) -> covers c) entries;
          cover "an empty archive" (entries = []);
          cover "several entries" (List.length entries >= 2);
          Law.round_trip archive string
            (saved ".npz" Nx_io.save_npz)
            Nx_io.load_npz (archive_of entries));
      cases "archives numpy wrote load, deflated or stored" ~id:fst
        [
          ( "archive.npz",
            [
              ("a", Nx.P (Nx.create Nx.float32 [| 1; 2 |] [| 1.5; -2. |]));
              ("b", Nx.P (Nx.arange Nx.int64 3 6 1));
            ] );
          ( "savez.npz",
            [
              ("a", Nx.P (Nx.create Nx.uint16 [| 3 |] [| 1; 2; 65535 |]));
              ("b", Nx.P (Nx.create Nx.bool [| 1; 2 |] [| true; false |]));
            ] );
        ]
        (fun (name, entries) ->
          equal archive (Archive.of_list entries)
            (Nx_io.load_npz (fixture name)));
      test
        "an archive of 65535 entries, the most a 16-bit count holds, loads as \
         itself, with or without a ZIP64 end record" (fun () ->
          let a =
            Archive.of_list (List.init 65535 (fun i -> (string_of_int i, one)))
          in
          let whole = read (saved ".npz" Nx_io.save_npz a) in
          equal archive a (Nx_io.load_npz (file whole));
          equal archive a (Nx_io.load_npz (file (without_zip64_end whole))));
      cases
        "entries of other names than an npy file's are ignored, whatever they \
         hold"
        ~id:fst
        [
          ( "a text file",
            ( "other_names.npz",
              ("w", Nx.P (Nx.create Nx.float64 [| 2 |] [| 1.5; -2. |])) ) );
          ( "a directory and a file compressed with bzip2",
            ( "skipped_entries.npz",
              ("weights/w", Nx.P (Nx.create Nx.int16 [| 2 |] [| 3; -4 |])) ) );
        ]
        (fun (_, (name, entry)) ->
          equal archive
            (Archive.of_list [ entry ])
            (Nx_io.load_npz (fixture name)));
      test "an entry named .npy alone names no array, and fails naming it"
        (fun () ->
          let path = fixture "suffix_only.npz" in
          fails ~path ~naming:{|".npy"|} (fun () -> Nx_io.load_npz path));
      cases "a file that is no npz file fails, naming it" ~id:fst
        [
          ("a text file", fun () -> "hello, nx\n");
          ("an npy file", fun () -> read (fixture "version2.npy"));
        ]
        (fun (_, contents) ->
          let path = file (contents ()) in
          fails ~path (fun () -> Nx_io.load_npz path));
      test "an entry numpy deflated with a stored block inside loads" (fun () ->
          let path = fixture "stored_block.npz" in
          equal ~msg:"deflated" int 8 (String.get_uint16_le (read path) 8);
          let w = Array.append (Array.make 1024 0) (xorshift 20_000) in
          equal archive
            (Archive.of_list
               [ ("w", Nx.P (Nx.create Nx.uint8 [| Array.length w |] w)) ])
            (Nx_io.load_npz path));
      (* Noise over two stored blocks, a full one of 65535 bytes and up to
         16 KiB more, after a run of zeros, which the encoder codes and which
         makes the entry's first 64 KiB shrink. Each case writes and syncs its
         file, so the entry is no longer than that needs. *)
      prop ~count:20
        "an entry of a run of zeros then noise, saved as npz, loads as itself"
        (Gen.triple
           (Gen.int_range 8_192 16_384)
           (Gen.int_range 65_536 81_920)
           Gen.nat)
        (fun (run, noise, seed) ->
          let random = Random.State.make [| seed |] in
          let b = Bigarray.(Array1.create int8_unsigned c_layout) (run + noise) in
          for i = 0 to run + noise - 1 do
            b.{i} <- (if i < run then 0 else Random.State.int random 256)
          done;
          let t = Nx.of_bigarray (Bigarray.genarray_of_array1 b) in
          let path = saved ".npz" save_npz [ ("w", Nx.P t) ] in
          equal ~msg:"deflated" int 8 (String.get_uint16_le (read path) 8);
          equal packed (Nx.P t) (find (Nx_io.load_npz path) "w"));
      test
        "a deflated entry whose data starts at no multiple of its element size \
         loads" (fun () ->
          equal packed
            (Nx.P (Nx.create Nx.float64 [| 5 |] [| 0.; 1.5; 3.; 4.5; 6. |]))
            (find (Nx_io.load_npz (fixture "unaligned.npz")) "w"));
      cases "an invalid name is refused" ~id:Fun.id
        [ "/w"; "a//b"; "./w"; "a/../b"; "\xff"; "a\\b"; "c:w" ] (fun name ->
          invalid ~by:"Nx_io.save_npz" (fun () ->
              save_npz (temp_file ()) [ (name, one) ]));
      test "a dtype npy lacks is refused, naming the entry" (fun () ->
          raises_match
            (Exn.invalid_arg ~sub:"Nx_io.save_npz: w: npy has no bfloat16")
            (fun () ->
              save_npz (temp_file ())
                [ ("w", Nx.P (Nx.zeros Nx.bfloat16 [| 1 |])) ]));
      (let s = Random.State.make [| 0x4e58 |] in
       let random _ = Random.State.int s 256 in
       cases
         "an incompressible entry is stored, over the file, and a compressible \
          one deflated, into memory of its own"
         ~id:fst
         [
           ("stored", (Nx.init Nx.uint8 [| 70_000 |] random, 0, Over_file));
           ("deflated", (Nx.zeros Nx.uint8 [| 70_000 |], 8, Own_memory));
         ]
         (fun (name, (t, method_, lies)) ->
           let path = saved ".npz" save_npz [ (name, Nx.P t) ] in
           (* The compression method of the archive's first entry. *)
           equal ~msg:"method" int method_ (String.get_uint16_le (read path) 8);
           let loaded = find (Nx_io.load_npz path) name in
           equal packed (Nx.P t) loaded;
           equal memory lies (memory_of loaded)));
      test "a deflated entry's donation writes in place" (fun () ->
          let n = 70_000 in
          let path =
            saved ".npz" save_npz [ ("w", Nx.P (Nx.zeros Nx.uint8 [| n |])) ]
          in
          equal ~msg:"deflated" int 8 (String.get_uint16_le (read path) 8);
          let load () = Nx.unpack Nx.uint8 (find (Nx_io.load_npz path) "w") in
          let y, grew = grown Rig.host load (fun x -> Nx.neg (Nx.donate x)) in
          less ~msg:"no memory of the result's size" int ~than:n grew;
          equal (array int) (Array.make n 0) (Nx.to_array y));
      test "an entry whose checksum does not match fails" (fun () ->
          let s = read (saved ".npz" save_npz [ ("w", one) ]) in
          (* The CRC-32 of the central directory's first entry. *)
          let rec crc i =
            if String.sub s i 4 = "PK\001\002" then i + 16 else crc (i + 1)
          in
          let path = file (flip_byte s (crc 0)) in
          fails ~path (fun () -> Nx_io.load_npz path));
      test "an entry whose stated checksum is 0 loads unchecked" (fun () ->
          let x = Nx.P (Nx.arange Nx.int16 0 5 1) in
          let b = Bytes.of_string (read (saved ".npz" save_npz [ ("w", x) ])) in
          (* The CRC-32 of the data descriptor and of the central directory. *)
          let rec zero signature at i =
            if Bytes.sub_string b i 4 = signature then
              Bytes.set_int32_le b (i + at) 0l
            else zero signature at (i + 1)
          in
          zero "PK\007\008" 4 0;
          zero "PK\001\002" 16 0;
          equal packed x (find (Nx_io.load_npz (file (Bytes.to_string b))) "w"));
      test
        "a deflated entry that declares more data than deflate can expand to \
         fails" (fun () ->
          let s = read (fixture "unaligned.npz") in
          let rec find signature i =
            if String.sub s i 4 = signature then i else find signature (i + 1)
          in
          (* The uncompressed size of the local and the central header. *)
          let b = Bytes.of_string s in
          Bytes.set_int32_le b (find "PK\003\004" 0 + 22) 0x40000000l;
          Bytes.set_int32_le b (find "PK\001\002" 0 + 24) 0x40000000l;
          let path = file (Bytes.to_string b) in
          fails ~path ~naming:"declares more data" (fun () ->
              Nx_io.load_npz path));
      test "every proper prefix fails, and any byte flipped loads or fails"
        (fun () ->
          let whole =
            read
              (saved ".npz" save_npz
                 [ ("a", Nx.P (Nx.arange Nx.int16 0 5 1)); ("b", one) ])
          in
          prefixes_fail Nx_io.load_npz whole;
          flips_load_or_fail Nx_io.load_npz whole);
    ]

let () = exit (run "nx.io NumPy" [ npy; concat; rows; npz ])
