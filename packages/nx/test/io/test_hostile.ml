(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Header fields a hostile file sets, whose refusal keeps a load from sizing the
   OCaml heap by them. Rig's budget, which the other suites load under, counts
   Rig's buffers alone: these tests run under a limit of the process's memory,
   so a regression that drops a refusal fails here by name rather than take the
   machine's memory. macOS's limits do not bound allocations, so the tests skip
   there. The files come from gen/generate.py. *)

open Windtrap
open Io_support
module H = Nx_io.Hdf5

(* Room past what the process holds at its start, as test_widths gives. *)
let room = 2_000_000_000

(* The outcome of the limit, which [run]'s workers set at their start. *)
let limit = ref 0

let limited f () =
  if !limit <> 0 then
    skip ~reason:(Printf.sprintf "no limit of the memory: error %d" !limit) ();
  f ()

let dataset file path =
  List.find (fun d -> H.path d = path) (snd (Nx_io.load_hdf5 (fixture file)))

let gguf =
  group "load_gguf"
    [
      (* The counts of keys and of arrays, and every length the header states,
         among the bytes flipped. *)
      test "any header byte flipped loads or fails"
        (limited (fun () ->
             let path = fixture "hostile.gguf" in
             equal ~msg:"the file's keys" int 4
               (List.length (Nx_io.Gguf.metadata (Nx_io.load_gguf path)));
             flips_load_or_fail Nx_io.load_gguf ~upto:300 (read path)));
    ]

let npz =
  group "load_npz"
    [
      test
        "a deflated entry that declares more data than deflate can expand to \
         fails"
        (limited (fun () ->
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
                 Nx_io.load_npz path)));
    ]

let pytorch =
  group "load_pytorch"
    [
      cases "byte counts that pass an int's range are refused" ~id:fst
        [
          ("an offset times a dtype's width", ("pt_offset_overflow.pt", "past"));
          ( "a storage's count times its width",
            ("pt_count_overflow.pt", "more bytes") );
        ]
        (fun (_, (name, naming)) ->
          limited
            (fun () ->
              let path = fixture name in
              fails ~path ~naming (fun () -> Nx_io.load_pytorch path))
            ());
    ]

let hdf5 =
  group "Hdf5"
    [
      test "values and chunk of chunks whose bytes pass any integer raise"
        (limited (fun () ->
             let d = dataset "hdf5_damaged.h5" "/h" in
             fails ~path:(fixture "hdf5_damaged.h5") ~naming:"/h: " (fun () ->
                 H.values d);
             fails ~path:(fixture "hdf5_damaged.h5") ~naming:"/h: " (fun () ->
                 H.chunk d)));
      test "rows of windows of more than 2^53 elements are refused"
        (limited (fun () ->
             let d = dataset "hdf5_huge.h5" "/huge" in
             let half = 1 lsl 27 in
             raises_match
               (Exn.invalid_arg
                  ~sub:
                    "Nx_io.Hdf5.rows: the windows hold more than 2^53 elements")
               (fun () -> H.rows d ~row:[| 0; 0 |] ~rows:[| half; half |])));
    ]

let () =
  limit := Nx_limit_support.limit_memory room;
  exit (run "nx.io hostile sizes" [ npz; gguf; pytorch; hdf5 ])
