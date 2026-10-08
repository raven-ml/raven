(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host loader over the fixtures' programs, whose sources say what they
   do. *)

open Windtrap
module Host = Device_host
module S = Device_host_support

let strf = Printf.sprintf
let timeout = 60.
let obj f = S.fixture ~dir:"fixtures" f
let link f = require_ok (Host.link ~entry:f (obj f))

(* [Host.call] on the buffers [bs], which it keeps alive until it returns. *)
let call ?split p (bs : S.words list) values =
  Host.call ?split p (Array.of_list (List.map S.address bs)) values;
  ignore (Sys.opaque_identity bs)

(* [f ()], run on a domain of its own, failing the test with [what] if it has
   not returned after 10 s: a call that waits forever fails the test instead of
   hanging it. *)
let finishes what f =
  let result = Atomic.make None in
  let d =
    Domain.spawn (fun () ->
        Atomic.set result (Some (try Ok (f ()) with e -> Error e)))
  in
  let until = Unix.gettimeofday () +. 10. in
  while Atomic.get result = None && Unix.gettimeofday () < until do
    Unix.sleepf 0.001
  done;
  match Atomic.get result with
  | None -> failf "%s did not finish within 10 s" what
  | Some r -> (
      Domain.join d;
      match r with Ok v -> v | Error e -> raise e)

(* Programs *)

let affine = lazy (link "affine")

let test_affine (n, a, c, input) =
  cover "no element" (n = 0);
  cover "one element" (n = 1);
  let p = Lazy.force affine in
  let inb = S.words n and out = S.words (n + 1) in
  Array.iteri (fun i x -> inb.{i} <- Int64.of_int x) input;
  out.{n} <- 7L;
  call p [ out; inb ] [| n; a; c |];
  let expected = Array.map (fun x -> (a * x) + c) input in
  let actual = Array.init n (fun i -> Int64.to_int out.{i}) in
  equal (array int) expected actual;
  equal ~msg:"the word past the last" int64 7L out.{n}

let affine_case =
  let open Gen in
  let* n = frequency [ (1, of_list [ 0; 1 ]); (4, int_range 0 64) ] in
  let+ a = small_int
  and+ c = small_int
  and+ input = array ~size:(constant n) small_int in
  (n, a, c, input)

let test_linked () =
  let p = link "linked" in
  let out = Bigarray.(Array1.create float64 c_layout 3)
  and inb = Bigarray.(Array1.of_array float64 c_layout [| 27.; 1.25 |]) in
  Host.call p [| S.address out; S.address inb |] [| 2 |];
  ignore (Sys.opaque_identity inb);
  equal ~msg:"cbrt 27" float_exact 3. out.{0};
  equal ~msg:"table.(2)" float_exact 3.125 out.{1};
  equal ~msg:"twice 1.25" float_exact 2.5 out.{2}

let test_got () =
  let p = link "got" in
  let out = Bigarray.(Array1.create float64 c_layout 1)
  and inb = Bigarray.(Array1.of_array float64 c_layout [| -8. |]) in
  Host.call p [| S.address out; S.address inb |] [||];
  ignore (Sys.opaque_identity inb);
  equal ~msg:"cbrt (-8)" float_exact (-2.) out.{0}

let program_tests =
  group ~timeout "programs"
    [
      prop "a program reads its values and writes its buffers" affine_case
        test_affine;
      test
        "a program calls the process's cbrt and its own function, and reads \
         its constants, with unwind tables"
        test_linked;
      test "a program calls a function whose address it reads from a word"
        test_got;
    ]

(* Splits *)

let blocks = lazy (link "blocks")
let nested = lazy (link "nested")

(* A split's blocks by its type's doc: block [i] of [b = min blocks extent]
   starts at [i * extent / b]. *)
let bounds ~extent ~blocks =
  let b = min blocks extent in
  List.init (b + 1) (fun i -> i * extent / b)

type case = { split : Host.split; values : int array }

let pp_case ppf { split = s; values } =
  Format.fprintf ppf "extent %d, blocks %d, lo %d, hi %d, values [%s]" s.extent
    s.blocks s.lo s.hi
    (String.concat "; " (Array.to_list (Array.map string_of_int values)))

let split_case ~min_values =
  let open Gen in
  let* extent =
    frequency
      [ (1, of_list [ 0; 1 ]); (3, int_range 0 100); (1, int_range 0 10_000) ]
  in
  let* blocks =
    frequency
      [
        (1, constant 1);
        (3, int_range 1 64);
        (1, int_range (extent + 1) (extent + 3));
      ]
  in
  let* n = int_range min_values 40 in
  let* lo = int_range 0 (n - 1) in
  let* hi = such_that (fun hi -> hi <> lo) (int_range 0 (n - 1)) in
  let+ values = array ~size:(constant n) small_int in
  { split = { extent; blocks; lo; hi }; values }

let split_case ~min_values = Gen.with_pp pp_case (split_case ~min_values)

(* The buffers of blocks.c: its counts, a log with room for [records] records,
   and its meta. *)
let blocks_buffers ~extent ~n ~offset ~lo ~hi ~records =
  let counts = S.words extent and log = S.words (2 + (records * (n + 2))) in
  log.{1} <- Int64.of_int (Bigarray.Array1.dim log);
  let meta = S.words 4 in
  List.iteri (fun i x -> meta.{i} <- Int64.of_int x) [ lo; hi; offset; n ];
  (counts, log, meta)

let records log ~n =
  let len = Int64.to_int log.{0} in
  List.init
    (len / (n + 2))
    (fun r ->
      let at = 2 + (r * (n + 2)) in
      Array.init (n + 2) (fun i -> Int64.to_int log.{at + i}))

(* Every iteration ran once, each call's range is whole consecutive blocks, and
   each call saw the values with its range at [lo] and [hi]. *)
let check_split { split = s; values } (counts, log, _) =
  let n = Array.length values in
  for i = 0 to s.extent - 1 do
    equal ~msg:(strf "the runs of iteration %d" i) int64 1L counts.{i}
  done;
  let bounds =
    if s.extent = 0 then [] else bounds ~extent:s.extent ~blocks:s.blocks
  in
  let check r =
    let first = r.(0) and last = r.(1) in
    mem ~msg:"a call's first iteration starts a block" int first bounds;
    mem ~msg:"a call's end ends a block" int last bounds;
    less ~msg:"a call's range" int ~than:last first;
    let expected =
      Array.mapi
        (fun i v -> if i = s.lo then first else if i = s.hi then last else v)
        values
    in
    equal ~msg:"the values a call saw" (array int) expected (Array.sub r 2 n)
  in
  List.iter check (records log ~n);
  if s.extent = 0 then
    equal ~msg:"the calls of a split of no iteration" int 0
      (List.length (records log ~n))

let covers { split = s; _ } =
  cover "no iteration" (s.extent = 0);
  cover "one iteration" (s.extent = 1);
  cover "one block" (s.blocks = 1);
  cover "more blocks than iterations" (s.blocks > s.extent && s.extent > 0)

let test_split ({ split = s; values } as c) =
  covers c;
  let p = Lazy.force blocks and n = Array.length values in
  let ((counts, log, meta) as bufs) =
    blocks_buffers ~extent:s.extent ~n ~offset:(-1) ~lo:s.lo ~hi:s.hi
      ~records:s.blocks
  in
  call ~split:s p [ counts; log; meta ] values;
  check_split c bufs

(* blocks.c through nested.c: split by [device_host_call], or unsplit, with its
   range given as [0, extent). *)
let test_entry (({ split = s; values } as c), split) =
  covers c;
  let n = Array.length values in
  let values =
    if split then values
    else
      Array.mapi
        (fun i v -> if i = s.lo then 0 else if i = s.hi then s.extent else v)
        values
  in
  let ((counts, log, meta) as bufs) =
    blocks_buffers ~extent:s.extent ~n ~offset:(-1) ~lo:s.lo ~hi:s.hi
      ~records:s.blocks
  in
  let inner =
    if split then [ s.extent; s.blocks; s.lo; s.hi ] else [ -1; 0; 0; 0 ]
  in
  let nested_values =
    Array.concat
      [
        [| Host.address (Lazy.force blocks); n |];
        Array.of_list inner;
        [| 0; 0; 0 |];
        values;
      ]
  in
  finishes "the call" (fun () ->
      call (Lazy.force nested) [ counts; log; meta ] nested_values);
  if split then check_split { split = s; values } bufs
  else begin
    equal ~msg:"the iterations of the unsplit call" (array int64)
      (Array.make s.extent 1L)
      (Array.init s.extent (fun i -> counts.{i}));
    equal ~msg:"the values of the unsplit call"
      (list (array int))
      [ Array.append [| 0; s.extent |] values ]
      (records log ~n)
  end

(* An outer split of nested.c, each block of which splits blocks.c over its own
   iterations, offset by its first. *)
let test_split_in_block { split = s; values } =
  let n = Array.length values in
  let offset = List.find (fun i -> i <> s.lo && i <> s.hi) [ 0; 1; 2 ] in
  let counts, log, meta =
    blocks_buffers ~extent:s.extent ~n ~offset ~lo:s.lo ~hi:s.hi
      ~records:(s.blocks * s.blocks)
  in
  let nested_values =
    Array.concat
      [
        [| Host.address (Lazy.force blocks); n; 0; s.blocks; s.lo; s.hi |];
        [| 0; 0; 1 |];
        values;
      ]
  in
  let outer = { Host.extent = s.extent; blocks = s.blocks; lo = 6; hi = 7 } in
  finishes "the split" (fun () ->
      call ~split:outer (Lazy.force nested) [ counts; log; meta ] nested_values);
  equal ~msg:"the runs of each iteration" (array int64) (Array.make s.extent 1L)
    (Array.init s.extent (fun i -> counts.{i}))

let test_refused (_, split, n) =
  let p = Lazy.force blocks in
  raises_match (Exn.invalid_arg ~substring:"Device_host.call: ") (fun () ->
      Host.call ~split p [||] (Array.make n 0))

let refusals =
  let s extent blocks lo hi = { Host.extent; blocks; lo; hi } in
  [
    ("a negative extent", s (-1) 1 0 1, 2);
    ("no block", s 4 0 0 1, 2);
    ("a negative lo", s 4 1 (-1) 1, 2);
    ("lo past the values", s 4 1 2 1, 2);
    ("hi past the values", s 4 1 0 2, 2);
    ("lo equal to hi", s 4 1 1 1, 2);
    ("no values", s 4 1 0 1, 0);
  ]

let split_tests =
  group ~timeout "splits"
    [
      prop "a split runs every iteration once, in calls of whole blocks"
        (split_case ~min_values:2) test_split;
      prop
        "device_host_call runs a split as call does, and an unsplit call once"
        (Gen.pair (split_case ~min_values:2) Gen.bool)
        test_entry;
      prop ~count:30 "a split begun inside a block of a split completes"
        (split_case ~min_values:3) test_split_in_block;
      cases
        ~name:(fun (n, _, _) -> n)
        "a split is refused with" refusals test_refused;
    ]

(* Domains *)

(* A split of blocks.c, and whether it ran every iteration once. *)
let split_once (extent, blocks_) =
  let counts, log, meta =
    blocks_buffers ~extent ~n:2 ~offset:(-1) ~lo:0 ~hi:1 ~records:blocks_
  in
  call
    ~split:{ extent; blocks = blocks_; lo = 0; hi = 1 }
    (Lazy.force blocks) [ counts; log; meta ] [| 0; 0 |];
  Array.for_all (fun c -> c = 1L) (Array.init extent (fun i -> counts.{i}))

(* A job of the host's pool, and whether it ran every unit once. *)
let job_once (total, chunks) =
  let counts = S.words total in
  S.count_job ~threads:(Host.workers ()) ~total ~chunks counts;
  Array.for_all (fun c -> c = 1L) (Array.init total (fun i -> counts.{i}))

let job =
  Gen.with_pp
    (fun ppf (total, chunks) ->
      Format.fprintf ppf "%d units, %d blocks" total chunks)
    (Gen.pair (Gen.int_range 0 2000) (Gen.int_range 1 64))

let domain_commands =
  [
    command "split" (job @-> returns bool) (fun _ -> true) split_once;
    command "pool job" (job @-> returns bool) (fun _ -> true) job_once;
  ]

(* Sets w[0] once w[1] is set, then whether the call had returned. *)
let test_released () =
  let p = link "waiting" in
  let w = S.words 2 in
  let returned = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        call p [ w ] [| 20_000_000_000 |];
        Atomic.set returned true)
  in
  let until = Unix.gettimeofday () +. 10. in
  while w.{1} = 0L && Unix.gettimeofday () < until do
    Unix.sleepf 0.001
  done;
  equal ~msg:"the program began" int64 1L w.{1};
  Gc.full_major ();
  let ran = not (Atomic.get returned) in
  w.{0} <- 1L;
  Domain.join d;
  equal ~msg:"the program ran while the domain collected" bool true ran

let domain_tests =
  group ~timeout "domains"
    [
      stateful ~domains:2 ~count:30
        "splits and pool jobs from two domains each run every unit once"
        domain_commands;
      test "the runtime is released while a program runs" test_released;
    ]

(* Code memory *)

let collect () =
  for _ = 1 to 3 do
    Gc.full_major ()
  done

let test_unmapped () =
  let address = Host.address (link "empty") in
  equal ~msg:"mapped while reachable" bool true (S.mapped address);
  collect ();
  equal ~msg:"mapped once unreachable" bool false (S.mapped address)

let test_survives () =
  let p = link "affine" in
  collect ();
  let out = S.words 1 and inb = S.words 1 in
  inb.{0} <- 5L;
  call p [ out; inb ] [| 1; 3; 1 |];
  equal int64 16L out.{0};
  equal ~msg:"mapped" bool true (S.mapped (Host.address p))

(* The permissions of the mapping of /proc/self/maps that holds [a]. *)
let permissions a =
  let holds line =
    match String.split_on_char ' ' line with
    | range :: perms :: _ -> (
        match String.split_on_char '-' range with
        | [ lo; hi ] ->
            let lo = int_of_string ("0x" ^ lo)
            and hi = int_of_string ("0x" ^ hi) in
            if lo <= a && a < hi then Some perms else None
        | _ -> None)
    | _ -> None
  in
  In_channel.with_open_text "/proc/self/maps" In_channel.input_lines
  |> List.find_map holds

let test_never_writable () =
  if not (Sys.file_exists "/proc/self/maps") then
    skip ~reason:"needs /proc/self/maps (Linux)" ();
  let p = link "affine" in
  equal (option string) (Some "r-xp") (permissions (Host.address p))

let memory_tests =
  group ~timeout "code memory"
    [
      test "an unreachable program's code is unmapped" test_unmapped;
      test "a reachable program's code survives collections and runs"
        test_survives;
      test "the code is never writable once linked" test_never_writable;
    ]

(* Refusals *)

let other = if S.machine = "x86_64" then "aarch64" else "x86_64"
let host_name = if S.machine = "x86_64" then "x86_64" else "arm64"

let other_machine =
  if other = "x86_64" then "machine 62 (x86_64)" else "machine 183 (arm64)"

let read f = In_channel.with_open_bin f In_channel.input_all

let refused =
  [
    ( "an object of another machine",
      "affine",
      read (strf "fixtures/affine_%s.o" other),
      strf "the object is for %s, expected %s" other_machine host_name );
    ("writable data", "writable", obj "writable", "section .bss is writable");
    ( "an undefined symbol",
      "undefined",
      obj "undefined",
      "symbol \"device_host_test_nowhere\" is defined by no library of the \
       process" );
    ( "a data symbol as the entry",
      "data_entry",
      obj "data_entry",
      "entry \"data_entry\" is no symbol of an executable section" );
    ( "a missing entry",
      "nowhere",
      obj "affine",
      "entry \"nowhere\" is no symbol of an executable section" );
  ]

(* [obj] with the header of its first section of type [kind], at [h], set by
   [set b h]. The ELF64 header holds [e_shoff] at 0x28, [e_shentsize] at 0x3a
   and [e_shnum] at 0x3c; a section header its type at 4. *)
let patch_section obj ~kind set =
  let b = Bytes.of_string obj in
  let shoff = Int64.to_int (Bytes.get_int64_le b 0x28) in
  let entsize = Bytes.get_uint16_le b 0x3a and n = Bytes.get_uint16_le b 0x3c in
  let header i = shoff + (i * entsize) in
  let kind_of i = Int32.to_int (Bytes.get_int32_le b (header i + 4)) in
  set b (header (List.find (fun i -> kind_of i = kind) (List.init n Fun.id)));
  Bytes.to_string b

let sht_progbits = 1
let sht_rela = 4
let sht_rel = 9
let set64 b at x = Bytes.set_int64_le b at (Int64.of_int x)

(* got's first relocation section as an SHT_REL one of its first entry: an ELF64
   Rel entry is a Rela entry without its addend, 16 bytes. *)
let test_rel () =
  let at = ref 0 in
  let o =
    patch_section (obj "got") ~kind:sht_rela (fun b h ->
        at :=
          Int64.to_int
            (Bytes.get_int64_le b
               (Int64.to_int (Bytes.get_int64_le b (h + 24))));
        Bytes.set_int32_le b (h + 4) (Int32.of_int sht_rel);
        set64 b (h + 32) 16;
        set64 b (h + 56) 16)
  in
  equal string
    (strf "relocation at 0x%x has its addend in its field (SHT_REL)" !at)
    (require_error (Host.link ~entry:"got" o))

(* affine's code, aligned to 1 MiB, above any system's page. *)
let test_aligned () =
  let o =
    patch_section (obj "affine") ~kind:sht_progbits (fun b h ->
        set64 b (h + 48) (1 lsl 20))
  in
  starts_with
    ~affix:"a section asks for an alignment of 1048576 bytes, above the page's "
    (require_error (Host.link ~entry:"affine" o))

let test_refusal (_, entry, o, msg) =
  equal string msg (require_error (Host.link ~entry o))

let refusal_tests =
  group ~timeout "refusals"
    [
      test "a link of what is not ELF is refused" (fun () ->
          is_error (Host.link ~entry:"f" "not an object"));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "a link is refused for" refused test_refusal;
      test "a link of relocations whose addends are in their fields is refused"
        test_rel;
      test "a link of code aligned above the page is refused" test_aligned;
    ]

let () =
  exit
    (run "device_host"
       [ program_tests; split_tests; domain_tests; memory_tests; refusal_tests ])
