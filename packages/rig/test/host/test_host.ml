(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host's code: images linked from the fixtures' objects, whose sources
   say what they do, and launches of their functions on Rig.host. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module Run = Sub.Run
module S = Rig_host_support
module R = Rig_support.Reader
module P = Rig_support.Polled

let strf = Printf.sprintf
let timeout = 60.

(* The code other suites also run lives in test/fixtures. *)
let obj f =
  match f with
  | "affine" | "calling" | "waiting" -> S.fixture ~dir:"../fixtures" f
  | _ -> S.fixture ~dir:"fixtures" f

let load f =
  require_ok ~pp:Format.pp_print_string (Rig.Image.load Rig.host (obj f))

(* A launch's grid: its groups and the threads of each, along x, y and z. *)
type grid = { groups : int * int * int; threads : int * int * int }

let one = { groups = (1, 1, 1); threads = (1, 1, 1) }

(* The point of one launch of [image]'s [f] on the host over [buffers], with
   [params] bytes of parameters that [set] stores. *)
let launch ?(refs = [||]) ?(params = 0) ?(shared = 0) ?(set = fun _ _ -> ())
    ?(grid = one) image f buffers =
  let s = Sub.make Rig.host [| Sub.launch ~refs image f ~params |] in
  let run = Run.make () and k = Sub.block s 0 in
  let gx, gy, gz = grid.groups and tx, ty, tz = grid.threads in
  Run.groups run k gx gy gz;
  Run.threads run k tx ty tz;
  Run.shared run k shared;
  set run k;
  Rig.submit s ~run ~buffers

let words = Array.map B.of_bigarray

(* Launches *)

let affine = lazy (load "affine")

(* affine.c over [n] elements, in groups of [t] threads. *)
let affine_launch ~n ~t ~a ~c out inb =
  let set run k =
    Run.int64 run k 16 n;
    Run.int64 run k 24 a;
    Run.int64 run k 32 c
  in
  launch (Lazy.force affine) "affine"
    ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
    ~params:40 ~set
    ~grid:{ groups = ((n + t - 1) / t, 1, 1); threads = (t, 1, 1) }
    (words [| out; inb |])

let test_affine (n, t, a, c, input) =
  cover "one element" (n = 1);
  cover "a last group in part past n" (n mod t <> 0);
  let inb = S.words n and out = S.words (n + 1) in
  Array.iteri (fun i x -> inb.{i} <- Int64.of_int x) input;
  out.{n} <- 7L;
  let p = affine_launch ~n ~t ~a ~c out inb in
  equal ~msg:"reached when submit returns" bool true (Rig.Point.reached p);
  let expected = Array.map (fun x -> (a * x) + c) input in
  equal (array int) expected (Array.init n (fun i -> Int64.to_int out.{i}));
  equal ~msg:"the word past the last" int64 7L out.{n}

let affine_case =
  let open Gen in
  let* n = frequency [ (1, constant 1); (4, int_range 1 200) ] in
  let+ t = int_range 1 9
  and+ a = small_int
  and+ c = small_int
  and+ input = array ~size:(constant n) small_int in
  (n, t, a, c, input)

let test_linked () =
  let out = Bigarray.(Array1.create float64 c_layout 3)
  and inb = Bigarray.(Array1.of_array float64 c_layout [| 27.; 1.25 |]) in
  ignore
    (launch (load "linked") "linked"
       ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
       ~params:24
       ~set:(fun run k -> Run.int64 run k 16 2)
       [| B.of_bigarray out; B.of_bigarray inb |]);
  equal ~msg:"cbrt 27" float_exact (Float.cbrt 27.) out.{0};
  equal ~msg:"table.(2)" float_exact 3.125 out.{1};
  equal ~msg:"twice 1.25" float_exact 2.5 out.{2}

(* The host runs work only after the work it follows: a launch that reads
   memory a Polled device's unrun launch writes waits for that work. *)
let test_follows () =
  let d, pd = P.open_ "host:follows" in
  let shared = B.create Rig.host 65536 in
  let functions = require_ok (Rig.Image.load d "functions") in
  let s = Sub.make d [| Sub.launch ~refs:Sub.[| write 0 ~at:0 |] functions "main" ~params:8 |] in
  let run = Run.make () and k = Sub.block s 0 in
  Run.groups run k 1 1 1;
  Run.threads run k 1 1 1;
  let written =
    Rig.submit s ~run ~buffers:[| require_some (B.borrow d shared) |]
  in
  equal ~msg:"queued before" int 1 (P.queued pd);
  let out = S.words 1 in
  ignore
    (launch (Lazy.force affine) "affine"
       ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
       ~params:40
       ~set:(fun run k -> Run.int64 run k 16 1)
       [| B.of_bigarray out; shared |]);
  equal ~msg:"the Polled work it follows" bool true (Rig.Point.reached written)

(* A profile taken around a launch records its span, named after its function:
   [take] reads the host's times before it returns. *)
let test_profiled () =
  let out = S.words 4 and inb = S.words 4 in
  let _, events =
    Rig.Profile.take (fun () -> affine_launch ~n:4 ~t:2 ~a:1 ~c:0 out inb)
  in
  equal (list string) [ "affine" ]
    (List.filter_map
       (function
         | Rig.Profile.Span s when Rig.equal s.device Rig.host -> Some s.name
         | _ -> None)
       events)

(* deep.c over [groups] groups of one thread, each on whichever thread the
   pool gives it, the caller's included: each element of its output. *)
let deep_launch groups =
  let out = S.words groups in
  ignore
    (launch (load "deep") "deep"
       ~refs:Sub.[| write 0 ~at:0 |]
       ~params:16
       ~set:(fun run k -> Run.int64 run k 8 0)
       ~grid:{ one with groups = (groups, 1, 1) }
       (words [| out |]));
  Array.init groups (fun i -> Int64.to_int out.{i})

let test_deep () =
  List.iter
    (fun groups ->
      equal ~msg:(strf "%d groups" groups) (array int) (Array.make groups 2)
        (deep_launch groups))
    [ 1; 4 * S.workers () ]

let program_tests =
  group ~timeout "launches"
    [
      prop "a launch runs its function on its parameters over every thread"
        affine_case test_affine;
      test "a profile records a launch's span" test_profiled;
      test
        "a function calls the process's cbrt and its own function, and reads \
         its constants"
        test_linked;
      test "a launch follows the work on the memory its refs name" test_follows;
      test
        "a function whose stack frame spans 256 KiB, written at its far end \
         first, runs on every thread"
        test_deep;
    ]

(* Splits *)

let pp_grid ppf { groups = gx, gy, gz; threads = tx, ty, tz } =
  Format.fprintf ppf "groups %dx%dx%d, threads %dx%dx%d" gx gy gz tx ty tz

let grid_case =
  let open Gen in
  let axis n = frequency [ (2, constant 1); (3, int_range 1 n) ] in
  let+ gx = frequency [ (1, constant 1); (3, int_range 1 200) ]
  and+ gy = axis 4
  and+ gz = axis 3
  and+ tx = axis 9
  and+ ty = axis 3
  and+ tz = axis 2 in
  { groups = (gx, gy, gz); threads = (tx, ty, tz) }

let grid_case = Gen.with_pp pp_grid grid_case
let size (x, y, z) = x * y * z

(* The records of blocks.c's log: each call's first thread and its end. *)
let records log =
  List.init
    (Int64.to_int log.{0} / 2)
    (fun r -> (Int64.to_int log.{2 + (2 * r)}, Int64.to_int log.{3 + (2 * r)}))

(* blocks.c over [g]: its counts and the calls it logged. Each run loads it,
   so that runs on two domains share no image. *)
let run_blocks g =
  let n = size g.groups * size g.threads in
  let counts = S.words n and log = S.words (2 + (2 * size g.groups)) in
  log.{1} <- Int64.of_int (Bigarray.Array1.dim log);
  ignore
    (launch (load "blocks") "blocks"
       ~refs:Sub.[| write 0 ~at:0; write 1 ~at:8 |]
       ~params:16 ~grid:g
       (words [| counts; log |]));
  (counts, records log)

(* Every thread ran once, and each call's range is whole consecutive groups. *)
let test_split g =
  let groups = size g.groups and t = size g.threads in
  cover "one group" (groups = 1);
  cover "one thread a group" (t = 1);
  cover "more groups than workers" (groups > S.workers ());
  let n = groups * t in
  let counts, calls = run_blocks g in
  equal ~msg:"each thread's runs" (array int64) (Array.make n 1L)
    (Array.init n (fun i -> counts.{i}));
  List.iter
    (fun (first, end_) ->
      equal ~msg:"a call's first thread starts a group" int 0 (first mod t);
      equal ~msg:"a call's end ends a group" int 0 (end_ mod t);
      less ~msg:"a call's range" int ~than:end_ first)
    calls;
  at_most ~msg:"calls" int ~than:groups (List.length calls)

let max_axis = (1 lsl 32) - 1

let refusals =
  [
    ("shared memory", 1, { one with groups = (1, 1, 1) });
    ("an empty grid", 0, { one with groups = (1, 0, 1) });
    ("an empty group", 0, { one with threads = (1, 1, 0) });
    ("more than 2^62 threads", 0, { one with groups = (max_axis, max_axis, 1) });
  ]

let test_refused (_, shared, grid) =
  let empty = load "empty" in
  raises_match Exn.invalid_arg (fun () ->
      launch ~shared ~grid empty "empty" [||])

let test_other_parts () =
  let b = B.create Rig.host 8 and b' = B.create Rig.host 8 in
  List.iter
    (fun (msg, part) ->
      raises_match ~msg Exn.invalid_arg (fun () ->
          Sub.make Rig.host [| part |]))
    [
      ("a copy", Sub.(copy ~src:(Buffer b) ~dst:(Buffer b')));
      ("a fill", Sub.fill 0n ~params:0);
      ("words", Sub.words b);
    ]

let split_tests =
  group ~timeout "splits"
    [
      prop "a launch runs every thread once, in calls of whole groups"
        grid_case test_split;
      cases
        ~name:(fun (n, _, _) -> n)
        "a launch never runs with" refusals test_refused;
      test "the host runs no copy, fill or words" test_other_parts;
    ]

(* Domains *)

(* A launch of blocks.c, and whether it ran every thread once. *)
let launch_once g =
  let counts, _ = run_blocks g in
  Array.for_all (fun c -> c = 1L)
    (Array.init (Bigarray.Array1.dim counts) (fun i -> counts.{i}))

(* A job of the host's pool, and whether it ran every unit once. *)
let job_once (total, chunks) =
  let counts = S.words total in
  S.count_job ~threads:(S.workers ()) ~total ~chunks counts;
  Array.for_all (fun c -> c = 1L) (Array.init total (fun i -> counts.{i}))

let job =
  Gen.with_pp
    (fun ppf (total, chunks) ->
      Format.fprintf ppf "%d units, %d chunks" total chunks)
    (Gen.pair (Gen.int_range 0 256) (Gen.int_range 1 64))

(* Grids of at most 64 groups of 4 threads: the commands race, so a few
   groups per worker suffice. *)
let small_grid =
  let open Gen in
  let+ gx = int_range 1 64 and+ tx = int_range 1 4 in
  { groups = (gx, 1, 1); threads = (tx, 1, 1) }

let small_grid = Gen.with_pp pp_grid small_grid

let domain_commands =
  [
    command "launch" (small_grid @-> returns bool) (fun _ -> true) launch_once;
    command "pool job" (job @-> returns bool) (fun _ -> true) job_once;
  ]

(* waiting.c's launch over [w] and [b], spinning until w[0] is set. *)
let waiting w ~rounds =
  launch (load "waiting") "waiting" ~refs:Sub.[| write 0 ~at:0 |] ~params:16
    ~set:(fun run k -> Run.int64 run k 8 rounds)
    [| w |]

(* Sets w[0] once w[1] is set, then whether the launch had returned. *)
let test_released () =
  let w = S.words 2 in
  let returned = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        ignore (waiting (B.of_bigarray w) ~rounds:20_000_000_000);
        Atomic.set returned true)
  in
  Rig_support.await "the launch running" (fun () -> w.{1} = 1L);
  Gc.full_major ();
  let ran = not (Atomic.get returned) in
  w.{0} <- 1L;
  Domain.join d;
  equal ~msg:"the launch ran while the domain collected" bool true ran

(* A launch keeps its buffers until it returns: waiting.c runs on memory only
   the submit reaches while another domain collects, and a buffer made
   meanwhile gets other memory. *)
let test_keeps_buffers () =
  let n = 1 lsl 16 in
  let at = Atomic.make 0 in
  let d =
    Domain.spawn (fun () ->
        let b = B.create Rig.host n in
        let a = R.host b in
        Rig_support.store a 0;
        Rig_support.store (a + 8) 0;
        Atomic.set at a;
        ignore (waiting b ~rounds:20_000_000_000))
  in
  Rig_support.await "the launch running" (fun () ->
      Atomic.get at <> 0 && Rig_support.load (Atomic.get at + 8) = 1);
  let a = Atomic.get at in
  Gc.full_major ();
  Gc.full_major ();
  let reused = R.host (B.create Rig.host n) = a in
  Rig_support.store a 1;
  Domain.join d;
  equal ~msg:"its memory reused while the launch ran" bool false reused

let domain_tests =
  group ~timeout "domains"
    [
      test "a launch keeps its buffers until it returns" test_keeps_buffers;
      stateful ~domains:2 ~count:30
        "launches and pool jobs from two domains each run every unit once"
        domain_commands;
      test "the runtime is released while a launch runs" test_released;
    ]

(* Code memory *)

(* Collects, then drains the host's releases as an allocation does. *)
let collect () =
  for _ = 1 to 3 do
    Gc.full_major ();
    ignore (B.create Rig.host 8)
  done

let test_unmapped () =
  let address =
    (fun () -> require_some (Rig.Image.entry (load "empty") "empty")) ()
  in
  collect ();
  equal ~msg:"unmapped once unreachable" bool false (S.executable address)

let test_survives () =
  let image = load "affine" in
  collect ();
  let out = S.words 1 and inb = S.words 1 in
  inb.{0} <- 5L;
  ignore
    (launch image "affine"
       ~refs:Sub.[| write 0 ~at:0; read 1 ~at:8 |]
       ~params:40
       ~set:(fun run k ->
         Run.int64 run k 16 1;
         Run.int64 run k 24 3;
         Run.int64 run k 32 1)
       (words [| out; inb |]));
  equal int64 16L out.{0};
  equal ~msg:"mapped" bool true
    (S.executable (require_some (Rig.Image.entry image "affine")))

(* waiting.c, which nothing holds but its running submit, while another
   domain collects. *)
let test_running () =
  let w = S.words 2 in
  let address = Atomic.make 0 in
  let d =
    Domain.spawn (fun () ->
        let image = load "waiting" in
        Atomic.set address (require_some (Rig.Image.entry image "waiting"));
        let s =
          Sub.make Rig.host
            [| Sub.launch ~refs:Sub.[| write 0 ~at:0 |] image "waiting" ~params:16 |]
        in
        let run = Run.make () and k = Sub.block s 0 in
        Run.groups run k 1 1 1;
        Run.threads run k 1 1 1;
        Run.int64 run k 8 20_000_000_000;
        ignore (Rig.submit s ~run ~buffers:[| B.of_bigarray w |]))
  in
  Rig_support.await "the launch running" (fun () -> w.{1} = 1L);
  collect ();
  let mapped = S.executable (Atomic.get address) in
  w.{0} <- 1L;
  Domain.join d;
  equal ~msg:"mapped while its launch runs" bool true mapped

(* A submission of calling.c, which calls waiting.c's function through an
   entry ref, with the function's address in [address]. Only the submission
   keeps waiting.c's image. *)
let calls_waiting address =
  let callee = load "waiting" in
  address := require_some (Rig.Image.entry callee "waiting");
  let refs = Sub.[| entry callee "waiting" ~at:0; write 0 ~at:8 |] in
  Sub.make Rig.host [| Sub.launch ~refs (load "calling") "calling" ~params:24 |]

(* The callee lives from make to submit and while the launch that calls it
   runs and another domain collects. Once the submission is unreachable and
   its work done, the callee is unmapped. *)
let test_keeps_callee () =
  let w = S.words 2 in
  let address = ref 0 in
  let s = calls_waiting address in
  collect ();
  equal ~msg:"mapped while a submission names it" bool true
    (S.executable !address);
  let d =
    Domain.spawn (fun () ->
        let run = Run.make () and k = Sub.block s 0 in
        Run.groups run k 1 1 1;
        Run.threads run k 1 1 1;
        Run.int64 run k 16 20_000_000_000;
        ignore (Rig.submit s ~run ~buffers:[| B.of_bigarray w |]))
  in
  Rig_support.await "the callee running" (fun () -> w.{1} = 1L);
  collect ();
  let mapped = S.executable !address in
  w.{0} <- 1L;
  Domain.join d;
  equal ~msg:"mapped while a launch calls it" bool true mapped;
  collect ();
  equal ~msg:"unmapped once its work is done and unreachable" bool false
    (S.executable !address)

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
  let image = load "affine" in
  let perms = permissions (require_some (Rig.Image.entry image "affine")) in
  ignore (Sys.opaque_identity image);
  equal (option string) (Some "r-xp") perms

let memory_tests =
  group ~timeout "code memory"
    [
      test "an unreachable image's code is unmapped" test_unmapped;
      test "a reachable image's code survives collections and runs"
        test_survives;
      test "an image's code survives collections while a launch of it runs"
        test_running;
      test "a callee in another image lives while a submission names it"
        test_keeps_callee;
      test "the code is never writable once linked" test_never_writable;
    ]

(* Functions and refusals *)

let test_functions () =
  let image = load "data_entry" in
  equal ~msg:"a function" bool true (Rig.Image.mem image "code");
  equal ~msg:"a data symbol" bool false (Rig.Image.mem image "data_entry");
  equal ~msg:"a missing name" bool false (Rig.Image.mem image "nowhere");
  raises_match ~msg:"a launch of a data symbol"
    (Exn.invalid_arg ~substring:"has no function \"data_entry\"") (fun () ->
      Sub.make Rig.host [| Sub.launch image "data_entry" ~params:0 |]);
  raises_match ~msg:"an entry ref to a data symbol"
    (Exn.invalid_arg ~substring:"has no function \"data_entry\"") (fun () ->
      let refs = [| Sub.entry image "data_entry" ~at:0 |] in
      Sub.make Rig.host [| Sub.launch ~refs image "code" ~params:8 |])

let other = if S.machine = "x86_64" then "aarch64" else "x86_64"
let host_name = if S.machine = "x86_64" then "x86_64" else "arm64"

let other_machine =
  if other = "x86_64" then "machine 62 (x86_64)" else "machine 183 (arm64)"

let read f = In_channel.with_open_bin f In_channel.input_all
let refused why = "CPU: " ^ why

let refusals =
  [
    ( "an object of another machine",
      read (strf "../fixtures/affine_%s.o" other),
      strf "the object is for %s, expected %s" other_machine host_name );
    ( "writable data",
      obj "writable",
      "section \".bss\" is writable and not empty; pass variables through the \
       parameters" );
    ( "an undefined symbol",
      obj "undefined",
      "symbol \"rig_host_test_nowhere\" is defined by no library of the process"
    );
  ]

let load_error o =
  require_error ~pp:(fun ppf _ -> Format.pp_print_string ppf "an image")
    (Rig.Image.load Rig.host o)

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

let sht_rela = 4
let sht_rel = 9
let set64 b at x = Bytes.set_int64_le b at (Int64.of_int x)

(* linked's first relocation section as an SHT_REL one of its first entry: an
   ELF64 Rel entry is a Rela entry without its addend, 16 bytes. *)
let test_rel () =
  let at = ref 0 in
  let o =
    patch_section (obj "linked") ~kind:sht_rela (fun b h ->
        at :=
          Int64.to_int
            (Bytes.get_int64_le b
               (Int64.to_int (Bytes.get_int64_le b (h + 24))));
        Bytes.set_int32_le b (h + 4) (Int32.of_int sht_rel);
        set64 b (h + 32) 16;
        set64 b (h + 56) 16)
  in
  equal string
    (refused
       (strf "relocation at 0x%x has its addend in its field (SHT_REL)" !at))
    (load_error o)

(* got's first relocation reads cbrt's address from a word of a global offset
   table: R_X86_64_REX_GOTPCRELX (42) or R_AARCH64_ADR_GOT_PAGE (311). *)
let test_got () =
  let kind = if S.machine = "x86_64" then 42 else 311 in
  starts_with
    ~affix:(refused (strf "relocation of type %d at 0x" kind))
    (load_error (obj "got"))

(* [obj] with the header of its section [name], at [h], set by [set b h]. The
   ELF64 header holds the index of the section of section names, [e_shstrndx],
   at 0x3e; a section header its name's offset in that section at 0 and its
   own offset at 24. *)
let patch_named obj name set =
  let b = Bytes.of_string obj in
  let shoff = Int64.to_int (Bytes.get_int64_le b 0x28) in
  let entsize = Bytes.get_uint16_le b 0x3a and n = Bytes.get_uint16_le b 0x3c in
  let header i = shoff + (i * entsize) in
  let names = header (Bytes.get_uint16_le b 0x3e) + 24 in
  let names = Int64.to_int (Bytes.get_int64_le b names) in
  let name_of i =
    let at = names + Int32.to_int (Bytes.get_int32_le b (header i)) in
    Bytes.sub_string b at (Bytes.index_from b at '\000' - at)
  in
  set b (header (List.find (fun i -> name_of i = name) (List.init n Fun.id)));
  Bytes.to_string b

(* An ELF64 section header holds its alignment, [sh_addralign], at 48. *)
let set_align align b h = set64 b (h + 48) align

(* Alignments at, around and above every host's page and Windows' allocation
   granularity of 64 KiB. *)
let alignments = [ 16; 4096; 16384; 65536; 1 lsl 17; 1 lsl 20 ]

(* aligned.c with its table's section aligned to [align]: the link places the
   table at a multiple of it, and the code reads it there. *)
let test_aligned align =
  let o = patch_named (obj "aligned") ".rodata" (set_align align) in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load Rig.host o)
  in
  let out = S.words 5 in
  ignore
    (launch image "aligned"
       ~refs:Sub.[| write 0 ~at:0 |]
       ~params:8
       ~grid:{ one with threads = (4, 1, 1) }
       (words [| out |]));
  equal ~msg:"the table's address modulo its alignment" int 0
    (Int64.to_int out.{0} land (align - 1));
  equal ~msg:"the table" (array int) [| 3; 1; 4; 1 |]
    (Array.init 4 (fun i -> Int64.to_int out.{1 + i}))

(* An alignment no address space holds: the system refuses the mapping, and
   the error names the alignment. *)
let test_unmappable () =
  let align = 1 lsl 61 in
  let why =
    load_error (patch_named (obj "aligned") ".rodata" (set_align align))
  in
  starts_with ~affix:"CPU: mapping " why;
  contains ~sub:(strf "at a multiple of %d bytes" align) why

let test_refusal (_, o, msg) = equal string (refused msg) (load_error o)

let refusal_tests =
  group ~timeout "loads"
    [
      test "an image's functions are the symbols of its code" test_functions;
      test "a load of what is not ELF is refused" (fun () ->
          ignore (load_error "not an object"));
      cases
        ~name:(fun (n, _, _) -> n)
        "a load is refused for" refusals test_refusal;
      test "a load of relocations whose addends are in their fields is refused"
        test_rel;
      test "a load of a relocation through a global offset table is refused"
        test_got;
      cases
        ~name:(strf "%d bytes")
        "a function reads data aligned to" alignments test_aligned;
      test "a load of an alignment no address space holds is refused"
        test_unmappable;
    ]

let () =
  exit
    (run "rig_host"
       [ program_tests; split_tests; domain_tests; memory_tests; refusal_tests ])
