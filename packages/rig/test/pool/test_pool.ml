(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The suite of rig_pool.h, through the probes of rig_pool_probe_stubs.c. *)

open Windtrap
module P = Rig_pool_probe

let strf = Printf.sprintf

(* A process this executable starts, to see the first call of rig_pool_cores
   under a pinned affinity. It must call nothing of the pool before, so this
   comes first. *)
let child_variable = "RIG_POOL_TEST_CHILD"

let () =
  match Sys.getenv_opt child_variable with
  | Some "affinity" ->
      let first, later = P.pinned_cores () in
      Printf.printf "%d %d\n%!" first later;
      exit 0
  | _ -> ()

let cores = P.cores ()

(* Jobs by rig_pool.h *)

(* c, the chunks of a job. *)
let chunk_count ~total ~chunks =
  if total <= 0L then 0L else Int64.min (Int64.max chunks 1L) total

(* t, the most threads of a job. *)
let thread_bound ~threads ~total ~chunks =
  let c = chunk_count ~total ~chunks in
  Int64.to_int
    (Int64.min
       (Int64.of_int (max threads 1))
       (Int64.min (Int64.of_int cores) c))

(* Chunk i is [floor (i * total / c), floor ((i + 1) * total / c)). With total =
   q c + r, floor (i * total / c) = i q + floor (i r / c), whose products stay
   below total and c * c. *)
let partition ~total ~chunks =
  let c = chunk_count ~total ~chunks in
  if c = 0L then []
  else
    let q = Int64.div total c and r = Int64.rem total c in
    let bound i = Int64.(add (mul i q) (div (mul i r) c)) in
    List.init (Int64.to_int c) (fun i ->
        let i = Int64.of_int i in
        (bound i, bound (Int64.succ i)))

(* The bounds of the chunks, floor (i * total / c) for 0 <= i <= c. *)
let chunk_bounds ~total ~chunks =
  match partition ~total ~chunks with
  | [] -> []
  | chunks -> 0L :: List.map snd chunks

(* [cut bounds ranges] is each of [ranges] cut at the [bounds] strictly inside
   it. Disjoint ranges of whole chunks that cover the units cut into the chunks,
   each once; a range that ends inside a chunk, or is empty, leaves a piece that
   is no chunk, and ranges that overlap leave a chunk twice. *)
let cut bounds ranges =
  let bounds = Array.of_list bounds in
  let n = Array.length bounds in
  (* The index of the first bound above [x]. *)
  let rec above x lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if bounds.(mid) > x then above x lo mid else above x (mid + 1) hi
  in
  let rec pieces i start hi =
    if i < n && bounds.(i) < hi then
      (start, bounds.(i)) :: pieces (i + 1) bounds.(i) hi
    else [ (start, hi) ]
  in
  List.concat_map (fun (lo, hi) -> pieces (above lo 0 n) lo hi) ranges

(* The first chunk of worker [w]'s strip of [c] chunks on [t] threads, floor (w
   c / t), as [partition] computes its bounds. *)
let strip_lo ~t c w =
  let t = Int64.of_int t and w = Int64.of_int w in
  Int64.(add (mul w (div c t)) (div (mul w (rem c t)) t))

(* The strip of chunk [i]. *)
let strip_of ~t c i =
  let rec go w = if strip_lo ~t c (w + 1) > i then w else go (w + 1) in
  go 0

(* The most chunks a run of strip [w] holds, max (1, floor (s / 8)) of
   rig_pool.h, and whether a job makes one call a chunk, at most 15 t chunks. *)
let run_bound ~t c w =
  let s = Int64.sub (strip_lo ~t c (w + 1)) (strip_lo ~t c w) in
  max 1 (Int64.to_int s / 8)

let single_runs ~t c = Int64.to_int c <= 15 * t

(* [chunk_at bounds lo] is the index of the last of the chunk [bounds] at most
   [lo]: the chunk that begins at unit [lo]. *)
let chunk_at bounds lo =
  let rec go lo_i hi_i =
    if hi_i - lo_i <= 1 then lo_i
    else
      let mid = (lo_i + hi_i) / 2 in
      if bounds.(mid) <= lo then go mid hi_i else go lo_i mid
  in
  Int64.of_int (go 0 (Array.length bounds))

(* Each call on [t] > 1 threads stays in its starting strip and holds at most
   that strip's run bound. *)
let check_runs ~t ~total ~chunks (job : P.job) =
  let c = chunk_count ~total ~chunks in
  let cuts = chunk_bounds ~total ~chunks in
  let bounds = Array.of_list cuts in
  List.iter
    (fun (call : P.call) ->
      let w = strip_of ~t c (chunk_at bounds call.lo) in
      let lo = bounds.(Int64.to_int (strip_lo ~t c w))
      and hi = bounds.(Int64.to_int (strip_lo ~t c (w + 1))) in
      let msg =
        strf "call [%Ld, %Ld) in strip [%Ld, %Ld)" call.lo call.hi lo hi
      in
      let n = List.length (cut cuts [ (call.lo, call.hi) ]) in
      at_most ~msg int64 ~than:hi call.hi;
      at_most ~msg int ~than:(run_bound ~t c w) n)
    job.calls

let chunks_ran (job : P.job) =
  List.sort compare (List.map (fun (c : P.call) -> (c.lo, c.hi)) job.calls)

let distinct l = List.sort_uniq compare l

(* Cores *)

let test_core_bounds () =
  let fast = P.performance_cores () in
  at_least ~msg:"cores" int ~than:1 cores;
  at_least ~msg:"performance cores" int ~than:1 fast;
  at_most ~msg:"performance cores" int ~than:cores fast

let test_macos_cores () =
  let physical = P.sysctl "hw.physicalcpu" in
  if physical < 0 then skip ~reason:"the host has no hw.physicalcpu" ();
  equal ~msg:"cores" int physical cores;
  let fast = P.sysctl "hw.perflevel0.physicalcpu" in
  if fast < 0 then skip ~reason:"the host has cores of one kind" ();
  equal ~msg:"performance cores" int fast (P.performance_cores ())

let test_other_performance_cores () =
  if P.sysctl "hw.perflevel0.physicalcpu" >= 0 then
    skip ~reason:"the host reports its performance cores" ();
  if Sys.file_exists "/sys/devices/system/cpu/cpu0/cpu_capacity" then
    skip ~reason:"the host reports its CPUs' capacities" ();
  if Sys.file_exists "/sys/devices/cpu_core/cpus" then
    skip ~reason:"the host lists its performance cores" ();
  equal int cores (P.performance_cores ())

(* A process of this executable pins itself to one CPU, reads the cores, then
   restores its affinity and reads them again. *)
let test_affinity () =
  let env =
    Array.append (Unix.environment ()) [| child_variable ^ "=affinity" |]
  in
  let exe = Sys.executable_name in
  let ((out, _, _) as process) =
    Unix.open_process_args_full exe [| exe |] env
  in
  (* The child's line ends with CRLF on Windows. *)
  In_channel.set_binary_mode out false;
  let line = In_channel.input_line out in
  ignore (Unix.close_process_full process);
  if line = Some "-1 -1" then skip ~reason:"the host has no affinity to pin" ();
  equal
    ~msg:"cores at the first call, on one CPU, and after the affinity returned"
    (option string) (Some "1 1") line

let test_windows_cores () =
  let active = P.active_processors () in
  if active < 0 then skip ~reason:"the host has no processor groups" ();
  equal ~msg:"cores" int active cores

(* A cgroup tree as Linux shows it to a process: proc/self/cgroup's lines,
   proc/self/mountinfo's mounts, and the quota files under the mount points, all
   under one directory. [cpus] is ceil q of rig_pool.h: the least ceil (quota /
   period) over the cgroups and their ancestors up to their mounts. *)
type tree = {
  name : string;
  cgroup : string list;
  mounts : string list;
  limits : (string * string) list;
  cpus : int;
}

let mount ?(root = "/") point =
  String.concat " "
    [ "35 24 0:30"; root; point; "rw,nosuid shared:9 - cgroup2 cgroup2 rw" ]

let host_mount = mount "/sys/fs/cgroup"
let session = "/user.slice/user-1000.slice/session-2.scope"
let cpu_max cgroup = "sys/fs/cgroup" ^ cgroup ^ "/cpu.max"
let quota q = strf "%d 100000\n" q
let no_quota = "max 100000\n"

(* cgroup v1's hierarchy of [controllers], mounted at [point]. *)
let v1_mount ?(root = "/") ?(controllers = "cpu,cpuacct") point =
  String.concat " "
    [
      "36 24 0:31";
      root;
      point;
      "rw shared:10 - cgroup cgroup rw," ^ controllers;
    ]

let v1_point = "/sys/fs/cgroup/cpu,cpuacct"
let v1_line = "4:cpu,cpuacct:" ^ session

(* cgroup v1's quota files of [cgroup], a quota of [q] microseconds a period of
   100 ms, -1 for none. *)
let cfs ?(point = v1_point) cgroup q =
  let dir = String.sub point 1 (String.length point - 1) ^ cgroup in
  [
    (dir ^ "/cpu.cfs_quota_us", strf "%d\n" q);
    (dir ^ "/cpu.cfs_period_us", "100000\n");
  ]

let trees =
  let on_host ~name limits cpus =
    {
      name;
      cgroup = [ "0::" ^ session ];
      mounts = [ host_mount ];
      limits;
      cpus;
    }
  in
  [
    on_host ~name:"no quota on the path"
      [ (cpu_max session, no_quota); (cpu_max "/user.slice", no_quota) ]
      (-1);
    on_host ~name:"a quota of 1.5 CPUs" [ (cpu_max session, quota 150000) ] 2;
    on_host ~name:"a quota under one CPU" [ (cpu_max session, quota 50000) ] 1;
    on_host ~name:"a quota of whole CPUs" [ (cpu_max session, quota 300000) ] 3;
    on_host ~name:"the tightest of the cgroup and its ancestors"
      [
        (cpu_max session, quota 400000);
        (cpu_max "/user.slice/user-1000.slice", quota 250000);
        (cpu_max "/user.slice", quota 800000);
      ]
      3;
    on_host ~name:"a cgroup without cpu.max"
      [ (cpu_max "/user.slice/user-1000.slice", quota 200000) ]
      2;
    on_host ~name:"a file above the mount point"
      [ ("sys/fs/cpu.max", quota 100000); (cpu_max session, no_quota) ]
      (-1);
    {
      name = "the root of a cgroup namespace";
      cgroup = [ "0::/" ];
      mounts = [ host_mount ];
      limits = [ ("sys/fs/cgroup/cpu.max", quota 150000) ];
      cpus = 2;
    };
    {
      name = "a mount of a cgroup below the hierarchy's root";
      cgroup = [ "0::/kubepods/pod1/c1" ];
      mounts = [ mount ~root:"/kubepods" "/sys/fs/cgroup" ];
      limits = [ ("sys/fs/cgroup/pod1/c1/cpu.max", quota 200000) ];
      cpus = 2;
    };
    {
      name = "a mount that does not hold the cgroup";
      cgroup = [ "0::" ^ session ];
      mounts = [ mount ~root:"/system.slice" "/mnt/system"; host_mount ];
      limits = [ ("mnt/system/cpu.max", quota 100000) ];
      cpus = -1;
    };
    {
      name = "the cgroup v2 line among v1 lines";
      cgroup = [ "12:cpu,cpuacct:/user.slice"; "0::" ^ session ];
      mounts = [ host_mount ];
      limits = [ (cpu_max session, quota 200000) ];
      cpus = 2;
    };
    {
      name = "cgroup v1's quota";
      cgroup = [ v1_line ];
      mounts = [ v1_mount v1_point ];
      limits = cfs session 150000;
      cpus = 2;
    };
    {
      name = "cgroup v1 without a quota";
      cgroup = [ v1_line ];
      mounts = [ v1_mount v1_point ];
      limits = cfs session (-1) @ cfs "/user.slice" (-1);
      cpus = -1;
    };
    {
      name = "the tightest of a cgroup v1 and its ancestors";
      cgroup = [ v1_line ];
      mounts = [ v1_mount v1_point ];
      limits = cfs session 400000 @ cfs "/user.slice" 250000;
      cpus = 3;
    };
    {
      name = "cgroup v1 reads no cpu.max";
      cgroup = [ v1_line ];
      mounts = [ v1_mount v1_point ];
      limits =
        [ ("sys/fs/cgroup/cpu,cpuacct" ^ session ^ "/cpu.max", quota 100000) ];
      cpus = -1;
    };
    {
      name = "a cgroup v1 hierarchy without the cpu controller";
      cgroup = [ "3:cpuset:" ^ session; "5:cpuacct:" ^ session ];
      mounts =
        [
          v1_mount ~controllers:"cpuset" "/sys/fs/cgroup/cpuset";
          v1_mount ~controllers:"cpuacct" "/sys/fs/cgroup/cpuacct";
        ];
      limits =
        cfs ~point:"/sys/fs/cgroup/cpuset" session 100000
        @ cfs ~point:"/sys/fs/cgroup/cpuacct" session 100000;
      cpus = -1;
    };
    {
      name = "a mount of a cgroup v1 below the hierarchy's root";
      cgroup = [ "4:cpu,cpuacct:/docker/c1" ];
      mounts = [ v1_mount ~root:"/docker/c1" v1_point ];
      limits = cfs "" 200000;
      cpus = 2;
    };
    {
      name = "the tighter of cgroup v2 and v1";
      cgroup = [ v1_line; "0::" ^ session ];
      mounts = [ v1_mount v1_point; mount "/sys/fs/cgroup/unified" ];
      limits =
        ("sys/fs/cgroup/unified" ^ session ^ "/cpu.max", quota 300000)
        :: cfs session 200000;
      cpus = 2;
    };
  ]

(* mountinfo quotes its path fields; proc/self/cgroup keeps the path's
   whitespace and backslashes. Both hierarchies use those same spellings. *)
let path_trees =
  let paths =
    [
      ( "a space in the mount point",
        "/",
        "/pool\\040space",
        "/job",
        "/pool space/job" );
      ( "a tab in the mount point",
        "/",
        "/pool\\011tab",
        "/job",
        "/pool\ttab/job" );
      ( "a newline in the mount point",
        "/",
        "/pool\\012line",
        "/job",
        "/pool\nline/job" );
      ( "a backslash in the mount point",
        "/",
        "/pool\\134slash",
        "/job",
        "/pool\\slash/job" );
      ( "spaces in the cgroup path",
        "/",
        "/pool",
        "/team one/job ",
        "/pool/team one/job " );
      ( "a tab in the cgroup path",
        "/",
        "/pool",
        "/team\tone/job",
        "/pool/team\tone/job" );
      ( "a literal octal sequence in the cgroup path",
        "/",
        "/pool",
        "/team\\040one/job",
        "/pool/team\\040one/job" );
      ( "a space in the mount root",
        "/team\\040one",
        "/pool",
        "/team one/job",
        "/pool/job" );
      ( "a tab in the mount root",
        "/team\\011one",
        "/pool",
        "/team\tone/job",
        "/pool/job" );
      ( "a backslash in the mount root",
        "/team\\134one",
        "/pool",
        "/team\\one/job",
        "/pool/job" );
      ( "an octal sequence decoded once",
        "/team\\134040one",
        "/pool",
        "/team\\040one/job",
        "/pool/job" );
    ]
  in
  List.concat_map
    (fun (name, root, point, path, dir) ->
      [
        {
          name = "cgroup v2: " ^ name;
          cgroup = [ "0::" ^ path ];
          mounts = [ mount ~root point ];
          limits =
            [
              ( String.sub dir 1 (String.length dir - 1) ^ "/cpu.max",
                quota 150000 );
            ];
          cpus = 2;
        };
        {
          name = "cgroup v1: " ^ name;
          cgroup = [ "4:cpu,cpuacct:" ^ path ];
          mounts = [ v1_mount ~root point ];
          limits = cfs ~point:dir "" 150000;
          cpus = 2;
        };
      ])
    paths

(* [with_files files f] is [f dir], with each (path, contents) of [files]
   written under [dir], beside the executable in _build, and removed after. A
   killed run leaves [dir], so it is cleared first. The native and bytecode
   suites, which may run at once, each have their own. *)
let with_files files f =
  let dir = Sys.executable_name ^ ".cgroup" in
  let rec make path =
    if not (Sys.file_exists path) then (
      make (Filename.dirname path);
      Sys.mkdir path 0o755)
  in
  let rec remove path =
    if Sys.is_directory path then (
      Array.iter (fun e -> remove (Filename.concat path e)) (Sys.readdir path);
      Sys.rmdir path)
    else Sys.remove path
  in
  if Sys.file_exists dir then remove dir;
  Sys.mkdir dir 0o755;
  List.iter
    (fun (path, contents) ->
      let file = Filename.concat dir path in
      make (Filename.dirname file);
      Out_channel.with_open_bin file (fun oc ->
          Out_channel.output_string oc contents))
    files;
  Fun.protect ~finally:(fun () -> remove dir) (fun () -> f dir)

let test_cgroup tree =
  let lines l = String.concat "" (List.map (fun s -> s ^ "\n") l) in
  let files =
    ("proc/self/cgroup", lines tree.cgroup)
    :: ("proc/self/mountinfo", lines tree.mounts)
    :: tree.limits
  in
  equal ~msg:"ceil q, or -1 without a quota" int tree.cpus
    (with_files files P.cgroup_cpus)

(* The CPUs [cpus] of a tree whose CPU i has capacity [capacities.(i)], none
   where it is negative, and whose hybrid CPU lists its performance cores as
   [core] and its efficiency cores as [atom], where given. [fast] is what
   rig_pool_host.h counts. *)
type cpu_tree = {
  name : string;
  capacities : int array;
  core : string option;
  atom : string option;
  cpus : int array;
  fast : int;
}

let cpu_trees =
  let all n = Array.init n Fun.id in
  let big_little = [| 1024; 1024; 1024; 1024; 446; 446; 446; 446 |] in
  let sized name capacities cpus fast =
    { name; capacities; core = None; atom = None; cpus; fast }
  in
  (* An Intel Core Ultra 5 235 under Linux 6.12: CPUs 0-5 are its
     performance cores, 6-13 its efficiency cores, and no CPU has a
     capacity. *)
  let hybrid name cpus fast =
    {
      name;
      capacities = Array.make 14 (-1);
      core = Some "0-5\n";
      atom = Some "6-13\n";
      cpus;
      fast;
    }
  in
  [
    sized "cores of one size" (Array.make 4 1024) (all 4) 4;
    sized "big and little cores" big_little (all 8) 4;
    sized "three sizes, the middle one more than half the largest"
      [| 1024; 900; 900; 900; 280; 280; 280; 280 |]
      (all 8) 4;
    sized "a capacity of exactly half the largest" [| 1024; 512 |] (all 2) 1;
    sized "a capacity just over half the largest" [| 1024; 513 |] (all 2) 2;
    sized "an affinity of little cores" big_little [| 4; 5; 6; 7 |] 4;
    sized "an affinity of a big and a little core" big_little [| 1; 5 |] 1;
    sized "a CPU without a capacity" [| 1024; -1 |] (all 2) (-1);
    sized "no CPU" [||] [||] 0;
    hybrid "a hybrid CPU without capacities" (all 14) 6;
    hybrid "an affinity of a hybrid CPU's efficiency cores"
      [| 6; 7; 8; 9; 10; 11; 12; 13 |]
      8;
    hybrid "an affinity across a hybrid CPU's kinds of core" [| 4; 5; 6; 7 |] 2;
    {
      (hybrid "a hybrid CPU listing several ranges" (all 14) 5) with
      core = Some "0-1,4,8-9\n";
    };
    {
      (hybrid "a hybrid CPU whose CPUs have capacities" (all 8) 4) with
      capacities = big_little;
      core = Some "0-1\n";
    };
    {
      (hybrid "a hybrid CPU where one CPU has a capacity" (all 14) 6) with
      capacities = Array.init 14 (fun cpu -> if cpu = 0 then 1024 else -1);
    };
    {
      (hybrid "a hybrid CPU's malformed list" (all 14) (-1)) with
      core = Some "0-\n";
    };
  ]

let test_cpu_tree tree =
  let capacity cpu c =
    if c < 0 then None
    else
      Some
        (strf "sys/devices/system/cpu/cpu%d/cpu_capacity" cpu, strf "%d\n" c)
  in
  let list kind = Option.map (fun l -> (strf "sys/devices/%s/cpus" kind, l)) in
  let files =
    List.filter_map Fun.id
      (list "cpu_core" tree.core :: list "cpu_atom" tree.atom
      :: Array.to_list (Array.mapi capacity tree.capacities))
  in
  equal ~msg:"the performance cores among the CPUs, or -1" int tree.fast
    (with_files files (fun root -> P.performance_cpus root tree.cpus))

let cores_tests =
  group ~timeout:P.timeout "cores"
    [
      test "the cores and the performance cores are within their bounds"
        test_core_bounds;
      test
        "on macOS the cores are the physical cores and the performance cores \
         those of hw.perflevel0"
        test_macos_cores;
      test "where the host reports no performance cores, they are the cores"
        test_other_performance_cores;
      test "on Windows the cores are the active processors of every group"
        test_windows_cores;
      test
        "on Linux the cores are bounded by the affinity at the first call, and \
         a later change is not seen"
        test_affinity;
      cases
        ~name:(fun (t : tree) -> t.name)
        "Linux's cgroup bound is the least ceil (quota / period) up the \
         process's cgroups of v2 and of v1's cpu controller, read from any \
         tree"
        trees test_cgroup;
      cases
        ~name:(fun (t : tree) -> t.name)
        "Linux's cgroup paths preserve their names" path_trees
        (fun tree ->
          if Sys.win32 then
            skip ~reason:"Windows cannot represent these Linux path names" ();
          test_cgroup tree);
      cases
        ~name:(fun (t : cpu_tree) -> t.name)
        "Linux's performance cores are the CPUs of capacity more than half the \
         largest of theirs, or without capacities those a hybrid CPU lists, \
         read from any tree"
        cpu_trees test_cpu_tree;
    ]

(* Chunks *)

let pp_int64 ppf x = Format.fprintf ppf "%LdL" x

let int64_extremes =
  Int64.
    [
      min_int;
      succ min_int;
      -1L;
      0L;
      1L;
      div max_int 2L;
      succ (div max_int 2L);
      pred max_int;
      max_int;
    ]

let threads_gen =
  Gen.frequency
    [
      (6, Gen.int_range (-2) (cores + 2));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ Int32.(to_int min_int); -1; 0; 1; Int32.(to_int max_int) ] );
    ]

let total_gen =
  Gen.frequency
    [
      (4, Gen.int64_range (-2L) 300L);
      (2, Gen.int64_range 0L 100_000L);
      (2, Gen.of_list ~pp:pp_int64 int64_extremes);
    ]

let chunks_gen =
  Gen.frequency
    [
      (4, Gen.int64_range (-2L) 70L);
      (1, Gen.int64_range 0L 4096L);
      (1, Gen.of_list ~pp:pp_int64 int64_extremes);
    ]

let pp_job ppf (threads, total, chunks) =
  Format.fprintf ppf "threads %d, total %LdL, chunks %LdL" threads total chunks

let job_gen = Gen.with_pp pp_job (Gen.triple threads_gen total_gen chunks_gen)

(* A drawn job records its every call. *)
let max_drawn_chunks = 16_384L

let assume_recordable (_, total, chunks) =
  assume (chunk_count ~total ~chunks <= max_drawn_chunks)

let job_examples =
  [
    (4, 0L, 3L);
    (3, -5L, 2L);
    (0, 10L, 0L);
    (cores + 3, 5L, 8L);
    (2, Int64.max_int, 3L);
    (2, 32L, 32L);
    (2, 33L, 33L);
    (1, 1000L, 7L);
  ]

let test_bounds ((threads, total, chunks) as job) =
  assume_recordable job;
  let c = chunk_count ~total ~chunks
  and t = thread_bound ~threads ~total ~chunks in
  cover "no unit" (total <= 0L);
  cover "fewer than one chunk" (total > 0L && chunks < 1L);
  cover "more chunks than units" (total > 0L && chunks > total);
  cover "units times chunks past 64 bits"
    (c > 1L && total > Int64.div Int64.max_int c);
  cover "fewer than one thread" (total > 0L && threads < 1);
  cover "more threads than cores" (c > Int64.of_int cores && threads > cores);
  cover "a job on one thread" (t = 1);
  if cores > 1 then cover "a job on several threads" (t > 1);
  if cores > 1 then
    cover "a job on several threads that may claim runs of several chunks"
      (t > 1 && not (single_runs ~t c));
  let ran = P.record ~threads ~total ~chunks in
  let ranges = chunks_ran ran in
  at_most ~msg:"calls" int ~than:(Int64.to_int c) ran.count;
  equal ~msg:"the calls' ranges cut at the chunk bounds"
    (list (pair int64 int64))
    (partition ~total ~chunks)
    (cut (chunk_bounds ~total ~chunks) ranges);
  if t = 1 then
    equal ~msg:"the calls of a job on one thread"
      (list (pair int64 int64))
      [ (0L, total) ]
      ranges
  else if t > 1 then begin
    check_runs ~t ~total ~chunks ran;
    if single_runs ~t c then
      equal ~msg:"calls of a job of at most 15 chunks a thread" int
        (Int64.to_int c) ran.count
  end

(* Calls in the order they began, which is the order of their claims on one
   thread. A call's place in its thread's order is the distance from the
   thread's worker to the call's strip, then the call's first chunk. *)
let test_claim_order ((threads, total, chunks) as job) =
  assume_recordable job;
  let t = thread_bound ~threads ~total ~chunks
  and c = chunk_count ~total ~chunks in
  let ran = P.record ~threads ~total ~chunks in
  let threads = distinct (List.map (fun (c : P.call) -> c.thread) ran.calls) in
  let bounds = Array.of_list (chunk_bounds ~total ~chunks) in
  let place (call : P.call) =
    let i = chunk_at bounds call.lo in
    let w = if t > 1 then strip_of ~t c i else 0 in
    ((((w - call.worker) mod t) + t) mod t, i)
  in
  let places th =
    List.filter_map
      (fun (c : P.call) -> if c.thread = th then Some (place c) else None)
      ran.calls
  in
  (* A job of more chunks than threads, on more than one thread, has a thread
     make several calls. *)
  if cores > 1 then
    cover "a thread ran several chunks"
      (List.exists (fun th -> List.length (places th) >= 2) threads);
  List.iter
    (fun th ->
      equal
        ~msg:(strf "(strip after its own, chunk) of thread %d's calls" th)
        (list (pair int int64))
        (distinct (places th))
        (places th))
    threads

(* 16 chunks on two threads go one a call; 4096 start with a run of 256. *)
let test_balance chunks =
  P.needs_two_cores ();
  equal ~msg:"the other chunks ran while chunk 0's call lasted" bool true
    (P.balance chunks)

let top = Int64.max_int

let wide_jobs =
  [
    (top, 2L, [ (0L, 4611686018427387903L); (4611686018427387903L, top) ]);
    ( top,
      3L,
      [
        (0L, 3074457345618258602L);
        (3074457345618258602L, 6148914691236517204L);
        (6148914691236517204L, top);
      ] );
    ( Int64.pred top,
      7L,
      [
        (0L, 1317624576693539400L);
        (1317624576693539400L, 2635249153387078801L);
        (2635249153387078801L, 3952873730080618202L);
        (3952873730080618202L, 5270498306774157603L);
        (5270498306774157603L, 6588122883467697004L);
        (6588122883467697004L, 7905747460161236405L);
        (7905747460161236405L, Int64.pred top);
      ] );
    (5L, 8L, [ (0L, 1L); (1L, 2L); (2L, 3L); (3L, 4L); (4L, 5L) ]);
    (3L, top, [ (0L, 1L); (1L, 2L); (2L, 3L) ]);
  ]

let test_wide_job (total, chunks, expected) =
  let ran = chunks_ran (P.record ~threads:cores ~total ~chunks) in
  equal ~msg:"the calls' ranges cut at the stated bounds"
    (list (pair int64 int64))
    expected
    (cut (0L :: List.map snd expected) ran)

let chunk_tests =
  group ~timeout:P.timeout "chunks"
    [
      prop ~count:300 ~examples:job_examples
        "a job calls ranges of whole chunks of the stated bounds that cover \
         its units once, one range on one thread, runs within their bound on \
         several, and nothing for no unit"
        job_gen test_bounds;
      cases
        ~name:(fun (total, chunks, _) ->
          strf "total %Ld, chunks %Ld" total chunks)
        "a job whose total times its chunks passes 64 bits, or of more chunks \
         than units, calls ranges of the stated chunks"
        wide_jobs test_wide_job;
      prop ~examples:job_examples
        "each thread claims its own strip's chunks, then those of the strips \
         after it in turn, each in index order"
        job_gen test_claim_order;
      cases ~name:(strf "%d chunks")
        "a thread that finishes early runs the chunks a slower one would have, \
         one at a time or in runs"
        [ 16; 4096 ] test_balance;
    ]

(* Workers *)

let test_workers ((threads, total, chunks) as job) =
  assume_recordable job;
  let t = thread_bound ~threads ~total ~chunks in
  let ran = P.record ~threads ~total ~chunks in
  let pairs =
    distinct (List.map (fun (c : P.call) -> (c.worker, c.thread)) ran.calls)
  in
  let workers = distinct (List.map fst pairs)
  and threads_used = distinct (List.map snd pairs) in
  cover "fewer than one thread" (total > 0L && threads < 1);
  cover "more threads than cores" (threads > cores && t = cores);
  List.iter
    (fun w ->
      at_least ~msg:"a worker index" int ~than:0 w;
      less ~msg:"a worker index" int ~than:t w)
    workers;
  List.iter
    (fun (w, th) ->
      if th = 0 then equal ~msg:"the calling thread's worker" int 0 w)
    pairs;
  equal ~msg:"workers, each on one thread" (list int) workers
    (List.map fst pairs);
  equal ~msg:"threads, each one worker" (list int) threads_used
    (List.sort compare (List.map snd pairs));
  equal ~msg:"calls begun while another of their worker ran" int 0 ran.overlaps

let test_visibility () =
  let bodies, caller = P.visibility ~jobs:2000 ~threads:cores in
  equal ~msg:"values bodies read stale" int 0 bodies;
  equal ~msg:"values the caller read stale" int 0 caller

let worker_tests =
  group ~timeout:P.timeout "workers"
    [
      prop ~examples:job_examples
        "a call's worker is below the job's threads, the caller's 0, one per \
         thread, and its calls never overlap"
        job_gen test_workers;
      test
        "bodies see the caller's writes made before the job, and the caller \
         the bodies' once it returns"
        test_visibility;
    ]

(* Scheduling *)

let test_one_thread_at_once () =
  P.needs_two_cores ();
  P.while_held (fun () ->
      let ran =
        P.finishes "a job of one thread" (fun () ->
            P.record ~threads:1 ~total:100L ~chunks:10L)
      in
      equal ~msg:"(lo, hi, worker, thread) of each call"
        (list (quad int64 int64 int int))
        [ (0L, 100L, 0, 0) ]
        (List.map
           (fun (c : P.call) -> (c.lo, c.hi, c.worker, c.thread))
           ran.calls))

let test_nested () =
  let outer = 16 and inner = 8 in
  equal
    ~msg:
      "(outer units run, inner jobs not run in one call, misplaced inner \
       calls, inner units not run once)"
    (quad int int int int) (outer, 0, 0, 0)
    (P.finishes "a job whose bodies begin jobs" (fun () ->
         P.nested ~threads:cores ~outer ~inner))

(* Nothing signals that a job waits, so the test samples: once the domain is
   about to begin its job, none of its chunks has run 50 ms later. *)
let test_waits () =
  P.needs_two_cores ();
  let beginning = Atomic.make false in
  let waiting =
    P.while_held (fun () ->
        let d =
          Domain.spawn (fun () ->
              Atomic.set beginning true;
              P.counted ~threads:2 ~total:4L ~chunks:4L)
        in
        P.within 10. "the domain did not begin its job" (fun () ->
            Atomic.get beginning);
        Unix.sleepf 0.05;
        equal ~msg:"chunks run while another thread's job ran" int 0
          (P.counted_calls ());
        d)
  in
  P.within 10. "the waiting job did not run once the other ended" (fun () ->
      P.counted_calls () = 4);
  Domain.join waiting

(* A job's chunks, each once, and whether its calls on several threads hold runs
   within their bound. Jobs of up to 96 chunks reach runs of several chunks on
   up to 6 threads. *)
let job_commands =
  let small_job =
    Gen.with_pp pp_job
      (Gen.triple
         (Gen.int_range (-1) (cores + 1))
         (Gen.int64_range (-1L) 2000L)
         (Gen.int64_range (-1L) 96L))
  in
  [
    command "job"
      (small_job @-> returns (list (pair int64 int64)))
      (fun (threads, total, chunks) ->
        let t = thread_bound ~threads ~total ~chunks
        and c = chunk_count ~total ~chunks in
        if cores > 1 then
          cover "a job on several threads in runs of several chunks"
            (t > 1 && not (single_runs ~t c));
        partition ~total ~chunks)
      (fun (threads, total, chunks) ->
        let ran = P.record ~threads ~total ~chunks in
        let t = thread_bound ~threads ~total ~chunks in
        if t > 1 then check_runs ~t ~total ~chunks ran;
        cut (chunk_bounds ~total ~chunks) (chunks_ran ran));
  ]

let scheduling_tests =
  group ~timeout:P.timeout "scheduling"
    [
      test "a job of one thread runs at once while another thread's job runs"
        test_one_thread_at_once;
      test
        "a job begun from a body of a job of more than one thread runs at once \
         on the body's thread as worker 0, in one call"
        test_nested;
      test
        "a job of more than one thread waits for another thread's job to end, \
         sampled for 50 ms"
        test_waits;
      stateful ~domains:2 ~count:30
        "jobs from two domains each call their chunks once" job_commands;
    ]

let () =
  exit
    (run "rig_pool.h"
       [ cores_tests; chunk_tests; worker_tests; scheduling_tests ])
