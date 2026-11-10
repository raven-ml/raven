(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A host's arch: the level of the x86-64 psABI its processor has, from the
   psABI's table "Micro-Architecture Levels", with each feature's CPUID bit
   and the XCR0 state its detection reads from the Intel SDM (vol. 1,
   §5.1.16.1, §12, §14.3, §15.2; vol. 2A, CMPXCHG16B); arm64's is "arm64"
   followed by the features the system states, each after a "+" in
   lexicographic order, as ACLE's "Function Multi Versioning" writes them. *)

open Windtrap

external runs_level : int -> int = "rig_test_runs_level"
external runs_fp16 : unit -> int = "rig_test_runs_fp16"

let levels = [ "x86-64"; "x86-64-v2"; "x86-64-v3"; "x86-64-v4" ]

type word = Ecx1 | Ebx7 | Ecx81 | Xcr0

(* Each level's features above the baseline: name, word, bit, and the flag
   Linux's /proc/cpuinfo shows for it, if any. Linux shows AVX only while the
   system enables XSAVE, and the AVX-512 features only while XCR0 holds their
   state (Documentation/arch/x86/cpuinfo.rst), so the registers' state has
   no flag of its own. *)
let features =
  [
    ( "x86-64-v2",
      [
        ("CMPXCHG16B", Ecx1, 13, Some "cx16");
        ("LAHF-SAHF", Ecx81, 0, Some "lahf_lm");
        ("POPCNT", Ecx1, 23, Some "popcnt");
        ("SSE3", Ecx1, 0, Some "pni");
        ("SSE4_1", Ecx1, 19, Some "sse4_1");
        ("SSE4_2", Ecx1, 20, Some "sse4_2");
        ("SSSE3", Ecx1, 9, Some "ssse3");
      ] );
    ( "x86-64-v3",
      [
        ("AVX", Ecx1, 28, Some "avx");
        ("AVX2", Ebx7, 5, Some "avx2");
        ("BMI1", Ebx7, 3, Some "bmi1");
        ("BMI2", Ebx7, 8, Some "bmi2");
        ("F16C", Ecx1, 29, Some "f16c");
        ("FMA", Ecx1, 12, Some "fma");
        ("LZCNT", Ecx81, 5, Some "abm");
        ("MOVBE", Ecx1, 22, Some "movbe");
        ("OSXSAVE", Ecx1, 27, None);
        ("XMM state", Xcr0, 1, None);
        ("YMM state", Xcr0, 2, None);
      ] );
    ( "x86-64-v4",
      [
        ("AVX512F", Ebx7, 16, Some "avx512f");
        ("AVX512BW", Ebx7, 30, Some "avx512bw");
        ("AVX512CD", Ebx7, 28, Some "avx512cd");
        ("AVX512DQ", Ebx7, 17, Some "avx512dq");
        ("AVX512VL", Ebx7, 31, Some "avx512vl");
        ("opmask state", Xcr0, 5, None);
        ("ZMM_Hi256 state", Xcr0, 6, None);
        ("Hi16_ZMM state", Xcr0, 7, None);
      ] );
  ]

let all = List.concat_map snd features

(* CPUID.01H:ECX, CPUID.07H.00H:EBX, CPUID.80000001H:ECX and XCR0. *)
type words = { ecx1 : int; ebx7 : int; ecx81 : int; xcr0 : int }

(* The words of a processor of the features [fs] alone. *)
let words fs =
  List.fold_left
    (fun w (_, word, bit, _) ->
      let b = 1 lsl bit in
      match word with
      | Ecx1 -> { w with ecx1 = w.ecx1 lor b }
      | Ebx7 -> { w with ebx7 = w.ebx7 lor b }
      | Ecx81 -> { w with ecx81 = w.ecx81 lor b }
      | Xcr0 -> { w with xcr0 = w.xcr0 lor b })
    { ecx1 = 0; ebx7 = 0; ecx81 = 0; xcr0 = 0 }
    fs

let level w =
  Rig.x86_level ~ecx1:w.ecx1 ~ebx7:w.ebx7 ~ecx81:w.ecx81 ~xcr0:w.xcr0

(* The features of [l] and the levels below it. *)
let up_to l =
  let rec go = function
    | [] -> []
    | (name, fs) :: rest -> if name = l then fs else fs @ go rest
  in
  if l = "x86-64" then [] else go features

let below l =
  let rec go prev = function
    | x :: rest -> if x = l then prev else go x rest
    | [] -> invalid_arg l
  in
  go "x86-64" levels

let named = Testable.make ~pp:Format.pp_print_string ~equal:String.equal

(* The table *)

let ordered () = equal (list string) levels Rig.x86_levels

let exact =
  cases ~id:Fun.id "a processor of exactly a level's features states it"
    levels
  @@ fun l -> equal named l (level (words (up_to l)))

(* With every feature but one, a processor states the level below that
   feature's: the levels are cumulative. *)
let missing =
  cases
    ~id:(fun (l, (f, _, _, _)) -> l ^ " " ^ f)
    "a processor lacking one feature states the level below it:"
    (List.concat_map (fun (l, fs) -> List.map (fun f -> (l, f)) fs) features)
  @@ fun (l, f) ->
  equal named (below l) (level (words (List.filter (( != ) f) all)))

(* The highest level whose features, and those of the levels below it, the
   words hold. *)
let highest w =
  let held (_, word, bit, _) =
    let v =
      match word with
      | Ecx1 -> w.ecx1
      | Ebx7 -> w.ebx7
      | Ecx81 -> w.ecx81
      | Xcr0 -> w.xcr0
    in
    (v lsr bit) land 1 = 1
  in
  List.fold_left
    (fun found l -> if List.for_all held (up_to l) then l else found)
    "x86-64" levels

(* Processors that have each feature nineteen times in twenty, so that every
   level is drawn, with any bits besides. *)
let processors =
  Gen.(
    map
      (fun (kept, noise) ->
        let w = words (List.filteri (fun i _ -> List.nth kept i) all) in
        {
          ecx1 = w.ecx1 lor (noise land 0xffff_ffff);
          ebx7 = w.ebx7 lor ((noise lsr 8) land 0xffff_ffff);
          ecx81 = w.ecx81 lor ((noise lsr 16) land 0xffff_ffff);
          xcr0 = w.xcr0 lor ((noise lsr 24) land 0xff);
        })
      (pair
         (list ~size:(constant (List.length all))
            (map (fun n -> n > 0) (int_range 0 19)))
         (frequency [ (3, constant 0); (1, int_range 0 max_int) ])))

let law =
  prop "a processor states the highest level whose features it has" processors
  @@ fun w ->
  let l = level w in
  List.iter (fun x -> cover ~share:0.05 x (l = x)) levels;
  equal named (highest w) l

(* This machine *)

let x86_host () =
  if Rig.instruction_set Rig.host <> "x86_64" then
    skip ~reason:"not an x86-64 host" ()

let host_names () =
  match Rig.instruction_set Rig.host with
  | "x86_64" -> mem string (Rig.arch Rig.host) levels
  | "arm64" -> (
      match String.split_on_char '+' (Rig.arch Rig.host) with
      | "arm64" :: features ->
          equal ~msg:"features in lexicographic order, once each"
            (list string)
            (List.sort_uniq String.compare features)
            features
      | _ -> fail ("another arch: " ^ Rig.arch Rig.host))
  | isa -> fail ("another instruction set: " ^ isa)

let arm64_host () =
  if Rig.instruction_set Rig.host <> "arm64" then
    skip ~reason:"not an arm64 host" ()

let features () = List.tl (String.split_on_char '+' (Rig.arch Rig.host))

(* What [sysctl -n name] prints, without its newline: [None] where the
   system has no such parameter or no sysctl. *)
let sysctl name =
  match
    Unix.open_process_args_in "/usr/sbin/sysctl" [| "sysctl"; "-n"; name |]
  with
  | exception Unix.Unix_error _ -> None
  | ic -> (
      let out = In_channel.input_all ic in
      match Unix.close_process_in ic with
      | Unix.WEXITED 0 -> Some (String.trim out)
      | _ -> None)

(* rig reads one feature, fp16, from macOS's hw.optional.arm.FEAT_FP16
   alone, a parameter no other system has: an arm64 host elsewhere has no
   features. *)
let half_arithmetic () =
  arm64_host ();
  let stated = sysctl "hw.optional.arm.FEAT_FP16" = Some "1" in
  equal ~msg:"the features" (list string)
    (if stated then [ "fp16" ] else [])
    (features ())

let runs_features () =
  arm64_host ();
  if not (List.mem "fp16" (features ())) then
    skip ~reason:"the host states no feature this test runs" ();
  match runs_fp16 () with
  | -1 -> skip ~reason:"no child forks here" ()
  | -2 -> fail "the child could not be waited for"
  | status -> equal ~msg:"the child's status" int 0 status

let cpuinfo_flags () =
  let lines =
    try In_channel.with_open_text "/proc/cpuinfo" In_channel.input_lines
    with Sys_error _ -> skip ~reason:"no /proc/cpuinfo" ()
  in
  match List.find_opt (String.starts_with ~prefix:"flags") lines with
  | None -> skip ~reason:"/proc/cpuinfo states no flags" ()
  | Some l -> (
      match String.split_on_char ':' l with
      | [ _; flags ] -> String.split_on_char ' ' (String.trim flags)
      | _ -> fail ("a flags line of another form: " ^ l))

(* Linux states the features it has enabled and supports; a level is the
   highest whose shown features all appear. *)
let matches_linux () =
  x86_host ();
  let flags = cpuinfo_flags () in
  let shown l =
    List.for_all
      (fun (_, _, _, flag) ->
        match flag with None -> true | Some f -> List.mem f flags)
      (up_to l)
  in
  let expected =
    List.fold_left (fun e l -> if shown l then l else e) "x86-64" levels
  in
  equal named expected (Rig.arch Rig.host)

(* Each instruction of the host's level, and of those below it, runs. *)
let runs_its_level () =
  x86_host ();
  let rec index i = function
    | x :: rest -> if x = Rig.arch Rig.host then i else index (i + 1) rest
    | [] -> fail "not a level"
  in
  match runs_level (index 0 levels) with
  | -1 -> skip ~reason:"no child forks here" ()
  | -2 -> fail "the child could not be waited for"
  | status -> equal ~msg:"the child's status" int 0 status

let software () =
  let d = Result.get_ok (Rig.software "arch:software") in
  equal (pair string string)
    (Rig.arch Rig.host, Rig.instruction_set Rig.host)
    (Rig.arch d, Rig.instruction_set d)

let () =
  exit
    (run "rig.arch"
       [
         group "the x86-64 levels"
           [ test "are the psABI's, in order" ordered; exact; missing; law ];
         group "this machine"
           [
             test "names its instruction set and its level" host_names;
             test "states the level Linux's flags show" matches_linux;
             test "runs each instruction of its level" runs_its_level;
             test "states the half arithmetic macOS lists" half_arithmetic;
             test "runs the half arithmetic it states" runs_features;
             test "a software device's arch is the host's" software;
           ];
       ])
