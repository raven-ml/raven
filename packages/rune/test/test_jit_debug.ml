(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What RUNE_JIT_DEBUG prints, read from search.exe and declines.exe, each run
   as a process of its own, since the setting is read once per process. *)

open Windtrap

(* The lines of what [exe] prints on its standard output and standard error, run
   with RUNE_JIT_DEBUG=1 and a fresh disk cache, that hold one of [subs]. *)
let reported exe ~subs =
  let out = temp_file () in
  let cache = Filename.concat (temp_dir ()) "cache" in
  let fd = Unix.openfile out [ Unix.O_WRONLY; O_TRUNC ] 0o600 in
  let env =
    Array.append
      [| "RUNE_JIT_DEBUG=1"; "CACHEDB=" ^ cache |]
      (Unix.environment ())
  in
  let exe = Filename.concat (Sys.getcwd ()) exe in
  let pid = Unix.create_process_env exe [| exe |] env Unix.stdin fd fd in
  Unix.close fd;
  (match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED 0 -> ()
  | Unix.WEXITED n -> failf "%s exited %d" exe n
  | Unix.WSIGNALED n | Unix.WSTOPPED n -> failf "%s ended by signal %d" exe n);
  let holds line sub =
    let n = String.length sub in
    let rec at i =
      i + n <= String.length line && (String.sub line i n = sub || at (i + 1))
    in
    at 0
  in
  In_channel.with_open_bin out In_channel.input_lines
  |> List.filter (fun line -> List.exists (holds line) subs)
  |> String.concat "\n"

(* Each kernel a search optimises prints the width it searches at, the
   function's or the settings' (rune.mli's Search). In search.ml's order: the
   function given width 1; none for the function given none; none for width 0
   under BEAM=1; the function given width 2, once, though called again under
   BEAM=3 and JITBEAM=3; the three kernels of a gradient's programs derived from
   a function given width 1; none for a function given width 1 inlined in one
   given none; and the four kernels of a concatenation given width 2, one a
   piece. The pieces run as one launch, which rune.mli does not state: the case
   stays because a search once skipped a launch's kernels. *)
let searches () =
  Rune_support.Compiled.here ();
  expect (reported "search.exe" ~subs:[ "searched a kernel" ])
  @@ __POS_OF__
       {|
    rune.jit: searched a kernel at width 1
    rune.jit: searched a kernel at width 2
    rune.jit: searched a kernel at width 1
    rune.jit: searched a kernel at width 1
    rune.jit: searched a kernel at width 1
    rune.jit: searched a kernel at width 2
    rune.jit: searched a kernel at width 2
    rune.jit: searched a kernel at width 2
    rune.jit: searched a kernel at width 2
  |}

(* Programs over values on two software devices. Each call reports whether its
   program compiled or ran evaluated (Rune.evaluations) and whether it gave
   eager's values. Over several devices a program compiles unless it holds what
   rune.mli's Evaluated programs lists, such as a check in a loop's step or stop
   or a switch's branch, or an addition to a total in a branch or a remat's
   function, and its first call to find it declined prints why, naming the
   construct: a check names the innermost loop's step or stop or switch's branch
   that holds it, a remat between them included, and an addition the construct
   that holds it. The first call of a compiled program prints each move between
   devices: the two that gather a cut argument whole on each device, and the
   stop of an iterate, which lies on another device than the program's, its
   first argument's, and moves there before each trip. *)
let declines () =
  Rune_support.Compiled.here ();
  expect (reported "declines.exe" ~subs:[ "values"; "uncompiled"; "moves" ])
  @@ __POS_OF__
       {|
    a cut argument: compiled, eager's values
    a cut argument donated: compiled, eager's values
    rune.jit: Nx.place moves 8 bytes from declines-1 to declines-0
    rune.jit: Nx.place moves 8 bytes from declines-0 to declines-1
    a cut argument placed whole on each device: compiled, eager's values
    rune.jit: evaluates its program uncompiled: a check in Rune.switch's branch over several devices
    a check inside a switch's branch: evaluated, eager's values
    rune.jit: evaluates its program uncompiled: a check in a scan's step over several devices
    a check inside a scan's step: evaluated, eager's values
    rune.jit: evaluates its program uncompiled: a check in Rune.switch's branch over several devices
    a check inside a remat inside a switch's branch: evaluated, eager's values
    rune.jit: evaluates its program uncompiled: a total in Rune.switch over several devices
    a total inside a switch's branch: evaluated, eager's values
    rune.jit: evaluates its program uncompiled: a total in Rune.remat over several devices
    a total inside a remat's function: evaluated, eager's values
    rune.jit: evaluates its program uncompiled: a check in an iterate's stop over several devices
    a check inside an iterate's stop: evaluated, eager's values
    rune.jit: Rune.iterate' moves its stop, 1 byte, from declines-0 to declines-1 each trip
    an iterate whose stop lies on another device: compiled, eager's values
  |}

let () =
  exit
    (run "Rune.jit's debug reports"
       [
         test "a search reports the width of each kernel it searches" searches;
         test "a program over two devices reports its declines and moves"
           declines;
       ])
