(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every optimisation beam search can select must preserve kernel semantics,
   on the CPU renderer. See {!Tolk_opt_fuzz}. *)

open Windtrap

let device = lazy (Tolk_cpu.create "CPU:opt-correctness")

(* A sequence that raises is one beam search discards, so it is reported as a
   capability gap rather than failing the suite. Only a sequence that runs and
   returns the wrong values is a defect beam can select silently. *)
let report label (r : Tolk_opt_fuzz.result) =
  Printf.eprintf "[%s] %d sequences compared, %d rejected\n%!" label
    r.sequences (List.length r.rejected);
  if r.rejected <> [] then
    Printf.eprintf "[%s] %d sequence(s) applied but would not compile or run:\n%s\n%!"
      label (List.length r.rejected)
      (String.concat "\n"
         (List.map (fun (o, d) -> Printf.sprintf "  %s -> %s" o d) r.rejected))

let workload_test device w =
  slow (Tolk_opt_fuzz.name w) (fun () ->
      let r = Tolk_opt_fuzz.check (Lazy.force device) w in
      report (Tolk_opt_fuzz.name w) r;
      if r.miscompiled <> [] then
        fail
          (Printf.sprintf
             "%s: %d of %d compared opt sequences returned wrong values:\n%s"
             (Tolk_opt_fuzz.name w)
             (List.length r.miscompiled)
             r.sequences
             (String.concat "\n"
                (List.map
                   (fun (o, d) -> Printf.sprintf "  %s -> %s" o d)
                   r.miscompiled))))

let () =
  run __FILE__
    [
      group "beam actions preserve semantics (CPU)"
        (List.map (workload_test device) Tolk_opt_fuzz.workloads);
    ]
