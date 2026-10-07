(* nx_pool_run, called from another library's stubs, runs every chunk once and
   every unit once, alone or nested. *)

let failures = ref 0

let check name expected got =
  if expected <> got then (
    incr failures;
    Printf.printf "FAIL %s: expected %d, got %d\n" name expected got)

let () =
  let cores = Pool_smoke.cores () in
  let fast = Pool_smoke.performance_cores () in
  if cores < 1 || fast < 1 || fast > cores then (
    incr failures;
    Printf.printf "FAIL cores %d, performance cores %d\n" cores fast);
  List.iter
    (fun (threads, total, chunks) ->
      List.iter
        (fun nested ->
          let name =
            Printf.sprintf "threads %d, total %d, chunks %d%s" threads total
              chunks
              (if nested then ", nested" else "")
          in
          let sum, calls = Pool_smoke.sum threads total chunks nested in
          let expected_calls =
            if total <= 0 then 0 else min (max chunks 1) total
          in
          check (name ^ ": sum") (max 0 (total * (total - 1) / 2)) sum;
          check (name ^ ": chunks") expected_calls calls)
        [ false; true ])
    [
      (1, 1000, 7);
      (cores, 100_000, 8 * cores);
      (cores + 3, 5, 8);
      (0, 10, 0);
      (4, 0, 3);
      (2, 3, 3);
      (64, 1 lsl 16, 1 lsl 12);
    ];
  if !failures > 0 then exit 1
