(** Automatic system metrics during a computation.

    Starts a system monitor that logs CPU and memory usage in the background
    while a CPU-intensive computation runs. *)

open Munin

(* A simple CPU-bound computation: count primes up to n. *)
let count_primes n =
  let count = ref 0 in
  for i = 2 to n do
    let is_prime = ref true in
    let j = ref 2 in
    while !j * !j <= i && !is_prime do
      if i mod !j = 0 then is_prime := false;
      incr j
    done;
    if !is_prime then incr count
  done;
  !count

let () =
  let root = "_munin" in
  let session =
    Session.start ~root ~experiment:"compute" ~name:"prime-sieve"
      ~params:[ ("limit", `Int 5_000_000) ]
      ()
  in

  (* Start system monitoring with a short interval for this demo. *)
  let monitor = Munin_sys.start session ~interval:0.5 () in

  (* Run the computation, logging progress. *)
  let steps = 10 in
  let per_step = 500_000 in
  for i = 1 to steps do
    let limit = i * per_step in
    let n = count_primes limit in
    Session.log_metrics session ~step:i
      [ ("primes_found", Float.of_int n); ("limit", Float.of_int limit) ]
  done;

  Munin_sys.stop monitor;
  Session.finish session ();

  (* Check what system metrics were recorded. *)
  let run = Session.run session in
  let keys = Run.metric_keys run in
  let sys_keys =
    List.filter (fun k -> String.length k > 4 && String.sub k 0 4 = "sys/") keys
  in
  Printf.printf "run: %s\n" (Run.id run);
  Printf.printf "system metrics: %s\n" (String.concat ", " sys_keys);
  List.iter
    (fun key ->
      let history = Run.metric_history run key in
      let n = List.length history in
      if n > 0 then
        let last = (List.nth history (n - 1)).value in
        Printf.printf "  %s: %d samples, last=%.2f\n" key n last)
    sys_keys
