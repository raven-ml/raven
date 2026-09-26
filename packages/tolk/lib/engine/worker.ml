(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Admission is shared; domains and task ownership belong to one synchronous
   batch. A nested batch runs inline rather than waiting on its own permits. *)
let running = Domain.DLS.new_key (fun () -> false)
let lock = Mutex.create ()
let changed = Condition.create ()
let capacity = ref None
let available = ref 0

let reserve ~limit requested =
  Mutex.protect lock (fun () ->
      (match !capacity with
       | None -> capacity := Some limit; available := limit
       | Some _ -> ());
      while !available = 0 do Condition.wait changed lock done;
      let count = min requested !available in
      available := !available - count;
      count)

let release count =
  Mutex.protect lock (fun () ->
      available := !available + count;
      Condition.broadcast changed)

let map f tasks =
  let requested = Helpers.Context_var.get Helpers.parallel in
  let tasks = Array.of_list tasks in
  let count = Array.length tasks in
  if requested <= 0 || count < 2 || Domain.DLS.get running then
    Array.map f tasks
  else
    let workers = reserve ~limit:requested count in
    Fun.protect ~finally:(fun () -> release workers) (fun () ->
        let context = Helpers.Context_var.snapshot () in
        let next = Atomic.make 0 in
        let failure = Atomic.make None in
        let results = Array.make count None in
        let record_failure exn =
          let backtrace = Printexc.get_raw_backtrace () in
          ignore (Atomic.compare_and_set failure None (Some (exn, backtrace)))
        in
        let work () =
          Domain.DLS.set running true;
          try Helpers.Context_var.with_snapshot context (fun () ->
              let rec run () =
                if Option.is_none (Atomic.get failure) then begin
                  let index = Atomic.fetch_and_add next 1 in
                  if index < count then begin
                    results.(index) <- Some (f tasks.(index));
                    run ()
                  end
                end
              in
              run ())
          with exn -> record_failure exn
        in
        let started = ref [] in
        (try
           for _ = 1 to workers do
             let domain = Domain.spawn work in
             started := domain :: !started
           done
         with exn -> record_failure exn);
        (* Keep draining even if a join is interrupted: started workers must
           finish before their inputs or admission permits can be released. *)
        List.iter (fun domain ->
            let rec join () =
              try Domain.join domain with
              | Sys.Break as exn -> record_failure exn; join ()
              | exn -> record_failure exn
            in
            join ()) (List.rev !started);
        match Atomic.get failure with
        | Some (exn, backtrace) -> Printexc.raise_with_backtrace exn backtrace
        | None -> Array.map Option.get results)
