(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Domains live for one call: an idle domain would still join every minor
   collection. A new domain starts with its spawner's settings. *)
let spawned = Atomic.make 0

let rec reserve wanted =
  let n = Atomic.get spawned in
  let free = Setting.value Setting.parallel - 1 - n in
  let k = max 0 (min wanted free) in
  if k = 0 || Atomic.compare_and_set spawned n (n + k) then k
  else reserve wanted

let release k = ignore (Atomic.fetch_and_add spawned (-k))

(* [body ()] on the calling domain while up to [wanted] other domains run
   [work], returning once they all have. The runtime refuses a domain past its
   limit, which other domains of the program may have reached: the call then
   does with fewer. *)
let alongside wanted work body =
  let domains = ref [] in
  let rec spawn k =
    if k > 0 then
      match Domain.spawn work with
      | d ->
          domains := d :: !domains;
          spawn (k - 1)
      | exception Failure _ -> release k
  in
  Fun.protect
    ~finally:(fun () ->
      List.iter Domain.join !domains;
      release (List.length !domains))
    (fun () ->
      spawn (reserve wanted);
      body ())

let map f l =
  let tasks = Array.of_list l in
  let count = Array.length tasks in
  let results = Array.make count None and next = Atomic.make 0 in
  (* The raising application of least index. Indices are claimed in order, so
     every one below it has been claimed and runs to its end. *)
  let failure = Atomic.make None in
  let rec fail i e bt =
    match Atomic.get failure with
    | Some (j, _, _) when (j < i) [@mutate off "an index fails at most once"] ->
        ()
    | seen ->
        if not (Atomic.compare_and_set failure seen (Some (i, e, bt))) then
          fail i e bt
  in
  let rec work () =
    let i = Atomic.fetch_and_add next 1 in
    let runs =
      match Atomic.get failure with
      | Some (j, _, _) -> i < j
      | None -> i < count
    in
    if runs then begin
      (match f tasks.(i) with
      | y -> results.(i) <- Some y
      | exception e -> fail i e (Printexc.get_raw_backtrace ()));
      work ()
    end
  in
  alongside (count - 1) work work;
  match Atomic.get failure with
  | Some (_, e, bt) -> Printexc.raise_with_backtrace e bt
  | None -> List.init count (fun i -> Option.get results.(i))

let iter f g l =
  let tasks = Array.of_list l in
  let count = Array.length tasks in
  let results = Array.make count None and next = Atomic.make 0 in
  let ready = Mutex.create () and filled = Condition.create () in
  (* No index from [limit] on starts. An application of [f] that raises at [i]
     lowers it to [i + 1]: every index below has been claimed and runs to its
     end, since indices are claimed in order. The consumer lowers it to [0] when
     it raises. *)
  let limit = Atomic.make count in
  let rec lower n =
    let l = Atomic.get limit in
    if n < l && not (Atomic.compare_and_set limit l n) then lower n
  in
  let apply i =
    match f tasks.(i) with
    | y -> Ok y
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        lower (i + 1);
        Error (e, bt)
  in
  let rec work () =
    let i = Atomic.fetch_and_add next 1 in
    if i < Atomic.get limit then begin
      let r = apply i in
      Mutex.protect ready (fun () ->
          results.(i) <- Some r;
          Condition.broadcast filled);
      work ()
    end
  in
  (* The image of [i]: the consumer applies [f] itself to an index no domain has
     claimed, rather than wait for one to. *)
  let image i =
    if Atomic.compare_and_set next i (i + 1) then apply i
    else
      Mutex.protect ready (fun () ->
          while Option.is_none results.(i) do
            Condition.wait filled ready
          done;
          Option.get results.(i))
  in
  let consume () =
    for i = 0 to count - 1 do
      match image i with
      | Ok y -> g tasks.(i) y
      | Error (e, bt) -> Printexc.raise_with_backtrace e bt
    done
  in
  alongside (count - 1) work (fun () ->
      Fun.protect ~finally:(fun () -> lower 0) consume)
