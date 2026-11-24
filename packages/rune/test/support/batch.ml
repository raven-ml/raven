(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module T = Nx.Ptree

type ('r, 'd) call =
  | Call :
      ('a, 'd) T.t * 'a * ('b, 'd) T.t * ('a -> 'b) * ('b -> 'r list)
      -> ('r, 'd) call

let call s x s' f read = Call (s, x, s', f, fun y -> [ read y ])

let join (Call (s, x, s', f, read)) (Call (t, y, t', g, read')) =
  Call
    ( T.pair s t,
      (x, y),
      T.pair s' t',
      (fun (a, b) -> (f a, g b)),
      fun (c, d) -> read c @ read' d )

(* Joined as a balanced tree, so that the structure's depth is logarithmic in
   the number of calls. *)
let rec joined = function
  | [] -> invalid_arg "Batch.joined"
  | [ c ] -> c
  | cs ->
      let half = List.length cs / 2 in
      let left = List.filteri (fun i _ -> i < half) cs in
      let right = List.filteri (fun i _ -> i >= half) cs in
      join (joined left) (joined right)

let has_infix ~affix s =
  let n = String.length affix in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = affix || at (i + 1))
  in
  at 0

(* The joined calls' results read, or what their trace or program raised. *)
let compile calls =
  let (Call (s, x, s', f, read)) = joined calls in
  let before = Rune.evaluations () in
  let y =
    match Rune.jit T.(s @-> returns s') f x with
    | y -> Ok y
    | exception e -> Error e
  in
  (* The calls' devices are their arguments', which a call does not name. A
     program that raises the failure of the host's compiler, as a missing
     Clang's, or that is evaluated where the host's programs cannot compile,
     skips the test; any other compiler's failure is the test's to judge. *)
  let evaluated = Rune.evaluations () > before in
  (match (y, Tolk_engine.compiles Rig.host) with
  | Error (Tolk.Renderer.Compiler.Compile_error msg), Error reason
    when has_infix ~affix:reason msg ->
      Windtrap.skip ~reason ()
  | _, Error reason when evaluated -> Windtrap.skip ~reason ()
  | _, (Ok () | Error _) -> ());
  if evaluated then Windtrap.fail "the batch's program was evaluated";
  Result.map read y

let run = function
  | [] -> []
  | calls -> ( match compile calls with Ok rs -> rs | Error e -> raise e)

let results = function
  | [] -> []
  | calls -> (
      match compile calls with
      | Ok rs -> List.map (fun r () -> r) rs
      | Error e -> List.map (fun _ () -> raise e) calls)

let table ~per call rows =
  if per < 1 then invalid_arg "Batch.table: per < 1";
  let n = List.length rows in
  let programs =
    Array.init
      ((n + per - 1) / per)
      (fun k ->
        lazy
          (let mine = List.filteri (fun i _ -> i / per = k) rows in
           Array.of_list (run (List.map call mine))))
  in
  List.mapi
    (fun i row -> (row, fun () -> (Lazy.force programs.(i / per)).(i mod per)))
    rows

let law ~per call examples check =
  let kept = table ~per call examples in
  fun c ->
    match List.assq_opt c kept with
    | Some r -> check c r
    | None -> check c (fun () -> List.hd (run [ call c ]))

type 'd check =
  | Check : ('r, 'd) call * bool * ((unit -> 'r) -> unit) -> 'd check

let check ?(alone = false) c k = Check (c, alone, k)

(* A check's call whose result read is its check, ready to run. *)
let checking (Check (Call (s, x, s', f, read), _, k)) =
  Call (s, x, s', f, fun y -> List.map (fun r () -> k (fun () -> r)) (read y))

let checked = function
  | [] -> ()
  | cs -> (
      match compile (List.map checking cs) with
      | Ok ks -> List.iter (fun k -> k ()) ks
      | Error e -> List.iter (fun (Check (_, _, k)) -> k (fun () -> raise e)) cs
      )

let checks cs =
  let alone, joint = List.partition (fun (Check (_, a, _)) -> a) cs in
  checked joint;
  List.iter (fun c -> checked [ c ]) alone

let each ?(alone = fun _ -> false) call k cs =
  checks (List.map (fun c -> check ~alone:(alone c) (call c) (k c)) cs)

let drawn n gen = Windtrap.Gen.list ~size:(Windtrap.Gen.constant n) gen

let rec chunks n = function
  | [] -> []
  | xs ->
      let mine = List.filteri (fun i _ -> i < n) xs in
      mine :: chunks n (List.filteri (fun i _ -> i >= n) xs)
