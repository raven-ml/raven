(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

(* Fixtures *)

(* The database of golden/tz/zoneinfo/, written by gen/tz.py. *)
let db () = Error.get_ok (Tz.of_dir "golden/tz/zoneinfo")
let zone name = Error.get_ok (Tz.find (db ()) name)
let per_s = 1_000_000L

let floor_s t =
  let q = Int64.div t per_s in
  if Int64.rem t per_s < 0L then Int64.pred q else q

(* The transitions of a zone, from golden/tz/transitions.txt:
   [(instant, offset before, offset after)]. *)
let transitions name =
  In_channel.with_open_text "golden/tz/transitions.txt" In_channel.input_lines
  |> List.filter_map (fun line ->
      Scanf.sscanf line "%s %Ld %d %d" (fun z u before after ->
          if String.equal z name then Some (u, before, after) else None))

let walls_col ts = Column.of_options (Type.wall Us) ts
let instants_col ts = Column.of_options (Type.instant Us) ts
let stamp t = Option.get (Time.of_us t)
let ticks_of c = Array.map (Option.map Time.to_us) (Column.options Kind.instant c)
let wall_ticks_of c = Array.map (Option.map Time.to_us) (Column.options Kind.wall c)
let one name c = v [ (name, c) ]

let compute e t =
  Result.map
    (fun r -> column r "out")
    (Query.run (Query.select Expr.[ "out" := e ] (Query.of_table t)))

(* Generators *)

(* Ticks of a second within three hours of [u], with a fraction. *)
let near u =
  let open Gen in
  let+ d = int_range (-10_800) 10_800 and+ frac = int64_range 0L 999_999L in
  Int64.add (Int64.mul (Int64.add u (Int64.of_int d)) per_s) frac

let anywhere =
  Gen.map (Int64.mul per_s) (Gen.int64_range (-2_000_000_000L) 8_000_000_000L)

(* Wall ticks near the zone's folds and gaps, on either offset, across decades,
   and anywhere. *)
let wall_ticks name =
  let rows = transitions name in
  let around (u, before, after) =
    Gen.bind
      (Gen.of_list [ before; after ])
      (fun o -> near (Int64.add u (Int64.of_int o)))
  in
  Gen.frequency [ (5, Gen.bind (Gen.of_list rows) around); (1, anywhere) ]

let policies : Expr.Temporal.policy list = [ `Earlier; `Later; `Null; `Fail ]

let pp_policy ppf (p : Expr.Temporal.policy) =
  Format.pp_print_string ppf
    (match p with
    | `Earlier -> "Earlier"
    | `Later -> "Later"
    | `Null -> "Null"
    | `Fail -> "Fail")

let pp_case ppf (name, am, gap, ws) =
  Format.fprintf ppf "%s ambiguous:%a gap:%a [%a]" name pp_policy am pp_policy
    gap
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_char ppf ';')
       (fun ppf t -> Format.fprintf ppf "%Ld" t))
    (Array.to_list ws)

let localize_case =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list [ "Europe/Paris"; "America/Ojinaga" ]) (fun name ->
         let open Gen in
         let+ am = of_list policies
         and+ gap = of_list policies
         and+ ws = array ~size:(int_range 1 32) (wall_ticks name) in
         (name, am, gap, ws)))

(* [expected z am gap w] is what the contract of [Temporal.localize] gives the
   wall ticks [w]: [`Value], null or a failure. *)
let expected z am gap w =
  let s = floor_s w in
  let at o = `Value (Int64.sub w (Int64.mul (Int64.of_int o) per_s)) in
  let by (p : Expr.Temporal.policy) ~earlier ~later =
    match p with
    | `Earlier -> at earlier
    | `Later -> at later
    | `Null -> `Null
    | `Fail -> `Fail
  in
  match Tz.local z s with
  | Unique o -> at o
  | Ambiguous { before; after } -> by am ~earlier:before ~later:after
  | Gap { before; after } -> by gap ~earlier:after ~later:before

let value = function `Value v -> Some v | `Null | `Fail -> None

let localize_law =
  prop "localize agrees with Tz.local at every policy" ~count:300 localize_case
    (fun (name, am, gap, ws) ->
      let z = zone name in
      let rows = Array.map (expected z am gap) ws in
      let has f = Array.exists (fun w -> f (Tz.local z (floor_s w))) ws in
      cover "a fold" (has (function Tz.Ambiguous _ -> true | _ -> false));
      cover "a gap" (has (function Tz.Gap _ -> true | _ -> false));
      let e = Expr.Temporal.localize z ~ambiguous:am ~gap (Col.wall "w") in
      let out =
        compute e (one "w" (walls_col (Array.map (fun w -> Some (stamp w)) ws)))
      in
      if Array.mem `Fail rows then ignore (require_error out)
      else
        equal
          (array (option int64))
          (Array.map value rows)
          (ticks_of (require_ok ~pp:Error.pp out)))

(* Wall and localize *)

let pp_instants ppf (name, ts) =
  Format.fprintf ppf "%s [%s]" name
    (String.concat ";" (Array.to_list (Array.map Int64.to_string ts)))

let instant_case =
  let open Gen in
  with_pp pp_instants
  @@ let* name = of_list [ "Europe/Paris"; "America/Ojinaga" ] in
  let rows = transitions name in
  let tick =
    frequency
      [ (3, bind (of_list rows) (fun (u, _, _) -> near u)); (1, anywhere) ]
  in
  let+ ts = array ~size:(int_range 1 32) tick in
  (name, ts)

let wall_of z t =
  Int64.add t (Int64.mul (Int64.of_int (Tz.offset_s z (floor_s t))) per_s)

let wall_laws =
  [
    prop "wall reads each instant at its zone's offset" instant_case
      (fun (name, ts) ->
        let z = zone name in
        let col = instants_col (Array.map (fun t -> Some (stamp t)) ts) in
        let out =
          require_ok ~pp:Error.pp
            (compute (Expr.Temporal.wall z (Col.instant "t")) (one "t" col))
        in
        equal
          (array (option int64))
          (Array.map (fun t -> Some (wall_of z t)) ts)
          (wall_ticks_of out));
    prop "localize of wall is the instant where the reading is unambiguous"
      instant_case (fun (name, ts) ->
        let z = zone name in
        let col = instants_col (Array.map (fun t -> Some (stamp t)) ts) in
        let e =
          Expr.Temporal.(
            localize z ~ambiguous:`Null ~gap:`Fail (wall z (Col.instant "t")))
        in
        let out = require_ok ~pp:Error.pp (compute e (one "t" col)) in
        let unique t =
          match Tz.local z (floor_s (wall_of z t)) with
          | Unique _ -> Some t
          | Ambiguous _ | Gap _ -> None
        in
        equal
          (array (option int64))
          (Array.map unique ts)
          (ticks_of out));
    prop "fields and floors of wall agree with Tz.offset_s" instant_case
      (fun (name, ts) ->
        let z = zone name in
        let col = instants_col (Array.map (fun t -> Some (stamp t)) ts) in
        let w = Expr.Temporal.wall z (Col.instant "t") in
        let hour e = Expr.Temporal.field `Hour e in
        let floor_day e = Expr.Temporal.floor (Days 1) e in
        let day = Int64.mul 86_400L per_s in
        let local_day t =
          let v = wall_of z t in
          let q = Int64.div v day in
          Int64.mul (if Int64.rem v day < 0L then Int64.pred q else q) day
        in
        let local_hour t =
          let into = Int64.sub (wall_of z t) (local_day t) in
          Int64.to_int (Int64.div into (Int64.mul 3600L per_s))
        in
        let r e k = Column.options k (require_ok ~pp:Error.pp (compute e (one "t" col))) in
        equal
          (array (option int))
          (Array.map (fun t -> Some (local_hour t)) ts)
          (r (hour w) Kind.int);
        equal
          (array (option int64))
          (Array.map (fun t -> Some (local_day t)) ts)
          (Array.map (Option.map Time.to_us) (r (floor_day w) Kind.wall)));
  ]

let units_laws =
  let case =
    let open Gen in
    let+ name, ts = instant_case
    and+ unit_ = of_list Type.[ S; Ms; Ns ] in
    (name, Array.map floor_s ts, unit_)
  in
  prop "wall and localize read whole seconds alike in every unit" case
    (fun (name, secs, unit_) ->
      let z = zone name in
      let stamps = Array.map (fun s -> Option.get (Time.of_s s)) secs in
      let run ty k e c =
        Array.map
          (Option.map Time.to_s)
          (Column.options k
             (require_ok ~pp:Error.pp (compute e (one "t" (Column.v ty c)))))
      in
      let walled =
        run (Type.instant unit_) Kind.wall
          (Expr.Temporal.wall z (Col.instant "t"))
          stamps
      in
      let offset s = Int64.add s (Int64.of_int (Tz.offset_s z s)) in
      equal (array (option int64)) (Array.map (fun s -> Some (offset s)) secs) walled;
      let back =
        run (Type.wall unit_) Kind.instant
          (Expr.Temporal.localize z ~ambiguous:`Null ~gap:`Null (Col.wall "t"))
          (Array.map (fun w -> Option.get (Time.of_s (Option.get w))) walled)
      in
      let unique s =
        match Tz.local z (offset s) with Unique _ -> Some s | _ -> None
      in
      equal (array (option int64)) (Array.map unique secs) back)

let wall_of_localize =
  prop "wall of localize reads back every reading that occurs" localize_case
    (fun (name, _, _, ws) ->
      let z = zone name in
      List.iter
        (fun am ->
          let col = walls_col (Array.map (fun w -> Some (stamp w)) ws) in
          let e =
            Expr.Temporal.(
              wall z (localize z ~ambiguous:am ~gap:`Null (Col.wall "w")))
          in
          let out =
            wall_ticks_of (require_ok ~pp:Error.pp (compute e (one "w" col)))
          in
          let expected w =
            match Tz.local z (floor_s w) with Gap _ -> None | _ -> Some w
          in
          equal (array (option int64)) (Array.map expected ws) out)
        [ `Earlier; `Later ])

let wall_errors =
  [
    test "an instant has no calendar: floor and field name Temporal.wall"
      (fun () ->
        let t = one "t" (instants_col [| Some (stamp 0L) |]) in
        let names_wall f =
          match compute (f (Col.instant "t")) t with
          | exception Invalid_argument m -> contains ~sub:"Temporal.wall" m
          | Ok _ | Error _ -> Windtrap.fail "expected a problem"
        in
        names_wall (fun e -> Expr.Temporal.floor (Days 1) e);
        names_wall (fun e -> Expr.Temporal.offset (Months 1) e);
        names_wall (fun e -> Expr.Temporal.field `Year e));
    test "an exact step needs no calendar and moves an instant" (fun () ->
        let t = one "t" (instants_col [| Some (stamp 90_000_000L) |]) in
        let e =
          Expr.Temporal.floor (Exact (Time.Span.minutes 1)) (Col.instant "t")
        in
        equal
          (array (option int64))
          [| Some 60_000_000L |]
          (ticks_of (require_ok ~pp:Error.pp (compute e t))));
    test "a result outside its ticks is a data error" (fun () ->
        let z = zone "Europe/Paris" in
        let t =
          one "t" (Column.v (Type.instant Ns) [| Time.of_ns Int64.max_int |])
        in
        ignore
          (require_error (compute (Expr.Temporal.wall z (Col.instant "t")) t)));
    test "a null row does not fail, whatever its ticks" (fun () ->
        let z = zone "Europe/Paris" in
        let values = Nx.P (Nx.create Nx.int64 [| 1 |] [| Int64.max_int |]) in
        let validity = Some (Nx.create Nx.bit [| 1 |] [| false |]) in
        let c =
          match
            Column.of_layout (Any (Type.instant Ns)) (Fixed { validity; values })
          with
          | Ok c -> c
          | Error (row, why) -> failf "row %d: %s" row why
        in
        let out =
          require_ok ~pp:Error.pp
            (compute (Expr.Temporal.wall z (Col.instant "t")) (one "t" c))
        in
        equal (array (option int64)) [| None |] (wall_ticks_of out));
    test "a policy of Fail names the reading" (fun () ->
        let z = zone "Europe/Paris" in
        let w = walls_col [| Some (stamp 1_729_996_200_000_000L) |] in
        let e =
          Expr.Temporal.localize z ~ambiguous:`Fail ~gap:`Fail (Col.wall "t")
        in
        contains ~sub:"2024-10-27T02:30:00 repeats in Europe/Paris"
          (Format.asprintf "%a" Error.pp
             (require_error (compute e (one "t" w)))));
  ]

(* Ordered logs *)

(* A fold of [name], one per year in the test window: [(T, before, after)]. *)
let folds name =
  List.filter
    (fun (u, before, after) ->
      after < before && Int64.compare u 946_684_800L > 0
      && Int64.compare u 1_893_456_000L < 0)
    (transitions name)

let log_zones =
  [
    "Europe/Paris";
    "America/New_York";
    "Australia/Lord_Howe";
    "America/Ojinaga";
  ]

(* A scenario drops rows from a log's series of UTC truths. *)
type scenario = Complete | Iid of float | Outage of int | Dup of float

(* [log name fold cadence scenario seed] is the truths and the wall clocks a
   logger writes around [fold]: a series of the given cadence, in seconds,
   twelve hours each side, thinned by [scenario]. *)
let log z (t0, _, _) cadence scenario rand =
  let t0 = Int64.to_int t0 in
  let first = t0 - 43_200 in
  let aligned = first - (first mod cadence) in
  let cut =
    match scenario with
    | Outage len -> t0 - 1800 + rand 3600 - (len / 2)
    | _ -> min_int
  in
  let kept s =
    match scenario with
    | Complete -> true
    | Iid p -> Float.of_int (rand 1_000_000) /. 1e6 >= p
    | Outage len -> s < cut || s >= cut + len
    | Dup _ -> true
  in
  let n = 86_400 / cadence in
  List.concat_map
    (fun i ->
      let s = aligned + (i * cadence) in
      let t = Int64.mul (Int64.of_int s) per_s in
      let row = (t, wall_of z t) in
      if not (kept s) then []
      else
        match scenario with
        | Dup p when Float.of_int (rand 1_000_000) /. 1e6 < p -> [ row; row ]
        | _ -> [ row ])
    (List.init n Fun.id)

let fold_length (_, before, after) = before - after

let run_ordered z am gap ws =
  let w = walls_col (Array.map (fun t -> Some (stamp t)) ws) in
  compute
    (Expr.Temporal.localize_ordered z ~ambiguous:am ~gap (Col.wall "w"))
    (one "w" w)

let log_case =
  let open Gen in
  let* name = of_list log_zones in
  let* f = of_list (folds name) in
  let+ cadence = of_list [ 60; 300; 600; 900; 1800; 3600; 7200; 10_800 ]
  and+ scenario =
    of_list
      [
        Complete; Iid 0.05; Iid 0.3; Outage 600; Outage 3600; Outage 10_800;
        Dup 0.2;
      ]
  and+ seed = int_range 0 1_000_000 in
  (name, f, cadence, scenario, seed)

let pp_log ppf (n, (t, _, _), c, _, s) =
  Format.fprintf ppf "%s %Ld cadence %d seed %d" n t c s

let scan (name, f, cadence, scenario, seed) =
  let z = zone name in
  let rand = Random.State.int (Random.State.make [| seed |]) in
  let rows = Array.of_list (log z f cadence scenario rand) in
  (z, Array.map fst rows, Array.map snd rows)

let log_gen = Gen.with_pp pp_log log_case

(* A gap of a zone, and the policy that moves a reading inside it. *)
let gap_case =
  let open Gen in
  let* name = of_list log_zones in
  let gaps =
    List.filter
      (fun (u, before, after) ->
        before < after && Int64.compare u 946_684_800L > 0
        && Int64.compare u 1_893_456_000L < 0)
      (transitions name)
  in
  let+ g = of_list gaps and+ am = of_list [ `Earlier; `Later ] in
  (name, g, am)

let ordered_laws =
  [
    (* Every row is correct or the column is refused, at every cadence, with
       rows lost, or duplicated in a complete series. *)
    prop ~count:400 "with Fail a logged series is exact or refused" log_gen
      (fun c ->
        let z, truth, ws = scan c in
        let out = run_ordered z `Fail `Fail ws in
        let in_fold w =
          match Tz.local z (floor_s w) with Ambiguous _ -> true | _ -> false
        in
        cover "exact across a fold" (Result.is_ok out && Array.exists in_fold ws);
        cover "refused" (Result.is_error out);
        match out with
        | Error _ -> ()
        | Ok out ->
            equal
              (array (option int64))
              (Array.map Option.some truth)
              (ticks_of out));
    prop ~count:300 "a policy decides exactly the rows that order leaves open"
      log_gen (fun c ->
        let z, truth, ws = scan c in
        let refused = Result.is_error (run_ordered z `Fail `Fail ws) in
        cover "a policy decides a row" refused;
        let with_policy p =
          ticks_of (require_ok ~pp:Error.pp (run_ordered z p `Fail ws))
        in
        let settled = with_policy `Null in
        let guess later w =
          match Tz.local z (floor_s w) with
          | Ambiguous { before; after } ->
              let o = if later then after else before in
              Int64.sub w (Int64.mul (Int64.of_int o) per_s)
          | Unique _ | Gap _ -> Windtrap.fail "a null row of no fold"
        in
        let check later out =
          Array.iteri
            (fun i s ->
              match (s, out.(i)) with
              | Some v, r -> equal (option int64) (Some v) r
              | None, r -> equal (option int64) (Some (guess later ws.(i))) r)
            settled
        in
        Array.iteri
          (fun i s -> Option.iter (equal int64 truth.(i)) s)
          settled;
        check false (with_policy `Earlier);
        check true (with_policy `Later));
    prop "a row moved by a gap leaves the other rows' instants" gap_case
      (fun (name, gap_at, am) ->
        let z = zone name in
        let (t0, before, after) = gap_at in
        let at i = Int64.mul (Int64.add t0 (Int64.of_int (i * 900))) per_s in
        let rows = List.init 16 (fun i -> at (i - 8)) in
        let inside =
          Int64.mul
            (Int64.add t0 (Int64.of_int (before + ((after - before) / 2))))
            per_s
        in
        let walls =
          List.concat_map
            (fun t -> if t = at 0 then [ inside; wall_of z t ] else [ wall_of z t ])
            rows
        in
        let moved =
          match am with
          | `Earlier -> Int64.sub inside (Int64.mul (Int64.of_int after) per_s)
          | `Later -> Int64.sub inside (Int64.mul (Int64.of_int before) per_s)
        in
        let expected =
          List.concat_map
            (fun t -> if t = at 0 then [ moved; t ] else [ t ])
            rows
        in
        let out = run_ordered z `Fail (am :> Expr.Temporal.policy) (Array.of_list walls) in
        equal (array (option int64))
          (Array.of_list (List.map Option.some expected))
          (ticks_of (require_ok ~pp:Error.pp out)));
    prop "a complete series at a cadence at or below the fold is exact" log_gen
      (fun (name, f, cadence, _, _) ->
        let z = zone name in
        if cadence <= fold_length f then
          let rows = log z f cadence Complete (fun _ -> 0) in
          let ws = Array.map snd (Array.of_list rows) in
          let out = require_ok ~pp:Error.pp (run_ordered z `Fail `Fail ws) in
          equal
            (array (option int64))
            (Array.map (fun (t, _) -> Some t) (Array.of_list rows))
            (ticks_of out));
    prop "a complete series above the fold's cadence fails under Fail" log_gen
      (fun (name, f, cadence, _, _) ->
        let z = zone name in
        if cadence > fold_length f then
          let rows = log z f cadence Complete (fun _ -> 0) in
          let ws = Array.map snd (Array.of_list rows) in
          let ambiguous w =
            match Tz.local z (floor_s w) with Ambiguous _ -> true | _ -> false
          in
          if Array.exists ambiguous ws then
            ignore (require_error (run_ordered z `Fail `Fail ws)));
  ]

let paris_cases =
  let z = zone "Europe/Paris" in
  let w y mo d h mi =
    let day =
      Time.Date.to_days (Option.get (Time.Date.of_civil (y, mo, d)))
    in
    Int64.mul per_s (Int64.of_int ((day * 86_400) + (h * 3600) + (mi * 60)))
  in
  let d = w 2024 10 27 and u = w 2024 10 26 in
  let run ?(am = `Fail) ?(gap = `Fail) ws =
    match run_ordered z am gap (Array.of_list ws) with
    | Ok c -> Array.to_list (ticks_of c)
    | Error e -> Windtrap.failf "%a" Error.pp e
  in
  let some = List.map Option.some in
  [
    test "a repeated hour written half-hourly" (fun () ->
        equal
          (list (option int64))
          (some
             [
               u 23 30;
               d 0 0;
               d 0 30;
               d 1 0;
               d 1 30;
               d 2 0;
             ])
          (run
             [
               d 1 30; d 2 0; d 2 30; d 2 0; d 2 30; d 3 0;
             ]));
    test "an hourly logger writes the repeated hour once in each pass"
      (fun () ->
        equal
          (list (option int64))
          (some [ u 23 0; d 0 0; d 1 0; d 2 0 ])
          (run [ d 1 0; d 2 0; d 2 0; d 3 0 ]));
    test "a tie away from any fold is allowed" (fun () ->
        equal
          (list (option int64))
          (some [ d 8 0; d 8 0 ])
          (run [ d 9 0; d 9 0 ]));
    test "a duplicate inside the first pass with no second pass follows \
          ~ambiguous" (fun () ->
        let rows = [ d 1 50; d 2 20; d 2 20; d 2 30; d 3 5 ] in
        equal
          (list (option int64))
          (some [ u 23 50; d 0 20; d 0 20; d 0 30; d 2 5 ])
          (run ~am:`Earlier rows);
        ignore (require_error (run_ordered z `Fail `Fail (Array.of_list rows))));
    test "two equal rows at a finer cadence are a duplicate" (fun () ->
        let rows = [ d 1 45; d 2 20; d 2 20; d 3 5 ] in
        ignore (require_error (run_ordered z `Fail `Fail (Array.of_list rows)));
        equal
          (list (option int64))
          (some [ u 23 45; d 0 20; d 0 20; d 2 5 ])
          (run ~am:`Earlier rows));
    test "two equal rows at a frame's edge say nothing of their pass" (fun () ->
        List.iter
          (fun ws -> ignore (require_error (run_ordered z `Fail `Fail ws)))
          [
            [| d 2 0; d 2 0 |];
            [| d 2 0; d 2 0; d 3 0 |];
            [| d 1 0; d 2 0; d 2 0 |];
          ]);
    test "three equal rows are not two passes" (fun () ->
        ignore
          (require_error
             (run_ordered z `Fail `Fail [| d 1 0; d 2 0; d 2 0; d 2 0; d 3 0 |])));
    test "a duplicate after the step stays in the second pass" (fun () ->
        equal
          (list (option int64))
          (some [ d 0 0; d 0 30; d 1 0; d 1 0; d 1 30 ])
          (run [ d 2 0; d 2 30; d 2 0; d 2 0; d 2 30 ]));
    test "frames follow over's order, whatever the row order" (fun () ->
        let walls = [| d 2 30; d 1 30; d 2 0; d 2 0; d 2 30 |] in
        let tbl =
          v
            [
              ("i", Column.v Type.int64 [| 2; 0; 1; 3; 4 |]);
              ("w", Column.v (Type.wall Us) (Array.map stamp walls));
            ]
        in
        let e =
          Expr.(
            over ~order:[ Order.asc "i" ]
              (Temporal.localize_ordered z ~ambiguous:`Fail ~gap:`Fail
                 (Col.wall "w")))
        in
        equal
          (array (option int64))
          [|
            Some (d 0 30);
            Some (u 23 30);
            Some (d 0 0);
            Some (d 1 0);
            Some (d 1 30);
          |]
          (ticks_of (require_ok ~pp:Error.pp (compute e tbl))));
    test "a row moved by the gap policy leaves the state" (fun () ->
        let w' y mo d' h mi =
          let day =
            Time.Date.to_days (Option.get (Time.Date.of_civil (y, mo, d')))
          in
          Int64.mul per_s
            (Int64.of_int ((day * 86_400) + (h * 3600) + (mi * 60)))
        in
        let d = w' 2024 3 31 in
        equal
          (list (option int64))
          (some [ d 0 30; d 1 30; d 1 0 ])
          (run ~gap:`Later [ d 1 30; d 2 30; d 3 0 ]));
    test "a row going backwards is a data error" (fun () ->
        ignore (require_error (run_ordered z `Fail `Fail [| d 12 0; d 11 0 |])));
    test "nulls leave the state and stay null" (fun () ->
        let c =
          Column.of_options (Type.wall Us)
            [|
              Some (stamp (d 1 30));
              None;
              Some (stamp (d 2 0));
              Some (stamp (d 2 30));
              None;
              Some (stamp (d 2 0));
            |]
        in
        let out =
          compute
            (Expr.Temporal.localize_ordered z ~ambiguous:`Fail ~gap:`Fail
               (Col.wall "w"))
            (one "w" c)
        in
        equal
          (array (option int64))
          [| Some (u 23 30); None; Some (d 0 0); Some (d 0 30); None; Some (d 1 0) |]
          (ticks_of (require_ok ~pp:Error.pp out)));
    test "frames resolve independently under over" (fun () ->
        let walls = [| d 1 30; d 2 0; d 2 30; d 2 0 |] in
        let tbl =
          v
            [
              ("site", Column.v Type.string [| "a"; "b"; "a"; "b" |]);
              ("w", Column.v (Type.wall Us) (Array.map stamp walls));
            ]
        in
        let e =
          Expr.(
            over ~by:[ "site" ]
              (Temporal.localize_ordered z ~ambiguous:`Earlier ~gap:`Fail
                 (Col.wall "w")))
        in
        let out = require_ok ~pp:Error.pp (compute e tbl) in
        equal
          (array (option int64))
          [| Some (u 23 30); Some (d 0 0); Some (d 0 30); Some (d 0 0) |]
          (ticks_of out));
  ]

(* Fixed zones *)

let fixed =
  [
    test "a fixed zone reads its offset, and names itself" (fun () ->
        let z = Tz.fixed 19_800 in
        equal string "+05:30" (Tz.name z);
        equal int 19_800 (Tz.offset_s z 0L);
        equal string "-00:45:30" (Tz.name (Tz.fixed (-2730)));
        equal bool true (Tz.equal z (Tz.fixed 19_800));
        equal bool false (Tz.equal z (Tz.fixed 19_801));
        equal bool true (Tz.equal Tz.utc (Tz.fixed 0)));
    test "offsets past 32 bits are refused" (fun () ->
        raises (Invalid_argument "Tz.fixed: 2147483648 is not a 32-bit offset")
          (fun () -> ignore (Tz.fixed 2_147_483_648));
        raises (Invalid_argument "Tz.fixed: -2147483648 is not a 32-bit offset")
          (fun () -> ignore (Tz.fixed (-2_147_483_648))));
    test "a zone equals another of the same rules, whatever its name" (fun () ->
        equal bool true (Tz.equal (zone "Europe/Paris") (zone "Europe/Paris"));
        equal bool false
          (Tz.equal (zone "Europe/Paris") (zone "America/New_York"));
        equal bool true (Tz.equal Tz.utc (zone "UTC"));
        equal bool true (Tz.equal (Tz.fixed (-18_000)) (zone "Etc/GMT+5")));
  ]

let () =
  exit
    (run "talon.zoned"
       [
         group "localize" [ localize_law ];
         group "wall and localize" (units_laws :: wall_of_localize :: wall_laws);
         group "errors" wall_errors;
         group "ordered" ordered_laws;
         group "ordered cases" paris_cases;
         group "fixed zones" fixed;
       ])
