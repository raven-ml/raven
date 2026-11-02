(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type key = {
  kind : [ `Metric | `Reading ];
  goal : Metric.goal option;
  stats : Metric.stats;
}

let add_sample b (s : Metric.sample) =
  Json.add_object b
    [
      ("step", fun b -> Json.add_int b s.step);
      ("time", fun b -> Json.add_float b s.time);
      ("value", fun b -> Json.add_float b s.value);
    ]

let str s b = Json.add_string b s

let add_key b k =
  let st = k.stats in
  Json.add_object b
    ([
       ( "kind",
         str (match k.kind with `Metric -> "metric" | `Reading -> "reading") );
     ]
    @ (match k.goal with
      | None -> []
      | Some g -> [ ("goal", str (Record.goal_name g)) ])
    @ [
        ("count", fun b -> Json.add_int b st.count);
        ("last", fun b -> add_sample b st.last);
      ]
    @ (match st.finite with
      | None -> []
      | Some f ->
          [
            ( "finite",
              fun b ->
                Json.add_object b
                  [
                    ("least", fun b -> add_sample b f.least);
                    ("greatest", fun b -> add_sample b f.greatest);
                    ("mean", fun b -> Json.add_float b f.mean);
                  ] );
          ])
    @
    match st.non_finite with
    | None -> []
    | Some (n, first) ->
        [
          ( "non_finite",
            fun b ->
              Json.add_object b
                [
                  ("count", fun b -> Json.add_int b n);
                  ("first", fun b -> add_sample b first);
                ] );
        ])

let encode ~length keys =
  let b = Buffer.create 1024 in
  Json.add_object b
    [
      ("format", fun b -> Json.add_int b Store_format.version);
      ("rules", fun b -> Json.add_int b Store_format.rules);
      ("length", fun b -> Json.add_int b length);
      ( "keys",
        fun b ->
          Json.add_object b
            (List.map (fun (k, v) -> (k, fun b -> add_key b v)) keys) );
    ];
  Buffer.add_char b '\n';
  Buffer.contents b

let ( let* ) = Option.bind

let sample j : Metric.sample option =
  let* step = Json.field "step" Json.int j in
  let* time = Json.field "time" Json.float j in
  let* value = Json.field "value" Json.float j in
  Some { Metric.step; time; value }

let key j =
  let* kind =
    match Json.mem "kind" j with
    | Some (Json.String "metric") -> Some `Metric
    | Some (Json.String "reading") -> Some `Reading
    | _ -> None
  in
  let* goal = Json.opt "goal" Record.goal_of j in
  let* count = Json.field "count" Json.int j in
  let* last = Json.field "last" sample j in
  let* finite =
    Json.opt "finite"
      (fun f ->
        let* least = Json.field "least" sample f in
        let* greatest = Json.field "greatest" sample f in
        let* mean = Json.field "mean" Json.float f in
        Some { Metric.least; greatest; mean })
      j
  in
  let* non_finite =
    Json.opt "non_finite"
      (fun n ->
        let* c = Json.field "count" Json.int n in
        let* first = Json.field "first" sample n in
        Some (c, first))
      j
  in
  Some { kind; goal; stats = { Metric.count; last; finite; non_finite } }

let decode s =
  match Json.of_string s with
  | Error _ -> None
  | Ok j ->
      let* format = Json.field "format" Json.int j in
      let* rules = Json.field "rules" Json.int j in
      if format <> Store_format.version || rules <> Store_format.rules then None
      else
        let* length = Json.field "length" Json.int j in
        let* keys =
          match Json.mem "keys" j with
          | Some (Json.Object members) ->
              List.fold_right
                (fun (k, v) acc ->
                  let* acc in
                  let* v = key v in
                  Some ((k, v) :: acc))
                members (Some [])
          | _ -> None
        in
        Some (length, keys)

let float_eq a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let sample_eq (a : Metric.sample) (b : Metric.sample) =
  a.step = b.step && float_eq a.time b.time && float_eq a.value b.value

let opt_eq eq a b =
  match (a, b) with None, None -> true | Some a, Some b -> eq a b | _ -> false

let finite_eq (a : Metric.finite) (b : Metric.finite) =
  sample_eq a.least b.least
  && sample_eq a.greatest b.greatest
  && float_eq a.mean b.mean

let key_eq a b =
  a.kind = b.kind && a.goal = b.goal
  && a.stats.count = b.stats.count
  && sample_eq a.stats.last b.stats.last
  && opt_eq finite_eq a.stats.finite b.stats.finite
  && opt_eq
       (fun (n, s) (n', s') -> n = n' && sample_eq s s')
       a.stats.non_finite b.stats.non_finite

let equal a b =
  List.length a = List.length b
  && List.for_all2 (fun (k, x) (k', y) -> String.equal k k' && key_eq x y) a b
