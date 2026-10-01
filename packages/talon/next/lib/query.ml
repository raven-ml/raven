(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A step's expressions are bound to its input's schema, and a join's condition
   is bound by [Join.check]: the evaluator reads types from them, and the plan
   prints and compares them. Steps keep only bound arguments, so that a step the
   optimizer builds is a step like any other. *)
type node =
  | Of_table of Table.t
  | Of_source of Source.t
  | Select of { outputs : (string * Expr.packed) list; input : t }
  | Derive of { outputs : (string * Expr.packed) list; input : t }
  | Filter of { predicate : (bool, Expr.row) Expr.t; input : t }
  | Sort of { keys : Order.t list; input : t }
  | Slice of { offset : int; length : int; input : t }
  | Aggregate of {
      by : string list;
      outputs : (string * Expr.packed) list;
      input : t;
    }
  | Join of {
      kind : Join.kind;
      each_left : Join.count;
      each_right : Join.count;
      on : Join.cond;
      left : t;
      right : t;
    }
  | Append of { input : t; rest : t }
  | Unnest of { columns : string list; input : t }

and t = { node : node; schema : Schema.t }

let schema q = q.schema
let of_table t = { node = Of_table t; schema = Table.schema t }
let of_source (s : Source.t) = { node = Of_source s; schema = s.schema }

(* Formatting *)

let plural n what = Printf.sprintf "%d %s%s" n what (if n = 1 then "" else "s")
let pp_name = Type.pp_quoted

(* [kept (n, e)] is [Some n] iff [e] reads the column [n] unchanged, as [keep]
   binds it. *)
let kept (n, Expr.Packed e) =
  match Expr.node e with
  | Read (_, n') when String.equal n n' -> Some n
  | _ -> None

(* [pp_outputs] formats bound outputs, each run of kept columns as one
   [keep]. *)
let pp_outputs ppf outputs =
  let rec groups = function
    | [] -> []
    | o :: os -> (
        match kept o with
        | Some n ->
            let rec run ns = function
              | o :: os when Option.is_some (kept o) -> run (fst o :: ns) os
              | os -> (List.rev ns, os)
            in
            let ns, os = run [ n ] os in
            (fun ppf ->
              Format.fprintf ppf "@[<hov 2>keep@ (names@ %a)@]"
                (Type.pp_list pp_name) ns)
            :: groups os
        | None ->
            let n, Expr.Packed e = o in
            (fun ppf ->
              Format.fprintf ppf "@[<hov 2>%a :=@ %a@]" pp_name n Expr.pp e)
            :: groups os)
  in
  Type.pp_list (fun ppf g -> g ppf) ppf (groups outputs)

let kind_name : Join.kind -> string = function
  | Inner -> "Inner"
  | Left -> "Left"
  | Full -> "Full"
  | Semi -> "Semi"
  | Anti -> "Anti"

let count_name : Join.count -> string = function
  | Any -> "Any"
  | At_most_one -> "At_most_one"
  | One -> "One"
  | At_least_one -> "At_least_one"

let pp_on ppf (on : Join.cond) =
  match (on :> Join.atom list) with
  | [] | [ Position ] -> Join.pp ppf on
  | _ -> Format.fprintf ppf "@[<hov 1>(%a)@]" Join.pp on

let pp_step ppf q =
  let pf fmt = Format.fprintf ppf fmt in
  match q.node with
  | Of_table t ->
      pf "table (%s, %s)"
        (plural (List.length (Schema.columns q.schema)) "column")
        (plural (Table.rows t) "row")
  | Of_source s -> (
      let columns = plural (List.length (Schema.columns q.schema)) "column" in
      match s.rows with
      | None -> pf "%s (%s)" s.name columns
      | Some n -> pf "%s (%s, %s)" s.name columns (plural n "row"))
  | Select { outputs; _ } -> pf "@[<hov 2>select %a@]" pp_outputs outputs
  | Derive { outputs; _ } -> pf "@[<hov 2>derive %a@]" pp_outputs outputs
  | Filter { predicate; _ } -> pf "@[<hov 2>filter %a@]" Expr.pp_arg predicate
  | Sort { keys; _ } -> pf "@[<hov 2>sort %a@]" (Type.pp_list Order.pp) keys
  | Slice { offset; length; _ } ->
      let signed n =
        if n < 0 then Printf.sprintf "(%d)" n else string_of_int n
      in
      pf "@[<hov 2>slice@ ~offset:%s@ ~length:%d@]" (signed offset) length
  | Aggregate { by; outputs; _ } ->
      pf "@[<hov 2>aggregate ~by:%a %a@]" (Type.pp_list pp_name) by pp_outputs
        outputs
  | Join { kind; each_left; each_right; on; _ } ->
      pf "@[<hov 2>join ~on:%a" pp_on on;
      if kind <> Join.Inner then pf "@ ~kind:%s" (kind_name kind);
      if each_left <> Join.Any then pf "@ ~each_left:%s" (count_name each_left);
      if each_right <> Join.Any then
        pf "@ ~each_right:%s" (count_name each_right);
      pf "@]"
  | Append _ -> pf "append"
  | Unnest { columns; _ } ->
      pf "@[<hov 2>unnest %a@]" (Type.pp_list pp_name) columns

let inputs q =
  match q.node with
  | Of_table _ | Of_source _ -> []
  | Select { input; _ }
  | Derive { input; _ }
  | Filter { input; _ }
  | Sort { input; _ }
  | Slice { input; _ }
  | Aggregate { input; _ }
  | Unnest { input; _ } ->
      [ input ]
  | Join { left; right; _ } -> [ left; right ]
  | Append { input; rest } -> [ input; rest ]

(* [lines width pp] is what [pp] formats at the margin [width], line by line.
   Below 40 columns, as deep in a tree, Format would break a step at every
   argument, so a step overruns the margin there instead. *)
let lines width pp =
  let b = Buffer.create 128 in
  let ppf = Format.formatter_of_buffer b in
  Format.pp_set_margin ppf (Int.max width 40);
  pp ppf;
  Format.pp_print_flush ppf ();
  String.split_on_char '\n' (Buffer.contents b)

(* Format draws no tree, so each step is formatted alone, at the margin that its
   prefix leaves, and its lines are prefixed by hand: the first by [first], the
   others, and the step's inputs, by [prefix]. Both are [width] columns wide. *)
let pp ppf q =
  let margin = Format.pp_get_margin ppf () in
  let rec step ~first ~prefix ~width q =
    match lines (margin - width) (fun ppf -> pp_step ppf q) with
    | [] -> assert false
    | line :: rest ->
        Format.fprintf ppf "@,%s%s" first line;
        List.iter (fun l -> Format.fprintf ppf "@,%s%s" prefix l) rest;
        let rec children = function
          | [] -> ()
          | [ q ] -> child "└ " "  " q
          | q :: qs ->
              child "├ " "│ " q;
              children qs
        and child lead below q =
          step ~first:(prefix ^ lead) ~prefix:(prefix ^ below)
            ~width:(width + 2) q
        in
        children (inputs q)
  in
  Format.fprintf ppf "@[<v>query →";
  if not (List.is_empty (Schema.columns q.schema)) then
    Format.fprintf ppf " %a" Schema.pp q.schema;
  step ~first:"" ~prefix:"" ~width:0 q;
  Format.fprintf ppf "@]"

(* Reports *)

(* The type for the entries of a report: a problem of an argument, or an output
   or a predicate as written, with its problems. *)
type entry =
  | Arg of Problem.t
  | Written of (Format.formatter -> unit) * Problem.t list

let pp_input ppf (name, s) =
  let columns = Schema.columns s in
  let shown = List.filteri (fun i _ -> i < 8) columns in
  Format.fprintf ppf "%s (%s)" name (plural (List.length columns) "column");
  if not (List.is_empty columns) then
    Format.fprintf ppf ": %a%s" Schema.pp (Schema.v shown)
      (if List.compare_length_with columns 8 > 0 then ", …" else "")

let has_problem = function Arg _ -> true | Written (_, ps) -> ps <> []

(* [fail verb inputs entries] raises the report of the problems of [entries]. *)
let fail verb inputs entries =
  let entries = List.filter has_problem entries in
  let count =
    List.fold_left
      (fun n -> function
        | Arg _ -> n + 1 | Written (_, ps) -> n + List.length ps)
      0 entries
  in
  let b = Buffer.create 256 in
  let line indent s = Printf.bprintf b "\n%s%s" indent s in
  let problem indent p = line indent (Format.asprintf "%a" Problem.pp p) in
  Printf.bprintf b "%s: %s" verb (plural count "problem");
  List.iter
    (function
      | Arg p -> problem "  " p
      | Written (pp, ps) ->
          List.iter (line "  ") (lines 76 pp);
          List.iter (problem "    ") ps)
    entries;
  List.iter (fun i -> line "  " (Format.asprintf "%a" pp_input i)) inputs;
  invalid_arg (Buffer.contents b)

let check verb inputs entries =
  if List.exists has_problem entries then fail verb inputs entries

let arg fmt = Format.kasprintf (fun s -> Arg (Problem.v "%s" s)) fmt

let column_type (Expr.Packed e) =
  match Expr.typing e with
  | Column ty -> Type.Any ty
  | Extension d -> Type.Any d.type_
  | Value ->
      assert false (* [Expr.bind_out] gives every output a column type. *)

let columns outputs = List.map (fun (n, e) -> (n, column_type e)) outputs

(* [bind_outs s os] is the bound outputs of [os] over [s], the names of the
   outputs, and the entries of their problems, then of two outputs of one name.
   An output [n := e] that fails to bind still has the name [n], so that its
   collisions are reported with its problems. *)
let bind_outs s os =
  let bind o =
    let entry ps = Written ((fun ppf -> Expr.pp_out ppf o), ps) in
    match Expr.bind_out s o with
    | Ok outs -> (outs, List.map fst outs, entry [])
    | Error ps -> ([], Option.to_list (Expr.out_name o), entry ps)
  in
  let bound = List.map bind os in
  let outs = List.concat_map (fun (outs, _, _) -> outs) bound in
  let names = List.concat_map (fun (_, names, _) -> names) bound in
  let twice =
    List.map
      (arg "the output %a appears twice." pp_name)
      (Problem.repeated names)
  in
  (outs, names, List.map (fun (_, _, e) -> e) bound @ twice)

let input q = [ ("input", q.schema) ]

(* [missing what s n] is the problem of the name [n] if [s] lacks it, after
   [what]. *)
let missing what s n =
  match Schema.find s n with
  | Some _ -> []
  | None -> [ arg "%s%a" what Problem.pp (Problem.missing n s) ]

(* [first ns] is [ns] without the names it already holds, in order. *)
let first ns =
  List.rev
    (List.fold_left (fun ns n -> if List.mem n ns then ns else n :: ns) [] ns)

(* Verbs *)

let select os q =
  let outputs, _, entries = bind_outs q.schema os in
  check "select" (input q) entries;
  { node = Select { outputs; input = q }; schema = Schema.v (columns outputs) }

let derive os q =
  let outputs, _, entries = bind_outs q.schema os in
  check "derive" (input q) entries;
  let replaced (n, t) =
    match List.assoc_opt n outputs with
    | Some e -> (n, column_type e)
    | None -> (n, t)
  in
  let added =
    List.filter (fun (n, _) -> Option.is_none (Schema.find q.schema n)) outputs
  in
  {
    node = Derive { outputs; input = q };
    schema =
      Schema.v (List.map replaced (Schema.columns q.schema) @ columns added);
  }

let filter p q =
  match Expr.bind_predicate q.schema p with
  | Ok predicate ->
      { node = Filter { predicate; input = q }; schema = q.schema }
  | Error ps ->
      fail "filter" (input q) [ Written ((fun ppf -> Expr.pp ppf p), ps) ]

(* The first problem of a key on an extension column says how to sort by its
   storage; a later one is that the key is repeated. *)
let sort keys q =
  let entry seen ((k : Order.t), p) =
    let recipe =
      match Schema.find q.schema k.name with
      | Some (Any (Ext _)) when not (List.mem k.name seen) ->
          Format.asprintf
            " For example: derive [ \"k\" := Ext.storage e (Ext.col e %a) ] |> \
             sort [ asc \"k\" ] |> select [ keep Sel.(all - names [ \"k\" ]) \
             ]."
            pp_name k.name
      | _ -> ""
    in
    (k.name :: seen, arg "%a: %a%s" Order.pp k Problem.pp p recipe)
  in
  check "sort" (input q)
    (snd (List.fold_left_map entry [] (Order.check keys q.schema)));
  { node = Sort { keys; input = q }; schema = q.schema }

let slice ~offset ~length q =
  if length < 0 then
    fail "slice" (input q) [ arg "the length %d is negative." length ];
  { node = Slice { offset; length; input = q }; schema = q.schema }

let aggregate ~by os q =
  let keys =
    List.concat_map (missing "~by: " q.schema) (first by)
    @ List.map (arg "~by: %a is named twice." pp_name) (Problem.repeated by)
  in
  let outputs, names, entries = bind_outs q.schema os in
  let shadows =
    List.filter_map
      (fun n ->
        if List.mem n by then
          Some (arg "the output %a has the name of a key." pp_name n)
        else None)
      names
  in
  check "aggregate" (input q) (keys @ entries @ shadows);
  let key n = (n, Option.get (Schema.find q.schema n)) in
  {
    node = Aggregate { by; outputs; input = q };
    schema = Schema.v (List.map key by @ columns outputs);
  }

let join ?(kind = Join.Inner) ?(each_left = Join.Any) ?(each_right = Join.Any)
    ~on right left =
  let inputs = [ ("left", left.schema); ("right", right.schema) ] in
  match Join.check kind left.schema right.schema on with
  | Ok (on, schema) ->
      { node = Join { kind; each_left; each_right; on; left; right }; schema }
  | Error ps -> fail "join" inputs (List.map (fun p -> Arg p) ps)

let append rest q =
  let change : Schema.change -> entry = function
    | Removed (n, Any t) -> arg "%a (%a) is not in rest." pp_name n Type.pp t
    | Added (n, Any t) -> arg "%a (%a) is only in rest." pp_name n Type.pp t
    | Retyped (n, Any t0, Any t1) ->
        arg "%a is %a in the input and %a in rest." pp_name n Type.pp t0 Type.pp
          t1
  in
  check "append"
    [ ("input", q.schema); ("rest", rest.schema) ]
    (List.map change (Schema.diff q.schema rest.schema));
  { node = Append { input = q; rest }; schema = q.schema }

let unnest columns q =
  let column n =
    match Schema.find q.schema n with
    | None -> missing "" q.schema n
    | Some (Any (List _)) -> []
    | Some (Any t) -> [ arg "%a is %a, not a list." pp_name n Type.pp t ]
  in
  check "unnest" (input q)
    ((if List.is_empty columns then [ arg "no column to unnest." ] else [])
    @ List.concat_map column (first columns)
    @ List.map (arg "%a is named twice." pp_name) (Problem.repeated columns));
  let element (n, (Type.Any t as a)) =
    match t with
    | List e when List.mem n columns -> (n, Type.Any e)
    | _ -> (n, a)
  in
  {
    node = Unnest { columns; input = q };
    schema = Schema.v (List.map element (Schema.columns q.schema));
  }

(* Comparing *)

let outputs_equal o0 o1 =
  List.equal
    (fun (n0, Expr.Packed e0) (n1, Expr.Packed e1) ->
      String.equal n0 n1 && Expr.same e0 e1)
    o0 o1

let rec equal q0 q1 =
  q0 == q1
  || (match (q0.node, q1.node) with
       | Of_table t0, Of_table t1 -> Table.equal t0 t1
       | Of_source s0, Of_source s1 -> s0 == s1
       | Select { outputs = o0; _ }, Select { outputs = o1; _ }
       | Derive { outputs = o0; _ }, Derive { outputs = o1; _ } ->
           outputs_equal o0 o1
       | Filter f0, Filter f1 -> Expr.same f0.predicate f1.predicate
       | Sort s0, Sort s1 -> List.equal Order.equal s0.keys s1.keys
       | Slice s0, Slice s1 ->
           Int.equal s0.offset s1.offset && Int.equal s0.length s1.length
       | Aggregate a0, Aggregate a1 ->
           List.equal String.equal a0.by a1.by
           && outputs_equal a0.outputs a1.outputs
       | Join j0, Join j1 ->
           j0.kind = j1.kind
           && j0.each_left = j1.each_left
           && j0.each_right = j1.each_right
           && Join.equal j0.on j1.on
       | Append _, Append _ -> true
       | Unnest u0, Unnest u1 -> List.equal String.equal u0.columns u1.columns
       | ( ( Of_table _ | Of_source _ | Select _ | Derive _ | Filter _ | Sort _
           | Slice _ | Aggregate _ | Join _ | Append _ | Unnest _ ),
           _ ) ->
           false)
     && List.equal equal (inputs q0) (inputs q1)
