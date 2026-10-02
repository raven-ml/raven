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
  | Of_source of {
      source : Source.t;
      columns : string list;
      filters : (bool, Expr.row) Expr.t list;
      limit : int option;
    }
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
let node q = q.node

let column_type (Expr.Packed e) =
  match Expr.typing e with
  | Column ty -> Type.Any ty
  | Extension d -> Type.Any d.type_
  | Value ->
      assert false (* [Expr.bind_out] gives every output a column type. *)

let columns outputs = List.map (fun (n, e) -> (n, column_type e)) outputs

(* [derived s outputs] is the columns of [s] with [outputs] in place of the
   columns of their names, then the others. *)
let derived s outputs =
  let replaced (n, t) =
    match List.assoc_opt n outputs with
    | Some e -> (n, column_type e)
    | None -> (n, t)
  in
  let added =
    List.filter (fun (n, _) -> Option.is_none (Schema.find s n)) outputs
  in
  List.map replaced (Schema.columns s) @ columns added

let make node =
  let schema =
    match node with
    | Of_table t -> Table.schema t
    | Of_source { source; columns; _ } ->
        Schema.v
          (List.filter
             (fun (n, _) -> List.mem n columns)
             (Schema.columns source.schema))
    | Select { outputs; _ } -> Schema.v (columns outputs)
    | Derive { outputs; input } -> Schema.v (derived input.schema outputs)
    | Filter { input; _ }
    | Sort { input; _ }
    | Slice { input; _ }
    | Append { input; _ } ->
        input.schema
    | Aggregate { by; outputs; input } ->
        let key n = (n, Option.get (Schema.find input.schema n)) in
        Schema.v (List.map key by @ columns outputs)
    | Join { kind; on; left; right; _ } ->
        Join.columns kind left.schema right.schema on
    | Unnest { columns; input } ->
        let element (n, (Type.Any t as a)) =
          match t with
          | List e when List.mem n columns -> (n, Type.Any e)
          | _ -> (n, a)
        in
        Schema.v (List.map element (Schema.columns input.schema))
  in
  { node; schema }

let of_table t = make (Of_table t)

let of_source (source : Source.t) =
  make
    (Of_source
       {
         source;
         columns = Schema.names source.schema;
         filters = [];
         limit = None;
       })

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
  | Of_source { source = s; columns; filters; limit } ->
      let all = Schema.names s.schema in
      pf "@[<hov 2>%s (%s" s.name (plural (List.length all) "column");
      Option.iter (fun n -> pf ", %s" (plural n "row")) s.rows;
      pf ")";
      if not (List.equal String.equal columns all) then
        pf "@ ~columns:%a" (Type.pp_list pp_name) columns;
      if not (List.is_empty filters) then
        pf "@ ~filters:%a" (Type.pp_list Expr.pp) filters;
      Option.iter (pf "@ ~limit:%d") limit;
      pf "@]"
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

let map_inputs f = function
  | (Of_table _ | Of_source _) as n -> n
  | Select r -> Select { r with input = f r.input }
  | Derive r -> Derive { r with input = f r.input }
  | Filter r -> Filter { r with input = f r.input }
  | Sort r -> Sort { r with input = f r.input }
  | Slice r -> Slice { r with input = f r.input }
  | Aggregate r -> Aggregate { r with input = f r.input }
  | Join r -> Join { r with left = f r.left; right = f r.right }
  | Append { input; rest } -> Append { input = f input; rest = f rest }
  | Unnest r -> Unnest { r with input = f r.input }

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

(* [shared q] is the steps that [q] reaches more than once. *)
let shared q =
  let seen = ref [] and twice = ref [] in
  let rec visit q =
    if List.memq q !seen then
      begin if not (List.memq q !twice) then twice := q :: !twice
      end
    else begin
      seen := q :: !seen;
      List.iter visit (inputs q)
    end
  in
  visit q;
  !twice

(* Format draws no tree, so each step is formatted alone, at the margin that its
   prefix leaves, and its lines are prefixed by hand: the first by [first], the
   others, and the step's inputs, by [prefix]. Both are [width] columns wide. A
   shared step's label precedes its first line, and stands alone the next
   times. *)
let pp ppf q =
  let margin = Format.pp_get_margin ppf () in
  let shared = shared q and labels = ref [] in
  let rec step ~first ~prefix ~width q =
    match List.assq_opt q !labels with
    | Some k -> Format.fprintf ppf "@,%s#%d" first k
    | None -> (
        let label =
          if not (List.memq q shared) then ""
          else begin
            let k = List.length !labels + 1 in
            labels := (q, k) :: !labels;
            Printf.sprintf "#%d " k
          end
        in
        let pad = String.make (String.length label) ' ' in
        match
          lines
            (margin - width - String.length label)
            (fun ppf -> pp_step ppf q)
        with
        | [] -> assert false
        | line :: rest ->
            Format.fprintf ppf "@,%s%s%s" first label line;
            List.iter (fun l -> Format.fprintf ppf "@,%s%s%s" prefix pad l) rest;
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
            children (inputs q))
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
  make (Select { outputs; input = q })

let derive os q =
  let outputs, _, entries = bind_outs q.schema os in
  check "derive" (input q) entries;
  make (Derive { outputs; input = q })

let filter p q =
  match Expr.bind_predicate q.schema p with
  | Ok predicate -> make (Filter { predicate; input = q })
  | Error ps ->
      fail "filter" (input q) [ Written ((fun ppf -> Expr.pp ppf p), ps) ]

let sort keys q =
  let entry ((k : Order.t), p) = arg "%a: %a" Order.pp k Problem.pp p in
  check "sort" (input q) (List.map entry (Order.check keys q.schema));
  make (Sort { keys; input = q })

let slice ~offset ~length q =
  if length < 0 then
    fail "slice" (input q) [ arg "the length %d is negative." length ];
  make (Slice { offset; length; input = q })

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
  make (Aggregate { by; outputs; input = q })

let join ?(kind = Join.Inner) ?(each_left = Join.Any) ?(each_right = Join.Any)
    ~on right left =
  let inputs = [ ("left", left.schema); ("right", right.schema) ] in
  match Join.check kind left.schema right.schema on with
  | Ok on -> make (Join { kind; each_left; each_right; on; left; right })
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
  make (Append { input = q; rest })

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
  make (Unnest { columns; input = q })

let check_values e q =
  match Expr.bind_value q.schema e with
  | Ok b -> b
  | Error ps ->
      fail "values" (input q) [ Written ((fun ppf -> Expr.pp ppf e), ps) ]

(* Comparing *)

let outputs_equal o0 o1 =
  List.equal
    (fun (n0, Expr.Packed e0) (n1, Expr.Packed e1) ->
      String.equal n0 n1 && Expr.same e0 e1)
    o0 o1

(* [step_equal ~input ~table q0 q1] is [true] iff [q0] and [q1] are the same
   step with equal arguments, inputs compared with [input] and tables with
   [table]. *)
let step_equal ~input ~table q0 q1 =
  (match (q0.node, q1.node) with
    | Of_table t0, Of_table t1 -> table t0 t1
    | Of_source s0, Of_source s1 ->
        s0.source == s1.source
        && List.equal String.equal s0.columns s1.columns
        && List.equal Expr.same s0.filters s1.filters
        && Option.equal Int.equal s0.limit s1.limit
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
  && List.equal input (inputs q0) (inputs q1)

let rec equal q0 q1 =
  q0 == q1 || step_equal ~input:equal ~table:Table.equal q0 q1

let rec same q0 q1 = q0 == q1 || step_equal ~input:same ~table:( == ) q0 q1
