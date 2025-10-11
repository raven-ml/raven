(* schedule.ml *)

open Ir
module S = Ir.Scheduled
module SymVar = Ir.SymVar

(* wiring helpers *)

let compute_dependents (items : S.schedule_item array) : unit =
  let n = Array.length items in
  let deps_rev : int list array = Array.make n [] in
  Array.iter
    (fun (it : S.schedule_item) ->
      List.iter
        (fun (p : int) ->
          if p >= 0 && p < n then deps_rev.(p) <- it.item_id :: deps_rev.(p))
        it.depends_on)
    items;
  Array.iteri
    (fun i (it : S.schedule_item) ->
      items.(i) <- { it with dependents = List.rev deps_rev.(i) })
    items

let topological_order (items : S.schedule_item array) : int list =
  let n = Array.length items in
  let indeg = Array.make n 0 in
  Array.iter
    (fun (it : S.schedule_item) ->
      indeg.(it.item_id) <- List.length it.depends_on)
    items;
  let q = Queue.create () in
  for i = 0 to n - 1 do
    if indeg.(i) = 0 then Queue.add i q
  done;
  let order = ref [] in
  while not (Queue.is_empty q) do
    let u = Queue.pop q in
    order := u :: !order;
    List.iter
      (fun v ->
        indeg.(v) <- indeg.(v) - 1;
        if indeg.(v) = 0 then Queue.add v q)
      items.(u).dependents
  done;
  List.rev !order

(* analysis helpers *)

let find_critical_path (g : S.graph_t) : int list =
  let n = Array.length g.schedule_items in
  if n = 0 then []
  else
    let cost = Array.make n 1 in
    Array.iter
      (fun (a : S.item_analysis) ->
        if a.item_id < n then cost.(a.item_id) <- max 1 a.est_ns)
      g.analysis;
    let dist = Array.make n min_int in
    let prev = Array.make n (-1) in
    (* indegree by depends_on *)
    let indeg = Array.make n 0 in
    Array.iter
      (fun (it : S.schedule_item) ->
        indeg.(it.item_id) <- List.length it.depends_on)
      g.schedule_items;
    let q = Queue.create () in
    for i = 0 to n - 1 do
      if indeg.(i) = 0 then (
        dist.(i) <- cost.(i);
        Queue.add i q)
    done;
    while not (Queue.is_empty q) do
      let u = Queue.pop q in
      List.iter
        (fun v ->
          (if dist.(u) <> min_int then
             let cand = dist.(u) + cost.(v) in
             if cand > dist.(v) then (
               dist.(v) <- cand;
               prev.(v) <- u));
          indeg.(v) <- indeg.(v) - 1;
          if indeg.(v) = 0 then Queue.add v q)
        g.schedule_items.(u).dependents
    done;
    let end_id = ref 0 in
    for i = 1 to n - 1 do
      if dist.(i) > dist.(!end_id) then end_id := i
    done;
    let rec build acc u = if u = -1 then acc else build (u :: acc) prev.(u) in
    build [] !end_id

let sum_estimated_runtime_ns (g : S.graph_t) : int =
  List.fold_left
    (fun acc id ->
      match
        Array.find_opt (fun (a : S.item_analysis) -> a.item_id = id) g.analysis
      with
      | Some a -> acc + max 1 a.est_ns
      | None -> acc + 1)
    0 g.critical_path

let estimate_peak_memory (g : S.graph_t) : int =
  let mem_at_item (it : S.schedule_item) : int =
    match it.operation with
    | S.S_Kernel { inputs; outputs; _ } ->
        let sum l =
          List.fold_left (fun acc b -> acc + b.S.alloc.size_bytes) 0 l
        in
        sum inputs + sum outputs
    | S.S_Memory_Transfer { size_bytes; _ } -> size_bytes
    | _ -> 0
  in
  Array.fold_left (fun acc it -> max acc (mem_at_item it)) 0 g.schedule_items

(* assembly *)

let assemble ~(items : S.schedule_item list) ~(dependencies : S.dependency list)
    ~(fusion : S.fusion_opportunity list) ~(analysis : S.item_analysis list)
    ~(vars_metadata : (Var.t, var_metadata) Hashtbl.t)
    ~(symbolic_vars : SymVar.t list) : S.graph_t =
  let arr =
    Array.of_list
      (List.sort
         (fun (a : S.schedule_item) (b : S.schedule_item) ->
           compare a.item_id b.item_id)
         items)
  in
  compute_dependents arr;
  let analysis_arr = Array.of_list analysis in
  let g0 : S.graph_t =
    {
      schedule_items = arr;
      dependencies;
      fusion_opportunities = fusion;
      analysis = analysis_arr;
      critical_path = [];
      total_memory_usage = 0;
      estimated_runtime_ns = 0;
      vars_metadata;
      symbolic_vars;
    }
  in
  let cp = find_critical_path g0 in
  let total_ns = sum_estimated_runtime_ns { g0 with critical_path = cp } in
  let peak = estimate_peak_memory g0 in
  {
    g0 with
    critical_path = cp;
    estimated_runtime_ns = total_ns;
    total_memory_usage = peak;
  }

module PairTbl = Hashtbl.Make (struct
  type t = int * int

  let equal (a1, b1) (a2, b2) = a1 = a2 && b1 = b2
  let hash = Hashtbl.hash
end)

let build (g : Ir.graph_t) : S.graph_t =
  (* 1) Cluster the high-level graph *)
  let clusters : Grouper.cluster_t list = Grouper.group g in

  (* 2) Kernelize clusters -> scheduled kernel items (+ basic analysis) *)
  let items0, analysis = Kernelize.of_specs clusters in

  (* 3) Producer map: which item produces each Var.t *)
  let prod_of_var : (Var.t, int) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun (it : S.schedule_item) ->
      match it.operation with
      | S.S_Kernel { outputs; _ } ->
          List.iter
            (fun (b : S.buffer_info) ->
              Hashtbl.replace prod_of_var b.S.buf_var it.item_id)
            outputs
      | _ -> ())
    items0;

  let dedup_sorted (xs : int list) = List.sort_uniq compare xs in

  (* 4) Fill depends_on for each item from producers of its inputs *)
  let items_with_deps : S.schedule_item list =
    List.map
      (fun (it : S.schedule_item) ->
        match it.operation with
        | S.S_Kernel { inputs; _ } ->
            let deps =
              List.fold_left
                (fun acc (b : S.buffer_info) ->
                  match Hashtbl.find_opt prod_of_var b.S.buf_var with
                  | Some pid when pid <> it.item_id -> pid :: acc
                  | _ -> acc)
                [] inputs
              |> dedup_sorted
            in
            { it with depends_on = deps }
        | _ -> it)
      items0
  in

  (* 5) Group dependency edges (producer, consumer) -> dep_vars *)
  let groups : Var.Set.t PairTbl.t = PairTbl.create 32 in
  List.iter
    (fun (it : S.schedule_item) ->
      match it.operation with
      | S.S_Kernel { inputs; _ } ->
          List.iter
            (fun (b : S.buffer_info) ->
              match Hashtbl.find_opt prod_of_var b.S.buf_var with
              | Some src when src <> it.item_id ->
                  let key = (src, it.item_id) in
                  let cur =
                    match PairTbl.find_opt groups key with
                    | Some s -> s
                    | None -> Var.Set.empty
                  in
                  PairTbl.replace groups key (Var.Set.add b.S.buf_var cur)
              | _ -> ())
            inputs
      | _ -> ())
    items_with_deps;

  let dependencies : S.dependency list =
    PairTbl.fold
      (fun (src, dst) vars acc ->
        {
          S.dep_from = src;
          dep_to = dst;
          dep_vars = Var.Set.elements vars;
          kind = `Data;
        }
        :: acc)
      groups []
  in

  (* Gather symbolic vars from metadata to surface in the scheduled graph. *)
  let collected_symbolics =
    let tbl : (int, SymVar.t) Hashtbl.t = Hashtbl.create 16 in
    Hashtbl.iter
      (fun _ meta ->
        match meta.shape_expr with
        | Some expr ->
            Array.iter
              (function
                | Shape_expr.Var v ->
                    let id = Shape_expr.Var.id v in
                    if not (Hashtbl.mem tbl id) then
                      let raw_name = Shape_expr.Var.name v in
                      let name =
                        if String.length raw_name = 0 then
                          Printf.sprintf "v%d" id
                        else raw_name
                      in
                      Hashtbl.add tbl id
                        {
                          SymVar.name;
                          min_val = Shape_expr.Var.min v;
                          max_val = Shape_expr.Var.max v;
                        }
                | _ -> ())
              expr
        | None -> ())
      g.vars_metadata;
    Hashtbl.fold (fun id sym acc -> (id, sym) :: acc) tbl []
    |> List.sort (fun (a, _) (b, _) -> compare a b)
    |> List.map snd
  in
  let module SymKeySet = Set.Make (struct
    type t = string * int * int

    let compare = compare
  end) in
  let existing_keys =
    List.fold_left
      (fun acc (sym : SymVar.t) ->
        let SymVar.{ name; min_val; max_val } = sym in
        SymKeySet.add (name, min_val, max_val) acc)
      SymKeySet.empty g.symbolic_vars
  in
  let _, added_rev =
    List.fold_left
      (fun (seen, acc_rev) (sym : SymVar.t) ->
        let SymVar.{ name; min_val; max_val } = sym in
        let key = (name, min_val, max_val) in
        if SymKeySet.mem key seen then (seen, acc_rev)
        else (SymKeySet.add key seen, sym :: acc_rev))
      (existing_keys, []) collected_symbolics
  in
  let symbolic_vars = g.symbolic_vars @ List.rev added_rev in

  (* 6) Assemble final scheduled graph *)
  assemble ~items:items_with_deps ~dependencies ~fusion:[] ~analysis
    ~vars_metadata:g.vars_metadata ~symbolic_vars
