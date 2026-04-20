(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module T = Tensor

let debug = Helpers.getenv "DEBUG" 0
let scache_enabled = Helpers.getenv "SCACHE" 1

(* Schedule linearizer *)

(* Follow src[0] chains through movement ops until hitting a data
   source: After, Buffer, Param, Mselect, Mstack, or Bind. *)
let rec unwrap_src (node : T.t) : T.t =
  match T.view node with
  | After _ | Buffer _ | Param _ | Mselect _ | Mstack _ | Bind _ -> node
  | v ->
      match T.children_of v with
      | src :: _ -> unwrap_src src
      | [] -> node

(* Build kernel dependency graph from the scheduled tensor graph and
   topologically sort it into a LINEAR node.

   In an After node, deps.(0) is the kernel (Call or End) and
   deps.(1..) are WAR dependencies from rangeify. *)
let create_schedule (sink : T.t) : T.t =
  (* Phase 1: build dependency graph.
     children_map.(k) = kernels that depend on k.
     in_degree.(k) = number of unresolved dependencies of k. *)
  let children_map : (int, T.t list) Hashtbl.t = Hashtbl.create 64 in
  let in_degree : (int, int) Hashtbl.t = Hashtbl.create 64 in
  let degree_nodes : (int, T.t) Hashtbl.t = Hashtbl.create 64 in
  let key n = T.tag n in
  let add_child producer consumer =
    let pk = key producer in
    let prev = match Hashtbl.find_opt children_map pk with
      | Some l -> l | None -> [] in
    Hashtbl.replace children_map pk (consumer :: prev);
    let ck = key consumer in
    let deg = match Hashtbl.find_opt in_degree ck with
      | Some n -> n | None -> 0 in
    Hashtbl.replace in_degree ck (deg + 1)
  in
  let ensure_in_degree k =
    let tag = key k in
    if not (Hashtbl.mem in_degree tag) then begin
      Hashtbl.replace in_degree tag 0;
      Hashtbl.replace degree_nodes tag k
    end
  in
  let slice = T.backward_slice sink in
  List.iter (fun u ->
    match T.view u with
    | After { deps; _ } -> begin
        match deps with
        | [] -> ()
        | k :: war_deps ->
            (match T.view k with
             (* Skip unprocessed STORE+AFTER inside precompiled CALL bodies *)
             | Store _ -> ()
             | Call _ | End _ ->
                 ensure_in_degree k;
                 (match T.view k with
                  | End { value; _ } ->
                      (match T.view value with
                       | Call _ -> ()
                       | v -> invalid_arg (Format.asprintf
                           "END src[0] should be CALL, not %a" T.pp_view v))
                  | _ -> ());
                 let kernel_deps = match T.view k with
                   | End { value; _ } ->
                       (match T.view value with
                        | Call { args; _ } -> args
                        | _ -> [])
                   | Call { args; _ } -> args
                   | _ -> []
                 in
                 List.iter (fun s ->
                   let s = unwrap_src s in
                   match T.view s with
                   | After { deps = s_deps; _ } ->
                       (match s_deps with
                        | s_kernel :: _ -> add_child s_kernel k
                        | [] -> ())
                   | Mselect _ | Mstack _ ->
                       List.iter (fun ss ->
                         let ss = match T.view ss with
                           | Mselect { src; _ } -> src | _ -> ss in
                         match T.view ss with
                         | Buffer _ | Param _ -> ()
                         | After { deps = ss_deps; _ } ->
                             (match ss_deps with
                              | ss_kernel :: _ -> add_child ss_kernel k
                              | [] -> ())
                         | v -> invalid_arg (Format.asprintf
                             "expected AFTER, got %a" T.pp_view v))
                         (T.children s)
                   | Buffer _ | Param _ | Bind _ -> ()
                   | v -> invalid_arg (Format.asprintf
                       "input to kernel must be AFTER, BUFFER, PARAM, \
                        MSELECT, MSTACK, or BIND, not %a" T.pp_view v))
                   (kernel_deps @ war_deps)
             | v -> invalid_arg (Format.asprintf
                 "AFTER deps[0] should be CALL or END, not %a" T.pp_view v))
      end
    | _ -> ())
    slice;
  (* Phase 2: BFS topological sort. *)
  let queue = Queue.create () in
  Hashtbl.iter (fun tag deg ->
    if deg = 0 then
      Queue.add (Hashtbl.find degree_nodes tag) queue)
    in_degree;
  let linearized = ref [] in
  while not (Queue.is_empty queue) do
    let rk = Queue.pop queue in
    (match T.view rk with
     | Linear { srcs } ->
         linearized := List.rev_append srcs !linearized
     | _ ->
         let k = match T.view rk with
           | End { value; _ } -> value | _ -> rk in
         (match T.view k with
          | Call { callee; args; info; dtype } ->
              let buf_nodes = List.filter (fun s ->
                match T.view s with Bind _ -> false | _ -> true) args in
              let buf_nodes = List.map unwrap_src buf_nodes in
              let new_call = T.call ~callee ~args:buf_nodes ~info ~dtype in
              linearized := new_call :: !linearized
          | v -> invalid_arg (Format.asprintf
              "unexpected op in queue: %a" T.pp_view v)));
    let succs = match Hashtbl.find_opt children_map (key rk) with
      | Some l -> l | None -> [] in
    List.iter (fun x ->
      let xk = key x in
      let deg = Hashtbl.find in_degree xk - 1 in
      Hashtbl.replace in_degree xk deg;
      if deg = 0 then Queue.add x queue)
      succs
  done;
  T.linear (List.rev !linearized)

(* Convert a Linear node to ExecItem list.
   [buffers] maps tensor nodes to runtime Buffer.t values. *)
let linear_to_schedule (linear : T.t)
    ~(buffers : T.t -> Device.Buffer.t option) : Realize.Exec_item.t list =
  let srcs = match T.view linear with
    | Linear { srcs } -> srcs
    | _ -> invalid_arg "linear_to_schedule: expected Linear node"
  in
  let schedule = ref [] in
  List.iter (fun si ->
    match T.view si with
    | Call { callee; args; _ } ->
        (* Create subbuffer views if the callee is a Buffer_view *)
        (match callee with
         | Ref ref_node -> begin
             match T.view ref_node with
             | Buffer_view { size; offset; dtype; _ } -> begin
                 match args with
                 | _dst :: base_node :: _ ->
                     (match buffers base_node with
                      | Some base ->
                          let _view = Device.Buffer.view base ~size ~dtype
                            ~offset:(offset * Dtype.itemsize dtype) in
                          (* XXX: register view in buffer table *)
                          ()
                      | None -> ())
                 | _ -> ()
               end
             | _ -> ()
           end
         | Ast _ -> ());
        let buf_nodes = List.filter (fun s ->
          match T.view s with Bind _ -> false | _ -> true) args in
        let bufs = List.map buffers buf_nodes in
        (* XXX: multi-device expansion not yet implemented *)
        schedule :=
          Realize.Exec_item.make ~ast:si ~bufs () :: !schedule
    | _ -> ())
    srcs;
  List.rev !schedule

(* Resolve PARAM nodes to actual buffers. *)
let post_sched_cache_rule ~(param_bufs : T.t list) (node : T.t)
    : T.t option =
  match T.view node with
  | Param { slot; _ } ->
      if slot >= 0 && slot < List.length param_bufs then
        Some (List.nth param_bufs slot)
      else None
  | _ -> None

(* Resolve CALL(LINEAR, ...) by substituting PARAMs with buffer
   arguments. Flatten nested LINEAR nodes. *)
let resolve_linear_call_rule (node : T.t) : T.t option =
  match T.view node with
  | Call { callee = Ref ref_node; args; _ } -> begin
      match T.view ref_node with
      | Linear _ ->
          Some (T.graph_rewrite
            (post_sched_cache_rule ~param_bufs:args)
            ref_node)
      | _ -> None
    end
  | Linear { srcs } ->
      let has_nested = List.exists (fun s ->
        match T.view s with Linear _ -> true | _ -> false) srcs in
      if has_nested then
        let flat = List.concat_map (fun s ->
          match T.view s with
          | Linear { srcs = inner } -> inner
          | _ -> [s]) srcs
        in
        Some (T.linear flat)
      else None
  | _ -> None

(* Schedule cache *)

let schedule_cache : (int, T.t) Hashtbl.t = Hashtbl.create 64

(* Convert a tensor-level SINK into a LINEAR node. *)
let lower_sink_to_linear ~get_kernel_graph (sink : T.t) : T.t option =
  match T.view sink with
  | Sink { kernel_info = Some _; _ } -> None
  | Sink _ ->
      let st = Unix.gettimeofday () in
      let cache_key = T.tag sink in
      let linear =
        if scache_enabled <> 0 then
          match Hashtbl.find_opt schedule_cache cache_key with
          | Some cached -> cached
          | None ->
              let kernel_graph = get_kernel_graph sink in
              let r = create_schedule kernel_graph in
              Hashtbl.replace schedule_cache cache_key r;
              r
        else
          create_schedule (get_kernel_graph sink)
      in
      if (debug >= 1 && (match T.view linear with
            | Linear { srcs } -> List.length srcs > 1
            | _ -> false))
         || debug >= 3
      then begin
        let n = match T.view linear with
          | Linear { srcs } -> List.length srcs | _ -> 0 in
        Printf.eprintf "scheduled %5d kernels in %8.2f ms\n%!"
          n ((Unix.gettimeofday () -. st) *. 1000.)
      end;
      Some linear
  | _ -> None

(* Full schedule pipeline: tensor graph → ExecItem list + var_vals.

   [big_sink] is either a raw SINK (legacy) or a CALL produced by
   {!Allocations_next.transform_to_call}.

   1. Lower each SINK to a LINEAR via get_kernel_graph + create_schedule.
      [enter_calls] lets us descend into CALL bodies from allocations.
   2. Resolve CALL(LINEAR, ...) by substituting PARAMs with buffers.
   3. Extract bound variable values from BIND nodes.
   4. Convert the final LINEAR to ExecItems. *)
let complete_create_schedule_with_vars ~get_kernel_graph
    ~(buffers : T.t -> Device.Buffer.t option)
    (big_sink : T.t)
    : Realize.Exec_item.t list * (string * int) list =
  (* Step 1: lower SINKs to LINEARs *)
  let graph = T.graph_rewrite ~enter_calls:true (fun node ->
    lower_sink_to_linear ~get_kernel_graph node)
    big_sink
  in
  (* Step 2: resolve CALL(LINEAR, ...) *)
  let graph = T.graph_rewrite resolve_linear_call_rule graph in
  (* Step 3: find the LINEAR result *)
  let linear = match T.view graph with
    | Linear _ -> graph
    | _ -> graph
  in
  (* Step 4: extract var_vals from BIND nodes.
     When big_sink is a CALL from allocations, BINDs are among the
     args; when it is a raw SINK, they are among the srcs. *)
  let var_vals = ref [] in
  let extract_binds nodes =
    List.iter (fun src ->
      match T.view src with
      | Bind { var; value = Some v; _ } -> begin
          match T.view var, T.view v with
          | Define_var { name; _ }, Const { value; _ } ->
              (match Const.view value with
               | Int n ->
                   let n = Int64.to_int n in
                   (match List.assoc_opt name !var_vals with
                    | Some prev when prev <> n ->
                        invalid_arg (Printf.sprintf
                          "bind mismatch on %s, %d <> %d" name prev n)
                    | _ -> var_vals := (name, n) :: !var_vals)
               | _ -> ())
          | _ -> ()
        end
      | _ -> ()) nodes
  in
  (match T.view big_sink with
   | Sink { srcs; _ } -> extract_binds srcs
   | Call { args; _ } -> extract_binds args
   | _ -> ());
  (* Step 5: convert LINEAR to ExecItems *)
  let schedule = linear_to_schedule linear ~buffers in
  (schedule, !var_vals)
