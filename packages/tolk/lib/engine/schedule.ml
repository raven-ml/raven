(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop
module U = Uop

let debug = Helpers.getenv "DEBUG" 0
let scache_enabled = Helpers.getenv "SCACHE" 1
let no_memory_planner = Helpers.getenv "NO_MEMORY_PLANNER" 0 <> 0
let next_post_sched_buffer_slot = ref (-1)

let round_up n align = (n + align - 1) / align * align

(* Schedule linearizer *)

let is_op op n = Ops.equal (U.op n) op

let is_global_addrspace = function
  | Dtype.Global -> true
  | Dtype.Local | Dtype.Reg | Dtype.Alu -> false

let after_parts n =
  match U.op n, U.children n with
  | Ops.After, src :: deps -> Some (src, deps)
  | _ -> None

let split_after after =
  match after_parts after with
  | None -> invalid_arg "split_after: expected AFTER"
  | Some (_, deps) ->
      let rec loop kernels after_deps = function
        | [] -> List.rev kernels, List.rev after_deps
        | dep :: deps ->
            (match U.op dep with
             | Ops.Call | Ops.End -> loop (dep :: kernels) after_deps deps
             | Ops.After -> loop kernels (dep :: after_deps) deps
             | Ops.Store -> loop kernels after_deps deps
             | _ ->
                 invalid_arg
                   (Format.asprintf
                      "AFTER source should be CALL, END, STORE, or AFTER, not %a"
                      Ops.pp (U.op dep)))
      in
      loop [] [] deps

let linear_srcs n =
  match U.op n with
  | Ops.Linear -> Some (U.children n)
  | _ -> None

let call_arg_uops args = List.filter (fun s -> not (is_op Ops.Bind s)) args

let gate_kernel_sink n =
  match U.op n with
  | Ops.Linear -> false
  | Ops.Sink when Option.is_some (U.as_kernel_info n) -> false
  | _ -> true

(* Follow src[0] chains through movement ops until hitting a data
   source: After, Buffer, Param, Mselect, Mstack, or Bind. *)
let rec unwrap_src (node : U.t) : U.t =
  match U.op node with
  | Ops.After | Ops.Buffer | Ops.Param | Ops.Mselect | Ops.Mstack | Ops.Bind ->
      node
  | _ ->
      match U.children node with
      | src :: _ -> unwrap_src src
      | [] -> node

let call_arg_buffer_node node = U.buf_uop (unwrap_src node)

(* Unwrap a kernel input to the buffer states (After, Buffer, or Param) it
   resolves to. Mselect/Mstack join per-device states, Bind is not a buffer
   dependency. *)
let states s =
  let rec loop s =
    let s = unwrap_src s in
    match U.op s with
    | Ops.Mselect | Ops.Mstack -> List.concat_map loop (U.children s)
    | Ops.Bind -> []
    | Ops.After | Ops.Buffer | Ops.Param -> [ s ]
    | _ ->
        invalid_arg
          (Format.asprintf
             "input to kernel must resolve to a buffer state, not %a" Ops.pp
             (U.op s))
  in
  loop s

(* Build kernel dependency graph from the scheduled tensor graph and
   topologically sort it into a LINEAR node.

   In an After node, children after src.(0) are partitioned like tinygrad:
   Call/End nodes are kernels, After nodes are dependencies, and Store
   nodes are ignored. RAW edges run a kernel after the kernels that produced
   the states it reads; WAR edges run a kernel reading buffer state S before
   any later write that supersedes S. *)
let create_schedule (sink : U.t) : U.t =
  let children_map : (int, U.t list) Hashtbl.t = Hashtbl.create 64 in
  let in_degree : (int, int) Hashtbl.t = Hashtbl.create 64 in
  let degree_nodes : (int, U.t) Hashtbl.t = Hashtbl.create 64 in
  let degree_order = ref [] in
  let key n = U.tag n in
  let add_child producer consumer =
    let pk = key producer in
    let prev =
      match Hashtbl.find_opt children_map pk with
      | Some l -> l
      | None -> []
    in
    Hashtbl.replace children_map pk (consumer :: prev);
    let ck = key consumer in
    let deg =
      match Hashtbl.find_opt in_degree ck with
      | Some n -> n
      | None -> 0
    in
    Hashtbl.replace in_degree ck (deg + 1)
  in
  let ensure_in_degree k =
    let tag = key k in
    if not (Hashtbl.mem in_degree tag) then begin
      Hashtbl.replace in_degree tag 0;
      Hashtbl.replace degree_nodes tag k;
      degree_order := k :: !degree_order
    end
  in
  let kernel_deps k =
    match U.as_end k with
    | Some { value; _ } ->
        if not (is_op Ops.Call value) then
          invalid_arg
            (Format.asprintf "END src[0] should be CALL, not %a" U.pp value);
        (match U.as_call value with Some { args; _ } -> args | None -> [])
    | None -> (match U.as_call k with Some { args; _ } -> args | None -> [])
  in
  (* buffer -> (After, prior state, new kernels) records for WAR analysis. *)
  let writes : (int, (U.t * U.t * U.t list) list) Hashtbl.t =
    Hashtbl.create 64
  in
  let add_write buf_tag entry =
    let prev =
      match Hashtbl.find_opt writes buf_tag with Some l -> l | None -> []
    in
    Hashtbl.replace writes buf_tag (entry :: prev)
  in
  (* (reader After, reader kernel, buffer state read) triples. *)
  let reads = ref [] in
  let slice = U.toposort ~gate:gate_kernel_sink ~enter_calls:false sink in
  List.iter
    (fun u ->
      match after_parts u with
      | None -> ()
      | Some (src, _) ->
          let kernels, after_deps = split_after u in
          let prev_state = unwrap_src src in
          let prev_kernels =
            match U.op prev_state with
            | Ops.After -> List.map key (fst (split_after prev_state))
            | _ -> []
          in
          let new_kernels =
            List.filter (fun k -> not (List.mem (key k) prev_kernels)) kernels
          in
          add_write (key (U.buf_uop u)) (u, prev_state, new_kernels);
          List.iter
            (fun k ->
              ensure_in_degree k;
              let read_states = List.concat_map states (kernel_deps k) in
              List.iter (fun st -> reads := (u, k, st) :: !reads) read_states;
              (* RAW: run k after the kernels producing the states it reads
                 or joins. *)
              let dep_states =
                read_states @ List.concat_map states after_deps
              in
              List.iter
                (fun st ->
                  if is_op Ops.After st then
                    List.iter (fun t -> add_child t k) (fst (split_after st)))
                dep_states)
            kernels)
    slice;
  (* WAR: a kernel reading buffer state S must run before another write that
     supersedes S. An After only supersedes its immediate prior state; join
     members already present in that prior state are ordering deps, not
     writes. *)
  List.iter
    (fun (u, k, s) ->
      match Hashtbl.find_opt writes (key (U.buf_uop s)) with
      | None -> ()
      | Some entries ->
          List.iter
            (fun (a, prev_state, write_kernels) ->
              if U.equal a u || not (U.equal prev_state s) then ()
              else
                List.iter
                  (fun t ->
                    if not (U.equal t k) && not (U.in_backward_slice t k) then
                      add_child k t)
                  write_kernels)
            (List.rev entries))
    (List.rev !reads);
  (* BFS topological sort. *)
  let queue = Queue.create () in
  List.iter
    (fun node ->
      let tag = key node in
      match Hashtbl.find_opt in_degree tag with
      | Some 0 -> Queue.add (Hashtbl.find degree_nodes tag) queue
      | Some _ | None -> ())
    (List.rev !degree_order);
  let linearized = ref [] in
  while not (Queue.is_empty queue) do
    let rk = Queue.pop queue in
    (match linear_srcs rk with
     | Some srcs ->
         linearized := List.rev_append srcs !linearized
     | None ->
         let k = match U.as_end rk with
           | Some { value; _ } -> value | None -> rk in
         (match U.as_call k with
          | Some { body; args; info } ->
              let buf_nodes = call_arg_uops args in
              let buf_nodes = List.map call_arg_buffer_node buf_nodes in
              let new_call = U.call ~body ~args:buf_nodes ~info in
              linearized := new_call :: !linearized
          | None ->
              invalid_arg
                (Format.asprintf "unexpected op in queue: %a" U.pp k)));
    let succs =
      match Hashtbl.find_opt children_map (key rk) with
      | Some l -> List.rev l
      | None -> []
    in
    List.iter
      (fun x ->
        let xk = key x in
        let deg = Hashtbl.find in_degree xk - 1 in
        Hashtbl.replace in_degree xk deg;
        if deg = 0 then Queue.add x queue)
      succs
  done;
  U.linear (List.rev !linearized)

(* Resolve cached LINEAR calls. *)

type post_sched_cache_ctx = {
  param_bufs : U.t list;
  created_buffers : (int, U.t) Hashtbl.t;
}

let param_slot_arg args idx =
  if idx >= 0 && idx < List.length args then Some (List.nth args idx)
  else None

let create_post_sched_buffer ctx b =
  let tag = U.tag b in
  match Hashtbl.find_opt ctx.created_buffers tag with
  | Some ret -> ret
  | None ->
      let ret =
        match U.as_buffer b with
        | Some { buffer; shape } ->
            let slot = !next_post_sched_buffer_slot in
            decr next_post_sched_buffer_slot;
            U.buffer ~slot ~dtype:(U.dtype b)
              ~shape ?name:buffer.name ~addrspace:buffer.addrspace
              ?axis:buffer.axis ?device:buffer.device ()
        | None -> assert false
      in
      Hashtbl.replace ctx.created_buffers tag ret;
      ret

let post_sched_cache_rule ctx node =
  match U.as_param node, U.as_buffer node with
  | Some { param = { slot; _ }; _ }, _ -> param_slot_arg ctx.param_bufs slot
  | _, Some { buffer; _ } ->
      if buffer.slot >= 0 && is_global_addrspace buffer.addrspace then
        Some (create_post_sched_buffer ctx node)
      else None
  | None, None -> None

(* Resolve CALL(LINEAR, ...) by substituting PARAMs with buffer
   arguments. Flatten nested LINEAR nodes. *)
let resolve_linear_call_rule (node : U.t) : U.t option =
  match U.as_call node with
  | Some { body; args; _ } ->
      if is_op Ops.Linear body then
        let ctx =
          { param_bufs = call_arg_uops args;
            created_buffers = Hashtbl.create 16 }
        in
        Some (U.graph_rewrite ~walk:true (post_sched_cache_rule ctx) body)
      else None
  | None when is_op Ops.Linear node ->
      let srcs = U.children node in
      let has_nested = List.exists (fun s -> is_op Ops.Linear s) srcs in
      if has_nested then
        let flat =
          List.concat_map
            (fun s ->
              match linear_srcs s with
              | Some inner -> inner
              | None -> [ s ])
            srcs
        in
        Some (U.linear flat)
      else None
  | _ -> None

(* UOp-level memory planner.

   Mirrors tinygrad.schedule.memory.memory_plan_rewrite: internal buffers in a
   LINEAR are replaced by slices into per-device arenas before the schedule is
   returned. The runtime planner still handles concrete Device buffers, but this
   pass is what makes the scheduled UOp graph itself match tinygrad. *)

let collect_bufs u =
  let rec loop acc u =
    match U.op u with
    | Ops.Buffer -> u :: acc
    | Ops.Mselect | Ops.Mstack ->
        List.fold_left loop acc (U.children u)
    | _ -> acc
  in
  loop [] u

let buffer_numel b =
  U.max_shard_shape b |> List.fold_left ( * ) 1

let buffer_nbytes b =
  buffer_numel b * Dtype.itemsize (U.dtype b)

let buffer_device = function
  | U.Single d -> Some d
  | U.Multi _ | U.Index _ -> None

let plannable_buffer held b =
  match U.as_buffer b, U.device_of b with
  | Some { buffer = { addrspace = Dtype.Global; _ }; _ }, Some device -> (
      match buffer_device device with
      | Some dev ->
          (not (List.exists (( == ) b) held))
          && not
               (String.starts_with ~prefix:"DISK" dev
               || String.starts_with ~prefix:"TINYFS" dev)
      | None -> false)
  | _ -> false

let rec call_buffers held call =
  match U.as_end call with
  | Some { value; ranges } ->
      call_buffers held value
      @ (ranges
         |> List.concat_map collect_bufs
         |> List.filter (plannable_buffer held))
  | None -> (
      match U.as_call call with
      | None -> []
      | Some { body; args; _ } ->
          (body :: args)
          |> List.concat_map collect_bufs
          |> List.filter (plannable_buffer held))

let rec call_is_copy call =
  match U.as_end call with
  | Some { value; _ } -> call_is_copy value
  | None -> (
      match U.as_call call with
      | Some { body; _ } -> is_op Ops.Copy body
      | None -> false)

type memory_lane = string * int

let memory_lane copy_bufs b =
  match U.device_of b with
  | Some device -> (
      match buffer_device device with
      | Some dev ->
          (dev, if Hashtbl.mem copy_bufs (U.tag b) then 1 else 0)
      | None -> invalid_arg "memory_plan_rewrite: multi-device buffer")
  | None -> invalid_arg "memory_plan_rewrite: buffer without device"

let create_arena_buffer ~device ~nbytes =
  let slot = !next_post_sched_buffer_slot in
  decr next_post_sched_buffer_slot;
  U.buffer ~slot ~dtype:Dtype.int8
    ~shape:(U.const (Const.int Dtype.Val.weakint nbytes))
    ~device:(U.Single device) ()

let memory_plan_rewrite linear held_bufs =
  let substitute_memory_plan replacements linear =
    U.graph_rewrite ~enter_calls:true ~bottom_up:true
      (fun node -> List.assq_opt node replacements)
      linear
  in
  match linear_srcs linear with
  | None -> linear
  | Some calls ->
      if no_memory_planner then linear
      else
        let first = Hashtbl.create 16 in
        let last = Hashtbl.create 16 in
        let buffers = Hashtbl.create 16 in
        let copy_bufs = Hashtbl.create 16 in
        List.iteri
          (fun i call ->
             let bufs = call_buffers held_bufs call in
             List.iter
               (fun b ->
                  let tag = U.tag b in
                  if not (Hashtbl.mem first tag) then begin
                    Hashtbl.replace first tag i;
                    Hashtbl.replace buffers tag b
                  end;
                  Hashtbl.replace last tag i)
               bufs;
             if call_is_copy call then
               List.iter
                 (fun b -> Hashtbl.replace copy_bufs (U.tag b) ())
                 bufs)
          calls;
        if Hashtbl.length first = 0 then linear
        else
          let hold b =
            let tag = U.tag b in
            if Hashtbl.mem copy_bufs tag then
              Hashtbl.find last tag - Hashtbl.find first tag + 1
            else 0
          in
          let events =
            Hashtbl.fold
              (fun tag b acc ->
                 ((Hashtbl.find first tag, 1), b)
                 :: ((Hashtbl.find last tag + 1 + hold b, 0), b)
                 :: acc)
              buffers []
            |> List.sort (fun (a, _) (b, _) -> compare a b)
          in
          let total_memory =
            Hashtbl.fold
              (fun _ b acc -> acc + round_up (buffer_nbytes b) 256)
              buffers 0
            * 2
          in
          let planners : (memory_lane, int * Tlsf.t) Hashtbl.t =
            Hashtbl.create 8
          in
          let get_planner lane =
            match Hashtbl.find_opt planners lane with
            | Some planner -> planner
            | None ->
                let planner =
                  ( 0,
                    Tlsf.create ~size:total_memory ~block_size:256
                      ~lv2_cnt:32 () )
                in
                Hashtbl.replace planners lane planner;
                planner
          in
          let offsets = Hashtbl.create 16 in
          List.iter
            (fun ((_, is_alloc), b) ->
               let tag = U.tag b in
               let lane = memory_lane copy_bufs b in
               let peak, tlsf = get_planner lane in
               let offset =
                 if is_alloc = 1 then begin
                   let offset =
                     Tlsf.alloc tlsf (round_up (buffer_nbytes b) 256) ()
                   in
                   Hashtbl.replace offsets tag offset;
                   offset
                 end
                 else begin
                   let offset = Hashtbl.find offsets tag in
                   Tlsf.free tlsf offset;
                   offset
                 end
               in
               Hashtbl.replace planners lane
                 (max peak (offset + buffer_nbytes b), tlsf))
            events;
          let arenas = Hashtbl.create 8 in
          Hashtbl.iter
            (fun (device, _ as lane) (peak, _) ->
               let nbytes = round_up peak 256 in
               if nbytes > 0 then
                 Hashtbl.replace arenas lane
                   (create_arena_buffer ~device ~nbytes))
            planners;
          let replacements =
            Hashtbl.fold
              (fun tag b acc ->
                 let lane = memory_lane copy_bufs b in
                 let arena = Hashtbl.find arenas lane in
                 let offset =
                   U.const
                     (Const.int Dtype.Val.weakint
                        (Hashtbl.find offsets tag))
                 in
                 let slice =
                   U.slice ~src:arena ~offset ~size:(buffer_numel b)
                     ~dtype:(U.dtype b)
                 in
                 (b, slice) :: acc)
              buffers []
          in
          substitute_memory_plan replacements linear

(* Schedule cache *)

let schedule_cache : (string, U.t) Hashtbl.t = Hashtbl.create 64

let schedule_cache_key function_ = U.semantic_key function_

(* Convert a tensor-level SINK into a LINEAR node. *)
let lower_sink_to_linear ~get_kernel_graph (sink : U.t) : U.t option =
  match U.op sink with
  | Ops.Sink when Option.is_some (U.as_kernel_info sink) -> None
  | Ops.Sink ->
      let st = Unix.gettimeofday () in
      let cache_key = schedule_cache_key sink in
      let cache_hit = ref false in
      let linear =
        if scache_enabled <> 0 then
          match Hashtbl.find_opt schedule_cache cache_key with
          | Some cached ->
              cache_hit := true;
              cached
          | None ->
              let kernel_graph = get_kernel_graph sink in
              let r = create_schedule kernel_graph in
              Hashtbl.replace schedule_cache cache_key r;
              r
        else
          create_schedule (get_kernel_graph sink)
      in
      if
        (debug >= 1
        &&
        match linear_srcs linear with
        | Some srcs -> List.length srcs > 1
        | None -> false)
        || debug >= 3
      then begin
        let n =
          match linear_srcs linear with
          | Some srcs -> List.length srcs
          | None -> 0
        in
        Printf.eprintf "scheduled %5d kernels in %8.2f ms | %s %s\n%!"
          n ((Unix.gettimeofday () -. st) *. 1000.)
          (if scache_enabled <> 0 && !cache_hit then "cache hit"
           else "CACHE MISS")
          (String.sub cache_key 0 (min 8 (String.length cache_key)))
      end;
      Some linear
  | _ -> None

let variables_of_kernel_body body =
  U.toposort ~enter_calls:true body
  |> List.filter_map (fun node ->
         match U.as_param node with
         | Some { param = { slot = -1; name = Some name; _ }; _ } -> Some name
         | _ -> None)
  |> List.sort_uniq String.compare

(* Full schedule pipeline: tensor graph -> Linear + var_vals. *)
let create_linear_with_vars ?(memory_plan = true) ~get_kernel_graph
    (big_sink : U.t)
    : U.t * (string * int) list =
  (* Step 1: lower SINKs to LINEARs *)
  let graph =
    U.graph_rewrite ~enter_calls:true
      (fun node -> lower_sink_to_linear ~get_kernel_graph node)
      big_sink
  in
  let held_bufs =
    match U.as_call graph with
    | Some { args; _ } ->
        List.filter (fun arg -> is_op Ops.Buffer arg) args
    | None -> []
  in
  (* Step 2: resolve CALL(LINEAR, ...) into the LINEAR result *)
  let linear = U.graph_rewrite resolve_linear_call_rule graph in
  let linear = if memory_plan then memory_plan_rewrite linear held_bufs else linear in
  let linear_srcs =
    match linear_srcs linear with
    | Some srcs -> srcs
    | None -> invalid_arg "create_linear_with_vars: expected Linear node"
  in
  (* Step 4: extract var_vals from used BIND nodes. *)
  let used_vars =
    List.concat_map
      (fun si ->
        match U.as_call si with
        | Some { body; _ } -> variables_of_kernel_body body
        | None -> [])
      linear_srcs
    |> List.sort_uniq String.compare
  in
  let var_vals = ref [] in
  let extract_binds nodes =
    List.iter (fun src ->
      match U.as_bind src with
      | Some { var; value = v } -> begin
          match U.as_param var, U.op v, U.arg v with
          | Some { param = { name = Some name; _ }; _ }, Ops.Const,
            U.Arg.Value value ->
              if List.mem name used_vars then
                (match Const.view value with
                 | Int n ->
                     let n = Int64.to_int n in
                     (match List.assoc_opt name !var_vals with
                      | Some prev when prev <> n ->
                          invalid_arg (Printf.sprintf
                            "bind mismatch on %s, %d <> %d" name prev n)
                      | Some _ -> ()
                      | None -> var_vals := (name, n) :: !var_vals)
                 | _ -> ())
          | _ -> ()
        end
      | _ -> ()) nodes
  in
  (match U.op big_sink, U.as_call big_sink with
   | Ops.Sink, _ -> extract_binds (U.children big_sink)
   | _, Some { args; _ } -> extract_binds args
   | _ -> ());
  (linear, !var_vals)
