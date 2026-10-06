(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let with_ctx = Pattern_matcher.with_ctx
let ops = Op.Set.of_list
let var = Upat.var
let setting = Setting.value

(* Ordered tables: the keys in the order they were first added. *)
module Ordered = struct
  type 'a t = { tbl : 'a Tbl.t; mutable keys : Ops.t list }

  let create () = { tbl = Tbl.create 16; keys = [] }
  let find_opt t k = Tbl.find_opt t.tbl k

  let replace t k x =
    if not (Tbl.mem t.tbl k) then t.keys <- k :: t.keys;
    Tbl.replace t.tbl k x

  let bindings t = List.rev_map (fun k -> (k, Tbl.find t.tbl k)) t.keys
end

(* Linearizing *)

(* The data source a kernel reads: a kernel's output, a buffer, or a value on
   several devices. *)
let rec unwrap_src s =
  match (op s, src s) with
  | (Op.After | Op.Buffer | Op.Alloc | Op.Param | Op.Mselect | Op.Mstack), _
  | _, [] ->
      s
  | _, x :: _ -> unwrap_src x

(* A buffer's state is an after or the storage; shard selections and stacks join
   the states of each device. *)
let rec states s =
  let s = unwrap_src s in
  match op s with
  | Op.Mselect | Op.Mstack -> List.concat_map states (src s)
  | Op.After | Op.Buffer | Op.Alloc | Op.Param -> [ s ]
  | o ->
      invalid_arg
        (Format.asprintf "a kernel's input is a buffer state, not %a" Op.pp o)

(* An empty argument of a precompiled call is its constant (Rangeify): the body,
   scheduled on its own, reaches no element of it, and it is no state. *)
let empty_argument k s =
  op (unwrap_src s) = Op.Const
  && match arg k with Call c -> c.precompile | _ -> false

(* A loop around a call: a range's end, or a back edge. *)
let is_loop k = op k = Op.End || op k = Op.Backedge

let split_after after =
  let effects = List.tl (src after) in
  let kernels, rest =
    List.partition (fun s -> op s = Op.Call || is_loop s) effects
  in
  let deps, rest = List.partition (fun s -> op s = Op.After) rest in
  (match List.find_opt (fun s -> op s <> Op.Store) rest with
  | Some s ->
      invalid_arg
        (Format.asprintf
           "an after orders a call, a loop, a store or an after, not %a" Op.pp
           (op s))
  | None -> ());
  (kernels, deps)

let create_schedule sched_sink =
  (* The dependency graph of kernels: edges go from a kernel to those that must
     run after it. *)
  let children = Tbl.create 64 and in_degree = Ordered.create () in
  let add_child t k =
    Tbl.replace children t
      (k :: Option.value (Tbl.find_opt children t) ~default:[]);
    Ordered.replace in_degree k (Option.get (Ordered.find_opt in_degree k) + 1)
  in
  (* Each superseded state, with the afters that write it and their new kernels,
     and each read of a state; both last first. *)
  let writes = Tbl.create 64 and reads = ref [] in
  List.iter
    (fun u ->
      if op u = Op.After then begin
        let kernels, after_deps = split_after u in
        let prev_state = unwrap_src (nth u 0) in
        let prev_kernels =
          if op prev_state = Op.After then fst (split_after prev_state) else []
        in
        let fresh =
          List.filter (fun k -> not (List.memq k prev_kernels)) kernels
        in
        Tbl.replace writes prev_state
          ((u, fresh)
          :: Option.value (Tbl.find_opt writes prev_state) ~default:[]);
        List.iter
          (fun k ->
            if Ordered.find_opt in_degree k = None then
              Ordered.replace in_degree k 0;
            let call =
              if is_loop k then begin
                if op (nth k 0) <> Op.Call then
                  invalid_arg "a loop of a kernel loops over a call";
                nth k 0
              end
              else k
            in
            let kernel_deps =
              List.filter
                (fun s -> not (empty_argument call s))
                (List.tl (src call))
            in
            let read_states = List.concat_map states kernel_deps in
            reads :=
              List.rev_append
                (List.map (fun st -> (u, k, st)) read_states)
                !reads;
            (* Read after write: a kernel runs after the kernels that produced
               the states it reads or joins. *)
            List.iter
              (fun st ->
                if op st = Op.After then
                  List.iter (fun t -> add_child t k) (fst (split_after st)))
              (read_states @ List.concat_map states after_deps))
          kernels
      end)
    (toposort ~calls:Enter ~gate:gate_kernel_sink sched_sink);
  (* Write after read: a kernel reading a state runs before any other write that
     supersedes it. An after supersedes only the state before it: the kernels it
     shares with that state order, and do not write. *)
  List.iter
    (fun (u, k, s) ->
      List.iter
        (fun (a, write_kernels) ->
          if a != u then
            List.iter
              (fun t ->
                if t != k && not (Nodes.mem t (backward_slice ~calls:Skip k))
                then add_child k t)
              write_kernels)
        (List.rev (Option.value (Tbl.find_opt writes s) ~default:[])))
    (List.rev !reads);
  let queue = Queue.create () in
  List.iter
    (fun (k, d) -> if d = 0 then Queue.push k queue)
    (Ordered.bindings in_degree);
  let linearized = ref [] in
  let loops r = Axis_type.equal (axis_type r) Axis_type.Loop in
  (* A call's argument is the storage it reads, or a view of it that moves with
     the loops the call runs in. *)
  let rec argument s =
    if
      Op.Set.mem (op s) Op.Set.movement
      && List.exists loops (Nodes.to_list (ranges s))
    then replace s ~src:(argument (nth s 0) :: List.tl (src s))
    else buf_uop (unwrap_src s)
  in
  while not (Queue.is_empty queue) do
    let rk = Queue.pop queue in
    let k = if is_loop rk then nth rk 0 else rk in
    if op k <> Op.Call then invalid_arg "a scheduled kernel is a call";
    let args =
      List.filter_map
        (fun s ->
          if is_bound_var s then None
          else if empty_argument k s then Some s
          else Some (argument s))
        (List.tl (src k))
    in
    let call = replace k ~src:(body k :: args) in
    (* A loop around a call stays, a back edge's condition read from the storage
       it names. *)
    let entry =
      match op rk with
      | Op.End when List.exists loops (List.tl (src rk)) ->
          replace rk ~src:(call :: List.tl (src rk))
      | Op.Backedge ->
          replace rk ~src:[ call; nth rk 1; buf_uop (unwrap_src (nth rk 2)) ]
      | _ -> call
    in
    linearized := entry :: !linearized;
    List.iter
      (fun x ->
        let d = Option.get (Ordered.find_opt in_degree x) - 1 in
        Ordered.replace in_degree x d;
        if d = 0 then Queue.push x queue)
      (List.rev (Option.value (Tbl.find_opt children rk) ~default:[]))
  done;
  if List.exists (fun (_, d) -> d <> 0) (Ordered.bindings in_degree) then
    invalid_arg "cycle detected in assign graph";
  v Op.Linear ~src:(List.rev !linearized)

(* A linear in a linear is inlined into it. *)
let pm_flatten_linear =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.op Op.Linear ~name:"lin" ~early_reject:[ Op.Linear ]) (fun m ->
          let lin = m "lin" in
          Some
            (replace lin
               ~src:
                 (List.concat_map
                    (fun c -> if op c = Op.Linear then src c else [ c ])
                    (src lin))));
    ])

(* Parameters and call-local storage *)

let param_of u =
  match arg u with
  | Param p -> p
  | _ -> invalid_arg "storage needs its argument"

let create_new_buffer (buffers, args) b =
  match Tbl.find_opt buffers b with
  | Some buf -> buf
  | None ->
      let device =
        match device b with
        | Some d -> d
        | None -> (
            match List.find_map device args with
            | Some d -> d
            | None -> invalid_arg "call-local storage needs a device")
      in
      let buf = new_buffer device (max_numel b) (dtype b) in
      Tbl.replace buffers b buf;
      buf

let pm_post_sched_cache =
  Pattern_matcher.v
    (fun () -> [
      (* Positional arguments are resolved outside kernel bodies; free variables
         have slot -1. *)
      rule_ctx (Upat.op Op.Param ~name:"x") (fun (_, args) m ->
          let slot = (param_of (m "x")).slot in
          if slot >= 0 then Some (List.nth args slot) else None);
      (* Call-local storage is bound to new buffers for this invocation. *)
      rule_ctx (Upat.op Op.Alloc ~name:"b") (fun ctx m ->
          Some (create_new_buffer ctx (m "b")));
    ])

(* Nested linear calls are lexical scopes: their positional parameters shadow
   the enclosing scope, while calls without scalar arguments, such as a
   precompiled allreduce, inherit it. *)
let rec resolve_linear_call ?(outer_binds = []) linear_call =
  let args = List.tl (src linear_call) in
  let linear =
    graph_rewrite ~calls:Skip ~pass:Once
      ~ctx:(Tbl.create 8, args)
      (body linear_call) (After_sources pm_post_sched_cache)
  in
  let binds =
    let local =
      List.concat
        (List.mapi
           (fun i x ->
             if op x = Op.Param && addrspace x = Some Dtype.Alu then
               [ (i, if is_variable x then unbound x else x) ]
             else [])
           args)
    in
    local @ List.filter (fun (i, _) -> not (List.mem_assoc i local)) outer_binds
  in
  let apply_binds si =
    match (op si, src si) with
    | Op.Call, b :: _ when op b = Op.Linear ->
        resolve_linear_call ~outer_binds:binds si
    (* Compiled parameters already have their slots in the program. *)
    | Op.Call, b :: _ when op b = Op.Program -> si
    | _ ->
        let subs =
          List.concat_map
            (fun s ->
              List.filter_map
                (fun v ->
                  Option.map
                    (fun x -> (v, x))
                    (List.assoc_opt (param_of v).slot binds))
                (variables s))
            (src si)
        in
        replace si
          ~src:
            (List.map
               (fun s -> substitute ~calls:Skip ~pass:Fixed_point s subs)
               (src si))
  in
  replace linear ~src:(List.map apply_binds (src linear))

let pm_resolve_linear_call =
  Pattern_matcher.append
    (Pattern_matcher.v
       (fun () -> [
         rule
           (Upat.op Op.Call ~name:"linear_call" ~allow_any_len:true
              ~src:[ Upat.op Op.Linear ])
           (fun m -> Some (resolve_linear_call (m "linear_call")));
       ]))
    pm_flatten_linear

(* Scheduling calls *)

let schedule_cache : (string, t) Hashtbl.t = Hashtbl.create 64
let schedule_cache_lock = Mutex.create ()

(* [fn] with its ranges numbered by their order in it. Whoever makes a loop
   numbers its range from a counter of its own, whose value depends on what the
   process did before, such as making a schedule or reading one back: the key
   is the same for every numbering of one function's ranges. A range's new
   number can be another range's old one, as when ranges numbered 1 and 0 trade
   numbers, so the renumbering is one walk: a renumbered range is not
   renumbered again. *)
let ranges_in_order fn =
  let ranges =
    List.filter (fun u -> op u = Op.Range) (toposort ~calls:Enter fn)
  in
  substitute ~calls:Enter ~pass:Once fn
    (List.mapi
       (fun k r ->
         ( r,
           replace r
             ~arg:(Range { axis_id = [ k ]; axis_type = axis_type r }) ))
       ranges)

(* The key of a schedule, in memory and, with the setting scache at 2 or more,
   on disk: the function's, its ranges numbered in order, every setting and
   variable that shapes what compilation makes ([Setting.shaping]), and the
   digest of this library's sources, of which a schedule is a function. *)
let schedule_key fn =
  String.concat "\n"
    ([ Source_digest.digest; key (ranges_in_order fn) ]
    @ List.map (fun (k, v) -> k ^ "=" ^ v) (Setting.shaping ()))

let lower_sink_to_linear call =
  let fn = body call in
  let precompile = match arg call with Call c -> c.precompile | _ -> false in
  match (op fn, arg fn) with
  | Op.Sink, Kernel _ -> None
  | Op.Sink, _ when precompile ->
      let start = Unix.gettimeofday () in
      let cache_key = Digest.BLAKE256.string (schedule_key fn)
      and cached = setting Setting.scache >= 1 in
      let hit =
        if cached then
          Mutex.protect schedule_cache_lock (fun () ->
              Hashtbl.find_opt schedule_cache cache_key)
        else None
      in
      let make () =
        if setting Setting.spec <> 0 then
          Spec.type_verify ~calls:Enter Spec.tensor fn;
        create_schedule
          (Rangeify.get_kernel_graph (Prepare.prepare_rangeify fn))
      in
      let linear, kept =
        match hit with
        | Some linear -> (linear, true)
        | None ->
            let linear, kept =
              if setting Setting.scache >= 2 then
                Graph.cached ~table:"schedule_cache" ~key:cache_key
                  ~valid:(fun l -> op l = Op.Linear)
                  make
              else (make (), false)
            in
            if cached then
              Mutex.protect schedule_cache_lock (fun () ->
                  Hashtbl.replace schedule_cache cache_key linear);
            (linear, kept)
      in
      let n = List.length (src linear) and debug = setting Setting.debug in
      if (debug >= 1 && n > 1) || debug >= 3 then
        Format.printf "scheduled %5d kernels in %8.2f ms | %s %s@." n
          ((Unix.gettimeofday () -. start) *. 1000.)
          (if kept then " cache hit" else "CACHE MISS")
          (String.sub (Digest.BLAKE256.to_hex cache_key) 0 8);
      Some (replace call ~src:(linear :: List.tl (src call)))
  | _ -> None

let pm_schedule =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.op Op.Call ~name:"call") (fun m ->
          lower_sink_to_linear (m "call"));
    ])

(* Copies *)

let assert_all_same_devices ast =
  let devices =
    List.fold_left
      (fun acc x ->
        match (op x, device x) with
        | Op.Param, Some d when not (List.exists (equal_device d) acc) ->
            acc @ [ d ]
        | _ -> acc)
      []
      (toposort ~calls:Enter ast)
  in
  if List.length devices >= 2 then
    invalid_arg
      (Format.asprintf "all buffers must be on the same device: %a"
         (Format.pp_print_list
            ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
            pp_device)
         devices)

(* A kernel that copies between devices, or to or from a disk, is a copy. *)
let is_copy dst src =
  (not (Option.equal equal_device (device dst) (device src))) || on_disk dst

let copy_kernel_to_store call dst src =
  if is_copy dst src then
    Some (replace call ~src:(store dst src :: List.tl (Ops.src call)))
  else None

(* The copy kernels are code for copy engines. *)
let simplify_copy_kernel call ast dst src =
  if not (is_copy dst src) then None
  else
    let sink =
      graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Tbl.create 8) ast
        (After_sources
           (Pattern_matcher.concat
              [
                with_ctx Symbolic.sym;
                with_ctx Prepare.pm_mops;
                with_ctx Simplify.pm_flatten_range;
                Simplify.pm_simplify_ranges;
              ]))
    in
    Some (replace call ~src:(sink :: List.tl (Ops.src call)))

let pm_copy_from_store =
  let zero = Upat.op Op.Const ~arg:(Const (`Int Bigint.zero)) in
  let param name = Upat.op Op.Param ~name in
  let copy_call body =
    Upat.op Op.Call ~name:"call" ~allow_any_len:true ~src:[ Upat.sink [ body ] ]
  in
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.op Op.Call ~name:"call"
           ~src:[ Upat.op Op.Sink ~name:"ast"; var "dst"; var "src" ])
        (fun m -> simplify_copy_kernel (m "call") (m "ast") (m "dst") (m "src"));
      (* Copy kernels become bulk stores. *)
      rule
        (copy_call
           (Upat.store
              (Upat.index (param "dst") [ zero ])
              [ Upat.index (param "src") [ zero ] ]))
        (fun m -> copy_kernel_to_store (m "call") (m "dst") (m "src"));
      rule
        (copy_call
           (Upat.end_
              (Upat.store
                 (Upat.index (param "dst") [ Upat.op Op.Range ~name:"r" ])
                 [ Upat.index (param "src") [ Upat.op Op.Range ~name:"r" ] ])
              [ Upat.op Op.Range ~name:"r" ]))
        (fun m -> copy_kernel_to_store (m "call") (m "dst") (m "src"));
      (* Any other kernel stays on one device. *)
      rule
        (Upat.op Op.Call ~allow_any_len:true
           ~src:[ Upat.op Op.Sink ~name:"ast" ])
        (fun m ->
          assert_all_same_devices (m "ast");
          None);
    ])

(* Callify: the tensor graph becomes a call, with its state scoped *)

type callify_ctx = {
  mutable replacements : t list; (* Last first. *)
  allocs : t Ordered.t;
  views : unit Tbl.t;
  mutable stores : t list; (* Last first. *)
}

let callify_ctx () =
  {
    replacements = [];
    allocs = Ordered.create ();
    views = Tbl.create 8;
    stores = [];
  }

let all_int shape =
  List.for_all (function Int _ -> true | Sym _ -> false) shape

(* Movements and bitcasts of a buffer are a view of it when they select a
   contiguous range of its elements. *)
let rec view_of ctx c src =
  if not (all_int (shape c)) then None
  else
    let rec storage b =
      if op b = Op.Bitcast then storage (base (nth b 0)) else b
    in
    let buf = storage (base src) in
    if op buf = Op.Unshard then
      match device c with
      | Some (Single _) -> None
      | _ -> (
          let unshard =
            graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() src
              (After_sources Multi.multi_pm)
          in
          if op unshard <> Op.Unshard then None
          else
            let shard = nth unshard 0 in
            match view_of ctx shard shard with
            | None -> None
            | Some view ->
                let s = sharding unshard in
                Some
                  (Shape.unshard ~ranges:(List.map snd s) view (List.map fst s)))
    else if op buf <> Op.Buffer then None
    else
      match Prepare.contiguous_view src with
      | Some (buf, offset) when op buf = Op.Buffer ->
          let n = max_numel src * element_size src / element_size buf in
          let view =
            bitcast
              (shrink buf [ Some (Int offset, Int (offset + n)) ])
              (dtype src)
          in
          Option.iter (fun ctx -> Tbl.replace ctx.views view ()) ctx;
          let view = reshape view (shape c) in
          if op c = Op.Copy || op c = Op.Store then
            Some (replace c ~src:(view :: List.tl (Ops.src c)))
          else Some view
      | _ -> None

let contiguous_mops_to_view c src = view_of None c src

let is_store_after u =
  op u = Op.After
  && (op (unsharded_base (nth u 0)) <> Op.Alloc || op (nth u 1) = Op.Store)

(* Scheduling rewrites belong in Prepare; only storage and interface
   normalisation belongs here. *)
let pm_callify_ctx_collect =
  Pattern_matcher.v
    (fun () -> [
      (* Movements and bitcasts of a buffer that collapse to a contiguous range
         are a shrink of it. *)
      rule_ctx
        (Upat.v
           ~op:(ops Op.[ Copy; Stage ])
           ~name:"c" ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:(Op.Set.union Op.Set.movement (ops [ Op.Bitcast ]))
                 ~name:"src" ();
             ]
           ())
        (fun ctx m -> view_of (Some ctx) (m "c") (m "src"));
      rule_ctx
        (Upat.op Op.Store ~name:"c" ~allow_any_len:true
           ~src:[ Upat.op Op.Bitcast ~name:"src"; Upat.wild ])
        (fun ctx m -> view_of (Some ctx) (m "c") (m "src"));
      (* Effects are collected after their sources are rewritten, without
         entering call bodies. *)
      rule_ctx (Upat.op Op.After ~name:"u") (fun ctx m ->
          let u = m "u" in
          if is_store_after u then ctx.stores <- u :: ctx.stores;
          None);
    ])

(* Call-local storage gets slots of its own scope, so that equal calls hash
   alike for the schedule cache. Fresh slots count up from 0; negative slots are
   already the scope's own. *)
let canonicalize_alloc ctx b =
  let p = param_of b in
  if p.slot >= 0 && Ordered.find_opt ctx.allocs b = None then
    Ordered.replace ctx.allocs b
      (replace b
         ~arg:(Param { p with slot = -1 - List.length ctx.allocs.keys }));
  Ordered.find_opt ctx.allocs b

let rec canonicalize_call_body c =
  let body =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(callify_ctx ()) (body c)
      (Before_sources (Lazy.force pm_canonicalize_alloc))
  in
  replace c ~src:(body :: List.tl (src c))

and pm_canonicalize_alloc =
  lazy
    (Pattern_matcher.v
       (fun () -> [
         rule (Upat.op Op.Call ~name:"c") (fun m ->
             Some (canonicalize_call_body (m "c")));
         rule_ctx (Upat.op Op.Alloc ~name:"b") (fun ctx m ->
             canonicalize_alloc ctx (m "b"));
       ]))

(* Forced as the module initialises, on one domain: the compilers' domains
   would race to force it first, and a lazy value that two domains force at
   once raises. *)
let pm_canonicalize_alloc = Lazy.force pm_canonicalize_alloc

let replace_input_buffer ctx b =
  ctx.replacements <- b :: ctx.replacements;
  param_like b (List.length ctx.replacements - 1)

let pm_replace_buf =
  Pattern_matcher.v
    (fun () -> [
      (* Global buffers become parameters, which normalises the cache key;
         variables are scalar parameters and do not match. *)
      rule_ctx (Upat.op Op.Buffer ~name:"b") (fun ctx m ->
          let b = m "b" in
          if addrspace b = Some Dtype.Global then
            Some (replace_input_buffer ctx b)
          else None);
      (* So do the views of buffers found above. *)
      rule_ctx
        (Upat.v ~op:(ops Op.[ Shrink; Bitcast ]) ~name:"b" ())
        (fun ctx m ->
          let b = m "b" in
          if Tbl.mem ctx.views b then Some (replace_input_buffer ctx b)
          else None);
      (* Bound variables become parameters without their value, so that
         different values hit the same cache entry. *)
      rule_ctx (Upat.op Op.Param ~name:"b") (fun ctx m ->
          let b = m "b" in
          if is_bound_var b then Some (replace_input_buffer ctx b) else None);
    ])

let transform_to_call big_sink =
  if setting Setting.spec <> 0 then
    Spec.type_verify ~calls:Enter Spec.tensor big_sink;
  (* The stores are collected before these rewrites change node identities. *)
  let ctx = callify_ctx () in
  ignore
    (graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx big_sink
       (After_sources pm_callify_ctx_collect));
  let ret =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx
      (sink (List.rev ctx.stores))
      (Before_sources (Pattern_matcher.concat
         [
           pm_canonicalize_alloc;
           pm_replace_buf;
           with_ctx remove_all_tags;
         ]))
  in
  call ~precompile:true ret (List.rev ctx.replacements)

(* Schedules *)

let create_linear_with_vars ?(capturing = false) big_sink =
  let big_sink = transform_to_call big_sink in
  (* The call's sources are the values to realize. *)
  let linear_call =
    graph_rewrite ~calls:Enter ~pass:Fixed_point ~ctx:() big_sink
      (After_sources pm_schedule)
  in
  (* The linear call is resolved recursively, and its storage allocated. *)
  let linear =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() linear_call
      (After_sources pm_resolve_linear_call)
  in
  let linear =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() linear
      (After_sources pm_copy_from_store)
  in
  let used_vars =
    List.concat_map
      (fun si -> List.map expr (variables (nth si 0)))
      (src linear)
  in
  let var_vals =
    List.fold_left
      (fun acc b ->
        if not (is_bound_var b) then acc
        else
          let nm = expr b in
          let value = Dtype.Value.to_int (Option.get (param_of b).bound) in
          if not (List.mem nm used_vars) then acc
          else
            match List.assoc_opt nm acc with
            | Some v when v <> value ->
                invalid_arg
                  (Printf.sprintf "bind mismatch on %s, %d <> %d" nm v value)
            | Some _ -> acc
            | None -> acc @ [ (nm, value) ])
      []
      (List.tl (src big_sink))
  in
  if capturing then (linear, var_vals)
  else
    let held_bufs =
      List.filter (fun b -> op b = Op.Buffer) (List.tl (src linear_call))
    in
    (Memory.memory_plan_rewrite ~held_bufs linear, var_vals)
