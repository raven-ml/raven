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

(* The data source a kernel reads: a kernel's output or a buffer. *)
let rec unwrap_src s =
  match (op s, src s) with
  | (Op.After | Op.Buffer | Op.Alloc | Op.Param), _ | _, [] -> s
  | _, x :: _ -> unwrap_src x

(* A buffer's state is an after or the storage. *)
let states s =
  let s = unwrap_src s in
  match op s with
  | Op.After | Op.Buffer | Op.Alloc | Op.Param -> [ s ]
  | o ->
      invalid_arg
        (Format.asprintf "a kernel's input is a buffer state, not %a" Op.pp o)

(* A scalar argument of a precompiled call is no state: a weak integer
   expression of the loops around the call, which its body's scalar parameter
   holds on each trip. *)
let scalar_argument k s =
  (match arg k with Call c -> c.precompile | _ -> false)
  && match addrspace s with Some Dtype.Alu -> true | _ -> false

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

(* Copies before overwrites

   A kernel that reads a state and waits, through a value computed after it,
   for a kernel that overwrites it, as [b + (a * 3)] does where [b] read [a]
   before [a] was assigned, runs neither before nor after the overwrite. It
   reads instead a copy of the state into storage of its own, which a kernel
   makes before the overwrite:

     a --copy--> a'          a' = a as it was
     a --assign--> a         the overwrite, after the copy reads a
     a', a*3 --reader-->     the reader reads a' *)

(* The next free slot of the graph's storage. *)
let next_slot sink =
  let slot u =
    match (op u, arg u) with
    | (Op.Alloc | Op.Buffer | Op.Param), Param p -> p.slot
    | _ -> -1
  in
  let slots = List.map slot (toposort ~calls:Skip sink) in
  1 + List.fold_left max (-1) slots

(* The state [s] copied into new storage of slot [slot] by a kernel of its
   own. *)
let copy_state ~slot s =
  let buf = buf_uop s in
  let device =
    match device buf with
    | Some d -> d
    | None -> invalid_arg "a copied buffer state needs a device"
  in
  let n = max_numel buf in
  (* The copy lies as the state's storage does, which the reader's kernel
     reads it at. *)
  let align, phase = Shape.storage_phase buf in
  let arg = param_arg ~slot ~device ~phase ~align (dtype buf) in
  let dst = v Op.Alloc ~src:[ size_src (Some n) ] ~arg:(Param arg) in
  let r = range (Int n) [ 0 ] in
  let into = Shape.param_like dst 0 and from = Shape.param_like buf 1 in
  let body =
    sink ~kernel:(kernel_info ())
      [ end_ (store (index into [ r ]) (index from [ r ])) [ r ] ]
  in
  after dst [ call body [ dst; s ] ]

(* [d], a source of a kernel, reading [c] where it reads the state [s]: its
   path to the state is the one {!states} follows. *)
let rec swap s c d =
  if d == s then c
  else
    match (op d, src d) with
    | (Op.After | Op.Buffer | Op.Alloc | Op.Param), _ | _, [] -> d
    | _, x :: rest -> replace d ~src:(swap s c x :: rest)

(* [sink] with each kernel of [hazards], a list of a kernel and a state it
   reads, reading a copy of the state. *)
let copy_before_overwrites sink hazards =
  let slot = ref (next_slot sink) and copies = ref [] in
  let copy s =
    match List.assq_opt s !copies with
    | Some c -> c
    | None ->
        let c = copy_state ~slot:!slot s in
        incr slot;
        copies := (s, c) :: !copies;
        c
  in
  let rebuilt =
    List.fold_left
      (fun acc (k, s) ->
        let k' = Option.value (List.assq_opt k acc) ~default:k in
        let call = if is_loop k' then nth k' 0 else k' in
        let call =
          replace call
            ~src:(body call :: List.map (swap s (copy s)) (List.tl (src call)))
        in
        let k'' =
          if is_loop k' then replace k' ~src:(call :: List.tl (src k'))
          else call
        in
        (k, k'') :: List.remove_assq k acc)
      [] hazards
  in
  (* A rebuilt kernel may read another, directly or through the copy of a
     state: each is replaced wherever it appears, in rebuilt kernels and
     copies too. *)
  substitute ~calls:Skip ~pass:Fixed_point sink rebuilt

(* A kernel as the schedule runs it: a call of its storage, in a loop where it
   runs in one. *)
let scheduled rk =
  let k = if is_loop rk then nth rk 0 else rk in
  if op k <> Op.Call then invalid_arg "a scheduled kernel is a call";
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
  let args =
    List.filter_map
      (fun s ->
        if is_bound_var s then None
        else if scalar_argument k s then Some s
        else Some (argument s))
      (List.tl (src k))
  in
  let call = replace k ~src:(body k :: args) in
  (* A loop around a call stays, a back edge's condition read from the storage
     it names. *)
  match op rk with
  | Op.End when List.exists loops (List.tl (src rk)) ->
      replace rk ~src:(call :: List.tl (src rk))
  | Op.Backedge ->
      replace rk ~src:[ call; nth rk 1; buf_uop (unwrap_src (nth rk 2)) ]
  | _ -> call

module Iset = Set.Make (Int)

(* The kernels in an order of the graph [children], [in_degree] counting the
   kernels each waits for: each as it becomes ready. *)
let ready_order ~children in_degree =
  let queue = Queue.create () in
  List.iter
    (fun (k, d) -> if d = 0 then Queue.push k queue)
    (Ordered.bindings in_degree);
  let linearized = ref [] in
  while not (Queue.is_empty queue) do
    let rk = Queue.pop queue in
    linearized := scheduled rk :: !linearized;
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

(* The kernels in an order of the graph [children], [in_degree] counting the
   kernels each waits for and numbering them in the order the walk of the sink
   meets them.

   A ready kernel runs by its number, so that a value is consumed soon after it
   is made, as the program wrote it, and one that frees storage and on no device
   makes more than it frees runs before it. The storage a kernel writes,
   [written], lives from the first kernel that takes it to the last
   ({!Memory.taken}); other storage, which only kernels read, lives throughout.
   So a closing that slices a partial for each device sums each slice before it
   makes the next, and the last reader of a gathered value runs before the
   kernels that make more. A pick walks the ready kernels: [O(k²)] for [k]
   kernels at worst. *)
let storage_order ~written ~children in_degree =
  let kernels = Array.of_list (List.map fst (Ordered.bindings in_degree)) in
  let count = Array.length kernels in
  let number = Tbl.create count in
  Array.iteri (fun i k -> Tbl.replace number k i) kernels;
  let entries = Array.map scheduled kernels in
  let made_here b = Tbl.mem written b && not (on_disk b) in
  let storage =
    Array.map (fun e -> List.filter made_here (Memory.taken e)) entries
  in
  (* The kernels left to take each storage, and the storage made so far. *)
  let users = Tbl.create count and made = Tbl.create count in
  Array.iter
    (List.iter (fun b ->
         Tbl.replace users b (1 + Option.value (Tbl.find_opt users b) ~default:0)))
    storage;
  let bytes b = max_numel b * element_size b in
  let net = Hashtbl.create 4 in
  (* Whether kernel [i] frees storage and, where [~makes] is false, makes
     none, else makes on no device more than it frees. *)
  let frees ~makes i =
    Hashtbl.reset net;
    let add b n =
      let d = Option.value (device b) ~default:"" in
      Hashtbl.replace net d (n + Option.value (Hashtbl.find_opt net d) ~default:0)
    in
    let freed = ref false and making = ref false in
    List.iter
      (fun b ->
        if not (Tbl.mem made b) then begin
          add b (bytes b);
          making := true
        end
        else if Tbl.find users b = 1 then begin
          add b (-bytes b);
          freed := true
        end)
      storage.(i);
    !freed
    && (if makes then Hashtbl.fold (fun _ n ok -> ok && n <= 0) net true
        else not !making)
  in
  let ready = ref Iset.empty in
  Array.iteri
    (fun i k ->
      if Ordered.find_opt in_degree k = Some 0 then ready := Iset.add i !ready)
    kernels;
  (* A kernel holds what it makes beside what it reads: one that only frees
     runs before one that makes storage too. *)
  let pick () =
    let seq = Iset.to_seq !ready in
    match Seq.find (frees ~makes:false) seq with
    | Some i -> i
    | None -> (
        match Seq.find (frees ~makes:true) seq with
        | Some i -> i
        | None -> Iset.min_elt !ready)
  in
  let linearized = ref [] in
  while not (Iset.is_empty !ready) do
    let i = pick () in
    ready := Iset.remove i !ready;
    List.iter
      (fun b ->
        Tbl.replace made b ();
        Tbl.replace users b (Tbl.find users b - 1))
      storage.(i);
    linearized := entries.(i) :: !linearized;
    List.iter
      (fun x ->
        let d = Option.get (Ordered.find_opt in_degree x) - 1 in
        Ordered.replace in_degree x d;
        if d = 0 then ready := Iset.add (Tbl.find number x) !ready)
      (List.rev (Option.value (Tbl.find_opt children kernels.(i)) ~default:[]))
  done;
  if List.exists (fun (_, d) -> d <> 0) (Ordered.bindings in_degree) then
    invalid_arg "cycle detected in assign graph";
  v Op.Linear ~src:(List.rev !linearized)

let linearize ~by_storage ~children sched_sink in_degree =
  if not by_storage then ready_order ~children in_degree
  else
    (* The storage the kernels write. *)
    let written = Tbl.create 16 in
    List.iter
      (fun u -> if op u = Op.After then Tbl.replace written (buf_uop u) ())
      (toposort ~calls:Skip ~gate:gate_kernel_sink sched_sink);
    storage_order ~written ~children in_degree

let rec schedule ~by_storage ~copies sched_sink =
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
                (fun s -> not (scalar_argument call s))
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
    (toposort ~calls:Skip ~gate:gate_kernel_sink sched_sink);
  (* Write after read: a kernel reading a state runs before any other write that
     supersedes it. An after supersedes only the state before it: the kernels it
     shares with that state order, and do not write. A kernel that also waits
     for such a write, through a value computed after it, reads a state no order
     keeps, as [b + (a * 3)] does where [b] read [a] before [a] was assigned. *)
  let hazards = ref [] in
  let overwrite k s t =
    if t == k then ()
    else if Nodes.mem t (backward_slice ~calls:Skip k) then begin
      if not (List.exists (fun (k', s') -> k' == k && s' == s) !hazards) then
        hazards := (k, s) :: !hazards
    end
    else add_child k t
  in
  List.iter
    (fun (u, k, s) ->
      List.iter
        (fun (a, write_kernels) ->
          if a != u then List.iter (overwrite k s) write_kernels)
        (List.rev (Option.value (Tbl.find_opt writes s) ~default:[])))
    (List.rev !reads);
  match List.rev !hazards with
  | [] -> linearize ~by_storage ~children sched_sink in_degree
  | _ when not copies ->
      invalid_arg
        "a kernel reads a buffer state that a kernel it waits for overwrites"
  | hazards ->
      schedule ~by_storage ~copies:false
        (copy_before_overwrites sched_sink hazards)

let create_schedule ?(by_storage = false) sched_sink =
  schedule ~by_storage ~copies:true sched_sink

(* A linear in a linear is inlined into it. *)
let flatten_linear () =
  rule (Upat.op Op.Linear ~name:"lin" ~early_reject:[ Op.Linear ]) (fun m ->
      let lin = m "lin" in
      Some
        (replace lin
           ~src:
             (List.concat_map
                (fun c -> if op c = Op.Linear then src c else [ c ])
                (src lin))))

let pm_flatten_linear = Pattern_matcher.v (fun () -> [ flatten_linear () ])

(* Parameters and call-local storage *)

let param_of u =
  match arg u with
  | Param p -> p
  | _ -> invalid_arg "storage needs its argument"

let create_new_buffer args b =
  let device =
    match device b with
    | Some d -> d
    | None -> (
        match List.find_map device args with
        | Some d -> d
        | None -> invalid_arg "call-local storage needs a device")
  in
  new_buffer device (max_numel b) (dtype b)

let pm_post_sched_cache =
  Pattern_matcher.v
    (fun () -> [
      (* Positional arguments are resolved outside kernel bodies; free variables
         have slot -1. *)
      rule_ctx (Upat.op Op.Param ~name:"x") (fun args m ->
          let slot = (param_of (m "x")).slot in
          if slot >= 0 then Some (List.nth args slot) else None);
      (* Call-local storage is bound to new buffers for this invocation. *)
      rule_ctx (Upat.op Op.Alloc ~name:"b") (fun ctx m ->
          Some (create_new_buffer ctx (m "b")));
    ])

(* A kernel's scalar parameters that its scope binds to values, such as a loop's
   trip, stay parameters: each takes a slot of the kernel past its arguments,
   in the order the kernel's body reaches them, and the call passes the value
   there. The kernel's program then depends on neither the value, nor the loop
   it comes from, nor the order the scope met it, and a launch computes the
   value from the loop's ranges. *)
let pass_values values call =
  let args = src_without_body call in
  let held =
    List.filter
      (fun v ->
        op v = Op.Param
        && addrspace v = Some Dtype.Alu
        &&
        let p = param_of v in
        p.name = None && List.mem_assoc p.slot values)
      (toposort ~calls:Skip (body call))
  in
  let n = List.length args in
  let slot j v = replace v ~arg:(Param { (param_of v) with slot = n + j }) in
  let slotted = List.mapi (fun j v -> (v, slot j v)) held in
  replace call
    ~src:
      (substitute ~calls:Skip ~pass:Once (body call) slotted
      :: args
      @ List.map (fun v -> List.assoc (param_of v).slot values) held)

(* Nested linear calls are lexical scopes: their positional parameters shadow
   the enclosing scope, while calls without scalar arguments inherit it. A
   scalar argument that is a variable is bound into each kernel that reads it in
   its slot, by its name, unbound: a run binds it. A variable bound to a value
   of a name in [several], which the schedule binds to several values, and any
   other scalar argument, is a value its kernels are passed ({!pass_values}):
   each call passes its own. A named variable keeps its slot: a custom kernel's
   takes its own. *)
let rec resolve_linear_call ~several ?(outer_binds = []) linear_call =
  let args = List.tl (src linear_call) in
  let linear =
    graph_rewrite ~calls:Skip ~pass:Once ~ctx:args (body linear_call)
      (After_sources pm_post_sched_cache)
  in
  let binds =
    let local =
      List.concat
        (List.mapi
           (fun i x ->
             if addrspace x <> Some Dtype.Alu then []
             else if is_bound_var x && List.mem (expr x) several then
               [ (i, x) ]
             else if is_variable x then [ (i, unbound x) ]
             else [ (i, x) ])
           args)
    in
    local @ List.filter (fun (i, _) -> not (List.mem_assoc i local)) outer_binds
  in
  let params, values =
    List.partition (fun (_, x) -> op x = Op.Param && not (is_bound_var x)) binds
  in
  let apply_binds si =
    match (op si, src si) with
    | Op.Call, b :: _ when op b = Op.Linear ->
        resolve_linear_call ~several ~outer_binds:binds si
    (* Compiled parameters already have their slots in the program. *)
    | Op.Call, b :: _ when op b = Op.Program -> si
    | _ ->
        let subs =
          List.concat_map
            (fun s ->
              List.filter_map
                (fun v ->
                  let p = param_of v in
                  if p.name <> None then None
                  else
                    Option.map (fun x -> (v, x)) (List.assoc_opt p.slot params))
                (variables s))
            (src si)
        in
        let si =
          replace si
            ~src:
              (List.map
                 (fun s -> substitute ~calls:Skip ~pass:Fixed_point s subs)
                 (src si))
        in
        if op si = Op.Call && op (body si) = Op.Sink then pass_values values si
        else si
  in
  replace linear ~src:(List.map apply_binds (src linear))

(* Its context is the names the schedule binds to several values. *)
let pm_resolve_linear_call =
  Pattern_matcher.v (fun () -> [
      rule_ctx
        (Upat.op Op.Call ~name:"linear_call" ~allow_any_len:true
           ~src:[ Upat.op Op.Linear ])
        (fun several m ->
          Some (resolve_linear_call ~several (m "linear_call")));
      flatten_linear ();
    ])

(* The names that the bound variables of [u] bind to several values. *)
let bound_several u =
  let values = Hashtbl.create 8 in
  List.iter
    (fun v ->
      if is_bound_var v then
        let x = Dtype.Value.to_int (Option.get (param_of v).bound) in
        let name = expr v in
        let seen = Option.value (Hashtbl.find_opt values name) ~default:[] in
        if not (List.mem x seen) then Hashtbl.replace values name (x :: seen))
    (toposort ~calls:Enter u);
  Hashtbl.fold
    (fun name xs acc -> if List.length xs > 1 then name :: acc else acc)
    values []

(* Scheduling calls *)

let schedule_cache : t Helpers.Memo.t = Helpers.Memo.create ()

(* [in_order fn] is the pairs that take each range and each storage of [fn] to a
   placeholder numbered by its order in it. Whoever makes a loop numbers its
   range, and whoever makes storage, such as a staged loop's carry or
   call-local storage that inlining renames, takes its slot, from a counter
   whose value depends on what the process did before, such as making a
   schedule or reading one back: the key is [fn] with its placeholders, and so
   is the schedule kept, in memory and on disk, so that one schedule serves
   every numbering of a function, each given back its own. A placeholder is no
   node a schedule makes: ranges are numbered below [0], storage below every
   slot of [fn] and below [-1], the slot of a variable. *)
let in_order fn =
  let nodes = toposort ~calls:Enter fn in
  let slot u = match arg u with Param p -> Some p.slot | _ -> None in
  let lowest =
    List.fold_left
      (fun m u -> min m (Option.value ~default:0 (slot u)))
      (-1) nodes
  in
  let ranges = List.filter (fun u -> op u = Op.Range) nodes in
  let storage =
    List.filter
      (fun u ->
        (op u = Op.Buffer || op u = Op.Alloc)
        && match slot u with Some s -> s >= 0 | None -> false)
      nodes
  in
  List.mapi
    (fun k r ->
      ( r,
        replace r ~arg:(Range { axis_id = [ -1 - k ]; axis_type = axis_type r })
      ))
    ranges
  @ List.mapi
      (fun k u ->
        let p = param_of u in
        (u, replace u ~arg:(Param { p with slot = lowest - 1 - k })))
      storage

(* The key of a schedule, in memory and, with the setting scache at 2 or more,
   on disk: the function's with its placeholders ([in_order]), every setting
   and variable that shapes what compilation makes ([Setting.shaping]), and the
   digest of this library's sources, of which a schedule is a function. *)
let schedule_key ~by_storage fn =
  String.concat "\n"
    ([ Source_digest.digest; key fn; string_of_bool by_storage ]
    @ List.map (fun (k, v) -> k ^ "=" ^ v) (Setting.shaping ()))

let lower_sink_to_linear ~by_storage call =
  let fn = body call in
  let precompile = match arg call with Call c -> c.precompile | _ -> false in
  match (op fn, arg fn) with
  | Op.Sink, Kernel _ -> None
  | Op.Sink, _ when precompile ->
      let start = Unix.gettimeofday () in
      let placed = in_order fn in
      let place g = substitute ~calls:Enter ~pass:Once g placed in
      let cache_key =
        Digest.BLAKE256.string (schedule_key ~by_storage (place fn))
      in
      (* Whether neither memory nor disk had the schedule. *)
      let made = ref false in
      let make () =
        made := true;
        if setting Setting.spec <> 0 then
          Spec.type_verify ~calls:Skip Spec.tensor fn;
        place
          (create_schedule ~by_storage
             (Rangeify.get_kernel_graph (Prepare.prepare_rangeify fn)))
      in
      let read_or_make () =
        if setting Setting.scache < 2 then make ()
        else
          fst
            (Graph.cached ~table:"schedule_cache" ~key:cache_key
               ~valid:(fun l -> op l = Op.Linear)
               make)
      in
      let linear =
        if setting Setting.scache < 1 then read_or_make ()
        else Helpers.Memo.get schedule_cache cache_key read_or_make
      in
      let n = List.length (src linear) and debug = setting Setting.debug in
      if (debug >= 1 && n > 1) || debug >= 3 then
        Format.printf "scheduled %5d kernels in %8.2f ms | %s %s@." n
          ((Unix.gettimeofday () -. start) *. 1000.)
          (if !made then "CACHE MISS" else " cache hit")
          (String.sub (Digest.BLAKE256.to_hex cache_key) 0 8);
      let linear =
        substitute ~calls:Enter ~pass:Once linear
          (List.map (fun (u, p) -> (p, u)) placed)
      in
      Some (replace call ~src:(linear :: List.tl (src call)))
  | _ -> None

let pm_schedule =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx (Upat.op Op.Call ~name:"call") (fun by_storage m ->
          lower_sink_to_linear ~by_storage (m "call"));
    ])

(* Copies *)

let assert_all_same_devices ast =
  let devices =
    List.fold_left
      (fun acc x ->
        match (op x, device x) with
        | Op.Param, Some d when not (List.exists (String.equal d) acc) ->
            acc @ [ d ]
        | _ -> acc)
      []
      (toposort ~calls:Skip ast)
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
  (not (Option.equal String.equal (device dst) (device src))) || on_disk dst

(* The element a copy kernel's index [i] of its range [r] starts at: [0] for
   [r], [c] for [r + c]. *)
let run_start r i =
  if i == r then Some 0
  else
    match (op i, Ops.src i) with
    | Op.Add, [ a; c ] when a == r -> (
        match arg c with Const (`Int z) -> Some (Bigint.to_int z) | _ -> None)
    | _ -> None

(* The copy kernel [call] of the [n] elements of [src] from [s] into those of
   [dst] from [d], a copy between the call's arguments where each run is its
   argument whole, and between views of them otherwise, as a copy into a run of
   storage is ({!Prepare}). *)
let copy_runs call dst src ~n ~d ~s =
  let args = Array.of_list (List.tl (Ops.src call)) in
  let flat p = ndim args.((param_of p).slot) = 1 in
  let whole p start = start = 0 && max_numel p = n in
  if not (is_copy dst src) then None
  else if not ((whole dst d || flat dst) && (whole src s || flat src)) then
    None
  else
    let run p start =
      if whole p start then p
      else begin
        let k = (param_of p).slot in
        args.(k) <- shrink args.(k) [ Some (Int start, Int (start + n)) ];
        replace p ~src:[ Ops.const ~dtype:(dtype (nth p 0)) (`Int (Bigint.of_int n)) ]
      end
    in
    let dst = run dst d and src = run src s in
    Some (replace call ~src:(store dst src :: Array.to_list args))

(* The copy kernels are code for copy engines. *)
(* A kernel of a variable is no copy, though the variable has no device. *)
let simplify_copy_kernel call ast dst src =
  if is_virtual dst || is_virtual src || not (is_copy dst src) then None
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
  let const name = Upat.op Op.Const ~name in
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
      (* Copy kernels become bulk stores. A copy of one element indexes each
         side at its element. *)
      rule
        (copy_call
           (Upat.store
              (Upat.index (param "dst") [ const "di" ])
              [ Upat.index (param "src") [ const "si" ] ]))
        (fun m ->
          match (arg (m "di"), arg (m "si")) with
          | Const (`Int d), Const (`Int s) ->
              copy_runs (m "call") (m "dst") (m "src") ~n:1
                ~d:(Bigint.to_int d) ~s:(Bigint.to_int s)
          | _ -> None);
      rule
        (copy_call
           (Upat.end_
              (Upat.store
                 (Upat.index (param "dst") [ var "di" ])
                 [ Upat.index (param "src") [ var "si" ] ])
              [ Upat.op Op.Range ~name:"r" ]))
        (fun m ->
          let r = m "r" in
          match
            (run_start r (m "di"), run_start r (m "si"), arg (nth r 0))
          with
          | Some d, Some s, Const (`Int n) ->
              copy_runs (m "call") (m "dst") (m "src") ~n:(Bigint.to_int n) ~d
                ~s
          | _ -> None);
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
let view_of ctx c src =
  if not (all_int (shape c)) then None
  else
    let rec storage b =
      if op b = Op.Bitcast then storage (base (nth b 0)) else b
    in
    let buf = storage (base src) in
    (* A store's value that reads its buffer keeps the buffer's identity, by
       which the kernel split stages it before the store. Its kernel reads
       other storage at a stored state, an after, which ends the walk: each
       store walks its own kernel's values, and a chain of outputs, each
       reading the last, costs its length. *)
    let reads_target () =
      let walked u = op u <> Op.After || storage_base u == buf in
      op c = Op.Store && reaches ~gate:walked ~calls:Skip (nth c 1) buf
    in
    if op buf <> Op.Buffer || reads_target () then None
    else
      (* A store's run may start at bound variables, which its call knows: its
         kernel then knows the start modulo what they move it by. *)
      let at = function
        | Int _ -> true
        | Sym _ -> op c = Op.Store
      in
      match Prepare.contiguous_view src with
      | Some (buf, offset) when op buf = Op.Buffer && at offset ->
          let n = max_numel src * element_size src / element_size buf in
          let view =
            bitcast
              (shrink buf [ Some (offset, Sint.(offset + Int n)) ])
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
  && (op (base (nth u 0)) <> Op.Alloc || op (nth u 1) = Op.Store)

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
      (* So do those a store stores into: stores of one shape into runs of
         one buffer at other offsets are then one program. *)
      rule_ctx
        (Upat.op Op.Store ~name:"c" ~allow_any_len:true
           ~src:
             [
               Upat.v
                 ~op:(Op.Set.union Op.Set.movement (ops [ Op.Bitcast ]))
                 ~name:"src" ();
               Upat.wild;
             ])
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
    Spec.type_verify ~calls:Skip Spec.tensor big_sink;
  (* The stores are collected before these rewrites change node identities. *)
  let ctx = callify_ctx () in
  ignore
    (graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx big_sink
       (After_sources pm_callify_ctx_collect));
  let ret =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx
      (sink (List.rev ctx.stores))
      (Before_sources (Pattern_matcher.concat [ pm_canonicalize_alloc; pm_replace_buf ]))
  in
  call ~precompile:true ret (List.rev ctx.replacements)

(* Calls that run as one

   Two kernel calls of a schedule run as one when neither touches what the other
   writes, so either order gives the same values: one kernel then computes both,
   and the work they share once. A call moves up to the first member of its
   group only past entries it does not conflict with, and only where that grows
   no storage's lifetime past what the scope allows: in a loop's body, every
   storage lives for the whole loop, so calls join freely; at the top level,
   only storage the schedule is given or storage of one element may start living
   earlier.

   k1 (a) ; k2 (b <- a) ; k3 (c) k3 joins k1: it touches neither a nor b k1+k3 ;
   k2 k2 reads what k1 writes: apart *)

(* The storage an entry takes, without the values it is passed. *)
let storage_taken e =
  List.filter (fun b -> addrspace b <> Some Dtype.Alu) (Memory.taken e)

(* A kernel call that can run with others: its device, its domain (the ranges
   its stores end, by position), the stores of its body, and the storage it
   writes. *)
type member = {
  device : string;
  domain : (Axis_type.t * t) list;
  ranges : t list;
  stores : t list;
  writes : t list;
}

module Domain = Hashtbl.Make (struct
  type t = string * (Axis_type.t * Ops.t) list

  let equal (d, a) (e, b) =
    String.equal d e
    && List.equal (fun (x, u) (y, w) -> Axis_type.equal x y && u == w) a b

  let hash (d, a) = Hashtbl.hash (d, List.map (fun (x, u) -> (x, Ops.hash u)) a)
end)

(* The member that the kernel call [si] is, if it is a kernel of the default
   kind whose body ends its stores over ranges of constant sizes and reduces
   nothing, and all of whose storage lies on one device. A size a call passes
   is a parameter whose slot the merge renumbers, so two such sizes cannot be
   compared before it. *)
let member si =
  let kernel = kernel_info () in
  match (op si, src si) with
  | Op.Call, b :: args when op b = Op.Sink -> (
      let plain = match arg b with Kernel k -> k = kernel | _ -> false in
      let root = match src b with [ r ] -> Some r | _ -> None in
      match root with
      | Some r when plain -> (
          let body, ranges =
            if op r = Op.End then (nth r 0, List.tl (src r)) else (r, [])
          in
          let stores =
            if is_group body then List.tl (src body)
            else if op body = Op.Store then [ body ]
            else []
          in
          let nodes = toposort ~calls:Skip b in
          let ranged u = op u = Op.Range && not (List.memq u ranges) in
          let args = Array.of_list args in
          let written st =
            match arg (buf_uop (nth st 0)) with
            | Param p when p.slot >= 0 && p.slot < Array.length args ->
                Some (storage_base args.(p.slot))
            | _ -> None
          in
          let writes = List.filter_map written stores in
          let storage = storage_taken si in
          let devices =
            List.sort_uniq String.compare (List.filter_map device storage)
          in
          if
            stores = []
            || List.exists (fun st -> op st <> Op.Store) stores
            || List.compare_lengths writes stores <> 0
            || List.exists
                 (fun r -> op r <> Op.Range || op (nth r 0) <> Op.Const)
                 ranges
            || List.exists (fun u -> op u = Op.Reduce || ranged u) nodes
            || List.exists (fun b -> Option.is_none (device b)) storage
          then None
          else
            match devices with
            | [ device ] ->
                let domain =
                  List.map (fun r -> (axis_type r, nth r 0)) ranges
                in
                Some { device; domain; ranges; stores; writes }
            | _ -> None)
      | _ -> None)
  | _ -> None

(* The calls [calls], each with its member, as one call: arguments that are one
   node are passed once, storage first, so that loads of them in several members
   are one node, and each member's ranges are the first's. *)
let merged calls =
  let storage = Ordered.create () and values = Ordered.create () in
  List.iter
    (fun (si, _) ->
      List.iter
        (fun a ->
          let table =
            if addrspace a = Some Dtype.Alu then values else storage
          in
          if Ordered.find_opt table a = None then Ordered.replace table a ())
        (List.tl (src si)))
    calls;
  let storage = List.map fst (Ordered.bindings storage)
  and values = List.map fst (Ordered.bindings values) in
  let args = storage @ values in
  let slots = Tbl.create 16 in
  List.iteri (fun i a -> Tbl.replace slots a i) args;
  let slot_of a = Tbl.find slots a in
  let first = snd (List.hd calls) in
  let rebuilt (si, m) =
    let own = Array.of_list (List.tl (src si)) in
    let params =
      List.filter_map
        (fun v ->
          match arg v with
          | Param p
            when op v = Op.Param
                 && p.name = None && p.slot >= 0
                 && p.slot < Array.length own ->
              Some
                ( v,
                  replace v ~arg:(Param { p with slot = slot_of own.(p.slot) })
                )
          | _ -> None)
        (toposort ~calls:Skip (body si))
    in
    let ranges = List.combine m.ranges first.ranges in
    let subs = List.filter (fun (x, y) -> x != y) (params @ ranges) in
    List.map (fun st -> substitute ~calls:Skip ~pass:Once st subs) m.stores
  in
  let stores = List.concat_map rebuilt calls in
  let root =
    if first.ranges = [] then stores else [ end_ (group stores) first.ranges ]
  in
  call (sink ~kernel:(kernel_info ()) root) args

(* A group of calls: the position of its first, its members, last first, the
   storage they are passed, and whether it still takes calls. *)
type group = {
  first : int;
  mutable members : (t * member) list;
  passed : unit Tbl.t;
  mutable open_ : bool;
}

(* The groups of one device and domain, in the order of their first calls, each
   with the next that may be open past it. *)
type groups = { gs : group Dynarray.t; next : int Dynarray.t }

(* The first open group of [k] from the [j]th, skipping closed ones once. *)
let rec open_from k j =
  if j >= Dynarray.length k.gs then j
  else if (Dynarray.get k.gs j).open_ then j
  else begin
    let n = open_from k (Dynarray.get k.next j) in
    Dynarray.set k.next j n;
    n
  end

(* The first group of [k] whose first call is at [bound] or after. *)
let at_or_after k bound =
  let rec go lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if (Dynarray.get k.gs mid).first < bound then go (mid + 1) hi
      else go lo mid
  in
  go 0 (Dynarray.length k.gs)

(* The entries of [linear] with the calls that can run as one merged, and the
   loops' bodies too. [held] is the storage the schedule is given, which
   outlives it; [free] is whether every storage lives for the whole scope, as in
   a loop's body. Each entry is weighed once, in order: it joins the earliest
   open group of its device and domain that starts past the last entry it
   conflicts with and past the first use of any storage whose life may not start
   earlier, the positions being those after the merges. *)
let rec merge_linear ~held ~free linear =
  let entries =
    Array.of_list
      (List.map
         (fun e ->
           match op e with
           | (Op.End | Op.Backedge) when op (nth e 0) = Op.Linear ->
               let body = merge_linear ~held ~free:true (nth e 0) in
               replace e ~src:(body :: List.tl (src e))
           | _ -> e)
         (src linear))
  in
  let n = Array.length entries in
  let limit = Setting.value Setting.max_kernel_buffers in
  let last_write = Tbl.create 64
  and last_read = Tbl.create 64
  and first_use = Tbl.create 64 in
  let at tbl b = Option.value (Tbl.find_opt tbl b) ~default:(-1) in
  let groups = Domain.create 16 in
  let opened = Array.make n None and joined = Array.make n false in
  let fits g si =
    limit = 0
    ||
    let fresh =
      List.sort_uniq Ops.compare
        (List.filter
           (fun a -> addrspace a <> Some Dtype.Alu && not (Tbl.mem g.passed a))
           (List.tl (src si)))
    in
    Tbl.length g.passed + List.length fresh <= limit
  in
  let join g si m =
    g.members <- (si, m) :: g.members;
    List.iter
      (fun a -> if addrspace a <> Some Dtype.Alu then Tbl.replace g.passed a ())
      (List.tl (src si))
  in
  for i = 0 to n - 1 do
    let e = entries.(i) in
    let taken = storage_taken e in
    let kernel = member e in
    let writes = match kernel with Some m -> m.writes | None -> taken in
    (* At the top level, only calls of one element join: the storage of a call
       of many elements would live longer. *)
    let m =
      match kernel with
      | Some m when free || m.ranges = [] -> Some m
      | _ -> None
    in
    let conflict =
      List.fold_left
        (fun c b ->
          let c = max c (at last_write b) in
          if List.memq b writes then max c (at last_read b) else c)
        (-1) taken
    in
    let bound =
      List.fold_left
        (fun lo b ->
          if free || Tbl.mem held b || Shape.max_numel b = 1 then lo
          else max lo (Option.value (Tbl.find_opt first_use b) ~default:i))
        (conflict + 1) taken
    in
    let pos =
      match m with
      | None -> i
      | Some m -> (
          let key = (m.device, m.domain) in
          let k =
            match Domain.find_opt groups key with
            | Some k -> k
            | None ->
                let k =
                  { gs = Dynarray.create (); next = Dynarray.create () }
                in
                Domain.replace groups key k;
                k
          in
          let rec pick j =
            let j = open_from k j in
            if j >= Dynarray.length k.gs then None
            else
              let g = Dynarray.get k.gs j in
              if fits g e then Some g
              else begin
                g.open_ <- false;
                pick (j + 1)
              end
          in
          match pick (at_or_after k bound) with
          | Some g ->
              join g e m;
              joined.(i) <- true;
              g.first
          | None ->
              let g =
                { first = i; members = []; passed = Tbl.create 8; open_ = true }
              in
              join g e m;
              Dynarray.add_last k.gs g;
              Dynarray.add_last k.next (Dynarray.length k.gs);
              opened.(i) <- Some g;
              i)
    in
    List.iter
      (fun b ->
        Tbl.replace last_read b (max pos (at last_read b));
        if not (Tbl.mem first_use b) then Tbl.replace first_use b pos;
        if List.memq b writes then
          Tbl.replace last_write b (max pos (at last_write b)))
      taken
  done;
  let out =
    List.filter_map
      (fun i ->
        if joined.(i) then None
        else
          match opened.(i) with
          | Some g when List.compare_length_with g.members 1 > 0 ->
              Some (merged (List.rev g.members))
          | _ -> Some entries.(i))
      (List.init n Fun.id)
  in
  replace linear ~src:out

(* The calls of [linear] that can run as one merged, at its top level and in its
   loops, [held] the storage the schedule is given. *)
let merge_calls ~held linear =
  let tbl = Tbl.create 16 in
  List.iter (fun b -> Tbl.replace tbl b ()) held;
  merge_linear ~held:tbl ~free:false linear

(* Schedules *)

let create_linear_with_vars ?(capturing = false) ?(by_storage = false) big_sink
    =
  let several = bound_several big_sink in
  let big_sink = transform_to_call big_sink in
  (* The call's sources are the values to realize. *)
  let linear_call =
    graph_rewrite ~calls:Enter ~pass:Fixed_point ~ctx:by_storage big_sink
      (After_sources pm_schedule)
  in
  (* The linear call is resolved recursively, and its storage allocated. *)
  let linear =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:several linear_call
      (After_sources pm_resolve_linear_call)
  in
  let linear =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() linear
      (After_sources pm_copy_from_store)
  in
  (* The storage the call is given, passed or viewed, outlives it. *)
  let held_bufs =
    List.filter_map
      (fun a ->
        let b = storage_base a in
        if op b = Op.Buffer then Some b else None)
      (List.tl (src linear_call))
  in
  let linear = merge_calls ~held:held_bufs linear in
  let used_vars =
    List.concat_map
      (fun si ->
        List.filter_map (fun v -> (param_of v).name) (variables (nth si 0)))
      (src linear)
  in
  let var_vals =
    List.fold_left
      (fun acc b ->
        if not (is_bound_var b) then acc
        else
          let nm = expr b in
          let value = Dtype.Value.to_int (Option.get (param_of b).bound) in
          (* A name of several values is no run's: its calls pass them. *)
          if
            List.mem nm several
            || (not (List.mem nm used_vars))
            || List.mem_assoc nm acc
          then acc
          else acc @ [ (nm, value) ])
      []
      (List.tl (src big_sink))
  in
  if capturing then (linear, var_vals)
  else (Memory.memory_plan_rewrite ~held_bufs linear, var_vals)
