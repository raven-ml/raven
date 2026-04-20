(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* JIT compilation.

   Three-phase execution: warmup (cnt=0) runs eagerly, capture (cnt=1)
   records the computation schedule, exec (cnt>=2) replays the compiled
   schedule with fresh input buffers.  On the first replay, the schedule
   may be condensed into graph executors when the device supports it. *)

open Tolk_ir
module K = Kernel
module T = Tensor
module B = Device.Buffer

let strf = Printf.sprintf
let debug = Helpers.getenv "DEBUG" 0
let jit_level = Helpers.getenv "JIT" 2
let jit_batch_size = Helpers.getenv "JIT_BATCH_SIZE" 0

(* Exceptions *)

exception Graph_exn of string
exception Jit_error of string

(* Types *)

let next_uid = ref 0
let fresh_uid () = let i = !next_uid in incr next_uid; i

(* Runner kind — replaces Python isinstance dispatch on Runner subclasses.
   Each variant carries enough to dispatch and to extract kind-specific
   data (e.g. Program_spec from a compiled kernel). *)
type prg =
  | Compiled of Realize.Compiled_runner.t
  | View_op of Realize.Runner.t
  | Buffer_copy of Realize.Runner.t
  | Buffer_xfer of Realize.Runner.t
  | Enc_dec of Realize.Runner.t
  | Graph of graph_runner

(* Execution item with mutable buffer slots for input substitution.
   [uid] provides stable identity across list rebuilds (replaces Python
   id() on ExecItem objects). *)
and exec_item = {
  uid : int;
  bufs : B.t option array;
  prg : prg;
  fixedvars : string list;
}

(* Graph runner — batches multiple kernels for accelerated dispatch.
   Stores precomputed replacement tables so the device graph only needs
   to update the values that actually change between calls. *)
and graph_runner = {
  gr_cache : exec_item list;
  gr_input_replace : ((int * int), int) Hashtbl.t;
  gr_var_replace : (int, (int * int) list) Hashtbl.t;
  gr_dims_replace : (int, int option * int option) Hashtbl.t;
  gr_dims_base : (int, int array * int array) Hashtbl.t;
  gr_vars : string array;
  gr_sym_dims : K.t array list;
  gr_w_dep : (int, (int * int * int) list) Hashtbl.t;
  gr_r_dep : (int, (int * int * int) list) Hashtbl.t;
  gr_runner : Realize.Runner.t;
}

(* A view input is a sub-buffer of an existing input that must be
   reconstructed from the base on every call. *)
type view_input = {
  vi_base_idx : int;
  vi_offset : int;
  vi_device : string;
  vi_size : int;
  vi_dtype : Dtype.t;
}

(* Validation token for ensuring inputs don't change shape between calls. *)
type input_info = {
  ii_size : int;
  ii_dtype : Dtype.t;
  ii_device : string;
}

(* Exec item helpers *)

let runner_of_prg = function
  | Compiled cr -> Realize.Compiled_runner.runner cr
  | View_op r | Buffer_copy r | Buffer_xfer r | Enc_dec r -> r
  | Graph gr -> gr.gr_runner

let run_ei ei var_vals ~jit =
  let runner = runner_of_prg ei.prg in
  let bufs = Array.to_list ei.bufs |> List.filter_map (fun b ->
    Option.map (fun buf -> B.ensure_allocated buf; buf) b) in
  ignore (Realize.Runner.call runner bufs var_vals
    ~wait:(not jit || debug >= 2) ~timeout:None)

(* Lower a Realize.Exec_item into our richer exec_item.  Compiles
   kernels via [get_runner] and wraps the result in the appropriate
   [prg] variant so we can dispatch on runner kind later. *)
let lower_realize_ei ~device ~get_program (rei : Realize.Exec_item.t)
    : exec_item =
  let bufs = Array.of_list (Realize.Exec_item.bufs rei) in
  let live_bufs = Array.to_list bufs |> List.filter_map Fun.id in
  let prg =
    match T.view (Realize.Exec_item.ast rei) with
    | Call { callee = Ast kernel; _ } ->
        Compiled (Realize.get_runner ~device ~get_program kernel)
    | Call { callee = Ref ref_node; _ } -> begin
        match T.view ref_node with
        | Buffer_view _ ->
            View_op (Realize.view_op ~device (List.hd live_bufs))
        | Copy _ ->
            let dest = List.nth live_bufs 0 in
            let src = List.nth live_bufs 1 in
            Buffer_copy (Realize.buffer_copy ~device
              ~total_sz:(B.nbytes dest)
              ~dest_device:(B.device dest)
              ~src_device:(B.device src))
        | _ -> failwith "lower_realize_ei: unsupported Ref callee"
      end
    | _ -> failwith "lower_realize_ei: expected Call node"
  in
  let fixedvars = List.map fst (Realize.Exec_item.var_vals rei) in
  { uid = fresh_uid (); bufs; prg; fixedvars }

(* Output buffers *)

(* Buffers written by an exec item.  For compiled kernels, output
   parameters that are not also inputs; for copies, the destination. *)
let get_out_buffers ei =
  match ei.prg with
  | Compiled cr ->
      let p = Realize.Compiled_runner.p cr in
      let ins = Program_spec.ins p in
      List.filter_map (fun out ->
        if List.mem out ins then None else ei.bufs.(out))
        (Program_spec.outs p)
  | Buffer_copy _ | Buffer_xfer _ | Enc_dec _ ->
      Option.to_list ei.bufs.(0)
  | View_op _ | Graph _ -> []

(* Buffer set *)

(* Set of buffers keyed by id, with an optional None sentinel for
   tracking "unknown" / cleared slots. *)
type buf_set = {
  mutable has_none : bool;
  tbl : (int, B.t) Hashtbl.t;
}

let buf_set () = { has_none = false; tbl = Hashtbl.create 32 }

let buf_set_mem s = function
  | None -> s.has_none
  | Some b -> Hashtbl.mem s.tbl (B.id b)

let buf_set_add s b = Hashtbl.replace s.tbl (B.id b) b

(* Propagate buffer dependencies forward through a cache: any exec item
   whose inputs overlap the seed set has its outputs added. *)
let update_depends depends cache =
  List.iter (fun ei ->
    if Array.exists (buf_set_mem depends) ei.bufs then
      List.iter (buf_set_add depends) (get_out_buffers ei))
    cache

(* Input replacement *)

(* Build (cache_idx, buf_idx) -> input_idx map.
   When [orig_valid_positions] is provided (keyed by exec_item uid),
   only positions valid during the original capture are included — this
   prevents aliasing bugs when graph batching reuses buffer slots. *)
let get_input_replace cache (input_bufs : B.t array)
    ?orig_valid_positions () =
  let idx_of_buf : (int, int) Hashtbl.t = Hashtbl.create 32 in
  Array.iteri (fun i buf ->
    Hashtbl.replace idx_of_buf (B.id buf) i) input_bufs;
  let result = Hashtbl.create 64 in
  List.iteri (fun j ei ->
    Array.iteri (fun i b ->
      match b with
      | None -> ()
      | Some buf ->
          match Hashtbl.find_opt idx_of_buf (B.id buf) with
          | None -> ()
          | Some idx ->
              let valid = match orig_valid_positions with
                | None -> true
                | Some vp ->
                    match Hashtbl.find_opt vp ei.uid with
                    | None -> false
                    | Some set -> List.mem i set
              in
              if valid then Hashtbl.replace result (j, i) idx)
      ei.bufs)
    cache;
  result

(* Graph runner *)

let is_sym_dim dim =
  let rec loop i =
    i < Array.length dim &&
    (match K.const_arg dim.(i) with None -> true | Some _ -> loop (i + 1))
  in
  loop 0

let dim_eq a b =
  Array.length a = Array.length b &&
  let rec loop i =
    i >= Array.length a || (K.tag a.(i) = K.tag b.(i) && loop (i + 1))
  in
  loop 0

let is_runtime_var p name =
  match Program_spec.core_id p with
  | Some ci ->
      let vars = Program_spec.vars p in
      ci.var_index < List.length vars &&
      (List.nth vars ci.var_index).name = name
  | None -> false

let create_graph_runner cache (input_bufs : B.t array)
    (var_vals : (string * int) list) ?orig_valid_positions () =
  let input_replace =
    get_input_replace cache input_bufs ?orig_valid_positions () in
  let vars =
    List.sort_uniq String.compare (List.map fst var_vals)
    |> Array.of_list in
  let var_index name =
    let rec loop i =
      if i >= Array.length vars then
        failwith (strf "graph_runner: unknown variable %S" name)
      else if String.equal vars.(i) name then i
      else loop (i + 1)
    in
    loop 0
  in
  (* Collect unique symbolic launch dimension vectors. *)
  let sym_dims = ref [] in
  let add_if_sym dim =
    if is_sym_dim dim && not (List.exists (dim_eq dim) !sym_dims) then
      sym_dims := dim :: !sym_dims
  in
  List.iter (fun ei ->
    match ei.prg with
    | Compiled cr ->
        let p = Realize.Compiled_runner.p cr in
        (match Program_spec.local_size p with
         | Some ls -> add_if_sym ls
         | None -> ());
        add_if_sym (Program_spec.global_size p)
    | _ -> ())
    cache;
  let sym_dims = List.rev !sym_dims in
  let find_sym_idx dim =
    if not (is_sym_dim dim) then None
    else
      let rec loop i = function
        | [] -> None
        | d :: rest -> if dim_eq d dim then Some i else loop (i + 1) rest
      in
      loop 0 sym_dims
  in
  (* Build per-kernel replacement tables. *)
  let var_replace = Hashtbl.create 16 in
  let dims_replace = Hashtbl.create 16 in
  let dims_base = Hashtbl.create 16 in
  let total_est = ref Program_spec.Estimates.zero in
  List.iteri (fun j ei ->
    total_est := Program_spec.Estimates.( + ) !total_est
      (Realize.Runner.estimates (runner_of_prg ei.prg));
    match ei.prg with
    | Compiled cr ->
        let p = Realize.Compiled_runner.p cr in
        (* Variables needing runtime substitution: not fixed, not runtime. *)
        let replace = ref [] in
        List.iteri (fun i (v : Program_spec.var) ->
          if not (List.mem v.name ei.fixedvars) &&
             not (is_runtime_var p v.name)
          then replace := (i, var_index v.name) :: !replace)
          (Program_spec.vars p);
        if !replace <> [] then
          Hashtbl.replace var_replace j (List.rev !replace);
        (* Symbolic launch dims. *)
        let g = Program_spec.global_size p in
        let gi = find_sym_idx g in
        let li = match Program_spec.local_size p with
          | Some ls -> find_sym_idx ls | None -> None in
        if gi <> None || li <> None then begin
          Hashtbl.replace dims_replace j (gi, li);
          let eval d = Array.map (fun s -> K.sym_infer s var_vals) d in
          let base_l = match Program_spec.local_size p with
            | Some ls -> eval ls | None -> [| 1; 1; 1 |] in
          Hashtbl.replace dims_base j (eval g, base_l)
        end
    | _ -> ())
    cache;
  let dev = Realize.Runner.dev (runner_of_prg (List.hd cache).prg) in
  (* Base runner — device-specific graph implementations override call. *)
  let runner = Realize.Runner.make
    ~display_name:(strf "<batched %d>" (List.length cache))
    ~device:dev ~estimates:!total_est
    (fun _bufs _var_vals ~wait:_ ~timeout:_ -> None) in
  { gr_cache = cache; gr_input_replace = input_replace;
    gr_var_replace = var_replace; gr_dims_replace = dims_replace;
    gr_dims_base = dims_base; gr_vars = vars; gr_sym_dims = sym_dims;
    gr_w_dep = Hashtbl.create 0; gr_r_dep = Hashtbl.create 0;
    gr_runner = runner }

(* (cache_idx, program_var_idx, value) for runtime variable updates. *)
let updated_vars gr var_vals =
  let vals = Array.map (fun name -> List.assoc name var_vals) gr.gr_vars in
  let acc = ref [] in
  Hashtbl.iter (fun j vidxs ->
    List.iter (fun (i, v) -> acc := (j, i, vals.(v)) :: !acc) vidxs)
    gr.gr_var_replace;
  !acc

(* (cache_idx, global, local) for symbolic launch dimension updates. *)
let updated_launch_dims gr var_vals =
  let dims = List.map (fun dim ->
    Array.map (fun s -> K.sym_infer s var_vals) dim)
    gr.gr_sym_dims |> Array.of_list in
  let acc = ref [] in
  Hashtbl.iter (fun j (gi, li) ->
    let base_g, base_l = Hashtbl.find gr.gr_dims_base j in
    let g = match gi with Some i -> dims.(i) | None -> base_g in
    let l = match li with Some i -> dims.(i) | None -> base_l in
    acc := (j, g, l) :: !acc)
    gr.gr_dims_replace;
  !acc

(* Interval-based read/write dependency tracking for suballocated buffers.
   Device-specific graph implementations call this to discover which
   previously-launched kernels a new dispatch must wait on. *)
let access_resources gr bufs ~write new_dep =
  let get tbl key =
    match Hashtbl.find_opt tbl key with Some l -> l | None -> [] in
  let overlaps st en s e = st < e && s < en in
  (* Phase 1: collect wait dependencies from overlapping ranges. *)
  let wait = Hashtbl.create 8 in
  Array.iteri (fun i buf ->
    let key = B.base_id buf in
    let s = B.offset buf in
    let e = s + B.nbytes buf in
    List.iter (fun (st, en, dep) ->
      if overlaps st en s e then Hashtbl.replace wait dep dep)
      (get gr.gr_w_dep key);
    if List.mem i write then
      List.iter (fun (st, en, dep) ->
        if overlaps st en s e then Hashtbl.replace wait dep dep)
        (get gr.gr_r_dep key))
    bufs;
  (* Phase 2: clip written intervals and insert new dependency. *)
  let clip entries s e =
    List.concat_map (fun (st, en, dep) ->
      (if st < min s en then [(st, min s en, dep)] else []) @
      (if max e st < en then [(max e st, en, dep)] else []))
      entries
  in
  Array.iteri (fun i buf ->
    let key = B.base_id buf in
    let s = B.offset buf in
    let e = s + B.nbytes buf in
    if List.mem i write then begin
      Hashtbl.replace gr.gr_w_dep key
        (clip (get gr.gr_w_dep key) s e @ [(s, e, new_dep)]);
      Hashtbl.replace gr.gr_r_dep key
        (clip (get gr.gr_r_dep key) s e)
    end else
      Hashtbl.replace gr.gr_r_dep key
        (get gr.gr_r_dep key @ [(s, e, new_dep)]))
    bufs;
  Hashtbl.fold (fun _ dep acc -> dep :: acc) wait []

let supports_exec_item devs ei =
  match ei.prg with
  | Compiled _ ->
      let n = List.length (List.sort_uniq (fun a b ->
        String.compare (Device.name a) (Device.name b)) devs) in
      n = 1
  | _ -> false

(* Multi-device variant: all devices must be the same backend type. *)
let multi_supports_exec_item devs ei =
  let backend name =
    match String.split_on_char ':' name with t :: _ -> t | [] -> name in
  match ei.prg with
  | Compiled _ | Buffer_xfer _ ->
      let buf_types = Array.to_list ei.bufs |> List.filter_map (fun b ->
        Option.map (fun buf -> backend (B.device buf)) b) in
      let dev_types = List.map (fun d -> backend (Device.name d)) devs in
      List.length (List.sort_uniq String.compare (buf_types @ dev_types)) = 1
  | _ -> false

(* Graph batching *)

(* Split the jit cache into batches for graph execution.  Consecutive
   compatible kernels are condensed into a single graph executor when
   the device provides a graph implementation.  The batch size doubles
   after each successful graph, allowing the accelerator to update
   later graphs while early ones are still running. *)
let apply_graph_to_jit cache (input_bufs : B.t array)
    (var_vals : (string * int) list) ?orig_valid_positions
    ?(max_batch_size = 0) () =
  let graph_one = Helpers.getenv "GRAPH_ONE_KERNEL" 0 <> 0 in
  let graphed = ref [] in
  let batch = ref [] in
  let batch_devs : Device.t list ref = ref [] in
  let max_bs = ref max_batch_size in
  let dedup_devs ds =
    List.sort_uniq (fun a b ->
      String.compare (Device.name a) (Device.name b)) ds in
  let flush () =
    begin try
      if !batch_devs = [] then
        raise (Graph_exn "no device for graph");
      if List.length !batch <= 1 && not graph_one then
        raise (Graph_exn "only one kernel doesn't graph");
      let dev = List.hd !batch_devs in
      (* Device graph construction: dev.graph(batch, input_bufs, var_vals).
         When graph support is added, the device will provide a constructor
         that returns a graph_runner wrapping the batched kernels. *)
      ignore (dev, input_bufs, var_vals, orig_valid_positions);
      raise (Graph_exn "device graph not yet implemented")
    with Graph_exn e ->
      graphed := List.rev_append !batch !graphed;
      if debug >= 2 then
        Printf.eprintf "JIT GRAPHing failed batch with %d kernels: %s\n%!"
          (List.length !batch) e
    end;
    batch := [];
    batch_devs := []
  in
  List.iter (fun ei ->
    let ji_dev = match ei.prg with
      | Compiled cr ->
          Some (Realize.Runner.dev (Realize.Compiled_runner.runner cr))
      | View_op _ -> None  (* silently skipped *)
      | _ -> None
    in
    (* Graphability requires a device with graph support.  When a device
       implements [graph], this check also calls [supports_exec_item]. *)
    let can_graph = match ji_dev with
      | Some _dev -> false  (* no device graph support yet *)
      | None -> false
    in
    let can_share = can_graph && !batch_devs <> [] in
    let can_extend = can_share &&
      (!max_bs = 0 || List.length !batch < !max_bs) in
    if not can_extend && !batch <> [] then flush ();
    if can_graph then begin
      batch := ei :: !batch;
      batch_devs := dedup_devs (match ji_dev with
        | Some d -> d :: !batch_devs | None -> !batch_devs)
    end else begin
      graphed := ei :: !graphed;
      batch_devs := []
    end)
    cache;
  if !batch <> [] then flush ();
  ignore max_bs;
  List.rev !graphed

(* Memory planning *)

(* Apply the internal memory planner to a jit cache, returning a new
   cache with buffer assignments optimized.  Buffers absent from the
   planner's assignment table keep their original allocation. *)
let plan_jit_memory jit_cache =
  let copies = List.filter_map (fun ei ->
    match ei.prg with
    | Buffer_copy _ | Buffer_xfer _ | Enc_dec _ ->
        (match ei.bufs.(0), (if Array.length ei.bufs > 1 then ei.bufs.(1)
                             else None) with
         | Some dst, Some src -> Some (dst, src)
         | _ -> None)
    | _ -> None) jit_cache in
  let buffers = List.map (fun ei ->
    Array.to_list ei.bufs |> List.filter_map Fun.id) jit_cache in
  let assigned =
    Memory.internal_memory_planner ~copies ~debug_prefix:"JIT " buffers in
  List.map (fun ei ->
    let new_bufs = Array.map (function
      | None -> None
      | Some buf ->
          let repl = match Hashtbl.find_opt assigned (B.id buf) with
            | Some b -> b | None -> buf in
          B.ensure_allocated repl;
          Some repl) ei.bufs in
    { ei with bufs = new_bufs; uid = fresh_uid () }) jit_cache

(* Captured JIT *)

type 'a captured_jit = {
  ret : 'a;
  jit_cache : exec_item array;
  input_replace : ((int * int), int) Hashtbl.t;
  extra_view_inputs : view_input list;
  expected_input_info : input_info array;
  mutable live_cache : exec_item list;
  mutable live_replace : ((int * int), int) Hashtbl.t;
  mutable first_run : bool;
  output_to_writer : (int, int) Hashtbl.t;
  input_to_max_reader : (int, int) Hashtbl.t;
}

(* Null out input buffer slots so their memory can be reused. *)
let clear_inputs t =
  Hashtbl.iter (fun (j, i) _ ->
    (List.nth t.live_cache j).bufs.(i) <- None)
    t.live_replace

(* Precompute read-after-write hazard detection tables.
   output_to_writer: buffer_id -> cache index that writes it.
   input_to_max_reader: input buffer index -> latest cache index
   that reads it (only when the buffer is NOT also an output of
   that same kernel, since same-kernel overlap is always safe). *)
let init_hazard_tables t =
  Hashtbl.clear t.output_to_writer;
  Array.iteri (fun j ei ->
    List.iter (fun b ->
      Hashtbl.replace t.output_to_writer (B.id b) j)
      (get_out_buffers ei))
    t.jit_cache;
  Hashtbl.clear t.input_to_max_reader;
  Hashtbl.iter (fun (j, i) idx ->
    let ei = t.jit_cache.(j) in
    let outs = get_out_buffers ei in
    let is_own_output = match ei.bufs.(i) with
      | None -> false
      | Some b ->
          List.exists (fun o -> B.id o = B.id b) outs
    in
    if not is_own_output then begin
      let prev = match Hashtbl.find_opt t.input_to_max_reader idx with
        | Some n -> n | None -> -1 in
      if j > prev then Hashtbl.replace t.input_to_max_reader idx j
    end)
    t.input_replace

let create_captured ret jit_cache input_replace extra_view_inputs
    expected_input_info =
  let jit_cache = Array.of_list jit_cache in
  let t = {
    ret; jit_cache; input_replace; extra_view_inputs; expected_input_info;
    live_cache = Array.to_list jit_cache;
    live_replace = input_replace;
    first_run = true;
    output_to_writer = Hashtbl.create 32;
    input_to_max_reader = Hashtbl.create 16;
  } in
  init_hazard_tables t;
  clear_inputs t;
  t

let free_intermediates t =
  let dep = buf_set () in
  dep.has_none <- true;
  update_depends dep (Array.to_list t.jit_cache);
  Hashtbl.iter (fun _ buf ->
    if B.is_allocated buf then B.deallocate buf)
    dep.tbl;
  (* Reset execution state. *)
  t.live_cache <- Array.to_list t.jit_cache;
  t.live_replace <- t.input_replace;
  t.first_run <- true;
  init_hazard_tables t;
  clear_inputs t

let replan_buffers_memory_layout t =
  (* Snapshot old buffers so we can copy data after remapping. *)
  let old_bufs : (int, B.t) Hashtbl.t = Hashtbl.create 32 in
  Array.iter (fun ei ->
    Array.iter (function
      | None -> ()
      | Some buf -> Hashtbl.replace old_bufs (B.id buf) buf)
      ei.bufs)
    t.jit_cache;
  (* Run memory planner over all buffers with ignore_checks. *)
  let all = [Array.to_list t.jit_cache |> List.concat_map (fun ei ->
    Array.to_list ei.bufs |> List.filter_map Fun.id)] in
  let assigned =
    Memory.internal_memory_planner ~ignore_checks:true all in
  (* Remap jit_cache buffers. *)
  let new_cache = Array.map (fun ei ->
    let new_bufs = Array.map (function
      | None -> None
      | Some buf ->
          Some (match Hashtbl.find_opt assigned (B.id buf) with
                | Some b -> b | None -> buf))
      ei.bufs in
    { ei with bufs = new_bufs }) t.jit_cache in
  (* Copy data from old to new for any reassigned buffer. *)
  Hashtbl.iter (fun old_id new_buf ->
    match Hashtbl.find_opt old_bufs old_id with
    | Some old_buf when B.is_allocated old_buf ->
        B.ensure_allocated new_buf;
        let tmp = Bytes.create (B.nbytes old_buf) in
        B.copyout old_buf tmp;
        B.copyin new_buf tmp
    | _ -> ())
    assigned;
  (* Reinitialize with the new cache. *)
  let cache_list = Array.to_list new_cache in
  Array.blit new_cache 0 t.jit_cache 0 (Array.length new_cache);
  t.live_cache <- cache_list;
  t.live_replace <- t.input_replace;
  t.first_run <- true;
  init_hazard_tables t;
  clear_inputs t

(* Execute the captured schedule with fresh input buffers. *)
let exec_captured t ~device (input_bufs : B.t array)
    (var_vals : (string * int) list) =
  (* Validate inputs match what was captured. *)
  let n_expected = Array.length t.expected_input_info in
  if Array.length input_bufs <> n_expected then
    raise (Jit_error (strf "input count mismatch: expected %d, got %d"
      n_expected (Array.length input_bufs)));
  Array.iteri (fun i info ->
    let buf = input_bufs.(i) in
    if B.size buf <> info.ii_size
       || not (Dtype.equal (B.dtype buf) info.ii_dtype)
       || B.device buf <> info.ii_device
    then
      raise (Jit_error (strf
        "input %d mismatch: expected (%d, %s, %s), got (%d, %s, %s)" i
        info.ii_size (Dtype.to_string info.ii_dtype) info.ii_device
        (B.size buf) (Dtype.to_string (B.dtype buf)) (B.device buf))))
    t.expected_input_info;
  (* Extend input_bufs with view inputs reconstructed from base buffers. *)
  let n = Array.length input_bufs in
  let n_extra = List.length t.extra_view_inputs in
  let bufs = Array.init (n + n_extra) (fun i ->
    if i < n then input_bufs.(i) else input_bufs.(0)) in
  Array.blit input_bufs 0 bufs 0 n;
  List.iteri (fun k vi ->
    let base = bufs.(vi.vi_base_idx) in
    let view = B.view base
      ~size:vi.vi_size ~dtype:vi.vi_dtype
      ~offset:(vi.vi_offset * Dtype.itemsize vi.vi_dtype) in
    B.ensure_allocated view;
    bufs.(n + k) <- view)
    t.extra_view_inputs;
  (* Copy aliased inputs to prevent read-after-write hazards.
     When an input is also written by a kernel and a later kernel
     reads the same input, snapshot the input before execution. *)
  for i = 0 to Array.length bufs - 1 do
    let ib = bufs.(i) in
    match Hashtbl.find_opt t.output_to_writer (B.id ib) with
    | None -> ()
    | Some writer ->
        let max_reader =
          match Hashtbl.find_opt t.input_to_max_reader i with
          | Some n -> n | None -> -1 in
        if max_reader >= writer then begin
          let copy = Device.create_buffer
            ~size:(B.size ib) ~dtype:(B.dtype ib) device in
          B.ensure_allocated copy;
          let tmp = Bytes.create (B.nbytes ib) in
          B.copyout ib tmp;
          B.copyin copy tmp;
          bufs.(i) <- copy
        end
  done;
  (* Assign input buffers into their live cache slots. *)
  Hashtbl.iter (fun (j, i) idx ->
    (List.nth t.live_cache j).bufs.(i) <- Some bufs.(idx))
    t.live_replace;
  (* First run: allocate intermediates and try graph batching. *)
  if t.first_run then begin
    Array.iter (fun ei ->
      Array.iter (function
        | Some buf -> B.ensure_allocated buf
        | None -> ())
        ei.bufs)
      t.jit_cache;
    if jit_level < 2 then begin
      (* Build valid positions from the capture-time input_replace. *)
      let orig_valid : (int, int list) Hashtbl.t = Hashtbl.create 32 in
      Hashtbl.iter (fun (j, i) _ ->
        let uid = t.jit_cache.(j).uid in
        let prev = match Hashtbl.find_opt orig_valid uid with
          | Some l -> l | None -> [] in
        if not (List.mem i prev) then
          Hashtbl.replace orig_valid uid (i :: prev))
        t.input_replace;
      t.live_cache <- apply_graph_to_jit (Array.to_list t.jit_cache)
        bufs var_vals ~orig_valid_positions:orig_valid
        ~max_batch_size:jit_batch_size ();
      (* Recompute input_replace: graph items have all positions valid,
         non-graph items keep their original valid positions. *)
      let valid : (int, int list) Hashtbl.t = Hashtbl.create 32 in
      List.iter (fun ei ->
        let positions = match ei.prg with
          | Graph _ -> List.init (Array.length ei.bufs) Fun.id
          | _ ->
              match Hashtbl.find_opt orig_valid ei.uid with
              | Some l -> l | None -> [] in
        Hashtbl.replace valid ei.uid positions)
        t.live_cache;
      t.live_replace <-
        get_input_replace t.live_cache bufs ~orig_valid_positions:valid ()
    end;
    t.first_run <- false
  end;
  if debug >= 1 && List.length t.live_cache >= 10 then
    Printf.eprintf "jit execs %d kernels\n%!" (List.length t.live_cache);
  List.iter (fun ei -> run_ei ei var_vals ~jit:true) t.live_cache;
  clear_inputs t;
  t.ret

(* Capture state *)

(* Non-empty during JIT capture.  The schedule machinery should call
   [add_linear] to record each linear into the active capture. *)
let capturing : T.t list ref option ref = ref None

let is_capturing () = Option.is_some !capturing

let add_linear linear =
  match !capturing with
  | None -> failwith "add_linear: not inside a JIT capture"
  | Some linears -> linears := linear :: !linears

(* TinyJit *)

type 'a tiny_jit = {
  fxn : (B.t array -> (string * int) list -> 'a) option;
  device : Device.t;
  get_program : Kernel.t -> Program_spec.t;
  prune : bool;
  optimize : bool;
  mutable captured : 'a captured_jit option;
  mutable cnt : int;
}

let captured t = t.captured
let jit_cache t = t.jit_cache

let create ~device ~get_program ?fxn ?captured
    ?(prune = false) ?(optimize = false) () =
  let cnt = if fxn = None then 2 else 0 in
  { fxn; device; get_program; prune; optimize; captured; cnt }

let reset t =
  if t.fxn = None then invalid_arg "can't reset without function";
  t.cnt <- 0;
  t.captured <- None

let call t (input_bufs : B.t array)
    (var_vals : (string * int) list)
    ~(buffers : T.t -> B.t option) =
  let ret =
    if jit_level = 0 || t.cnt = 0 then begin
      (* Warmup: execute eagerly. *)
      let fxn = Option.get t.fxn in
      fxn input_bufs var_vals
    end
    else if t.cnt = 1 then begin
      (* Capture: record the computation schedule. *)
      let fxn = Option.get t.fxn in
      if is_capturing () then
        raise (Jit_error "nested TinyJit is not supported");
      let linears = ref [] in
      capturing := Some linears;
      let ret = Fun.protect
        ~finally:(fun () -> capturing := None)
        (fun () -> fxn input_bufs var_vals) in
      let linears = List.rev !linears in
      if linears = [] then raise (Jit_error "didn't JIT anything!");
      if debug >= 1 then
        Printf.eprintf "JIT captured %d linears with %d inputs\n%!"
          (List.length linears) (Array.length input_bufs);
      (* Combine captured linears into a single schedule. *)
      let linear = T.linear (List.concat_map (fun l ->
        match T.view l with
        | Linear { srcs } -> srcs
        | _ -> [l]) linears) in
      (* Convert to exec items via schedule + lower. *)
      let realize_eis =
        Schedule.linear_to_schedule linear ~buffers in
      let jit_cache = List.map
        (lower_realize_ei ~device:t.device ~get_program:t.get_program)
        realize_eis in
      (* Track view inputs: sub-buffers whose base is an input. *)
      let extra_views = ref [] in
      let all_bufs = ref (Array.to_list input_bufs) in
      List.iter (fun ei ->
        Array.iter (fun b ->
          match b with
          | None -> ()
          | Some buf ->
              let base = B.base buf in
              if B.id buf <> B.id base then begin
                let base_idx = ref (-1) in
                List.iteri (fun k ib ->
                  if B.id ib = B.id base then base_idx := k)
                  !all_bufs;
                if !base_idx >= 0 then begin
                  all_bufs := !all_bufs @ [buf];
                  extra_views :=
                    { vi_base_idx = !base_idx;
                      vi_offset = B.offset buf;
                      vi_device = B.device buf;
                      vi_size = B.size buf;
                      vi_dtype = B.dtype buf } :: !extra_views
                end
              end)
          ei.bufs)
        jit_cache;
      (* Prune independent kernels (optional). *)
      let jit_cache =
        if t.prune then begin
          let dep = buf_set () in
          Array.iter (buf_set_add dep) input_bufs;
          update_depends dep jit_cache;
          let pruned, onetime = List.partition (fun ei ->
            List.exists (fun b -> Hashtbl.mem dep.tbl (B.id b))
              (get_out_buffers ei))
            jit_cache in
          if debug >= 1 then
            Printf.eprintf "pruned from %d -> %d kernels\n%!"
              (List.length jit_cache) (List.length pruned);
          (* Synchronize devices before running onetime kernels. *)
          let seen_devs = Hashtbl.create 4 in
          List.iter (fun ei ->
            Array.iter (function
              | None -> ()
              | Some buf ->
                  let dname = B.device buf in
                  if not (Hashtbl.mem seen_devs dname) then begin
                    Hashtbl.replace seen_devs dname ();
                    Device.synchronize t.device
                  end)
              ei.bufs)
            onetime;
          (* Run onetime kernels now; they won't be replayed. *)
          List.iter (fun ei -> run_ei ei var_vals ~jit:true) onetime;
          pruned
        end else jit_cache
      in
      (* Memory planning. *)
      let jit_cache = plan_jit_memory jit_cache in
      let input_arr = Array.of_list !all_bufs in
      let input_replace = get_input_replace jit_cache input_arr () in
      if debug >= 1 then begin
        let n_unique =
          let s = Hashtbl.create 16 in
          Hashtbl.iter (fun _ v -> Hashtbl.replace s v ()) input_replace;
          Hashtbl.length s in
        if n_unique <> Array.length input_bufs then
          Printf.eprintf "WARNING: some input tensors not found\n%!"
      end;
      (* Execute the schedule. *)
      List.iter (fun ei -> run_ei ei var_vals ~jit:false) jit_cache;
      (* Record input shapes for validation on subsequent calls. *)
      let expected_input_info = Array.map (fun buf ->
        { ii_size = B.size buf; ii_dtype = B.dtype buf;
          ii_device = B.device buf })
        input_bufs in
      t.captured <- Some (create_captured ret jit_cache input_replace
        (List.rev !extra_views) expected_input_info);
      if t.optimize then
        replan_buffers_memory_layout (Option.get t.captured);
      ret
    end
    else begin
      (* Exec: replay the captured schedule. *)
      let captured = Option.get t.captured in
      exec_captured captured ~device:t.device input_bufs var_vals
    end
  in
  t.cnt <- t.cnt + 1;
  ret
