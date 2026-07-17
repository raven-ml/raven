(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Persistent compilation cache for jitted traces.

   A compiled trace is a deterministic function of the traced computation —
   the PARAM-normalized CALL body produced by [Callify.transform_to_call]
   — and of the compilation environment: this executable (scheduling and
   codegen live in the binary), the device and its kernel compiler, and the
   codegen knobs that change lowering output. [key] digests exactly those;
   [store] saves the compiled LINEAR (kernels scheduled, lowered, and
   compiled to binaries) normalized back onto dense PARAM slots; [load]
   imports it and rebinds it to the fresh trace's buffer nodes. A warm
   process thus skips scheduling, lowering, and kernel compilation — tracing
   and buffer allocation always rerun and produce the buffers the stored
   schedule is rebound to.

   Entries live in [Tolk.Diskcache] table "rune_jit" under the platform
   cache directory (e.g. ~/.cache/tolk/rune_jit). JITCACHE=0 disables the
   cache entirely; it then never touches the disk. Every failure mode — a
   corrupt entry, an import from an incompatible format, a slot descriptor
   mismatch — degrades to a plain miss, and the recompile overwrites the
   entry.

   Symbolic variables: a stored entry records the names of the bound
   variables whose values the schedule needs, and [load] re-extracts the
   values from the fresh CALL's BIND arguments. Rune traces cannot produce
   BINDs today (no symbolic shapes), so the list is always empty; the code
   handles them anyway so the entry format does not change when they
   appear. *)

module U = Tolk_uop.Uop
module TD = Tolk_uop.Dtype
module Ops = Tolk_uop.Ops

let format_version = 1
let table = "rune_jit"

let env_int name default =
  match Sys.getenv_opt name with
  | Some s -> ( match int_of_string_opt s with Some v -> v | None -> default)
  | None -> default

(* Read per call, not at module initialization, so tests can toggle them
   with [Unix.putenv]. *)
let enabled () = env_int "JITCACHE" 1 <> 0
let debug () = env_int "RUNE_JIT_DEBUG" 0

let log event key =
  if debug () >= 1 then
    Printf.eprintf "rune.jit: compile cache %s %s\n%!" event key

(* Scheduling and code generation live in this binary: any change to it can
   change what a key would compile to, so the executable's digest versions
   every entry. Computed once per process. *)
let exe_digest = lazy (Digest.to_hex (Digest.file Sys.executable_name))

(* One entry per (key). [e_slots] validates the fresh CALL's arguments
   against the saving trace's before the graph is imported; [e_linear] is
   [Uop.export] of the compiled LINEAR with each CALL argument replaced by a
   PARAM carrying its position; [e_vars] are the symbolic variable names
   whose values must be re-extracted from the fresh CALL's BIND args. *)

type slot_desc = {
  sd_numel : int;  (* -1 when the argument has no tensor shape (a BIND) *)
  sd_dtype : TD.t;
  sd_device : U.device option;
  sd_is_bind : bool;
}

type entry = {
  e_slots : slot_desc array;
  e_linear : string;
  e_vars : string list;
}

let numel_of u =
  match U.max_shape u with
  | s -> List.fold_left ( * ) 1 s
  | exception Invalid_argument _ -> -1

let slot_desc u =
  {
    sd_numel = numel_of u;
    sd_dtype = U.dtype u;
    sd_device = U.device_of u;
    sd_is_bind = Ops.equal (U.op u) Ops.Bind;
  }

let slot_matches sd u =
  Bool.equal sd.sd_is_bind (Ops.equal (U.op u) Ops.Bind)
  && sd.sd_numel = numel_of u
  && TD.equal sd.sd_dtype (U.dtype u)
  && sd.sd_device = U.device_of u

(* Key *)

let key ~device call =
  if not (enabled ()) then None
  else
    match U.as_call call with
    | None -> None
    | Some { body; _ } ->
        (* The compiler's disk-cache key identifies the exact compilation
           target (e.g. "compile_cuda_sm_90"), so it fingerprints the
           architecture on top of the compiler name. *)
        let compiler_id =
          match Tolk.Renderer.compiler (Tolk.Device.renderer device) with
          | Some c ->
              Tolk.Compiler.name c ^ ":"
              ^ Option.value ~default:"" (Tolk.Compiler.cachekey c)
          | None -> ""
        in
        (* Exactly the environment knobs that change lowering output for a
           fixed binary: the optimization toggles read by [Tolk.Codegen]. *)
        let knobs =
          Printf.sprintf "NOOPT=%d,BEAM=%d,BEAM_ESTIMATE=%d"
            (env_int "NOOPT" 0) (env_int "BEAM" 0)
            (env_int "BEAM_ESTIMATE" 1)
        in
        Some
          (Digest.to_hex
             (Digest.string
                (String.concat "\x00"
                   [
                     string_of_int format_version;
                     Lazy.force exe_digest;
                     Tolk.Device.name device;
                     compiler_id;
                     knobs;
                     U.semantic_key body;
                   ])))

(* Save *)

(* The PARAM standing for CALL argument [i], as [Tolk.Jit.jit_lower] builds
   it for its own input substitution. *)
let param_of_arg i u =
  let shape =
    match U.as_buffer u with Some { shape; _ } -> Some shape | None -> None
  in
  U.param ~slot:i ~dtype:(U.dtype u) ?shape ?device:(U.device_of u) ()

(* A global buffer slot minted by this process would alias an unrelated
   buffer in an importing process (buffers hash-cons on their slot); after
   normalization the only such slots left must be the internal (negative)
   ones, which [load] renumbers. *)
let leaks_local_slot normalized =
  List.exists
    (fun n ->
      match U.as_buffer n with
      | Some { buffer = { slot; addrspace = TD.Global; _ }; _ } -> slot >= 0
      | _ -> false)
    (U.toposort ~enter_calls:true normalized)

let store ~key call linear var_vals =
  match U.as_call call with
  | None -> ()
  | Some { args; _ } ->
      let mappings = List.mapi (fun i u -> (u, param_of_arg i u)) args in
      let normalized = U.substitute ~walk:true mappings linear in
      if leaks_local_slot normalized then log "skip (unnormalized buffer)" key
      else begin
        let entry =
          {
            e_slots = Array.of_list (List.map slot_desc args);
            e_linear = U.export normalized;
            e_vars = List.map fst var_vals;
          }
        in
        Tolk.Diskcache.put ~table ~key entry;
        log "store" key
      end

(* Load *)

(* Mirrors the BIND extraction in [Schedule.create_linear_with_vars]: bind
   values live in the CALL's arguments and rebind on every compile. Raises
   [Not_found] when a stored name has no fresh bind, which [load] turns into
   a miss. *)
let vars_of_args names args =
  if names = [] then []
  else
    let bound =
      List.filter_map
        (fun u ->
          match U.as_bind u with
          | Some { var; value } -> (
              match (U.as_param var, U.op value, U.arg value) with
              | ( Some { param = { name = Some name; _ }; _ },
                  Ops.Const,
                  U.Arg.Value v ) -> (
                  match Tolk_uop.Const.view v with
                  | Tolk_uop.Const.Int n -> Some (name, Int64.to_int n)
                  | _ -> None)
              | _ -> None)
          | None -> None)
        args
    in
    List.map (fun name -> (name, List.assoc name bound)) names

let rebind entry args =
  let args_a = Array.of_list args in
  if
    Array.length entry.e_slots <> Array.length args_a
    || not (Array.for_all2 slot_matches entry.e_slots args_a)
  then None
  else begin
    let linear = U.import entry.e_linear in
    (* Imported internal buffers carry slots minted by the saving process;
       hash-consing on slots would silently alias them with this process's
       internal buffers, so rebuild each with a fresh local slot (same
       traversal as [Schedule.memory_plan_rewrite]'s substitution). *)
    let renumber =
      List.filter_map
        (fun n ->
          match U.as_buffer n with
          | Some { buffer; shape } when buffer.slot < 0 ->
              Some
                ( n,
                  U.buffer
                    ~slot:(Tolk.Schedule.fresh_internal_buffer_slot ())
                    ~dtype:(U.dtype n) ~shape ?name:buffer.name
                    ~addrspace:buffer.addrspace ?axis:buffer.axis
                    ?device:buffer.device () )
          | _ -> None)
        (U.toposort ~enter_calls:true linear)
    in
    let linear =
      if renumber = [] then linear
      else
        U.graph_rewrite ~name:"jitcache_renumber" ~enter_calls:true
          ~bottom_up:true
          (fun node -> List.assq_opt node renumber)
          linear
    in
    (* Rebind the schedule onto this trace's buffers: the inverse of the
       save-time normalization, matching PARAMs by slot as
       [Schedule.post_sched_cache_rule] does. Kernel-internal PARAMs live
       inside CALL bodies, which the default traversal does not enter, and
       replacements are final ([walk]) so a fresh BIND's own variable PARAM
       is not rewritten. *)
    let n = Array.length args_a in
    let linear =
      U.graph_rewrite ~name:"jitcache_args" ~walk:true
        (fun node ->
          match U.as_param node with
          | Some { param = { slot; _ }; _ } when slot >= 0 && slot < n ->
              Some args_a.(slot)
          | _ -> None)
        linear
    in
    Some (linear, vars_of_args entry.e_vars args)
  end

let load ~key call =
  match U.as_call call with
  | None -> None
  | Some { args; _ } -> (
      match (Tolk.Diskcache.get ~table ~key : entry option) with
      | None ->
          log "miss" key;
          None
      | Some entry -> (
          (* Any failure past this point — a malformed export blob, an
             argument mismatch — is a miss; the recompile overwrites the
             entry. *)
          match rebind entry args with
          | Some _ as hit ->
              log "hit" key;
              hit
          | None | (exception _) ->
              log "miss" key;
              None))
