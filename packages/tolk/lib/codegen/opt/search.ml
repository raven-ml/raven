(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Beam search kernel optimizer. Port of tinygrad/codegen/opt/search.py to
   the tolk_uop IR. *)

open Tolk_uop
module U = Uop
module P = Postrange

(* Environment *)

let beam_padto = Helpers.getenv "BEAM_PADTO" 0 <> 0
let tc = Helpers.getenv "TC" 1
let tc_opt = Helpers.getenv "TC_OPT" 2
let debug = Helpers.getenv "DEBUG" 0
let beam_debug = Helpers.getenv "BEAM_DEBUG" 0

let beam_log_surpass_max () = Helpers.getenv "BEAM_LOG_SURPASS_MAX" 0 <> 0
let beam_upcast_max () = Helpers.getenv "BEAM_UPCAST_MAX" 256
let beam_local_max () = Helpers.getenv "BEAM_LOCAL_MAX" 1024
let beam_uops_max () = Helpers.getenv "BEAM_UOPS_MAX" 3000
let beam_timeout_sec () = Helpers.getenv "BEAM_TIMEOUT_SEC" 10
let beam_strict_mode () = Helpers.getenv "BEAM_STRICT_MODE" 0 <> 0
let beam_dev_timeout () = Helpers.getenv "BEAM_DEV_TIMEOUT" 1 <> 0
(* [BEAM_PARALLEL] sets the number of domains compiling a beam step's
   candidates concurrently; 0 (the default) compiles sequentially. Only the
   CPU-side compile runs in parallel — the GPU timing phase below always runs
   one candidate at a time. *)
let beam_parallel = Helpers.Context_var.int ~key:"BEAM_PARALLEL" ~default:0
let cachelevel () = Helpers.getenv "CACHELEVEL" 1
let ignore_beam_cache () = Helpers.getenv "IGNORE_BEAM_CACHE" 0 <> 0

(* Minimum progress per beam step, in microseconds. *)
let beam_min_progress () =
  (match Sys.getenv_opt "BEAM_MIN_PROGRESS" with
   | Some s when s <> "" -> Float.of_string s
   | _ -> 0.01) /. 1e6

(* Actions *)

(* All candidate optimizations tried during beam search. *)
let actions =
  let open U.Opt in
  let acc = ref [] in
  let add opt = acc := opt :: !acc in
  let gen mk max_axis amounts =
    List.iter (fun amount ->
      for axis = 0 to max_axis do add (mk axis amount) done) amounts
  in
  List.iter (fun kind ->
      gen (fun axis amount -> Split { axis; amount; kind; top = false })
        9 [0; 2; 3; 4; 5; 7]) [ Axis_type.Upcast; Axis_type.Unroll ];
  gen (fun axis amount -> Split { axis; amount; kind = Axis_type.Local; top = false })
    7 [0; 2; 3; 4; 8; 13; 16; 29];
  gen (fun axis amount -> Split { axis; amount; kind = Axis_type.Local; top = true })
    7 [13; 16; 28; 29; 32; 49; 64; 256];
  if beam_padto then
    gen (fun axis amount -> Padto { axis; amount }) 6 [32];
  add (Split { axis = 0; amount = 32; kind = Axis_type.Local; top = false });
  add (Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = tc });
  for axis = 0 to 8 do
    add (Tc { axis; tc_select = -1; tc_opt; use_tc = tc })
  done;
  for axis_0 = 0 to 4 do
    for axis_1 = axis_0 + 1 to 4 do
      add (Swap { axis = axis_0; with_axis = axis_1 })
    done
  done;
  List.rev !acc

let is_tc = function U.Opt.Tc _ -> true | _ -> false

(* Action filtering *)

(* Skip actions that are equivalent to the zero-variant already in the list. *)
let is_noop a ax full_shape =
  ax < List.length full_shape
  && (match U.Opt.amount a, U.const_int_value (List.nth full_shape ax) with
      | Some amt, Some sz when sz = amt ->
          List.mem (U.Opt.with_amount a 0) actions
      | _ -> false)

(* Return valid actions for a scheduler state as (index, scheduler) pairs. *)
let get_kernel_actions ?(include_0 = true) ?max_up ~var_vals s =
  let max_up = Option.value max_up ~default:(beam_upcast_max ()) in
  let max_lcl = beam_local_max () in
  let dominated a =
    let axis = U.Opt.axis a in
    not (is_tc a) && (axis >= P.shape_len s || is_noop a axis (P.full_shape s))
  in
  let factor x =
    match U.const_int_value x with
    | Some sz -> sz
    | None -> U.sym_infer x var_vals
  in
  let upcast_and_local s2 =
    let up = ref 1 and lcl = ref 1 in
    List.iter2 (fun x t ->
      let sz = factor x in
      if t = Axis_type.Upcast || t = Axis_type.Unroll then
        up := !up * sz
      else if t = Axis_type.Warp || t = Axis_type.Local then
        lcl := !lcl * sz)
      (P.full_shape s2) (P.axis_types s2);
    let tc_up = match P.tensor_core s2 with
      | Some (tc : Tc.t) ->
          let m, n, k = tc.dims in m * n * k / tc.threads
      | None -> 1
    in
    (!up / tc_up, !lcl)
  in
  let acted = ref (if include_0 then [(0, s)] else []) in
  List.iteri (fun i a ->
    if not (dominated a) then
      let s2 = P.copy s in
      match P.apply_opt s2 a with
      | exception P.Opt_error _ -> ()
      | _ ->
          let up, lcl = upcast_and_local s2 in
          if up > max_up || lcl > max_lcl then begin
            if beam_log_surpass_max () then
              Printf.eprintf
                "too many upcast/local. up/tc_up=%d, max_up=%d, lcl=%d, \
                 max_lcl=%d\n%!"
                up max_up lcl max_lcl
          end else
            acted := (i + 1, s2) :: !acted)
    actions;
  List.rev !acted

(* Resolve symbolic global dims and shrink until they fit max_global_size by
   halving dims > 16 from the end. Returns (scaled_size, factor). *)
let get_test_global_size global_size var_vals max_global_size =
  let test = Array.map (fun sz -> U.sym_infer sz var_vals) global_size in
  let input_size = Array.fold_left ( * ) 1 test in
  let cont = ref true in
  while !cont && Array.fold_left ( * ) 1 test > max_global_size do
    cont := false;
    for j = Array.length test - 1 downto 0 do
      if not !cont && test.(j) > 16 then begin
        test.(j) <- test.(j) / 2;
        cont := true
      end
    done
  done;
  let scaled = Array.fold_left ( * ) 1 test in
  (test, Float.of_int input_size /. Float.of_int (max scaled 1))

(* Compilation *)

type compiled = { program : Program_spec.t; compile_time : float }

exception Compile_timeout

let with_compile_timeout ~use_timeout f =
  let prev =
    if use_timeout then
      let h =
        Sys.signal Sys.sigalrm
          (Sys.Signal_handle (fun _ -> raise Compile_timeout))
      in
      ignore (Unix.alarm (beam_timeout_sec ()));
      Some h
    else None
  in
  let cleanup () = match prev with
    | Some h -> ignore (Unix.alarm 0); Sys.set_signal Sys.sigalrm h
    | None -> ()
  in
  match f () with
  | v -> cleanup (); v
  | exception e -> cleanup (); raise e

(* Compile a single candidate: optimize -> lower -> check uop count -> compile.
   Returns (index, result) so callers can dispatch candidates in parallel and
   match results back. *)
let try_compile ~use_timeout ((idx, s) : int * P.t) (device : Device.t)
    : int * compiled option =
  let ren = P.ren s in
  let compile () =
    let st = Unix.gettimeofday () in
    let ast = P.get_optimized_ast ~name_override:"test" (P.copy s) in
    let ir = Linearizer.linearize (Codegen_lower.lower ren ast) in
    let uop_count = List.length ir in
    let beam_uops_max = beam_uops_max () in
    if beam_uops_max > 0 && uop_count >= beam_uops_max then begin
      if beam_log_surpass_max () then
        Printf.eprintf "too many uops. uop_count=%d, uops_max=%d\n%!"
          uop_count beam_uops_max;
      None
    end else
      let estimates = Program_spec.Estimates.of_program ir in
      let prog = Device.compile_program device ~name:"test" ~estimates ir in
      Some { program = prog; compile_time = Unix.gettimeofday () -. st }
  in
  let result =
    try with_compile_timeout ~use_timeout compile with
    | Compile_timeout ->
        if debug >= 2 then Printf.eprintf "*** BEAM COMPILE TIMEOUT\n%!";
        None
    | (Out_of_memory | Stack_overflow) as exn -> raise exn
    | Failure _ | Invalid_argument _ ->
        if debug >= 4 then
          Printf.eprintf "%s\n%!" (Printexc.get_backtrace ());
        None
    | _ when not (beam_strict_mode ()) -> None
  in
  (idx, result)

(* Compile a beam step's candidates, optionally across domains
   ([beam_parallel] sets the worker count; 0 is sequential). Workers only run
   the CPU-side compile (optimize, lower, render, nvrtc); the GPU timing phase
   runs afterwards in the main domain, one candidate at a time, so timings
   never contend for the device. In parallel mode the per-candidate alarm
   timeout is skipped: SIGALRM is process-global. *)
let compile_candidates ~device ~nworkers candidates =
  let n = List.length candidates in
  let compiled : compiled option array = Array.make n None in
  let compile_one ~use_timeout i cand =
    compiled.(i) <- snd (try_compile ~use_timeout (i, cand) device)
  in
  if nworkers <= 0 || n < 2 then begin
    List.iteri (compile_one ~use_timeout:true) candidates;
    compiled
  end else begin
    let nworkers = min nworkers (min 16 n) in
    let cands = Array.of_list candidates in
    let chunk = (n + nworkers - 1) / nworkers in
    let workers = ref [] and failure = ref None in
    let record_failure exn =
      match !failure with
      | None -> failure := Some (exn, Printexc.get_raw_backtrace ())
      | Some _ -> ()
    in
    (try
      for w = 0 to nworkers - 1 do
          let lo = w * chunk in
          let hi = min ((w + 1) * chunk) n in
          let worker = Domain.spawn (fun () ->
              for i = lo to hi - 1 do
                compiled.(i) <-
                  snd (try_compile ~use_timeout:false (i, cands.(i)) device)
              done) in
          workers := worker :: !workers
      done
    with exn -> record_failure exn);
    (* A failed spawn or join must not let another worker outlive the search
       scope and observe its caller's restored compilation context. *)
    List.iter (fun worker ->
        try Domain.join worker with exn -> record_failure exn)
      (List.rev !workers);
    match !failure with
    | None -> compiled
    | Some (exn, backtrace) -> Printexc.raise_with_backtrace exn backtrace
  end

(* Timing *)

type buffer_req = { slot : int; size : int; dtype : Dtype.t }

let buffer_reqs ast =
  let req_of_param u =
    match U.as_param u with
    | Some { param; _ } when param.slot >= 0 ->
        (match param.size with
         | Some size when size >= 0 ->
             Some { slot = param.slot; size; dtype = U.dtype u }
         | Some _ | None ->
             invalid_arg
               (Printf.sprintf
                  "beam_search: cannot allocate raw buffer for slot %d"
                  param.slot))
    | _ -> None
  in
  let sorted =
    P.bufs_from_ast ast
    |> List.filter_map req_of_param
    |> List.sort (fun a b -> Int.compare a.slot b.slot)
  in
  let rec dedup = function
    | [] -> []
    | r :: rest ->
        let same, rest =
          List.partition (fun r2 -> r2.slot = r.slot) rest
        in
        let size =
          List.fold_left (fun acc r2 -> max acc r2.size) r.size same
        in
        let dtype =
          List.fold_left
            (fun dtype r2 ->
              if Dtype.equal dtype r2.dtype then dtype
              else
                invalid_arg
                  (Printf.sprintf
                     "beam_search: conflicting dtypes for raw buffer slot %d"
                     r.slot))
            r.dtype same
        in
        { r with size; dtype } :: dedup rest
  in
  dedup sorted

let normalize_buffer_req device req buf =
  if Device.Buffer.size buf < req.size
     || not (Dtype.equal (Device.Buffer.dtype buf) req.dtype)
  then
    (* A beam timing buffer: bypass the LRU cache so a GC-collected
       replacement returns its memory to the driver instead of piling up
       in the allocator cache. *)
    Device.create_buffer ~size:req.size ~dtype:req.dtype
      ~spec:{ Device.Buffer_spec.default with nolru = true } device
  else buf

let indexed_rawbufs ~device ast rawbufs =
  let reqs = buffer_reqs ast in
  let raw_count = List.length rawbufs in
  let req_count = List.length reqs in
  let max_slot =
    List.fold_left (fun acc req -> max acc req.slot) (-1) reqs
  in
  let pair_compact () =
    try List.combine reqs rawbufs with
    | Invalid_argument _ ->
        invalid_arg
          (Printf.sprintf
             "beam_search: expected %d raw buffers, got %d"
             req_count raw_count)
  in
  let pairs =
    if raw_count = req_count then pair_compact ()
    else if raw_count > max_slot then
      List.map
        (fun req ->
          match List.nth_opt rawbufs req.slot with
          | Some buf -> (req, buf)
          | None ->
              invalid_arg
                (Printf.sprintf
                   "beam_search: raw buffer slot %d missing (%d buffers supplied)"
                   req.slot raw_count))
        reqs
    else pair_compact ()
  in
  List.map
    (fun (req, buf) -> (req.slot, normalize_buffer_req device req buf))
    pairs

(* Time a compiled program on device. Returns a list of timing samples. *)
let time_program ~device ~to_program p rawbufs_by_slot var_vals ~early_stop ~cnt ~clear_l2
    ~allow_test_size ~dev_timeout =
  let timeout =
    if dev_timeout && Float.is_finite early_stop then
      Some (Float.to_int (early_stop *. 1e3))
    else None
  in
  let factor = ref 1.0 in
  let p =
    if not allow_test_size then p
    else
      let scaled_global, f =
        get_test_global_size (Program_spec.global_size p) var_vals 65536
      in
      factor := f;
      Program_spec.with_global_dims scaled_global p
  in
  let info = Program_spec.program_info p in
  let args = List.init (List.fold_left max (-1) info.globals + 1) (fun slot ->
      match List.assoc_opt slot rawbufs_by_slot with
      | Some buf -> U.from_buffer buf
      | None when not (List.mem slot info.globals) -> U.noop ()
      | None -> invalid_arg (Printf.sprintf
          "beam_search: raw buffer slot %d missing (%d slots supplied)"
          slot (List.length rawbufs_by_slot))) in
  let kernel_info = U.{name = Program_spec.name p;
    applied_opts = Program_spec.applied_opts p; opts_to_apply = None;
    estimates = Some (Program_spec.Estimates.to_uop (Program_spec.estimates p)); beam = 0} in
  let program = U.program ~sink:(U.sink ~kernel_info (Program_spec.program p))
      ~linear:(U.linear (Program_spec.program p)) ~source:(U.source (Program_spec.src p))
      ~binary:(U.binary (Bytes.to_string (Option.get (Program_spec.lib p)))) ~info () in
  let call = U.call ~body:program ~args
      ~info:U.{grad_fxn = None; name = None; precompile = false;
        precompile_backward = false; dtype = Dtype.void; aux = None} in
  Realize.time_call ~device ~to_program ~var_vals ?timeout ~clear_l2 call
    (fun sample ->
      let tms = ref [] and stopped = ref false in
      for _ = 1 to cnt do
        if not !stopped then begin
          let tm = try sample () *. !factor with Assert_failure _ -> infinity in
          tms := tm :: !tms;
          if early_stop < List.fold_left min infinity !tms then stopped := true
        end
      done;
      List.rev !tms)

(* Beam search *)

let cache_key_of s amt allow_test_size ren =
  let ast_key = U.semantic_key (P.ast s) in
  let key =
    [
      ("ast", ast_key);
      ("amt", string_of_int amt);
      ("allow_test_size", string_of_bool allow_test_size);
      ("device", Renderer.device ren);
      ("target", Target.to_string (Renderer.target ren));
      ("suffix", Renderer.name ren);
    ]
  in
  String.concat "|"
    (List.map
       (fun (name, value) ->
         Printf.sprintf "%s:%d:%s" name (String.length value) value)
       key)

let apply_cached_opts s cached_opts =
  let ret = P.copy s in
  let skip = List.length (P.applied_opts s) in
  List.iteri
    (fun i opt -> if i >= skip then ignore (P.apply_opt ret opt))
    cached_opts;
  ret

let program_ops program var_vals =
  match (Program_spec.estimates program).ops with
  | Program_spec.Estimates.Int n -> Float.of_int n
  | Symbolic node -> Float.of_int (U.sym_infer node var_vals)

let beam_search ~to_program ?(allow_test_size = true) ?disable_cache
    (s : P.t) (rawbufs : Device.Buffer.t list) ~var_vals (amt : int)
    (device : Device.t) : P.t =
  List.iter (fun (_, name, lo, hi) ->
      match List.assoc_opt name var_vals with
      | None -> invalid_arg (Printf.sprintf "beam_search: missing variable %S" name)
      | Some value ->
          let value = Bound.int value in
          if Bound.lt value lo || Bound.lt hi value then
            invalid_arg (Printf.sprintf "beam_search: variable %S is outside its bounds" name))
    (U.symbolic_vars (P.ast s));
  let ren = P.ren s in
  let cache_key = cache_key_of s amt allow_test_size ren in
  let disable_cache =
    Option.value disable_cache ~default:(ignore_beam_cache ())
  in
  let cachelevel = cachelevel () in
  let cache_read_enabled = not disable_cache && cachelevel >= 1 in
  let cache_write_enabled = cachelevel >= 1 in
  let cached =
    if cache_read_enabled then
      (try Diskcache.get ~table:"beam_search" ~key:cache_key with _ -> None)
    else None
  in
  match cached with
  | Some cached_opts -> apply_cached_opts s cached_opts
  | None ->
      let beam = ref [(s, infinity)] in
      let seen_libs : (bytes, unit) Hashtbl.t = Hashtbl.create 256 in
      (* Compilation is reusable; eligibility is reconsidered each round.
         Only a binary accepted for timing enters [seen_libs]. *)
      let compiled_asts : compiled option U.Ref_tbl.t = U.Ref_tbl.create 256 in
      let nworkers = Helpers.Context_var.get beam_parallel in
      if beam_debug > 0 then
        Format.eprintf "BEAM_SEARCH:@\n%a@." U.pp (P.ast s);
      if debug >= 2 then
        Printf.eprintf
          "   0.00s:                from   1 ->   1 actions %s\n%!"
          (P.colored_shape s);
      let rawbufs_by_slot =
        indexed_rawbufs ~device (P.ast s) rawbufs
      in
      List.iter (fun (_, buf) -> Device.Buffer.ensure_allocated buf)
        rawbufs_by_slot;
      let st = Unix.gettimeofday () in
      let exiting = ref false in
      let time_one timed n_candidates i cand program compile_time =
        let early_stop = match !beam with
          | (_, best) :: _ -> best *. 3.0
          | [] -> 1.0
        in
        match
          time_program ~device ~to_program program rawbufs_by_slot var_vals ~early_stop
            ~cnt:3
            ~clear_l2:true ~allow_test_size
            ~dev_timeout:(beam_dev_timeout ())
        with
        | tms ->
            let best_tm = List.fold_left min infinity tms in
            timed := (cand, best_tm) :: !timed;
            if beam_debug > 1 then
              Printf.eprintf
                "%7.2fs: %5d %12e compile/%12e run      %4d/%4d   %s\n%!"
                (Unix.gettimeofday () -. st) i compile_time best_tm
                (List.length !timed) n_candidates (P.colored_shape cand)
            else if debug >= 2 then
              Printf.eprintf
                "\r%7.2fs: %12e      %4d/%4d         %s%!"
                (Unix.gettimeofday () -. st) best_tm (List.length !timed)
                n_candidates (P.colored_shape cand)
        | exception exn ->
            if beam_debug > 0 then
              Printf.eprintf "BEAM failed for opts: %s\n%s\n%!"
                (String.concat ", "
                   (List.map U.Opt.to_string (P.applied_opts cand)))
                (Printexc.to_string exn);
            (match exn with
             | Failure _ | Invalid_argument _ -> ()
             | _ -> raise exn)
      in
      let consume_one timed least_compute_ops n_candidates i cand compiled =
        match compiled with
        | None -> ()
        | Some { program; compile_time } ->
            let lib = match Program_spec.lib program with
              | Some l -> l
              | None -> assert false
            in
            if not (Hashtbl.mem seen_libs lib) then
              let this_ops = program_ops program var_vals in
              least_compute_ops := Float.min this_ops !least_compute_ops;
              if !least_compute_ops *. 1000.0 < this_ops then begin
                if beam_log_surpass_max () then
                  Printf.eprintf "too much compute. this=%e, least=%e\n%!"
                    this_ops !least_compute_ops
              end else begin
                Hashtbl.replace seen_libs lib ();
                time_one timed n_candidates i cand program compile_time
              end
      in
      while not !exiting do
        let candidates =
          List.concat_map
            (fun (si, _) ->
              List.map snd (get_kernel_actions ~include_0:false ~var_vals si))
            !beam
        in
        let pending = U.Ref_tbl.create (List.length candidates) in
        let uncompiled =
          List.filter
            (fun cand ->
              let ast = P.ast cand in
              if U.Ref_tbl.mem compiled_asts ast || U.Ref_tbl.mem pending ast then false
              else begin
                U.Ref_tbl.add pending ast ();
                true
              end)
            candidates
        in
        let timed = ref [] in
        let least_compute_ops = ref infinity in
        let n_candidates = List.length candidates in
        let compiled = compile_candidates ~device ~nworkers uncompiled in
        List.iteri (fun i cand -> U.Ref_tbl.add compiled_asts (P.ast cand) compiled.(i)) uncompiled;
        List.iteri
          (fun i cand ->
            consume_one timed least_compute_ops n_candidates i cand
              (U.Ref_tbl.find compiled_asts (P.ast cand)))
          candidates;
        let opts =
          List.sort (fun (_, t1) (_, t2) -> Float.compare t1 t2) !timed
        in
        let should_exit =
          match opts, !beam with
          | [], _ -> true
          | (_, t) :: _, _ when t < beam_min_progress () -> true
          | (_, ot) :: _, (_, bt) :: _ when bt -. ot < beam_min_progress () ->
              true
          | _ -> false
        in
        exiting := should_exit;
        if not should_exit then
          beam := List.filteri (fun i _ -> i < amt) opts
        else
          (match opts, !beam with
           | (s_best, t_best) :: _, (_, t_beam) :: _ when t_best < t_beam ->
               beam := [(s_best, t_best)]
           | _ -> ());
        if debug >= 2 then
          Printf.eprintf "\r%7.2fs: %12e from %3d -> %3d actions %s\n%!"
            (Unix.gettimeofday () -. st) (snd (List.hd !beam))
            n_candidates (List.length opts)
            (P.colored_shape (fst (List.hd !beam)))
      done;
      let result = fst (List.hd !beam) in
      if cache_write_enabled then
        Diskcache.put ~table:"beam_search" ~key:cache_key
          (P.applied_opts result);
      if beam_debug > 0 then
        Printf.eprintf "BEAM_SEARCH: final tm=%e, applied_opts=%s\n%!"
          (snd (List.hd !beam))
          (String.concat ", "
             (List.map U.Opt.to_string (P.applied_opts result)));
      result
