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
let nolocals = Helpers.getenv "NOLOCALS" 0 <> 0
let debug = Helpers.getenv "DEBUG" 0
let beam_debug = Helpers.getenv "BEAM_DEBUG" 0

let beam_log_surpass_max () = Helpers.getenv "BEAM_LOG_SURPASS_MAX" 0 <> 0
let beam_upcast_max () = Helpers.getenv "BEAM_UPCAST_MAX" 256
let beam_local_max () = Helpers.getenv "BEAM_LOCAL_MAX" 1024
let beam_uops_max () = Helpers.getenv "BEAM_UOPS_MAX" 3000
let beam_timeout_sec () = Helpers.getenv "BEAM_TIMEOUT_SEC" 10
let beam_strict_mode () = Helpers.getenv "BEAM_STRICT_MODE" 0 <> 0
let beam_dev_timeout () = Helpers.getenv "BEAM_DEV_TIMEOUT" 1 <> 0
let cachelevel () = Helpers.getenv "CACHELEVEL" 1
let ignore_beam_cache () = Helpers.getenv "IGNORE_BEAM_CACHE" 0 <> 0

let beam_min_progress () =
  (match Sys.getenv_opt "BEAM_MIN_PROGRESS" with
   | Some s -> (try Float.of_string s with Failure _ -> 0.01)
   | None -> 0.01) /. 1e6

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
  gen (fun axis amount -> Upcast { axis; amount }) 7 [0; 2; 3; 4; 5; 7];
  gen (fun axis amount -> Unroll { axis; amount }) 4 [0; 4; 7];
  gen (fun axis amount -> Local { axis; amount }) 5 [2; 3; 4; 8; 13; 16; 29];
  gen (fun axis amount -> Grouptop { axis; amount }) 2
    [13; 16; 28; 29; 32; 49; 64; 256];
  gen (fun axis amount -> Group { axis; amount }) 2 [0; 4; 8; 16];
  if beam_padto then
    gen (fun axis amount -> Padto { axis; amount }) 6 [32];
  add (Local { axis = 0; amount = 32 });
  add (Local { axis = 6; amount = 2 });
  add (Tc { axis = 0; tc_select = -1; tc_opt = 0; use_tc = tc });
  for axis = 0 to 8 do
    add (Tc { axis; tc_select = -1; tc_opt; use_tc = tc })
  done;
  for axis_0 = 0 to 4 do
    for axis_1 = axis_0 + 1 to 4 do
      add (Swap { axis = axis_0; with_axis = axis_1 })
    done
  done;
  gen (fun axis amount -> Thread { axis; amount }) 2
    [2; 3; 4; 5; 8; 12; 16; 24; 32; 64];
  if nolocals then add Nolocals;
  List.rev !acc

let is_tc = function U.Opt.Tc _ -> true | _ -> false

(* Symbolic evaluation and variable scraping *)

let symbolic_vars u =
  U.find_nodes
    (fun n ->
      match U.as_param n with
      | Some { param = { addrspace = Dtype.Alu; name = Some _;
                         vmin_vmax = Some _; _ }; _ } -> true
      | _ -> false)
    u
  |> List.filter_map (fun n ->
       match U.as_param n with
       | Some { param = { addrspace = Dtype.Alu; name = Some name;
                          vmin_vmax = Some (lo, hi); _ }; _ } ->
           Some (n, name, lo, hi)
       | _ -> None)

(* Substitute symbolic Param nodes by their var_vals entries and fold the result
   to an integer constant. *)
let sym_infer (u : U.t) (var_vals : (string * int) list) : int =
  let mappings =
    symbolic_vars u
    |> List.filter_map (fun (n, name, _, _) ->
         match List.assoc_opt name var_vals with
         | Some value -> Some (n, U.const_int value)
         | None -> None)
  in
  let folded = U.simplify (U.substitute mappings u) in
  match U.const_int_value folded with
  | Some n -> n
  | None -> invalid_arg "sym_infer: expression did not reduce to a constant"

(* Build name-keyed var_vals from symbolic Param nodes in the AST, using the
   midpoint of each variable's range. *)
let build_var_vals ast =
  symbolic_vars ast |> List.map (fun (_, name, lo, hi) -> (name, (lo + hi) / 2))

(* Action filtering *)

(* Skip actions that are equivalent to the zero-variant already in the list. *)
let is_noop a ax full_shape =
  ax < List.length full_shape
  && (match U.Opt.amount a, U.const_int_value (List.nth full_shape ax) with
      | Some amt, Some sz when sz = amt ->
          List.mem (U.Opt.with_amount a 0) actions
      | _ -> false)

(* Return valid actions for a scheduler state as (index, scheduler) pairs. *)
let get_kernel_actions ?(include_0 = true) ?max_up s =
  let max_up = Option.value max_up ~default:(beam_upcast_max ()) in
  let max_lcl = beam_local_max () in
  let var_vals = build_var_vals (P.ast s) in
  let dominated a =
    match U.Opt.axis a with
    | Some _ when not (is_tc a) ->
        (match P.real_axis s a (U.Opt.axis a) with
         | ax -> ax >= P.shape_len s || is_noop a ax (P.full_shape s)
         | exception P.Opt_error _ -> true)
    | _ -> false
  in
  let factor x =
    match U.const_int_value x with
    | Some sz -> sz
    | None ->
        (try sym_infer x var_vals with Invalid_argument _ -> U.vmax x)
  in
  let upcast_and_local s2 =
    let up = ref 1 and lcl = ref 1 in
    List.iter2 (fun x t ->
      let sz = factor x in
      if t = Axis_type.Upcast || t = Axis_type.Unroll then
        up := !up * sz
      else if t = Axis_type.Warp || t = Axis_type.Local
              || t = Axis_type.Group_reduce then
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
  let test = Array.map (fun sz -> sym_infer sz var_vals) global_size in
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

(* Timing *)

type buffer_req = { slot : int; size : int; dtype : Dtype.t }

let shape_arg_max_numel u =
  match U.src u with
  | [| shape; _ |] | [| shape |] when U.op shape <> Ops.Noop ->
      Some (List.fold_left ( * ) 1 (List.map U.vmax (U.as_shape shape)))
  | _ -> None

let buffer_reqs ast =
  let req_of_param u =
    match U.as_param u with
    | Some { param; _ } when param.slot >= 0 ->
        (match shape_arg_max_numel u with
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
  then Device.create_buffer ~size:req.size ~dtype:req.dtype device
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
let time_program ~device p rawbufs_by_slot var_vals ~early_stop ~cnt ~clear_l2
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
  let car = Realize.Compiled_runner.create ~device p in
  let input_bufs =
    List.map
      (fun slot ->
        match List.assoc_opt slot rawbufs_by_slot with
        | Some buf -> buf
        | None ->
            invalid_arg
              (Printf.sprintf
                 "beam_search: raw buffer slot %d missing (%d slots supplied)"
                 slot (List.length rawbufs_by_slot)))
      (Program_spec.globals p)
  in
  let tms = ref [] in
  let stopped = ref false in
  for _ = 1 to cnt do
    if not !stopped then begin
      if clear_l2 then Device.invalidate_caches device;
      let tm =
        try
          match
            Realize.Compiled_runner.call car input_bufs var_vals ~wait:true
              ~timeout
          with
          | Some t -> t *. !factor
          | None -> infinity
        with Assert_failure _ -> infinity
      in
      tms := tm :: !tms;
      if early_stop < List.fold_left min infinity !tms then stopped := true
    end
  done;
  List.rev !tms

(* Beam search *)

let cache_key_of s amt allow_test_size ren =
  let ast_key = U.semantic_key (P.ast s) in
  let key =
    [
      ("ast", ast_key);
      ("amt", string_of_int amt);
      ("allow_test_size", string_of_bool allow_test_size);
      ("device", Renderer.device ren);
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
  | Symbolic node -> Float.of_int (sym_infer node var_vals)

let beam_search ?(allow_test_size = true) ?disable_cache
    (s : P.t) (rawbufs : Device.Buffer.t list) (amt : int)
    (device : Device.t) : P.t =
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
      let var_vals = build_var_vals (P.ast s) in
      let st = Unix.gettimeofday () in
      let exiting = ref false in
      let time_one timed n_candidates i cand program compile_time =
        let early_stop = match !beam with
          | (_, best) :: _ -> best *. 3.0
          | [] -> 1.0
        in
        match
          time_program ~device program rawbufs_by_slot var_vals ~early_stop
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
      let step_one timed least_compute_ops n_candidates i cand =
        match try_compile ~use_timeout:true (i, cand) device with
        | _, None -> ()
        | _, Some { program; compile_time } ->
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
              List.map snd (get_kernel_actions ~include_0:false si))
            !beam
        in
        let timed = ref [] in
        let least_compute_ops = ref infinity in
        let n_candidates = List.length candidates in
        List.iteri
          (step_one timed least_compute_ops n_candidates)
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
