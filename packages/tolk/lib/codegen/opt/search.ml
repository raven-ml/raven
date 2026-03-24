(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Beam search kernel optimizer. *)

open Tolk_ir
module K = Kernel

(* Environment variables *)

let beam_upcast_max = Device.Context.int ~name:"BEAM_UPCAST_MAX" ~default:256
let beam_local_max = Device.Context.int ~name:"BEAM_LOCAL_MAX" ~default:1024
let beam_uops_max = Device.Context.int ~name:"BEAM_UOPS_MAX" ~default:3000
let beam_min_progress_raw =
  Device.Context.string ~name:"BEAM_MIN_PROGRESS" ~default:"0.01"
let beam_debug = Device.Context.int ~name:"BEAM_DEBUG" ~default:0
let beam_padto = Device.Context.int ~name:"BEAM_PADTO" ~default:0
let beam_timeout_sec = Device.Context.int ~name:"BEAM_TIMEOUT_SEC" ~default:10
let beam_dev_timeout = Device.Context.int ~name:"BEAM_DEV_TIMEOUT" ~default:1
let beam_strict_mode = Device.Context.int ~name:"BEAM_STRICT_MODE" ~default:0
let beam_log_surpass = Device.Context.int ~name:"BEAM_LOG_SURPASS_MAX" ~default:0
let nolocals = Device.Context.int ~name:"NOLOCALS" ~default:0
let tc_env = Device.Context.int ~name:"TC" ~default:1
let tc_opt_env = Device.Context.int ~name:"TC_OPT" ~default:2
let parallel_env = Device.Context.int ~name:"PARALLEL" ~default:0
let debug_env = Device.Context.int ~name:"DEBUG" ~default:0

(* Actions *)

let actions =
  let open K.Opt in
  let acc = ref [] in
  let add opt = acc := opt :: !acc in
  let gen mk max_axis amounts =
    List.iter (fun amount ->
      for axis = 0 to max_axis do add (mk axis amount) done) amounts
  in
  gen (fun axis amount -> Upcast { axis; amount }) 7
    [ 0; 2; 3; 4; 5; 7 ];
  gen (fun axis amount -> Unroll { axis; amount }) 4
    [ 0; 4; 7 ];
  gen (fun axis amount -> Local { axis; amount }) 5
    [ 2; 3; 4; 8; 13; 16; 29 ];
  gen (fun axis amount -> Grouptop { axis; amount }) 2
    [ 13; 16; 28; 29; 32; 49; 64; 256 ];
  gen (fun axis amount -> Group { axis; amount }) 2
    [ 0; 4; 8; 16 ];
  if Device.Context.get beam_padto <> 0 then
    gen (fun axis amount -> Padto { axis; amount }) 6 [ 32 ];
  add (Local { axis = 0; amount = 32 });
  add (Local { axis = 6; amount = 2 });
  let tc = Device.Context.get tc_env in
  let tc_opt = Device.Context.get tc_opt_env in
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
    [ 2; 3; 4; 5; 8; 12; 16; 24; 32; 64 ];
  if Device.Context.get nolocals <> 0 then add Nolocals;
  List.rev !acc

(* get_kernel_actions *)

let const_to_int_opt n =
  match K.const_to_int n with v -> Some v | exception _ -> None

(* Noop when shape equals amount and the zero-variant is in the action list. *)
let is_noop_action a ~resolved_axis full_shape actions_list =
  resolved_axis < List.length full_shape
  && (match a, const_to_int_opt (List.nth full_shape resolved_axis) with
      | K.Opt.Upcast { amount; axis }, Some sz when sz = amount ->
          List.mem (K.Opt.Upcast { axis; amount = 0 }) actions_list
      | Unroll { amount; axis }, Some sz when sz = amount ->
          List.mem (K.Opt.Unroll { axis; amount = 0 }) actions_list
      | Local { amount; axis }, Some sz when sz = amount ->
          List.mem (K.Opt.Local { axis; amount = 0 }) actions_list
      | Grouptop { amount; axis }, Some sz when sz = amount ->
          List.mem (K.Opt.Grouptop { axis; amount = 0 }) actions_list
      | Group { amount; axis }, Some sz when sz = amount ->
          List.mem (K.Opt.Group { axis; amount = 0 }) actions_list
      | Thread { amount; axis }, Some sz when sz = amount ->
          List.mem (K.Opt.Thread { axis; amount = 0 }) actions_list
      | Padto { amount; axis }, Some sz when sz = amount ->
          List.mem (K.Opt.Padto { axis; amount = 0 }) actions_list
      | _ -> false)

exception Dominated

let get_kernel_actions ?(include_0 = true) ?max_up s =
  let max_up = Option.value max_up
      ~default:(Device.Context.get beam_upcast_max) in
  let max_lcl = Device.Context.get beam_local_max in
  let acted = ref (if include_0 then [ (0, s) ] else []) in
  List.iteri
    (fun i a ->
      (try
        (match a with
         | K.Opt.Tc _ | Nolocals -> ()
         | _ ->
             let axis =
               match Postrange.real_axis s a with
               | ax -> ax
               | exception Postrange.Opt_error _ -> raise_notrace Dominated
             in
             if axis >= Postrange.shape_len s then raise_notrace Dominated;
             if is_noop_action a ~resolved_axis:axis
                 (Postrange.full_shape s) actions
             then raise_notrace Dominated);
        let s2 = Postrange.copy s in
        (match Postrange.apply_opt s2 a with
         | _ ->
             let up = ref 1 in
             let lcl = ref 1 in
             let tc_up =
               match Postrange.tensor_core s2 with
               | Some (tc : Renderer.tensor_core) ->
                   let m, n, k = tc.dims in
                   m * n * k / tc.threads
               | None -> 1
             in
             List.iter2
               (fun x t ->
                 match const_to_int_opt x with
                 | None -> ()
                 | Some sz ->
                     (match t with
                      | Axis_kind.Upcast | Unroll -> up := !up * sz
                      | Warp | Local | Group_reduce -> lcl := !lcl * sz
                      | _ -> ()))
               (Postrange.full_shape s2) (Postrange.axis_types s2);
             if !up / tc_up > max_up || !lcl > max_lcl then begin
               if Device.Context.get beam_log_surpass <> 0 then
                 Printf.eprintf
                   "too many upcast/local. up/tc_up=%d, max_up=%d, \
                    lcl=%d, max_lcl=%d\n%!"
                   (!up / tc_up) max_up !lcl max_lcl
             end
             else acted := (i + 1, s2) :: !acted
         | exception (Postrange.Opt_error _ | Invalid_argument _) -> ())
       with Dominated -> ()))
    actions;
  List.rev !acted

(* try_compile *)

type compiled = {
  program : Device.Program.t;
  binary : bytes;
  compile_time : float;
}

exception Compile_timeout

let try_compile ~use_timeout (s : Postrange.t) (device : Device.t)
    : compiled option =
  let ren = Postrange.ren s in
  let prev_handler =
    if use_timeout then begin
      let h = Sys.signal Sys.sigalrm
        (Sys.Signal_handle (fun _ -> raise Compile_timeout)) in
      ignore (Unix.alarm (Device.Context.get beam_timeout_sec));
      Some h
    end else None
  in
  let cleanup_timeout () = match prev_handler with
    | Some h -> ignore (Unix.alarm 0); Sys.set_signal Sys.sigalrm h
    | None -> ()
  in
  let result =
    try
      let ast =
        Postrange.get_optimized_ast ~name_override:"test" (Postrange.copy s)
      in
      let ir_program = Lowering.lower_and_linearize ren ast in
      let uops_max = Device.Context.get beam_uops_max in
      let uop_count = Tolk_ir.Program.length ir_program in
      if uops_max > 0 && uop_count >= uops_max then begin
        if Device.Context.get beam_log_surpass <> 0 then
          Printf.eprintf "too many uops. uop_count=%d, uops_max=%d\n%!"
            uop_count uops_max;
        None
      end
      else begin
        let estimates =
          match K.view ast with
          | Sink { kernel_info = Some ki; _ } when ki.estimates <> None ->
              Option.map Program_spec.Estimates.of_kernel ki.estimates
          | _ -> Some (Program_spec.Estimates.of_program ir_program)
        in
        let st = Unix.gettimeofday () in
        let dev_program =
          Device.compile_program device ?estimates ir_program
        in
        let compile_time = Unix.gettimeofday () -. st in
        Some { program = dev_program;
               binary = Device.Program.binary dev_program;
               compile_time }
      end
    with
    | Compile_timeout ->
        if Device.Context.get debug_env >= 2 then
          Printf.eprintf "*** BEAM COMPILE TIMEOUT\n%!";
        None
    | (Out_of_memory | Stack_overflow) as exn ->
        cleanup_timeout (); raise exn
    | Failure _ | Invalid_argument _ ->
        if Device.Context.get debug_env >= 4 then
          Printf.eprintf "%s\n%!" (Printexc.get_backtrace ());
        None
    | exn ->
        if Device.Context.get beam_strict_mode <> 0 then
          (cleanup_timeout (); raise exn)
        else None
  in
  cleanup_timeout ();
  result

(* get_test_global_size *)

let get_test_global_size global_size max_global_size =
  let test = Array.copy global_size in
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
  let scaled_size = Array.fold_left ( * ) 1 test in
  (test, Float.of_int input_size /. Float.of_int (max scaled_size 1))

(* time_program *)

let time_program program rawbufs var_vals device ~early_stop ~cnt
    ~clear_l2 ~allow_test_size ~dev_timeout =
  let timeout_ms =
    if dev_timeout && Float.is_finite early_stop then
      Some (max 1 (Float.to_int (early_stop *. 1e3)))
    else None
  in
  let queue = Device.queue device in
  let program, factor =
    if allow_test_size then
      let global, _local = Device.Program.launch_dims program var_vals in
      let scaled_global, factor = get_test_global_size global 65536 in
      (Device.Program.with_global_override scaled_global program, factor)
    else (program, 1.0)
  in
  let tms = ref [] in
  let stopped = ref false in
  for _ = 1 to cnt do
    if not !stopped then begin
      if clear_l2 then Device.invalidate_caches device;
      let tm =
        Device.Queue.timed_exec ?timeout_ms queue program rawbufs var_vals
      in
      tms := (tm *. factor) :: !tms;
      if early_stop < List.fold_left min infinity !tms then stopped := true
    end
  done;
  List.rev !tms

(* beam_search *)

let cachelevel = Device.Context.int ~name:"CACHELEVEL" ~default:1
let ignore_beam_cache = Device.Context.int ~name:"IGNORE_BEAM_CACHE" ~default:0

let beam_search ?(allow_test_size = true) ?(disable_cache = false)
    (s : Postrange.t) (rawbufs : Device.Buffer.t list) (amt : int)
    (device : Device.t) : Postrange.t =
  let ren = Postrange.ren s in
  let cache_key =
    let ast_key =
      Digest.to_hex
        (Digest.string (Marshal.to_string (Postrange.ast s) []))
    in
    Printf.sprintf "%s_%d_%b_%s_%s" ast_key amt allow_test_size
      (Renderer.device ren) (Renderer.name ren)
  in
  let cache_enabled =
    not disable_cache
    && Device.Context.get ignore_beam_cache = 0
    && Device.Context.get cachelevel >= 1
  in
  let cached =
    if cache_enabled then
      match
        (Diskcache.get ~table:"beam_search" ~key:cache_key
         : K.Opt.t list option)
      with
      | Some cached_opts ->
          let ret = Postrange.copy s in
          let skip = List.length (Postrange.applied_opts s) in
          List.iteri
            (fun i opt ->
              if i >= skip then ignore (Postrange.apply_opt ret opt))
            cached_opts;
          Some ret
      | None -> None
      | exception _ -> None
    else None
  in
  match cached with
  | Some ret -> ret
  | None ->
  let beam = ref [ (s, infinity) ] in
  let seen_libs : (bytes, unit) Hashtbl.t = Hashtbl.create 256 in
  let min_progress =
    (match Float.of_string_opt (Device.Context.get beam_min_progress_raw) with
     | Some v -> v
     | None -> 0.01)
    /. 1e6
  in
  let beam_dbg = Device.Context.get beam_debug in
  let dbg = Device.Context.get debug_env in
  List.iter Device.Buffer.ensure_allocated rawbufs;
  let var_vals =
    K.find_nodes
      (fun n -> match K.view n with Define_var _ -> true | _ -> false)
      (Postrange.ast s)
    |> List.map (fun n ->
        match K.view n with
        | Define_var { lo; hi; _ } -> (lo + hi) / 2
        | _ -> assert false)
  in
  if beam_dbg > 0 then
    Format.eprintf "BEAM_SEARCH:@\n%a@." K.pp (Postrange.ast s);
  if dbg >= 2 then
    Printf.eprintf "   0.00s:                from   1 ->   1 actions %s\n%!"
      (Postrange.colored_shape s);
  let st = Unix.gettimeofday () in
  let workers =
    match Device.Context.get_opt parallel_env with
    | Some n -> n
    | None ->
        if Renderer.has_local (Postrange.ren s) then
          Domain.recommended_domain_count () - 1
        else 0
  in
  let exiting = ref false in
  (try while not !exiting do
    let candidates =
      List.concat_map
        (fun (si, _) -> List.map snd (get_kernel_actions ~include_0:false si))
        !beam
    in
    let timed = ref [] in
    let least_compute_ops = ref infinity in
    let n_candidates = List.length candidates in
    let process_result i cand { program; binary; compile_time } =
      if not (Hashtbl.mem seen_libs binary) then begin
        let this_compute_ops =
          match (Device.Program.estimates program).ops with
          | Program_spec.Estimates.Int n -> Float.of_int n
          | Symbolic _ -> Float.infinity
        in
        least_compute_ops := Float.min this_compute_ops !least_compute_ops;
        if !least_compute_ops *. 1000.0 < this_compute_ops then begin
          if Device.Context.get beam_log_surpass <> 0 then
            Printf.eprintf "too much compute. this=%e, least=%e\n%!"
              this_compute_ops !least_compute_ops
        end
        else begin
          Hashtbl.replace seen_libs binary ();
          let early_stop = match !beam with
            | (_, best) :: _ -> best *. 3.0
            | [] -> 1.0
          in
          match
            time_program program rawbufs var_vals device ~early_stop
              ~cnt:3 ~clear_l2:true ~allow_test_size
              ~dev_timeout:(Device.Context.get beam_dev_timeout <> 0)
          with
          | tms ->
              let best_tm = List.fold_left min infinity tms in
              timed := (cand, best_tm) :: !timed;
              if beam_dbg > 1 then
                Printf.eprintf
                  "%7.2fs: %5d %12e compile/%12e run      %4d/%4d   %s\n%!"
                  (Unix.gettimeofday () -. st)
                  i compile_time best_tm (List.length !timed)
                  n_candidates (Postrange.colored_shape cand)
              else if dbg >= 2 then
                Printf.eprintf "\r%7.2fs: %12e      %4d/%4d         %s%!"
                  (Unix.gettimeofday () -. st)
                  best_tm (List.length !timed) n_candidates
                  (Postrange.colored_shape cand)
          | exception exn ->
              if beam_dbg > 0 then
                Printf.eprintf "BEAM failed for opts: %s\n%s\n%!"
                  (String.concat ", "
                     (List.map K.Opt.to_string
                        (Postrange.applied_opts cand)))
                  (Printexc.to_string exn);
              (* Only swallow Failure/Invalid_argument (~ RuntimeError). *)
              (match exn with
               | Failure _ | Invalid_argument _ -> ()
               | _ -> raise exn)
        end
      end
    in
    if workers <= 1 then
      List.iteri
        (fun i cand ->
          match try_compile ~use_timeout:true cand device with
          | None -> ()
          | Some c -> process_result i cand c)
        candidates
    else begin
      let candidates_arr = Array.of_list candidates in
      let n = Array.length candidates_arr in
      let next_idx = Atomic.make 0 in
      let mu = Mutex.create () in
      let cv = Condition.create () in
      let q = Queue.create () in
      let num_domains = min workers n in
      let domains =
        Array.init num_domains (fun _ ->
          Domain.spawn (fun () ->
            let rec loop () =
              let idx = Atomic.fetch_and_add next_idx 1 in
              if idx < n then begin
                let cand = candidates_arr.(idx) in
                let result = try_compile ~use_timeout:false cand device in
                Mutex.lock mu;
                Queue.push (idx, cand, result) q;
                Condition.signal cv;
                Mutex.unlock mu;
                loop ()
              end
            in
            loop ()))
      in
      let received = ref 0 in
      while !received < n do
        Mutex.lock mu;
        while Queue.is_empty q do Condition.wait cv mu done;
        let (idx, cand, compiled) = Queue.pop q in
        Mutex.unlock mu;
        incr received;
        match compiled with
        | None -> ()
        | Some c -> process_result idx cand c
      done;
      Array.iter Domain.join domains
    end;
    let opts =
      List.sort (fun (_, t1) (_, t2) -> Float.compare t1 t2) !timed
    in
    let should_exit = match (opts, !beam) with
      | [], _ -> true
      | (_, t) :: _, _ when t < min_progress -> true
      | (_, ot) :: _, (_, bt) :: _ when bt -. ot < min_progress -> true
      | _ -> false
    in
    exiting := should_exit;
    if not should_exit then
      beam := List.filteri (fun i _ -> i < amt) opts
    else begin
      match opts, !beam with
      | (s_best, t_best) :: _, (_, t_beam) :: _ when t_best < t_beam ->
          beam := [ (s_best, t_best) ]
      | _ -> ()
    end;
    if dbg >= 2 then
      Printf.eprintf "\r%7.2fs: %12e from %3d -> %3d actions %s\n%!"
        (Unix.gettimeofday () -. st)
        (snd (List.hd !beam))
        (List.length candidates) (List.length opts)
        (Postrange.colored_shape (fst (List.hd !beam)))
  done
   with Sys.Break -> raise Sys.Break);
  let result = fst (List.hd !beam) in
  if cache_enabled then
    Diskcache.put ~table:"beam_search" ~key:cache_key
      (Postrange.applied_opts result);
  if beam_dbg > 0 then
    Printf.eprintf "BEAM_SEARCH: final tm=%e, applied_opts=%s\n%!"
      (snd (List.hd !beam))
      (String.concat ", "
         (List.map K.Opt.to_string (Postrange.applied_opts result)));
  result
