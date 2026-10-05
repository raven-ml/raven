(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
module K = Postrange.Scheduler

let setting = Setting.value
let debug () = setting Setting.debug

(* The settings that pick a search's candidates and how it measures and stops,
   which shape what it finds. A strict search raises where another drops a
   candidate, and finds what the other finds when it does not raise. *)
let padto = Setting.bool ~reach:Output "BEAM_PADTO" false
let uops_max = Setting.int ~reach:Output "BEAM_UOPS_MAX" 3000
let upcast_max = Setting.int ~reach:Output "BEAM_UPCAST_MAX" 256
let local_max = Setting.int ~reach:Output "BEAM_LOCAL_MAX" 1024
let min_progress = Setting.float ~reach:Output "BEAM_MIN_PROGRESS" 0.01
let estimate = Setting.bool ~reach:Output "BEAM_ESTIMATE" true
let strict_mode = Setting.bool ~reach:Process "BEAM_STRICT_MODE" false
let log_surpass_max = Setting.bool ~reach:Process "BEAM_LOG_SURPASS_MAX" false
let beam_debug = Setting.int ~reach:Process "BEAM_DEBUG" 0
let upto n = List.init n Fun.id

let actions () =
  let split ?(top = false) target amounts axes =
    List.concat_map
      (fun amount ->
        List.map (fun axis -> Opt.Split { axis; amount; target; top }) axes)
      amounts
  in
  let use_tc = setting Setting.use_tc in
  let tc tc_opt axis = Opt.Tc { axis; tc_select = -1; tc_opt; use_tc } in
  List.concat
    [
      split Upcast [ 0; 2; 3; 4; 5; 7 ] (upto 10);
      split Unroll [ 0; 2; 3; 4; 5; 7 ] (upto 10);
      split Local [ 0; 2; 3; 4; 8; 13; 16; 29 ] (upto 8);
      split ~top:true Local [ 13; 16; 28; 29; 32; 49; 64; 256 ] (upto 8);
      (if setting padto then
         List.map (fun axis -> Opt.Padto { axis; amount = 32 }) (upto 7)
       else []);
      split Local [ 32 ] [ 0 ];
      [ tc 0 0 ];
      (* covers resnet kernels (3 global * 3 reduce) *)
      List.map (tc (setting Setting.beam_tc_opt)) (upto 9);
      List.concat_map
        (fun axis ->
          List.map
            (fun with_axis -> Opt.Swap { axis; with_axis })
            (List.init (4 - axis) (fun i -> axis + 1 + i)))
        (upto 5);
    ]

(* Products in integers of any size, as Python's: the sizes of a launch, and the
   lanes and threads of a kernel, can multiply past an int. *)
let zprod l = List.fold_left (fun z n -> Bigint.(z * of_int n)) Bigint.one l

let get_test_global_size global_size max_global_size vars =
  let input = List.map (fun s -> sym_infer s vars) global_size in
  let rec halve_last_above_16 = function
    | [] -> []
    | n :: rest when n > 16 -> (n / 2) :: rest
    | n :: rest -> n :: halve_last_above_16 rest
  in
  let rec shrink size =
    if Bigint.leq (zprod size) (Bigint.of_int max_global_size) then size
    else shrink (List.rev (halve_last_above_16 (List.rev size)))
  in
  let size = shrink input in
  (size, Bigint.to_float (zprod input) /. Bigint.to_float (zprod size))

let least = List.fold_left Float.min infinity
let most = List.fold_left Float.max neg_infinity

(* Timed up to [cnt] times, stopping once its least exceeds [early_stop]: the
   samples. *)
let time_program ~time ~early_stop ~allow_test_size ~vars ?(cnt = 3) prg =
  let prg, factor =
    match arg prg with
    | Program info when allow_test_size ->
        let global_size, factor =
          get_test_global_size info.global_size 65536 vars
        in
        let global_size = List.map (fun n -> Int n) global_size in
        (replace prg ~arg:(Program { info with global_size }), factor)
    | _ -> (prg, 1.)
  in
  let sample = time prg in
  let rec go samples cnt =
    let samples = (sample () *. factor) :: samples in
    if cnt = 1 || early_stop < least samples then samples
    else go samples (cnt - 1)
  in
  go [] cnt

(* A kernel linearized, dropped past the cap, then compiled: its program and
   compile time. *)
let try_compile k =
  let st = Unix.gettimeofday () in
  let ren = K.ren k in
  let on_device p =
    match arg p with
    | Param a when op p = Op.Param && addrspace p <> Some Dtype.Alu ->
        let device = Single ren.target.device in
        [ (p, replace p ~arg:(Param { a with device = Some device })) ]
    | _ -> []
  in
  match
    let ast = K.get_optimized_ast ~name_override:"test" (K.copy k) in
    let lin =
      Codegen.linearize
        (substitute ast (List.concat_map on_device (toposort ast)))
        ren
    in
    let uops = List.length (src (nth lin 1)) in
    let uops_max = setting uops_max in
    if uops_max > 0 && uops >= uops_max then (
      if setting log_surpass_max then
        Printf.printf "too many uops. len(uops)=%d, uops_max=%d\n%!" uops
          uops_max;
      None)
    else Some (Codegen.to_program lin ren, Unix.gettimeofday () -. st)
  with
  | compiled -> compiled
  | exception (Sys.Break as e) -> raise e
  | exception (Failure _ as e) ->
      if debug () >= 4 then print_endline (Printexc.to_string e);
      None
  | exception e when setting strict_mode -> raise e
  | exception _ -> None

(* The least and greatest product of [sizes] over their variables' values. *)
let product_bounds sizes =
  let bound f = function
    | Int n -> Bigint.of_int n
    | Sym u -> (
        match f u with `Int z -> z | _ -> invalid_arg "a size is no integer")
  in
  List.fold_left
    (fun (lo, hi) s -> Bigint.(lo * bound vmin s, hi * bound vmax s))
    (Bigint.one, Bigint.one) sizes

(* Whether a product of bounds [(lo, hi)] exceeds [limit], which its variables'
   values must decide. *)
let exceeds (lo, hi) limit =
  if Bigint.gt lo (Bigint.of_int limit) then true
  else if Bigint.leq hi (Bigint.of_int limit) then false
  else invalid_arg "the lanes or threads of a candidate depend on a variable"

let too_many ~max_up ~max_lcl k =
  let shape = K.full_shape k in
  let size types =
    product_bounds (List.map (List.nth shape) (K.axes_of k types))
  in
  let tc_up =
    let tc u =
      match arg u with
      | Wmma { dims = n, m, depth; threads; _ } -> Some (n * m * depth / threads)
      | _ -> None
    in
    Option.value ~default:1
      (List.find_map tc (Nodes.to_list (backward_slice (K.ast k))))
  in
  let up =
    let lo, hi = size [ Upcast; Unroll ] in
    Bigint.(fdiv lo (of_int tc_up), fdiv hi (of_int tc_up))
  and lcl = size [ Warp; Local ] in
  let too_many = exceeds up max_up || exceeds lcl max_lcl in
  if too_many && setting log_surpass_max then
    Printf.printf
      "too many upcast/local. up//tc_up=%s, max_up=%d, lcl=%s, max_lcl=%d\n%!"
      (Bigint.to_string (snd up))
      max_up
      (Bigint.to_string (snd lcl))
      max_lcl;
  too_many

let redundant actions k = function
  | Opt.Tc _ -> false
  | a when Opt.axis a >= K.shape_len k -> true
  | Opt.Split s ->
      Sint.equal (List.nth (K.full_shape k) s.axis) (Int s.amount)
      && List.exists (Opt.equal (Opt.Split { s with amount = 0 })) actions
  | _ -> false

let get_kernel_actions ?(include_0 = true) ?max_up k =
  let max_up = Option.value max_up ~default:(setting upcast_max) in
  let max_lcl = setting local_max and actions = actions () in
  let act i a =
    if redundant actions k a then None
    else
      let k' = K.copy k in
      match K.apply_opt k' a with
      | Ok _ when not (too_many ~max_up ~max_lcl k') -> Some (i + 1, k')
      | _ -> None
  in
  (if include_0 then [ (0, k) ] else [])
  @ List.filter_map Fun.id (List.mapi act actions)

(* The cache keeps each optimisation as five integers: its kind, axis and
   arguments. *)
let opt_ints = function
  | Opt.Tc t -> [ 0; t.axis; t.tc_select; t.tc_opt; t.use_tc ]
  | Split s ->
      let target =
        match s.target with Upcast -> 0 | Unroll -> 1 | Local -> 2
      in
      [ 1; s.axis; s.amount; target; Bool.to_int s.top ]
  | Padto p -> [ 2; p.axis; p.amount; 0; 0 ]
  | Swap s -> [ 3; s.axis; s.with_axis; 0; 0 ]

let encode_opts opts =
  String.concat " " (List.map string_of_int (List.concat_map opt_ints opts))

let decode_opts s =
  let malformed () =
    failwith ("malformed optimisations in the beam cache: " ^ s)
  in
  let int w =
    match int_of_string_opt w with Some n -> n | None -> malformed ()
  in
  let rec opts = function
    | [] -> []
    | 0 :: axis :: tc_select :: tc_opt :: use_tc :: rest ->
        Opt.Tc { axis; tc_select; tc_opt; use_tc } :: opts rest
    | 1 :: axis :: amount :: target :: top :: rest
      when 0 <= target && target <= 2 ->
        let target = List.nth Opt.[ Upcast; Unroll; Local ] target in
        Opt.Split { axis; amount; target; top = top <> 0 } :: opts rest
    | 2 :: axis :: amount :: _ :: _ :: rest ->
        Opt.Padto { axis; amount } :: opts rest
    | 3 :: axis :: with_axis :: _ :: _ :: rest ->
        Opt.Swap { axis; with_axis } :: opts rest
    | _ -> malformed ()
  in
  if s = "" then [] else opts (List.map int (String.split_on_char ' ' s))

let pp_opts =
  Format.(pp_print_list ~pp_sep:(fun ppf () -> pp_print_string ppf ", ") Opt.pp)

let binary prg =
  match arg (nth prg 3) with
  | Bytes lib -> lib
  | _ -> invalid_arg "a program without its binary"

let midpoint v =
  match (vmin v, vmax v) with
  | `Int lo, `Int hi -> Bigint.(to_int (fdiv (lo + hi) (of_int 2)))
  | _ -> invalid_arg ("the variable " ^ expr v ^ " has no integer bounds")

let beam_search ~time ?allow_test_size amt s =
  if amt < 1 then
    invalid_arg
      (Printf.sprintf "a beam search needs a positive width, not %d" amt);
  let allow_test_size =
    match allow_test_size with Some allow -> allow | None -> setting estimate
  in
  let beam_debug = setting beam_debug in
  let ren = K.ren s in
  (* What a search's result is a function of, but the times it measures: the
     kernel, the search, the renderer, its compiler, what shapes compilation,
     among which the settings that pick the candidates, and this library's
     sources. *)
  let key =
    String.concat "\x00"
      ([
         Source_digest.digest;
         key (K.ast s);
         string_of_int amt;
         string_of_bool allow_test_size;
         ren.name;
         Format.asprintf "%a" Helpers.Target.pp ren.target;
         Option.value (Renderer.Compiler.cachekey ren.compiler) ~default:"";
       ]
      @ List.map (fun (k, v) -> k ^ "=" ^ v) (Setting.shaping ()))
  in
  let cached =
    if Setting.value Setting.ignore_beam_cache then None
    else Helpers.Diskcache.get ~table:"beam_search" key
  in
  match cached with
  | Some opts ->
      let ret = K.copy s in
      let apply i o =
        if i >= List.length (K.applied_opts s) then
          match K.apply_opt ret o with Ok _ -> () | Error msg -> failwith msg
      in
      List.iteri apply (decode_opts opts);
      ret
  | None ->
      let vars =
        List.map (fun v -> (expr v, midpoint v)) (variables (K.ast s))
      in
      let time = time ~vars (K.ast s) in
      let min_progress = setting min_progress /. 1e6 in
      let seen_libs = Hashtbl.create 256 in
      (* Each kernel is compiled once: two sequences of actions can reach equal
         kernels, whose programs [Codegen.to_program] keys apart by the
         optimisations they record. *)
      let compiled = Ops.Tbl.create 256 in
      let compile ks =
        let met = Ops.Tbl.create 64 in
        let fresh =
          List.filter
            (fun k ->
              let ast = K.ast k in
              let first =
                not (Ops.Tbl.mem compiled ast || Ops.Tbl.mem met ast)
              in
              if first then Ops.Tbl.add met ast ();
              first)
            ks
        in
        List.iter2
          (fun k r -> Ops.Tbl.replace compiled (K.ast k) r)
          fresh
          (Worker.map try_compile fresh);
        List.map (fun k -> (k, Ops.Tbl.find compiled (K.ast k))) ks
      in
      (* [k]'s program timed with an early stop, or [None] if its timing
         failed. *)
      let sampled k prg ~early_stop =
        match time_program ~time ~vars ~early_stop ~allow_test_size prg with
        | samples -> Some samples
        | exception e -> (
            let bt = Printexc.get_raw_backtrace () in
            if beam_debug > 0 then
              Format.printf "BEAM failed for opts: [%a]@.%s@." pp_opts
                (K.applied_opts k) (Printexc.to_string e);
            match e with
            | Failure _ -> None
            | e -> Printexc.raise_with_backtrace e bt)
      in
      let st = Unix.gettimeofday () in
      let elapsed () = Unix.gettimeofday () -. st in
      if beam_debug > 0 then Format.printf "BEAM_SEARCH:@.%a@." pp (K.ast s);
      if debug () >= 2 then
        Printf.printf "   0.00s:                from   1 ->   1 actions %s\n%!"
          (K.colored_shape s);
      (* A round progresses when each sample of its fastest beats each of the
         beam's first by more than [min_progress]: the least of noisy times is
         biased low, and a search never answers a kernel slower than [s]. *)
      let progresses (_, c) (_, b) = most c +. min_progress < least b in
      let rec search beam =
        let best = least (snd (List.hd beam)) in
        let candidates =
          List.concat_map
            (fun (k, _) -> List.map snd (get_kernel_actions ~include_0:false k))
            beam
        in
        let n = List.length candidates in
        let timed = ref [] and least_compute_ops = ref infinity in
        let consider i (cand, compiled) =
          match compiled with
          | Some (prg, compile_et) when not (Hashtbl.mem seen_libs (binary prg))
            ->
              let this_compute_ops =
                match arg (nth prg 0) with
                | Kernel { estimates = Some e; _ } ->
                    Float.of_int (sym_infer e.ops vars)
                | _ -> 0.
              in
              least_compute_ops := Float.min this_compute_ops !least_compute_ops;
              (* filter out kernels that use 1000x more compute than the
                 smallest *)
              if !least_compute_ops *. 1000. < this_compute_ops then (
                if setting log_surpass_max then
                  Printf.printf "too much compute. %g when least is %g\n%!"
                    this_compute_ops !least_compute_ops)
              else (
                Hashtbl.add seen_libs (binary prg) ();
                match sampled cand prg ~early_stop:(best *. 3.) with
                | None -> ()
                | Some samples ->
                    timed := (cand, samples) :: !timed;
                    let tm = Helpers.time_to_str ~w:12 (least samples) in
                    let progress = List.length !timed in
                    if beam_debug > 1 then
                      Printf.printf
                        "%7.2fs: %5d %5d uops %s compile/%s run       \
                         %4d/%4d         %s\n\
                         %!"
                        (elapsed ()) i
                        (List.length (src (nth prg 1)))
                        (Helpers.time_to_str ~w:12 compile_et)
                        tm progress n (K.colored_shape cand)
                    else if debug () >= 2 then
                      Printf.printf
                        "\r%7.2fs: %s       %4d/%4d         %s\027[K%!"
                        (elapsed ()) tm progress n (K.colored_shape cand))
          | _ -> ()
        in
        List.iteri consider (compile candidates);
        let opts =
          List.stable_sort
            (fun (_, t0) (_, t1) -> Float.compare (least t0) (least t1))
            (List.rev !timed)
        in
        let exiting =
          match opts with
          | fastest :: _ -> not (progresses fastest (List.hd beam))
          | [] -> true
        in
        let beam =
          if exiting then beam else List.filteri (fun i _ -> i < amt) opts
        in
        (if debug () >= 2 then
           let tm = Helpers.time_to_str ~w:12 (least (snd (List.hd beam))) in
           Printf.printf "\r%7.2fs: %s from %3d -> %3d actions\027[K %s\n%!"
             (elapsed ())
             (if exiting then Helpers.colored Green tm else tm)
             n (List.length opts)
             (K.colored_shape (fst (List.hd beam))));
        if exiting then beam else search beam
      in
      (* [s] itself, timed with no early stop, starts the beam. *)
      let start =
        match compile [ s ] with
        | [ (_, Some (prg, _)) ] ->
            Hashtbl.add seen_libs (binary prg) ();
            Option.value ~default:[] (sampled s prg ~early_stop:infinity)
        | _ -> []
      in
      let k, samples = List.hd (search [ (s, start) ]) in
      Helpers.Diskcache.put ~table:"beam_search" key
        (encode_opts (K.applied_opts k));
      if beam_debug > 0 then
        Format.printf "BEAM_SEARCH: final tm=%s, applied_opts=[%a]@."
          (Helpers.time_to_str ~w:0 (least samples))
          pp_opts (K.applied_opts k);
      k
