(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module K = Postrange.Scheduler

let setting = Setting.value
let debug () = setting Setting.debug
let upto n = List.init n Fun.id

(* The level of the search's tensor cores: [2] admits every kernel that [0] and
   [1] admit, and more, and the search measures what it admits. *)
let tc_opt_all = 2

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
      split Local [ 0; 2; 3; 4; 8; 13; 16; 29 ] (upto 8);
      split ~top:true Local [ 13; 16; 28; 29; 32; 49; 64; 256 ] (upto 8);
      (if setting Setting.beam_padto then
         List.map (fun axis -> Opt.Padto { axis; amount = 32 }) (upto 7)
       else []);
      split Local [ 32 ] [ 0 ];
      [ tc 0 0 ];
      (* covers resnet kernels (3 global * 3 reduce) *)
      List.map (tc tc_opt_all) (upto 9);
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

let get_test_global_size input max_global_size =
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

(* The program timed for [prg]: with [allow_test_size], one launching fewer
   workgroups, and the factor that scales its times up. *)
let test_size ~allow_test_size ~value prg =
  match arg prg with
  | Program info ->
      let infer s = sym_compile s (fun v () -> value v) () in
      let size = List.map infer info.global_size in
      let global_size, factor =
        if allow_test_size then get_test_global_size size 65536 else (size, 1.)
      in
      let global_size = List.map (fun n -> Int n) global_size in
      let local_size = List.map (fun s -> Int (infer s)) info.local_size in
      (replace prg ~arg:(Program { info with global_size; local_size }), factor)
  | _ -> (prg, 1.)

(* A candidate is timed through its call: storage keeps its slots, named by
   the target's device, which [link] places on a device of the target, each
   positional scalar is passed its sample, and each variable is passed itself,
   which [time] binds. *)
let timed_call ~value prg =
  let info = match arg prg with Program p -> p | _ -> assert false in
  let device = info.target.device in
  let arguments =
    List.filter_map
      (fun u ->
        match arg u with
        | Param p when op u = Op.Param ->
            if p.addrspace <> Dtype.Alu then
              Some (p.slot, replace u ~arg:(Param { p with device = Some device }))
            else if p.name = None then
              Some
                (p.slot, const ~dtype:(dtype u) (`Int (Bigint.of_int (value u))))
            else Some (p.slot, u)
        | _ -> None)
      (kernel_params prg)
  in
  let n = List.fold_left (fun n (slot, _) -> max n (slot + 1)) 0 arguments in
  let argument slot =
    match List.assoc_opt slot arguments with
    | Some u -> u
    | None ->
        param ~shape:[ Int 1 ] ~device slot Dtype.Uint8
  in
  call prg (List.init n argument)

(* [linked] timed up to [cnt] times, stopping once its least exceeds
   [early_stop]: the samples. *)
let time_program ~time ~early_stop ~factor ~vars ?(cnt = 3) linked =
  let rec go samples cnt =
    let samples = (time ~vars linked *. factor) :: samples in
    if cnt = 1 || early_stop < least samples then samples
    else go samples (cnt - 1)
  in
  go [] cnt

(* The least times of [a] and [b], each a linked program and the factor that
   scales its times up, sampled three times each in turn, [a] first, so that
   both run at the clocks of one moment. A sample whose time raises [Failure] is
   infinite. *)
let time_pair ~time ~vars (a, fa) (b, fb) =
  let sample linked factor =
    match time ~vars linked with
    | t -> t *. factor
    | exception Failure _ -> infinity
  in
  let rec go n ta tb =
    if n = 0 then (ta, tb)
    else
      let ta = Float.min ta (sample a fa) in
      let tb = Float.min tb (sample b fb) in
      go (n - 1) ta tb
  in
  go 3 infinity infinity

(* A kernel linearized, dropped past the cap, then compiled for [ren]: its
   program and compile time. *)
let try_compile (ren : Renderer.t) k =
  let st = Unix.gettimeofday () in
  match
    let ast = K.get_optimized_ast ~name_override:"test" (K.copy k) in
    let lin = Codegen.linearize ast ren in
    let uops = List.length (src (nth lin 1)) in
    let uops_max = setting Setting.beam_uops_max in
    if uops_max > 0 && uops >= uops_max then (
      if setting Setting.beam_log_surpass_max then
        Printf.printf "too many uops. len(uops)=%d, uops_max=%d\n%!" uops
          uops_max;
      None)
    else
      (* Bound before the clock is read: a tuple's components are evaluated
         right to left. *)
      let prg = Codegen.compile lin ren in
      Some (prg, Unix.gettimeofday () -. st)
  with
  | compiled -> compiled
  | exception (Sys.Break as e) -> raise e
  | exception (Failure _ as e) ->
      if debug () >= 4 then print_endline (Printexc.to_string e);
      None
  | exception e when setting Setting.beam_strict_mode -> raise e
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
      (List.find_map tc (Nodes.to_list (backward_slice ~calls:Skip (K.ast k))))
  in
  let up =
    let lo, hi = size [ Upcast ] in
    Bigint.(fdiv lo (of_int tc_up), fdiv hi (of_int tc_up))
  and lcl = size [ Warp; Local ] in
  let too_many = exceeds up max_up || exceeds lcl max_lcl in
  if too_many && setting Setting.beam_log_surpass_max then
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

let get_kernel_actions ?max_up k =
  let max_up = Option.value max_up ~default:(setting Setting.beam_upcast_max) in
  let max_lcl = setting Setting.beam_local_max and actions = actions () in
  let act a =
    if redundant actions k a then None
    else
      let k' = K.copy k in
      match K.apply_opt k' a with
      | Ok _ when not (too_many ~max_up ~max_lcl k') -> Some (a, k')
      | _ -> None
  in
  List.filter_map act actions

(* The cache keeps each optimisation as five integers: its kind, axis and
   arguments. *)
let opt_ints = function
  | Opt.Tc t -> [ 0; t.axis; t.tc_select; t.tc_opt; t.use_tc ]
  | Split s ->
      let target = match s.target with Upcast -> 0 | Local -> 2 in
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
    | 1 :: axis :: amount :: ((0 | 2) as target) :: top :: rest ->
        let target = if target = 0 then Opt.Upcast else Local in
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
  | _ ->
      invalid_arg (Format.asprintf "the variable %a has no integer bounds" pp v)

(* Lowering may change a parameter's dtype. Its binding is still identified by
   its name, or by its slot within this kernel. *)
let scalar_key v =
  let p = match arg v with Param p -> p | _ -> assert false in
  match p.name with Some name -> `Name name | None -> `Slot p.slot

let beam_search ~link ~time ?allow_test_size amt s =
  if amt < 1 then
    invalid_arg
      (Printf.sprintf "a beam search needs a positive width, not %d" amt);
  let allow_test_size =
    match allow_test_size with
    | Some allow -> allow
    | None -> setting Setting.beam_estimate
  in
  let beam_debug = setting Setting.beam_debug in
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
         Helpers.Target.key ren.target;
         Option.value (Renderer.Compiler.cachekey ren.compiler) ~default:"";
       ]
      @ List.map (fun (k, v) -> k ^ "=" ^ v) (Setting.shaping ()))
  in
  (* A damaged entry is searched again, as one never kept. *)
  let cached =
    if Setting.value Setting.ignore_beam_cache then None
    else
      try Helpers.Diskcache.get ~table:"beam_search" key
      with Failure _ | Sys_error _ -> None
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
      let values =
        List.map (fun v -> (scalar_key v, midpoint v)) (variables (K.ast s))
      in
      let value v = List.assoc (scalar_key v) values in
      let vars =
        List.filter_map
          (function `Name n, x -> Some (n, x) | `Slot _, _ -> None)
          values
      in
      let min_progress = setting Setting.beam_min_progress /. 1e6 in
      let seen_libs = Hashtbl.create 256 in
      (* Candidates compile through the compiler's disk cache. *)
      let ren =
        Renderer.with_compiler
          (Renderer.Compiler.v
             ~disassemble:(Renderer.Compiler.disassemble ren.compiler)
             (Renderer.Compiler.compile_cached ren.compiler))
          ren
      in
      (* [f i k] applied to each kernel [k] of [ks], the [i]th, and its
         compilation, in the order the compilations end, while other domains
         compile the rest. *)
      let compile ks f =
        Worker.iter
          (fun (_, k) -> try_compile ren k)
          (fun (i, k) r -> f i k r)
          (List.mapi (fun i k -> (i, k)) ks)
      in
      (* [k]'s program linked, with the factor that scales its times up, and its
         samples with an early stop, or [None] if its linking or timing
         failed. *)
      let sampled k prg ~early_stop =
        match
          let prg, factor = test_size ~allow_test_size ~value prg in
          let linked = link (timed_call ~value prg) in
          ((linked, factor), time_program ~time ~vars ~early_stop ~factor linked)
        with
        | r -> Some r
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
      if beam_debug > 0 then
        Format.printf "BEAM_SEARCH:@.%s@."
          (Render.render_uir
             (toposort ~calls:Enter (replace (K.ast s) ~arg:No_arg)));
      if debug () >= 2 then
        Printf.printf "   0.00s:                from   1 ->   1 actions %s\n%!"
          (K.colored_shape s);
      (* [first] is the beam's first kernel's linked program and factor, [None]
         while the beam is [s], untimed. *)
      let rec search beam first =
        let best = least (snd (List.hd beam)) in
        let candidates =
          List.concat_map
            (fun (k, _) -> List.map snd (get_kernel_actions k))
            beam
        in
        let n = List.length candidates in
        let timed = ref [] and least_compute_ops = ref infinity in
        (* The fastest candidate so far, as [opts] orders them: its least sample
           and its linked program, the only one kept past its samples. *)
        let fastest = ref None in
        let consider i cand compiled =
          match compiled with
          | Some (prg, compile_et) when not (Hashtbl.mem seen_libs (binary prg))
            ->
              let this_compute_ops =
                match arg (nth prg 0) with
                | Kernel { estimates = Some e; _ } ->
                    Bigint.to_float
                      (sym_compile_z e.ops (fun v () -> value v) ())
                | _ -> 0.
              in
              least_compute_ops := Float.min this_compute_ops !least_compute_ops;
              (* filter out kernels that use 1000x more compute than the
                 smallest *)
              if !least_compute_ops *. 1000. < this_compute_ops then (
                if setting Setting.beam_log_surpass_max then
                  Printf.printf "too much compute. %g when least is %g\n%!"
                    this_compute_ops !least_compute_ops)
              else (
                Hashtbl.add seen_libs (binary prg) ();
                match sampled cand prg ~early_stop:(best *. 3.) with
                | None -> ()
                | Some (linked, samples) ->
                    timed := (cand, samples) :: !timed;
                    (match !fastest with
                    | Some (t, _) when Float.compare t (least samples) <= 0 ->
                        ()
                    | _ -> fastest := Some (least samples, linked));
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
        compile candidates consider;
        let opts =
          List.stable_sort
            (fun (_, t0) (_, t1) -> Float.compare (least t0) (least t1))
            (List.rev !timed)
        in
        (* The beam's first and the fastest, timed again in turn: a clock that
           moved since the beam's first was timed, rounds ago, moves both. *)
        let compared =
          match (!fastest, first) with
          | None, _ -> None
          | Some (t, _), None -> Some (infinity, t)
          | Some (t, linked), Some first ->
              let again = time_pair ~time ~vars first linked in
              if beam_debug > 0 then
                Printf.printf
                  "BEAM_SEARCH: the beam's first %s, %s before, against the \
                   fastest %s, %s before\n\
                   %!"
                  (Helpers.time_to_str ~w:0 (fst again))
                  (Helpers.time_to_str ~w:0 best)
                  (Helpers.time_to_str ~w:0 (snd again))
                  (Helpers.time_to_str ~w:0 t);
              Some again
        in
        (* The search ends when nothing was timed, when the fastest took less
           than [min_progress], or when it beat the beam's first by less, or not
           at all, as two infinite times do; it then keeps the fastest alone if
           it beat the beam's first. *)
        let exiting, keep =
          match compared with
          | None -> (true, false)
          | Some (incumbent, t) ->
              let exiting =
                t < min_progress || not (incumbent -. t >= min_progress)
              in
              (exiting, exiting && t < incumbent)
        in
        let beam =
          match opts with
          | fastest :: _ when keep -> [ fastest ]
          | _ when exiting -> beam
          | _ -> List.filteri (fun i _ -> i < amt) opts
        in
        (if debug () >= 2 then
           let tm = Helpers.time_to_str ~w:12 (least (snd (List.hd beam))) in
           Printf.printf "\r%7.2fs: %s from %3d -> %3d actions\027[K %s\n%!"
             (elapsed ())
             (if exiting then Helpers.colored Green tm else tm)
             n (List.length opts)
             (K.colored_shape (fst (List.hd beam))));
        if exiting then beam else search beam (Option.map snd !fastest)
      in
      (* [s] starts the beam, untimed, of an infinite time. *)
      let k, samples = List.hd (search [ (s, [ infinity ]) ] None) in
      Helpers.Diskcache.put ~table:"beam_search" key
        (encode_opts (K.applied_opts k));
      if beam_debug > 0 then
        Format.printf "BEAM_SEARCH: final tm=%s, applied_opts=[%a]@."
          (Helpers.time_to_str ~w:0 (least samples))
          pp_opts (K.applied_opts k);
      k
