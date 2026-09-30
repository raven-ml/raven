(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
module K = Postrange.Scheduler

let setting = Helpers.Context_var.value
let debug () = setting Helpers.debug

let split ?(top = false) k axis amount target =
  K.apply_opt k (Opt.Split { axis; amount; target; top })

(* A split that may not apply. *)
let try_split k axis amount target =
  try ignore (split k axis amount target) with Opt.Kernel_opt_error _ -> ()

(* A size is known when it is an integer, which a constant beyond an int also
   is. *)
let known_size = function
  | Int n -> Some (Z.of_int n)
  | Sym u when op u = Op.Const -> (
      match value u with `Int n -> Some n | _ -> None)
  | Sym _ -> None

let divisible s n =
  match known_size s with
  | Some s -> Z.(equal (rem s (of_int n)) zero)
  | None -> false

let shape_at k i = List.nth (K.full_shape k) i
let prod_at k axes = Sint.prod (List.map (shape_at k) axes)
let holds = function Sint.Known b -> b | Sint.Cond u -> to_bool u
let size_at k i = Option.get (known_size (shape_at k i))

let first_dividing r sizes =
  List.find_opt
    (fun sz -> Option.is_some (divides (nth r 0) (Z.of_int sz)))
    sizes

let last l = List.nth l (List.length l - 1)

let index_of x l =
  let rec go i = function
    | [] -> invalid_arg "the node is not an axis of the kernel"
    | y :: _ when y == x -> i
    | _ :: r -> go (i + 1) r
  in
  go 0 l

let idx b = get_idx (nth b 1)
let indexes r b = Nodes.mem r (backward_slice (idx b))

(* first try the tensor cores *)
let tensor_cores k =
  let use_tc = setting Helpers.use_tc and tc_opt = setting Helpers.tc_opt in
  let tc_select = setting Helpers.tc_select in
  let min_globals = setting Helpers.tc_min_globals in
  if use_tc > 0 && (List.length (K.reduce_axes k) = 1 || tc_opt >= 1) then
    List.find_map
      (fun axis ->
        let tk = K.copy k in
        match K.apply_opt tk (Opt.Tc { axis; tc_select; tc_opt; use_tc }) with
        | exception Opt.Kernel_opt_error _ -> None
        | rngs ->
            let rngs = Array.of_list rngs in
            let split i size target =
              let axis = index_of rngs.(i) (K.rngs tk) in
              rngs.(i) <- List.hd (split tk axis size target)
            in
            let upcast i sizes target =
              Option.iter
                (fun sz -> split i sz target)
                (first_dividing rngs.(i) sizes)
            in
            if min_globals <> 0 then begin
              (* attempt to upcast M, local N, upcast N, skipping upcast N if
                 we'd end up with too few globals *)
              upcast 1 [ 5; 4; 3; 2 ] Opt.Upcast;
              upcast 0 [ 4; 2 ] Opt.Local;
              match first_dividing rngs.(0) [ 5; 4; 3; 2 ] with
              | Some size
                when Sint.resolve ~default:false
                       Sint.(
                         prod_at tk (K.axes_of tk [ Global ])
                         >= Int size * Int min_globals) ->
                  split 0 size Opt.Upcast
              | _ -> ()
            end
            else begin
              (* attempt to upcast M, N, local N *)
              upcast 1 [ 5; 4; 3; 2 ] Opt.Upcast;
              upcast 0 [ 5; 4; 3; 2 ] Opt.Upcast;
              upcast 0 [ 4; 2 ] Opt.Local
            end;
            Some tk)
      [ 0; 1; 2 ]
  else None

(* should use matvec - TODO: adjust/tune based on the wide vs tall/large vs
   small mat *)
let matvec k =
  let blocksize = Helpers.getenv "MV_BLOCKSIZE" 4
  and threads_per_row = Helpers.getenv "MV_THREADS_PER_ROW" 8
  and rows_per_thread = Helpers.getenv "MV_ROWS_PER_THREAD" 4 in
  let ren = K.ren k in
  let mulop =
    match K.reduceop k with
    | Some r
      when match arg r with Reduce { op = Op.Add; _ } -> true | _ -> false ->
        Some (nth r 0)
    | _ -> None
  in
  match mulop with
  | Some mulop
    when ren.has_local
         && Helpers.getenv "MV" 1 <> 0
         && (blocksize > 1 || threads_per_row > 1 || rows_per_thread > 1)
         && List.length (K.full_shape k) >= 2
         && ren.has_shared
         && op mulop = Op.Mul
         && op (nth mulop 0) = Op.Index
         && op (nth mulop 1) = Op.Index -> (
      let idx0 = idx (nth mulop 0) and idx1 = idx (nth mulop 1) in
      match K.ranges_of k [ Reduce ] with
      | first_reduce_rng :: _
        when List.exists (( == ) first_reduce_rng) (split_uop idx0 Op.Add)
             && List.for_all
                  (fun r -> Nodes.mem r (ranges idx1))
                  (Nodes.to_list (ranges idx0)) ->
          K.axes_of k [ Global ]
          |> List.find_map (fun global_idx ->
              if
                Option.is_some
                  (divides (nth first_reduce_rng 0) (Z.of_int threads_per_row))
                && divisible (shape_at k global_idx)
                     (blocksize * rows_per_thread)
              then begin
                if debug () >= 3 then
                  Format.eprintf
                    "MATVEC: full_shape=%a %s MV_BLOCKSIZE=%d \
                     MV_THREADS_PER_ROW=%d MV_ROWS_PER_THREAD=%d@."
                    (Format.pp_print_list Sint.pp)
                    (K.full_shape k)
                    (Render.render first_reduce_rng)
                    blocksize threads_per_row rows_per_thread;
                if threads_per_row > 1 then
                  try_split k
                    (List.hd (K.axes_of k [ Reduce ]))
                    threads_per_row Opt.Local;
                if blocksize > 1 then
                  ignore (split k global_idx blocksize Opt.Local);
                if rows_per_thread > 1 then
                  ignore (split k global_idx rows_per_thread Opt.Upcast);
                Some k
              end
              else None)
      | _ -> None)
  | _ -> None

(* are we grouping? (requires local shape support) *)
let group k =
  if
    Sint.resolve ~default:false
      Sint.(prod_at k (K.upcastable_dims k) <= Int 2048)
  then
    ignore
      (List.find_opt
         (fun axis ->
           match split ~top:true k axis 16 Opt.Local with
           | _ -> true
           | exception Opt.Kernel_opt_error _ -> false)
         (List.filteri (fun i _ -> i < 3) (K.axes_of k [ Reduce ])))

(* if there are small dims with lots of valid masks, upcast them (they might be
   from Tensor.stack) *)
let upcast_masked k =
  let where_gate_rngs =
    List.concat_map
      (fun u ->
        if op u = Op.Where then Nodes.to_list (ranges (nth u 0)) else [])
      (Nodes.to_list (backward_slice (K.ast k)))
  in
  (* upcast leading axes first (hack-ish for winograd; we actually want to
     upcast masked axes with low stride first) *)
  let to_upcast =
    List.fold_left
      (fun to_upcast axis ->
        let is_masked = List.memq (List.nth (K.rngs k) axis) where_gate_rngs in
        let upcast =
          List.fold_left (fun p a -> Z.mul p (size_at k a)) Z.one to_upcast
        in
        let n = size_at k axis in
        if Z.(leq n (of_int 7)) && is_masked && Z.(leq (upcast * n) (of_int 49))
        then begin
          if debug () >= 4 then
            Format.eprintf "upcasting masked axis : %d@." axis;
          to_upcast @ [ axis ]
        end
        else to_upcast)
      [] (K.upcastable_dims k)
  in
  List.iter
    (fun axis -> ignore (split k axis 0 Opt.Upcast))
    (List.rev to_upcast)

(* potentially do more upcasts of non reduce axes based on a heuristic *)
let upcast_more k =
  let rec loop upcasted_axis =
    if
      Sint.resolve Sint.(prod_at k (K.upcastable_dims k) >= Int 1024)
      && holds Sint.(K.upcast_size k < Int 32)
    then begin
      (* consider all upcastable axes with 3 or 4 upcast *)
      let amounts = [ 3; 4 ] in
      let choice axis upcast_amount =
        (* if we haven't upcasted it, it mods, and buffer has stride 0 on axis
           while having no stride 0 in the upcasted axis already *)
        let rng = List.nth (K.rngs k) axis in
        let upcast = K.ranges_of k [ Upcast; Unroll ] in
        let bufs = K.bufs k in
        if
          List.mem axis upcasted_axis
          || (not (divisible (shape_at k axis) upcast_amount))
          || not
               (List.exists
                  (fun b ->
                    (not (indexes rng b))
                    && List.for_all (fun r2 -> indexes r2 b) upcast)
                  bufs)
        then None
        else
          let stride c =
            let const_val c =
              match value c with
              | #Dtype.value as v -> Dtype.Value.to_int v
              | `Invalid -> 0
            in
            if c == rng then 1
            else if op c = Op.Mul && nth c 0 == rng && op (nth c 1) = Op.Const
            then const_val (nth c 1)
            else if op c = Op.Mul && nth c 1 == rng && op (nth c 0) = Op.Const
            then const_val (nth c 0)
            else 0
          in
          let num_strides = List.length (List.filter (indexes rng) bufs) in
          let sum_strides =
            List.fold_left
              (fun s b ->
                List.fold_left
                  (fun s c -> s + stride c)
                  s
                  (split_uop (idx b) Op.Add))
              0 bufs
          in
          Some (num_strides, sum_strides, axis, upcast_amount)
      in
      let xb_choices =
        List.concat_map
          (fun axis -> List.filter_map (choice axis) amounts)
          (K.upcastable_dims k)
      in
      match List.sort Stdlib.compare xb_choices with
      | (_, _, axis, amount) :: _ ->
          if debug () >= 4 then
            Format.eprintf "more upcast axis : %d by %d@." axis amount;
          ignore (split k axis amount Opt.Upcast);
          loop (axis :: upcasted_axis)
      | [] -> ()
    end
  in
  loop []

(* if last reduce dim is small(ish), loop unroll the reduce. NOTE: this can fail
   on multireduce with mismatching dimensions, this is okay *)
let unroll k =
  try
    let small n = holds Sint.(K.upcast_size k <= Int n) in
    if
      K.unrollable_dims k <> []
      && (small 4 || K.axes_of k [ Unroll ] = [])
      && holds Sint.(K.upcast_size k < Int 64)
    then
      let s = size_at k (last (K.unrollable_dims k)) in
      let at_most n x = Z.(leq x (of_int n)) in
      if at_most 32 s then begin
        ignore (split k (last (K.unrollable_dims k)) 0 Opt.Unroll);
        (* if it's small, upcast a second reduce dimension too *)
        match K.unrollable_dims k with
        | [] -> ()
        | dims ->
            if at_most 3 s && at_most 3 (size_at k (last dims)) then
              ignore (split k (last dims) 0 Opt.Unroll)
      end
      else
        let axis = last (K.unrollable_dims k) in
        if divisible (shape_at k axis) 4 then ignore (split k axis 4 Opt.Unroll)
  with Opt.Kernel_opt_error _ -> ()

(* if nothing at all is upcasted and it's easy to, do an upcast *)
let upcast_one k =
  match K.upcastable_dims k with
  | [] -> ()
  | dims ->
      let axis = last dims in
      if K.upcasted k = 0 && divisible (shape_at k axis) 4 then
        ignore (split k axis 4 Opt.Upcast)

(* prioritize making expand axes local *)
let locals k =
  let ranking =
    List.filter_map
      (fun axis ->
        let r = List.nth (K.rngs k) axis in
        if op (nth r 0) = Op.Const then
          Some (List.exists (fun b -> not (indexes r b)) (K.bufs k), axis)
        else None)
      (K.axes_of k [ Global; Weak ])
  in
  let order (e0, a0) (e1, a1) =
    match Bool.compare e1 e0 with 0 -> Int.compare a1 a0 | c -> c
  in
  let to_local =
    List.fold_left
      (fun to_local (_, axis) ->
        let local_size = Helpers.prod (List.map snd to_local) in
        let candidates =
          (if axis = 0 then [ 32 ] else []) @ [ 16; 8; 4; 3; 2 ]
        in
        match
          List.find_opt
            (fun x -> divisible (shape_at k axis) x && local_size * x <= 128)
            candidates
        with
        | Some sz -> to_local @ [ (axis, sz) ]
        | None -> to_local)
      []
      (List.stable_sort order ranking)
  in
  let first_three = List.filteri (fun i _ -> i < 3) to_local in
  ignore
    (List.fold_left
       (fun deleted_shape (axis, local_sz) ->
         let axis = axis - deleted_shape in
         let will_delete_shape =
           Option.equal Z.equal
             (known_size (shape_at k axis))
             (Some (Z.of_int local_sz))
         in
         ignore (split k axis local_sz Opt.Local);
         if will_delete_shape then deleted_shape + 1 else deleted_shape)
       0
       (List.sort Stdlib.compare first_three))

let hand_coded_optimizations k =
  match tensor_cores k with
  | Some tk -> tk
  | None -> (
      (* make a copy so it does not mutate the input *)
      let k = K.copy k in
      match matvec k with
      | Some k -> k
      | None ->
          group k;
          (* no more opt if we are grouping *)
          if K.group_for_reduces k = 0 then begin
            upcast_masked k;
            upcast_more k;
            unroll k;
            upcast_one k;
            if (K.ren k).has_local then locals k
          end;
          k)
