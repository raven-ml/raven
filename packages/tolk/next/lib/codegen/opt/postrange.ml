(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
module V = Dtype.Value

let strf = Printf.sprintf

(* An optimisation that does not apply to a kernel refuses with a reason;
   [Scheduler.apply_opt] returns it. *)
exception Refused of string

let check cond msg = if not cond then raise_notrace (Refused msg)

let role : Opt.target -> Axis_type.t = function
  | Upcast -> Upcast
  | Unroll -> Unroll
  | Local -> Local

let split_targets : Opt.target -> Axis_type.t list = function
  | Upcast -> [ Global; Local; Weak ]
  | Unroll -> [ Reduce; Local ]
  | Local -> [ Global; Weak; Reduce ]

let pp_types ppf ts =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
    Axis_type.pp ppf ts

(* The size of a range, its greatest value plus one, exactly: it may exceed an
   int. *)
let size r = Z.succ (V.to_z (vmax r))
let z n = const (`Int n)
let sint_of_z n = if Z.fits_int n then Int (Z.to_int n) else Sym (z n)

(* A size is known when it is an integer, which a constant beyond an int also
   is. *)
let known_size = function
  | Int n -> Some (Z.of_int n)
  | Sym u when op u = Op.Const -> (
      match value u with `Int n -> Some n | _ -> None)
  | Sym _ -> None

let mem_ranges r u = Nodes.mem r (ranges u)
let sink_ranges us = Nodes.to_list (ranges (sink us))
let special_name u = match arg u with String s -> s | _ -> ""

module Scheduler = struct
  type t = {
    mutable ast : Ops.t;
    ren : Renderer.t;
    mutable applied_opts : Opt.t list;
    mutable next_range : int;
  }

  let v ast ren =
    let applied_opts =
      match arg ast with Kernel k -> k.applied_opts | _ -> []
    in
    let ids =
      List.filter_map
        (fun u -> if op u = Op.Range then Some (List.hd (axis_id u)) else None)
        (Nodes.to_list (backward_slice ast))
    in
    let last = match ids with [] -> 0 | i :: is -> List.fold_left max i is in
    { ast; ren; applied_opts; next_range = last + 1 }

  let copy k =
    let c = v k.ast k.ren in
    c.applied_opts <- k.applied_opts;
    c

  let ast k = k.ast
  let ren k = k.ren
  let applied_opts k = k.applied_opts
  let slice k = Nodes.to_list (backward_slice k.ast)
  let of_op k o = List.filter (fun u -> op u = o) (slice k)

  (* always in order by axistype. void RANGEs are loops, not opt axes. the
     DEVICE axis is launched, not an opt axis *)
  let rngs k =
    let axis u =
      op u = Op.Range
      && (not (Dtype.equal (dtype u) Dtype.Void))
      && V.(vmax u > of_int 0)
      && axis_type u <> Device
    in
    let key u = (Axis_type.position (axis_type u), axis_id u) in
    List.stable_sort
      (fun a b -> Stdlib.compare (key a) (key b))
      (List.filter axis (slice k))

  let shape_len k = List.length (rngs k)
  let full_shape k = List.map (fun r -> ssimplify (nth r 0)) (rngs k)
  let axis_types k = List.map axis_type (rngs k)

  let positions p l =
    List.concat (List.mapi (fun i x -> if p x then [ i ] else []) l)

  let ranges_of k types =
    List.filter (fun r -> List.mem (axis_type r) types) (rngs k)

  let axes_of k types = positions (fun t -> List.mem t types) (axis_types k)
  let reduceops k = of_op k Op.Reduce
  let reduceop k = match reduceops k with r :: _ -> Some r | [] -> None
  let bufs k = List.rev (of_op k Op.Index)

  let reduce_axes k =
    let red = List.concat_map (fun u -> List.tl (src u)) (reduceops k) in
    positions (fun r -> List.exists (mem_ranges r) red) (rngs k)

  let upcast_size k =
    let shape = full_shape k in
    Sint.prod (List.map (List.nth shape) (axes_of k [ Upcast; Unroll ]))

  let known_above_one shape i =
    match known_size (List.nth shape i) with
    | Some s -> Z.gt s Z.one
    | None -> false

  let upcastable_dims k =
    List.filter
      (known_above_one (full_shape k))
      (axes_of k [ Global; Local; Weak ])

  let unrollable_dims k =
    let types = axis_types k and shape = full_shape k in
    List.filter
      (fun i ->
        List.mem (List.nth types i) [ Axis_type.Local; Reduce ]
        && known_above_one shape i)
      (reduce_axes k)

  let upcasted k = List.length (axes_of k [ Upcast; Unroll ])

  let group_for_reduces k =
    let types = axis_types k in
    List.length
      (List.filter
         (fun i -> List.mem (List.nth types i) [ Axis_type.Warp; Local ])
         (reduce_axes k))

  (* Printing *)

  let output_rngs k =
    List.concat_map
      (fun s -> if op s = Op.End then sink_ranges (List.tl (src s)) else [])
      (src k.ast)

  (* exclude any output ranges from global that don't appear in all BUFFERIZE *)
  let globalizable_rngs k =
    let weak = List.filter (fun r -> axis_type r = Weak) (output_rngs k) in
    List.fold_left
      (fun ret x -> List.filter (fun r -> mem_ranges r x) ret)
      weak (of_op k Op.Stage)

  let colors k =
    let output = output_rngs k and globalizable = globalizable_rngs k in
    List.map
      (fun r ->
        match axis_type r with
        | Weak when not (List.memq r output) -> Helpers.Bright_black
        | Weak when not (List.memq r globalizable) -> Helpers.White
        | t -> Axis_type.color t)
      (rngs k)

  let colored_shape k =
    List.map2
      (fun r c -> Helpers.colored c (strf "%4s" (Render.render (nth r 0))))
      (rngs k) (colors k)
    |> String.concat " "

  (* Rewriting *)

  let convert_loop_to_global k =
    if k.ren.has_local then begin
      let globalizable = globalizable_rngs k in
      let global r =
        (r, replace r ~arg:(Range { axis_id = axis_id r; axis_type = Global }))
      in
      k.ast <-
        substitute k.ast
          (List.map global
             (List.filter (fun r -> List.memq r globalizable) (rngs k)))
    end

  let new_range k amount dtype t =
    let n = k.next_range in
    k.next_range <- n + 1;
    range ~axis_type:(role t) ~dtype (sint_of_z amount) [ n ]

  (* The size of [rng] divided by [amount], if [rng] can be split for
     [target]. *)
  let split_size k rng amount target =
    if not (List.mem (axis_type rng) (split_targets target)) then
      Error
        (Format.asprintf "a %a axis comes from a %a axis, not a %a one"
           Axis_type.pp (role target) pp_types (split_targets target)
           Axis_type.pp (axis_type rng))
    else
      match divides (nth rng 0) amount with
      | Some s -> Ok s
      | None ->
          Error
            (strf "%s does not divide %s in %s" (Z.to_string amount)
               (Render.render (nth rng 0))
               (colored_shape k))

  let shift_by ?(top = false) ?new_rng k rng amount target old_sz =
    let new_rng =
      match new_rng with
      | Some r -> r
      | None -> new_range k amount (dtype rng) target
    in
    let replaced_rng = replace rng ~src:[ old_sz ] in
    let sub_axis =
      if top then O.((new_rng * old_sz) + replaced_rng)
      else O.((replaced_rng * z amount) + new_rng)
    in
    k.ast <- substitute k.ast [ (rng, sub_axis) ];
    (replaced_rng, new_rng)

  let shift_to ?top ?new_rng k rng amount target =
    if
      (amount <= 1)
      [@mutate off "a split by 1 also fails, when its rewrite cycles"]
    then invalid_arg (strf "a split takes more than 1, not %d" amount);
    let amount = Z.of_int amount in
    match split_size k rng amount target with
    | Ok old_sz -> shift_by ?top ?new_rng k rng amount target old_sz
    | Error msg -> invalid_arg msg

  (* [shift_to], refusing where it cannot split. *)
  let split ?top ?new_rng k rng amount target =
    match split_size k rng amount target with
    | Ok old_sz -> shift_by ?top ?new_rng k rng amount target old_sz
    | Error msg -> raise_notrace (Refused msg)

  let rec apply ?(append_opt = true) k (opt : Opt.t) =
    let axis_rng axis =
      check
        (0 <= axis && axis < shape_len k)
        (strf "axis %d is not one of the kernel's %d axes" axis (shape_len k));
      List.nth (rngs k) axis
    in
    let ret =
      match opt with
      | Split { axis; amount; target; top } ->
          let rng = axis_rng axis in
          check
            (amount = 0 || amount > 1)
            (strf "a split takes 0 or more than 1, not %d" amount);
          if target = Local then check k.ren.has_local "locals needed for opt";
          let amt = if amount = 0 then size rng else Z.of_int amount in
          if target = Unroll then
            check Z.(leq amt (of_int 32)) "don't unroll more than 32";
          if target = Upcast then
            check Z.(leq amt (of_int 16)) "don't upcast more than 16";
          (* prevents METAL compiler hangs *)
          (match reduceop k with
          | Some r
            when (target = Local && List.mem axis (reduce_axes k))
                 || group_for_reduces k > 0 ->
              (* the sizes may exceed an int, so the product is a node *)
              let shape = full_shape k in
              let smem_sz =
                List.fold_left
                  (fun p a -> mul p (sint_to_uop (List.nth shape a)))
                  (z amt)
                  (axes_of k [ Upcast; Warp; Local ])
                |> fun p -> mul p (int (Dtype.itemsize (dtype r)))
              in
              check
                (to_bool O.(smem_sz <= int k.ren.shared_max))
                (strf "exceeds maximum shared memory size: needs %s, max %d"
                   (Render.render smem_sz) k.ren.shared_max)
          | _ -> ());
          if target = Unroll || axis_type rng = Reduce then begin
            let reduces =
              List.filter
                (fun u -> List.exists (mem_ranges rng) (List.tl (src u)))
                (reduceops k)
            in
            check (reduces <> [])
              (Format.asprintf "cannot %a an axis that's not in a REDUCE"
                 Axis_type.pp (role target));
            (* We currently dont support a group within another reduce *)
            if target = Local then
              check
                (not
                   (List.exists
                      (fun u ->
                        List.mem (axis_type u) [ Axis_type.Reduce; Unroll ])
                      (Nodes.to_list (ranges (List.hd reduces)))))
                "cannot have a group inside another reduce"
          end;
          let replaced, new_rng = split ~top k rng amt target in
          [ replaced; new_rng ]
      | Tc { axis; tc_select; tc_opt; use_tc } -> (
          check (k.applied_opts = []) "tensor core opts must be first";
          check (axis >= 0) "tensor core opts must have an axis";
          check
            (-1 <= tc_select && tc_select < List.length k.ren.tensor_cores)
            "tensor core opts must have valid tc_select";
          check
            (0 <= tc_opt && tc_opt <= 2)
            "tensor core opts must have valid tc_opt";
          check
            (0 < use_tc && use_tc <= 2)
            "use_tensor_cores value is not valid";
          match apply_tc_opt k use_tc axis tc_select tc_opt with
          | Some axes -> axes
          | None -> raise_notrace (Refused "no tensor core available"))
      | Padto { axis; amount } ->
          let rng = axis_rng axis in
          check (amount > 1) (strf "padto arg is a multiple > 1, not %d" amount);
          check (op (nth rng 0) = Op.Const) "only pad const axes";
          check
            (not (List.mem (axis_type rng) [ Axis_type.Upcast; Unroll; Warp ]))
            "cannot pad upcasted or warp";
          let amount = Z.of_int amount in
          let sz = size rng in
          let new_sz = Z.(cdiv sz amount * amount) in
          check
            Z.(gt sz (fdiv new_sz (of_int 4)))
            "pad adds more than quadruple the work";
          let replaced_rng =
            replace rng ~src:[ const_like (nth rng 0) (`Int new_sz) ]
          in
          let valid_ = O.(replaced_rng < z sz) in
          let pad_buf b =
            let i = nth b 1 in
            if mem_ranges rng i then
              [
                ( b,
                  replace b
                    ~src:
                      [ nth b 0; valid (get_idx i) O.(valid_ land get_valid i) ]
                );
              ]
            else []
          in
          let pad_reduce r =
            if List.exists (mem_ranges rng) (List.tl (src r)) then
              let op =
                match arg r with Reduce { op; _ } -> op | _ -> Op.Add
              in
              let zero =
                const ~dtype:(dtype r) (identity_element op (dtype r))
              in
              [
                ( r,
                  replace r ~src:(where valid_ (nth r 0) zero :: List.tl (src r))
                );
              ]
            else []
          in
          k.ast <-
            substitute k.ast
              (((rng, replaced_rng) :: List.concat_map pad_buf (bufs k))
              @ List.concat_map pad_reduce (reduceops k));
          [ replaced_rng ]
      | Swap { axis; with_axis } ->
          let rng = axis_rng axis in
          check
            (0 <= with_axis && with_axis < shape_len k)
            (strf "axis %d is not one of the kernel's %d axes" with_axis
               (shape_len k));
          let altrng = List.nth (rngs k) with_axis in
          check
            (axis_type rng = Global && axis_type altrng = Global)
            "swap only for globals";
          let renamed r other =
            replace r
              ~arg:(Range { axis_id = axis_id other; axis_type = axis_type r })
          in
          k.ast <-
            substitute ~walk:true k.ast
              [ (rng, renamed rng altrng); (altrng, renamed altrng rng) ];
          []
    in
    if append_opt then k.applied_opts <- k.applied_opts @ [ opt ];
    ret

  and apply_tc_opt k use_tc axis tc_select opt_level =
    let reduceop =
      match reduceops k with
      | r :: _ -> r
      | [] -> raise_notrace (Refused "no reduce ops for TensorCore")
    in
    let mul =
      let s = nth reduceop 0 in
      if op s = Op.Cast then nth s 0 else s
    in
    match arg reduceop with
    | Reduce { op = Op.Add; _ } when op mul = Op.Mul ->
        let cores =
          if tc_select = -1 then k.ren.tensor_cores
          else [ List.nth k.ren.tensor_cores tc_select ]
        in
        List.find_map (try_core k use_tc axis opt_level reduceop mul) cores
    | _ -> None

  and try_core k use_tc axis opt_level reduceop mul (tc : Tc.t) =
    let in0 = nth mul 0 and in1 = nth mul 1 in
    let device = k.ren.target.device in
    if
      (device = "CUDA" || device = "NV")
      && Dtype.equal tc.dtype_in Dtype.Float32
      && not (Helpers.Context_var.value Helpers.allow_tf32)
    then None
    else if
      not
        (Dtype.equal tc.dtype_in (dtype in0)
        && Dtype.equal tc.dtype_in (dtype in1)
        && Dtype.equal tc.dtype_out (dtype reduceop))
    then None
    else
      (* tensor cores have three ranges. X, Y, and REDUCE *)
      let by_id_desc rs =
        let first r = List.hd (axis_id r) in
        List.stable_sort (fun a b -> Int.compare (first b) (first a)) rs
      in
      let only a b =
        List.filter (fun u -> not (mem_ranges u b)) (Nodes.to_list (ranges a))
      in
      let in0_ranges = by_id_desc (only in0 in1)
      and in1_ranges = by_id_desc (only in1 in0)
      and red_ranges = by_id_desc (sink_ranges (List.tl (src reduceop))) in
      if Helpers.Context_var.value Helpers.debug >= 3 then begin
        let show rs =
          String.concat ", "
            (List.map
               (fun r ->
                 strf "(%d, %s)" (List.hd (axis_id r)) (Z.to_string (size r)))
               rs)
        in
        Format.eprintf "TC(%d): [%s] [%s] [%s]@." axis (show in0_ranges)
          (show in1_ranges) (show red_ranges)
      end;
      (* pick ranges. NOTE: in1 and in0 are switched because tc.dims is (N, M,
         K) *)
      let choices =
        List.concat_map
          (fun n ->
            List.concat_map
              (fun m -> List.map (fun r -> [| n; m; r |]) red_ranges)
              in0_ranges)
          in1_ranges
      in
      if axis >= List.length choices then None
      else
        let axes = List.nth choices axis in
        let rs = rngs k in
        check
          (not
             (List.exists
                (fun i ->
                  List.nth rs i == axes.(0) || List.nth rs i == axes.(1))
                (reduce_axes k)))
          "tensor core N and M can't be contracted";
        (* do optimizations and save the ranges *)
        let saved = k.ast in
        let warp = range ~axis_type:Warp (Int (Tc.threads tc)) [ -1 ] in
        let dims = match Tc.dims tc with n, m, d -> [| n; m; d |] in
        let shape () =
          Array.iteri
            (fun i a ->
              if not (Z.equal (Z.rem (size a) (Z.of_int dims.(i))) Z.zero) then begin
                if opt_level < 2 then
                  raise_notrace (Refused "tc padding requires opt_level >= 2");
                (* PADTO might fail *)
                let axis = index_of a (rngs k) in
                axes.(i) <-
                  List.hd
                    (apply ~append_opt:false k
                       (Padto { axis; amount = dims.(i) }))
              end)
            axes;
          (* we create the warp as a whole thing, in case some of these ranges
             are moved/removed later *)
          List.map
            (fun c ->
              let d = match c with Tc.N _ -> 0 | M _ -> 1 | K _ -> 2 in
              let replaced, r =
                match index_of_bit c tc.frag_c.lanes with
                | Some j ->
                    split k axes.(d) (Z.of_int 2) Local
                      ~new_rng:
                        (let p = 1 lsl j in
                         O.(warp // int p % int 2))
                | None ->
                    split k axes.(d) (Z.of_int 2)
                      (if d = 2 then Unroll else Upcast)
              in
              axes.(d) <- replaced;
              (c, r))
            (Tc.axis_coords tc)
        in
        match shape () with
        | exception Refused _ ->
            k.ast <- saved;
            None
        | ne ->
            if use_tc <> 2 then use_wmma k tc axes ne;
            Some (Array.to_list axes)

  and use_wmma k (tc : Tc.t) axes ne =
    let ne_of c = snd (List.find (fun (b, _) -> Tc.equal_bit b c) ne) in
    let reduceop =
      match
        List.filter
          (fun x -> List.memq axes.(2) (sink_ranges (List.tl (src x))))
          (reduceops k)
      with
      | [ r ] -> r
      | rs ->
          raise_notrace
            (Refused
               (strf "%d reductions run over the tensor core's K axis, not one"
                  (List.length rs)))
    in
    let r0 = nth reduceop 0 in
    let gate, mul =
      if op r0 = Op.Where then (Some (nth r0 0), nth r0 1) else (None, r0)
    in
    let mul = if op mul = Op.Cast then nth mul 0 else mul in
    let ins =
      match gate with
      | None -> src mul
      | Some g ->
          List.map
            (fun x -> where g x (const ~dtype:(dtype x) (`Int Z.zero)))
            (src mul)
    in
    let relabel_a, relabel_b = Tc.relabel tc in
    let srcs =
      List.map2
        (fun x rl ->
          substitute ~walk:true x
            (List.map (fun (a, b) -> (ne_of a, ne_of b)) rl))
        ins [ relabel_a; relabel_b ]
    in
    (* get upcast axes for the tensor cores *)
    let base_upcast_axes =
      List.map (fun c -> axis_id (ne_of c)) (Tc.base_upcast_axes tc)
    in
    let cnt (f : Tc.fragment) = List.length f.elements in
    let ca = cnt tc.frag_a and cb = cnt tc.frag_b and cc = cnt tc.frag_c in
    (* each operand upcasts its first upcast_cnt axes, the axes only A or B
       upcast are size 1 so the operands broadcast *)
    let upcast c =
      List.filteri (fun j _ -> j < max c (max ca cb)) base_upcast_axes
      |> List.mapi (fun j a -> (a, if j < c then 2 else 1))
    in
    let acc =
      consts ~dtype:tc.dtype_out (List.init (1 lsl cc) (fun _ -> `Float 0.))
    in
    let tc_uop =
      wmma
        ~upcast_axes:(upcast ca, upcast cb, upcast cc)
        (List.nth srcs 0) (List.nth srcs 1) ~acc ~dims:(Tc.dims tc)
        ~threads:(Tc.threads tc)
    in
    (* preserve extra reduces *)
    let k_ranges =
      List.filter_map (function Tc.K _, r -> Some r | _ -> None) ne
    in
    let reduce_ranges =
      List.filter
        (fun x -> op x = Op.Range && not (List.memq x k_ranges))
        (toposort (sink (List.tl (src reduceop))))
    in
    let tc_uop =
      match reduce_ranges with
      | [] -> tc_uop
      | rs ->
          Ops.v Op.Reduce ~src:(tc_uop :: rs)
            ~arg:(Reduce { op = Op.Add; num_axes = 0 })
    in
    k.ast <- substitute k.ast [ (reduceop, tc_uop) ]

  and index_of x l =
    let rec go i = function
      | [] -> invalid_arg "the node is not an axis of the kernel"
      | y :: _ when y == x -> i
      | _ :: r -> go (i + 1) r
    in
    go 0 l

  and index_of_bit b l =
    let rec go i = function
      | [] -> None
      | y :: _ when Tc.equal_bit y b -> Some i
      | _ :: r -> go (i + 1) r
    in
    go 0 l

  let apply_opt ?append_opt k opt =
    match apply ?append_opt k opt with
    | axes -> Ok axes
    | exception Refused msg -> Error msg

  let get_optimized_ast ?name_override k =
    let name =
      match name_override with
      | Some n -> n
      | None ->
          let k_type = if Option.is_some (reduceop k) then "r" else "E" in
          let special_uops =
            List.stable_sort
              (fun a b -> String.compare (special_name a) (special_name b))
              (of_op k Op.Special)
          in
          let special_ops =
            List.map
              (fun x ->
                let c =
                  if String.starts_with ~prefix:"g" (special_name x) then
                    Helpers.Blue
                  else Helpers.Cyan
                in
                Helpers.colored c (Z.to_string (size x)))
              special_uops
          in
          let axes =
            List.map2
              (fun x c -> Helpers.colored c (Render.render (nth x 0)))
              (rngs k) (colors k)
          in
          k_type
          ^ String.concat
              (Helpers.colored Helpers.Bright_black "_")
              (("" :: special_ops) @ axes)
    in
    k.ast <- graph_rewrite ~ctx:() k.ast Simplify.pm_flatten_range;
    replace k.ast
      ~arg:(Kernel (kernel_info ~name ~applied_opts:k.applied_opts ()))
      ~tag:(Some (Tag.Int 1))
end

let apply_opts ?beam ~hand_coded ast ren =
  if Option.is_some (tag ast) then ast
  else
    let info = match arg ast with Kernel i -> Some i | _ -> None in
    let k = Scheduler.v ast ren in
    Scheduler.convert_loop_to_global k;
    let k =
      match (info, beam) with
      | Some { opts_to_apply = Some opts; _ }, _ ->
          let apply o =
            match Scheduler.apply_opt k o with
            | Ok _ -> ()
            | Error msg -> invalid_arg msg
          in
          List.iter apply opts;
          k
      | _, Some search -> search k
      | _
        when (not (Helpers.Context_var.value Helpers.noopt))
             && match info with Some i -> i.applied_opts = [] | None -> true ->
          (* NOTE: hand_coded_optimizations doesn't support multiblock opts
             yet *)
          if op_in_backward_slice_with_self ast [ Op.Stage ] then k
          else hand_coded k
      | _ -> k
    in
    let name_override =
      match info with Some i when i.name <> "test" -> Some i.name | _ -> None
    in
    Scheduler.get_optimized_ast ?name_override k
