(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let rule_ctx = Pattern_matcher.rule_ctx
let ops l = Op.Set.of_list l
let sint = sint_to_uop
let zero = `Int Z.zero

(* Indexing context *)

type ctx = {
  realize_map : int list option Tbl.t;
      (* The nodes stored whole: marked, then the axes given new ranges. *)
  non_removable : unit Tbl.t;
  stored_through : unit Tbl.t;
      (* The pads a store's destination moves through: its writes outside their
         sources are dropped. *)
  range_map : (t list * t list) Tbl.t;
      (* Each node's ranges: those that index its sources, then its output. *)
  mutable range_idx : int;
}

(* A range of size 1 only ever takes the value 0. *)
let new_range ?(axis_type = Axis_type.Weak) ctx (s : sint) =
  match s with
  | Sym r when op r = Op.Range -> r
  | _ when Sint.(resolve (s <> Int 1)) ->
      let r = range ~axis_type s [ ctx.range_idx ] in
      ctx.range_idx <- ctx.range_idx + 1;
      r
  | _ -> int 0

let new_ranges ?axis_type ctx shape =
  List.rev
    (List.fold_left (fun acc s -> new_range ?axis_type ctx s :: acc) [] shape)

let always_contiguous =
  ops Op.[ After; Buffer; Alloc; Const; Mselect; Mstack; Param; Load; Call ]

let realize ctx u = Tbl.replace ctx.realize_map u None

let realize_srcs ctx rb =
  List.iter
    (fun s ->
      if not (Op.Set.mem (op (base s)) always_contiguous) then realize ctx s)
    (src rb)

(* An assign needs this only for a write-after-read hazard, the destination read
   by the value stored into it. *)
let realize_store_after_src ctx dest s =
  if List.memq (base dest) (toposort ~enter_calls:false s) then realize ctx s

let mark_stored_pads ctx dest =
  let rec go u =
    if Op.Set.mem (op u) Op.Set.movement then begin
      if op u = Op.Pad then Tbl.replace ctx.stored_through u ();
      go (nth u 0)
    end
  in
  go dest

(* A store's destination that moves through a pad is made its own: each of its
   movements is tagged, so that no read shares the nodes that D69 gates. *)
let stored_tag = Tag.String "stored"

let own_destination st =
  let rec pads u =
    Op.Set.mem (op u) Op.Set.movement && (op u = Op.Pad || pads (nth u 0))
  in
  let rec own u =
    if not (Op.Set.mem (op u) Op.Set.movement) then u
    else
      replace u ~src:(own (nth u 0) :: List.tl (src u)) ~tag:(Some stored_tag)
  in
  match src st with
  | dest :: rest when pads dest && tag dest = None ->
      Some (replace st ~src:(own dest :: rest))
  | _ -> None

let pm_own_stored_destinations =
  Pattern_matcher.v (fun () ->
      [
        rule_ctx (Upat.op Op.Store ~name:"st") (fun () m ->
            own_destination (m "st"));
      ])

let realize_custom_kernel_srcs ctx c =
  let rec strip s = if op s = Op.Reshape then strip (nth s 0) else s in
  List.iter
    (fun s ->
      let s = strip s in
      if not (Op.Set.mem (op s) always_contiguous) then begin
        realize ctx s;
        Tbl.replace ctx.non_removable s ()
      end)
    (List.tl (src c))

let pm_generate_realize_map =
  let mark f ctx m =
    f ctx m;
    None
  in
  Pattern_matcher.v (fun () ->
      [
        rule_ctx
          (Upat.op Op.Call ~name:"c" ~allow_any_len:true
             ~src:[ Upat.v ~op:(ops [ Op.Sink; Op.Program ]) () ])
          (mark (fun ctx m -> realize_custom_kernel_srcs ctx (m "c")));
        rule_ctx
          (Upat.op Op.Store ~name:"tr")
          (mark (fun ctx m -> realize ctx (m "tr")));
        rule_ctx
          (Upat.v ~op:(ops [ Op.Mselect; Op.Mstack ]) ~name:"rb" ())
          (mark (fun ctx m -> realize_srcs ctx (m "rb")));
        rule_ctx
          (Upat.op Op.Store ~src:[ Upat.var "dest"; Upat.var "src" ])
          (mark (fun ctx m ->
               realize_store_after_src ctx (m "dest") (m "src");
               mark_stored_pads ctx (m "dest")));
      ])

(* Applying ranges *)

(* The ranges of [x] as its source [s] sees them: without the axes [s] lacks,
   and [0] on the axes [x] broadcasts it over. *)
let broadcast_rngs x s rngs =
  if not (Op.Set.mem (op x) Op.Set.broadcastable) then rngs
  else
    let baxes = broadcast_axes (shape s) (shape x)
    and nleft = ndim x - ndim s in
    List.concat
      (List.mapi
         (fun j r ->
           if j < nleft then []
           else if List.mem j baxes then [ const_like r zero ]
           else [ r ])
         rngs)

(* The sources that hold data, as opposed to shapes, bounds and ranges. *)
let data_srcs op src =
  let first = match src with s :: _ -> [ s ] | [] -> [] in
  if Op.Set.mem op (ops Op.[ Param; Buffer; Alloc; Range; Special ]) then []
  else if
    Op.Set.mem op
      (Op.Set.union Op.Set.movement
         (ops Op.[ Index; Stage; Reduce; After; End; Backedge; Copy ]))
  then first
  else src

let storage = ops Op.[ Param; Buffer; Alloc; Mstack; Mselect; After ]

let create_bufferize_and_index_srcs ctx x =
  let data_src_count = List.length (data_srcs (op x) (src x)) in
  let rngs = Option.map fst (Tbl.find_opt ctx.range_map x) in
  List.mapi
    (fun i s ->
      let src_rngs =
        match rngs with Some r -> broadcast_rngs x s r | None -> []
      in
      if Op.Set.mem (op s) storage then
        if Option.is_some rngs && i < data_src_count then index s src_rngs
        else s
      else
        match Tbl.find_opt ctx.realize_map s with
        | None -> s
        | Some None -> invalid_arg "the realize map holds no ranges"
        | Some (Some _) when op s = Op.Store ->
            let closed = snd (Tbl.find ctx.range_map s) in
            Tbl.remove ctx.realize_map s;
            end_ s (List.filter (fun r -> op r = Op.Range) closed)
        | Some (Some _) ->
            let closed = snd (Tbl.find ctx.range_map s) in
            let removable =
              (not (Op.Set.mem (op s) always_contiguous))
              && not (Tbl.mem ctx.non_removable s)
            in
            let opts : bufferize_opts =
              { device = device s; addrspace = Dtype.Global; removable }
            in
            let staged = bufferize ~opts s closed in
            if Option.is_some rngs then index staged src_rngs else staged)
    (src x)

let create_bufferize_and_index_based_on_ranges ctx x =
  if op x = Op.Stage || op x = Op.Index then None
  else Some (replace x ~src:(create_bufferize_and_index_srcs ctx x))

(* The first source is taken from the list: rebuilding [x] on an indexed source
   would give it a shape its argument does not fit. *)
let convert_pad_to_where_to_keep_behavior_local ctx x =
  match Tbl.find_opt ctx.range_map x with
  | None -> None
  | Some (rngs, _) when Tbl.mem ctx.stored_through x ->
      (* A store through the pad writes only where its index falls within the
         source. The movements below would simplify that validity away (a
         reshape flattens the index), so the index into the storage carries it,
         and the pad is removed as any movement is. *)
      let valid = uprod (bool true) (List.map get_valid rngs) in
      let rec lowest u =
        let s = nth u 0 in
        if Op.Set.mem (op s) Op.Set.movement then lowest s else u
      in
      let m = lowest x in
      let ins, outs = Tbl.find ctx.range_map m in
      Tbl.replace ctx.range_map m
        (List.map (fun i -> Ops.valid (get_idx i) valid) ins, outs);
      None
  | Some (rngs, _) ->
      let valid = uprod (bool true) (List.map get_valid rngs) in
      let s = List.hd (create_bufferize_and_index_srcs ctx x) in
      Some (where valid s (const (Dtype.const (dtype x) zero)))

let convert_reduce_to_reduce_with_ranges ctx x =
  match arg x with
  | Reduce { op = rop; num_axes } when num_axes <> 0 -> (
      match Tbl.find_opt ctx.range_map x with
      | None -> invalid_arg "a reduction of leading axes has no ranges"
      | Some (rngs, _) ->
          let s = List.hd (create_bufferize_and_index_srcs ctx x) in
          Some
            (v Op.Reduce
               ~src:(s :: List.take num_axes rngs)
               ~arg:(Reduce { op = rop; num_axes = 0 })))
  | _ -> None

(* A tree of comparisons bounds the depth of a select among many sources, such
   as a large table of constants. *)
let rec stack_select r0 srcs lo hi =
  if hi - lo <= 8 then begin
    let ret = ref srcs.(hi - 1) in
    for k = hi - 2 downto lo do
      ret := where (eq r0 (int k)) srcs.(k) !ret
    done;
    !ret
  end
  else
    let mid = (lo + hi) / 2 in
    where
      (lt r0 (int mid))
      (stack_select r0 srcs lo mid)
      (stack_select r0 srcs mid hi)

(* A stack of shapes has no ranges, and the empty shape is void. The sources are
   taken from the list, since a stack of them may not fit its shape. *)
let convert_stack_to_where ctx x =
  match Tbl.find_opt ctx.range_map x with
  | Some (_, r0 :: _) when not (Dtype.equal (dtype x) Dtype.Void) ->
      let srcs = Array.of_list (create_bufferize_and_index_srcs ctx x) in
      let n = Array.length srcs in
      let ret = stack_select r0 srcs 0 n in
      Some (if n > 8 then where (lt r0 (int 0)) srcs.(n - 1) ret else ret)
  | _ -> None

let remove_movement_op_after_rangeify ctx x =
  if Tbl.mem ctx.range_map x || op (nth x 0) = Op.Index then Some (nth x 0)
  else None

let pm_apply_rangeify =
  let on o f =
    rule_ctx (Upat.v ~op:o ~name:"x" ()) (fun ctx m -> f ctx (m "x"))
  in
  Pattern_matcher.v (fun () ->
      [
        on (ops [ Op.Reduce ]) convert_reduce_to_reduce_with_ranges;
        on (ops [ Op.Pad ]) convert_pad_to_where_to_keep_behavior_local;
        on (ops [ Op.Stack ]) convert_stack_to_where;
        on Op.Set.all create_bufferize_and_index_based_on_ranges;
        on Op.Set.movement remove_movement_op_after_rangeify;
      ])

let pm_fix_deviceless =
  Pattern_matcher.v (fun () ->
      [
        rule_ctx (Upat.op Op.Stage ~name:"b") (fun device m ->
            let b = m "b" in
            match arg b with
            | Bufferize ({ device = None; _ } as o) ->
                Some (replace b ~arg:(Bufferize { o with device }))
            | _ -> None);
      ])

(* Movements *)

let apply_reshape in_shape out_shape urngs =
  let _, axes_in =
    List.fold_left2
      (fun (acc, axes) s r -> (Sint.(acc * s), mul (sint acc) r :: axes))
      (Int 1, []) (List.rev out_shape)
      (List.rev (src urngs))
  in
  let combined = usum (int 0) (List.rev axes_in) in
  let _, axes_out =
    List.fold_left
      (fun (c, axes) s -> (O.(c // sint s), O.(c % sint s) :: axes))
      (combined, []) (List.rev in_shape)
  in
  (* Simplifying merges what reshapes of reshapes would otherwise stack. *)
  graph_rewrite ~ctx:() (Ops.sink axes_out)
    (Pattern_matcher.concat
       Symbolic.[ symbolic; pm_simplify_valid; pm_drop_and_clauses ])

let pad_valid = Pattern_matcher.concat Symbolic.[ symbolic; pm_simplify_valid ]

let apply_movement_op in_shape m rngs =
  match m with
  | Shrink b ->
      List.map2
        (fun a (off, _) ->
          if Sint.equal off (Int 0) then a else add a (sint off))
        rngs b
  | Permute p -> List.map (List.nth rngs) (Helpers.argsort p)
  | Flip f ->
      List.map2
        (fun (a, s) f -> if f then sub (sint Sint.(s - Int 1)) a else a)
        (List.combine rngs in_shape)
        f
  | Expand added -> List.drop (List.length added) rngs
  | Pad b ->
      (* The validity is simplified on its own, so that the pad's selection
         wraps the new validity alone. *)
      List.map2
        (fun (r, sh) (off, sz) ->
          if Sint.equal sz sh && Sint.equal off (Int 0) then r
          else
            let inside =
              bitwise_and (ge r (sint off)) (lt r (sint Sint.(sh + off)))
            in
            valid (sub r (sint off)) (graph_rewrite ~ctx:() inside pad_valid))
        (List.combine rngs in_shape)
        b
  | Reshape out_shape ->
      (* Simplifying first puts the ranges in their canonical order. *)
      let sink = simplify (Ops.sink rngs) in
      let sub_array =
        List.mapi
          (fun i r ->
            ( r,
              replace r
                ~src:[ nth r 0 ]
                ~arg:
                  (Range { axis_id = [ i ]; axis_type = Axis_type.Placeholder })
            ))
          (Nodes.to_list (ranges sink))
      in
      let reshaped =
        apply_reshape in_shape out_shape (substitute sink sub_array)
      in
      src (substitute reshaped (List.map (fun (r, p) -> (p, r)) sub_array))

(* Rangeify *)

let rec transpose = function
  | [] -> []
  | ls when List.exists List.is_empty ls -> []
  | ls -> List.map List.hd ls :: transpose (List.map List.tl ls)

let render_ranges ~realized rngs_list =
  List.mapi
    (fun i rs ->
      let rng =
        if Helpers.all_same String.equal rs then List.hd rs
        else String.concat " -> " rs
      in
      let rng =
        match realized with
        | Some axes when List.mem i axes -> Helpers.colored Yellow rng
        | _ -> rng
      in
      "[" ^ rng ^ "]")
    (transpose (List.map (List.map Render.render) rngs_list))
  |> String.concat ""

let print_ranges rctx x ~consumers ~ending rngs out_rngs =
  let realized = Option.join (Tbl.find_opt rctx.realize_map x) in
  let disp =
    if op x = Op.Reshape || List.length rngs <> List.length out_rngs then
      render_ranges ~realized [ rngs ]
      ^ " -> "
      ^ render_ranges ~realized [ out_rngs ]
    else render_ranges ~realized [ rngs; out_rngs ]
  in
  let pp_shape ppf = function
    | None -> Format.pp_print_string ppf "None"
    | Some [ s ] -> Format.fprintf ppf "(%a,)" Sint.pp s
    | Some s ->
        Format.fprintf ppf "(%a)"
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
             Sint.pp)
          s
  in
  Format.printf "%s %2d %-20s %-35s %2d %s@."
    (if Tbl.mem rctx.realize_map x then "***" else "   ")
    consumers
    (Format.asprintf "%a" Op.pp (op x))
    (Format.asprintf "%a" pp_shape (shape_opt x))
    ending disp

(* A node that consumers index differently gets new ranges on every axis;
   otherwise it takes theirs, valid where any of theirs is. *)
let merge_consumer_rngs rctx x consumer_rngs =
  let axes = transpose consumer_rngs in
  let locals = List.map (List.map get_idx) axes in
  if List.for_all (Helpers.all_same Ops.equal) locals then
    List.map2
      (fun local rngs ->
        let minimum_valid = usum (bool false) (List.map get_valid rngs) in
        graph_rewrite ~ctx:()
          (valid (List.hd local) minimum_valid)
          Symbolic.symbolic)
      locals axes
  else begin
    Tbl.replace rctx.realize_map x (Some (List.init (List.length axes) Fun.id));
    new_ranges rctx (List.take (List.length axes) (shape x))
  end

(* Kernels are internal, and after, shard selections and shard stacks carry no
   ranges, as a sink does not. *)
let no_ranges = ops Op.[ Call; Linear; After; Mstack; Mselect ]

let assign_ranges rctx ~debug ~consumer_map ~ending_ranges x =
  let consumers = List.rev (Tbl.find consumer_map x) in
  let ending =
    ref
      (List.concat_map
         (fun u -> Option.value (Tbl.find_opt ending_ranges u) ~default:[])
         consumers)
  in
  (* The ranges the consumers iterate that this node is broadcast over. *)
  let ended =
    List.concat_map
      (fun c ->
        match Tbl.find_opt rctx.range_map c with
        | Some (rngs, _) when Op.Set.mem (op c) Op.Set.broadcastable ->
            List.map (List.nth rngs) (broadcast_axes (shape x) (shape c))
        | _ -> [])
      consumers
  in
  let broadcast_ending_ranges = Nodes.to_list (ranges (Ops.sink ended)) in
  (* The fusion decision: a reduction is stored before it is broadcast. *)
  if op x = Op.Reduce then ending := !ending @ broadcast_ending_ranges;
  let consumer_rngs =
    List.filter_map
      (fun c ->
        Option.map
          (fun (rngs, _) -> broadcast_rngs c x rngs)
          (Tbl.find_opt rctx.range_map c))
      consumers
  in
  let out_rngs =
    if Tbl.mem rctx.realize_map x then begin
      ending := [];
      Tbl.replace rctx.realize_map x (Some (List.init (ndim x) Fun.id));
      Some (new_ranges rctx (shape x))
    end
    else
      match consumer_rngs with
      | [] -> None
      | [ rngs ] -> Some rngs
      | _ -> Some (merge_consumer_rngs rctx x consumer_rngs)
  in
  Option.iter
    (fun out_rngs ->
      let out_rngs =
        if
          (not (List.is_empty !ending))
          && Op.Set.mem (op x)
               (Op.Set.union Op.Set.elementwise (ops [ Op.Reduce ]))
        then begin
          ending := [];
          if List.is_empty out_rngs then out_rngs
          else begin
            Tbl.replace rctx.realize_map x
              (Some (List.init (List.length out_rngs) Fun.id));
            new_ranges rctx (List.take (List.length out_rngs) (shape x))
          end
        end
        else out_rngs
      in
      ending := !ending @ broadcast_ending_ranges;
      let rngs =
        if Op.Set.mem (op x) Op.Set.movement then
          apply_movement_op (shape (nth x 0)) (marg x) out_rngs
        else if op x = Op.Stack then List.drop 1 out_rngs
        else out_rngs
      in
      (* An expand that injects a range does not end it. Ending the others is
         why convolutions are stored. *)
      (if op x = Op.Expand then
         match marg x with
         | Expand added
           when List.for_all
                  (function Int _ -> true | Sym s -> op s <> Op.Range)
                  (shape x) ->
             let injected = List.take (List.length added) out_rngs in
             ending := !ending @ Nodes.to_list (ranges (Ops.sink injected))
         | _ -> ());
      let rngs =
        match arg x with
        | Reduce { num_axes; _ } when num_axes <> 0 ->
            new_ranges ~axis_type:Axis_type.Reduce rctx
              (List.take num_axes (shape (nth x 0)))
            @ out_rngs
        | _ -> rngs
      in
      if debug then
        print_ranges rctx x ~consumers:(List.length consumers)
          ~ending:(List.length !ending) rngs out_rngs;
      Tbl.replace rctx.range_map x (rngs, out_rngs))
    out_rngs;
  Tbl.replace ending_ranges x !ending

let run_rangeify ?(debug = false) tsink =
  if debug then print_endline "**************************";
  let rctx =
    {
      realize_map = Tbl.create 64;
      non_removable = Tbl.create 8;
      stored_through = Tbl.create 8;
      range_map = Tbl.create 256;
      range_idx = 0;
    }
  in
  let tsink = graph_rewrite ~ctx:() tsink pm_own_stored_destinations in
  ignore (graph_rewrite ~ctx:rctx tsink pm_generate_realize_map);
  let tsink_toposort = toposort ~gate:gate_kernel_sink tsink in
  let consumer_map = Tbl.create 256 in
  List.iter (fun x -> Tbl.replace consumer_map x []) tsink_toposort;
  List.iter
    (fun c ->
      List.iter
        (fun x ->
          match Tbl.find_opt consumer_map x with
          | Some (c' :: _) when c' == c -> ()
          | Some cs -> Tbl.replace consumer_map x (c :: cs)
          | None -> ())
        (data_srcs (op c) (src c)))
    tsink_toposort;
  let ending_ranges = Tbl.create 256 in
  List.iter
    (fun x ->
      if not (Op.Set.mem (op x) no_ranges) then
        assign_ranges rctx ~debug ~consumer_map ~ending_ranges x)
    (List.rev tsink_toposort);
  let spec = min (Helpers.Context_var.value Helpers.spec) 2 in
  let tsink =
    Helpers.context
      [ B (Helpers.spec, spec) ]
      (fun () ->
        graph_rewrite ~bottom_up:true ~ctx:rctx tsink pm_apply_rangeify)
  in
  (* A value without a device that must be stored lives on the sink's. *)
  graph_rewrite ~ctx:(device tsink) tsink pm_fix_deviceless
