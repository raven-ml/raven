(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let rule_ctx = Pattern_matcher.rule_ctx
let ops l = Op.Set.of_list l
let sint = sint_to_uop
let zero = `Int Bigint.zero

(* Indexing context *)

type ctx = {
  realize_map : int list option Tbl.t;
      (* The nodes stored whole: marked, then the axes given new ranges. *)
  non_removable : unit Tbl.t;
  broadcast : unit Tbl.t;
      (* The values stored because a broadcast reads them. *)
  range_map : (t list * t list) Tbl.t;
      (* Each node's ranges: those that index its sources, then its output. *)
  mutable range_idx : int;
  shares : t list Tbl.t;
      (* Each store that shares its loops with others: the stores of its group,
         in order. *)
  loops : t list Tbl.t; (* A group's loops, by its first store. *)
  ends : t Tbl.t; (* A group's end, by its first store. *)
  mutable siblings : siblings list;
      (* The reductions that share their loops, newest group first. *)
}

and siblings = {
  sizes : sint list; (* The sizes of the axes they reduce. *)
  outer : t list; (* The ranges that index their outputs. *)
  reduced : t list; (* The ranges they reduce. *)
  reached : unit Tbl.t; (* What the members reach, but the pending ones. *)
  mutable pending : t list; (* The members whose reach is not walked yet. *)
}

(* A range of size 1 only ever takes the value 0. *)
let new_range ctx (s : sint) =
  match s with
  | Sym r when op r = Op.Range -> r
  | _ when Sint.(resolve (s <> Int 1)) ->
      let r = range s [ ctx.range_idx ] in
      ctx.range_idx <- ctx.range_idx + 1;
      r
  | _ -> int 0

let new_ranges ctx shape =
  List.rev (List.fold_left (fun acc s -> new_range ctx s :: acc) [] shape)

let always_contiguous =
  Op.Set.union Op.Set.defines (ops Op.[ After; Const; Load; Call ])

let realize ctx u = Tbl.replace ctx.realize_map u None
let storage = Op.Set.union Op.Set.defines (ops [ Op.After ])

(* A gather is an index by an integer value with axes: its source is read at
   loaded positions, so it is stored whole unless it is storage. A view of
   storage is stored too, and the stage, which reads one buffer, is inlined
   back. *)
let is_gather_src src = List.exists (fun i -> ndim i > 0) (List.tl src)
let is_gather u = op u = Op.Index && is_gather_src (src u)

let realize_gathered ctx g =
  let x = nth g 0 in
  if is_gather g && not (Op.Set.mem (op x) storage) then realize ctx x

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
          (Upat.op Op.Index ~name:"g" ~allow_any_len:true)
          (mark (fun ctx m -> realize_gathered ctx (m "g")));
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

(* The operations whose sources hold no data, and those whose first alone
   does. *)
let no_data = Op.Set.union Op.Set.defines (ops Op.[ Range; Special ])

let first_data =
  Op.Set.union Op.Set.movement
    (ops Op.[ Index; Stage; Reduce; After; End; Backedge; Copy ])

(* The sources that hold data, as opposed to shapes, bounds and ranges. *)
let data_srcs u =
  let op = op u and src = src u in
  let first = match src with s :: _ -> [ s ] | [] -> [] in
  if Op.Set.mem op no_data then []
  else if op = Op.Index && is_gather_src src then src
  else if is_group u then (* every store is data *) src
  else if Op.Set.mem op first_data then first
  else src

(* [s], a source of an indexed node read at [src_rngs] if [indexed]: storage
   indexed, a source stored whole staged over its own ranges and indexed, and
   any other source as it is. *)
(* The end of a group of stores: one end of all of them over their loops. *)
let end_shared ctx members =
  let first = List.hd members in
  match Tbl.find_opt ctx.ends first with
  | Some e -> e
  | None ->
      let closed = snd (Tbl.find ctx.range_map first) in
      List.iter (Tbl.remove ctx.realize_map) members;
      let e =
        end_ (group members) (List.filter (fun r -> op r = Op.Range) closed)
      in
      Tbl.replace ctx.ends first e;
      e

let bufferize_and_index ctx ~indexed s src_rngs =
  if Op.Set.mem (op s) storage then if indexed then index s src_rngs else s
  else if Tbl.mem ctx.shares s then end_shared ctx (Tbl.find ctx.shares s)
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
        let keep =
          if Op.Set.mem (op s) always_contiguous || Tbl.mem ctx.non_removable s
          then Whole
          else if Tbl.mem ctx.broadcast s then Broadcast
          else Removable
        in
        let opts : bufferize_opts =
          { device = device s; addrspace = Dtype.Global; keep }
        in
        let staged = bufferize ~opts s closed in
        if indexed then index staged src_rngs else staged

let create_bufferize_and_index_srcs ctx x =
  let data_src_count = List.length (data_srcs x) in
  let rngs = Option.map fst (Tbl.find_opt ctx.range_map x) in
  List.mapi
    (fun i s ->
      let src_rngs =
        match rngs with Some r -> broadcast_rngs x s r | None -> []
      in
      let indexed =
        Option.is_some rngs
        && (i < data_src_count || not (Op.Set.mem (op s) storage))
      in
      bufferize_and_index ctx ~indexed s src_rngs)
    (src x)

(* A gather reads its source at the loaded index, then at its own trailing
   ranges. *)
let convert_gather ctx x =
  match (Tbl.find_opt ctx.range_map x, src x) with
  | Some (rngs, _), [ g; l ] when is_gather x ->
      let n = ndim l in
      let l = bufferize_and_index ctx ~indexed:true l (List.take n rngs) in
      Some (bufferize_and_index ctx ~indexed:true g (l :: List.drop n rngs))
  | Some _, _ when is_gather x -> invalid_arg "a gather takes one index"
  | _ -> None

(* An index of a value by scalars, which select its leading axes. A gather
   indexes by a value with axes, and an index of a stage is already the read of
   its storage. *)
let scalar_index x =
  op x = Op.Index && (not (is_gather x)) && op (nth x 0) <> Op.Stage

(* A group's stores are its own: each indexes its sources as any node does. *)
let create_bufferize_and_index_based_on_ranges ctx x =
  if is_group x then None
  else Some (replace x ~src:(create_bufferize_and_index_srcs ctx x))

(* The first source is taken from the list: rebuilding [x] on an indexed source
   would give it a shape its argument does not fit. An index by scalars reads
   its source where they select, which is that source indexed, unless it is
   already the index ([scalar_index]). *)
let convert_pad_to_where_to_keep_behavior_local ctx x =
  match Tbl.find_opt ctx.range_map x with
  | None -> None
  | Some (rngs, out)
    when op x = Op.Index
         && ((not (scalar_index x)) || List.compare_lengths rngs out = 0) ->
      None
  | Some (rngs, _) ->
      let s = List.hd (create_bufferize_and_index_srcs ctx x) in
      if s == x then None
      else
        let valid = uprod (bool true) (List.map get_valid rngs) in
        Some (where valid s (const (Dtype.const (dtype x) zero)))

(* [constant u] is whether [u] is a constant through casts and movements. *)
let rec constant u =
  match op u with
  | Op.Const -> true
  | Op.Cast | Reshape | Expand | Permute | Shrink -> constant (nth u 0)
  | _ -> false

(* An integer division computed where its ranges are invalid, on a pad's
   padding, divides by 1 there, unless its divisor is a nonzero constant. The
   padding loads zeros whatever the divisor's bounds, and a division by zero
   traps on x86 even in a lane whose result the pad drops. A range's validity
   is read after the symbolic rules lift its gates to the top and merge them:
   a shrink after a pad leaves the gate below an addition, and a pad of a pad
   nests one gate in another. *)
let guard_division_on_padding ctx x =
  match Tbl.find_opt ctx.range_map x with
  | Some (rngs, _) when Dtype.is_int (dtype x) -> (
      let a, d =
        match create_bufferize_and_index_srcs ctx x with
        | [ a; d ] -> (a, d)
        | _ -> invalid_arg "a division has two sources"
      in
      let validity r =
        get_valid
          (graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() r
             (After_sources Shape.symbolic))
      in
      let may_fail c = not (Dtype.Value.to_bool (vmin c)) in
      let nonzero_const =
        constant d && Dtype.Value.(vmin d = vmax d && vmin d <> of_int 0)
      in
      match List.filter may_fail (List.map validity rngs) with
      | c :: cs when not nonzero_const ->
          let one = const (Dtype.const (dtype d) (`Int Bigint.one)) in
          Some (replace x ~src:[ a; where (uprod c cs) d one ])
      | _ -> Some (replace x ~src:[ a; d ]))
  | _ -> None

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

(* A tree of selections on the bits of [r0], the most significant first. *)
let rec stack_select r0 srcs =
  match srcs with
  | [ x ] -> x
  | _ ->
      let n = List.length srcs in
      let half = 1 lsl (Bigint.numbits (Bigint.of_int (n - 1)) - 1) in
      let lo = List.filteri (fun i _ -> i < half) srcs
      and hi = List.filteri (fun i _ -> i >= half) srcs in
      where
        (lt (mod_ r0 (int (2 * half))) (int half))
        (stack_select r0 lo) (stack_select r0 hi)

(* A stack of shapes has no ranges. The sources are taken from the list, since a
   stack of them may not fit its shape. *)
let convert_stack_to_where ctx x =
  match Tbl.find_opt ctx.range_map x with
  | Some (_, r0 :: _) ->
      Some (stack_select r0 (create_bufferize_and_index_srcs ctx x))
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
        on
          (ops [ Op.Pad; Op.Index ])
          convert_pad_to_where_to_keep_behavior_local;
        on (ops [ Op.Stack ]) convert_stack_to_where;
        on (ops [ Op.Index ]) convert_gather;
        on
          (ops [ Op.Truncdiv; Op.Truncmod; Op.Div; Op.Mod ])
          guard_division_on_padding;
        on
          (Op.Set.diff Op.Set.all (ops [ Op.Stage; Op.Index ]))
          create_bufferize_and_index_based_on_ranges;
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

let deviceless_stage u =
  match (op u, arg u) with
  | Op.Stage, Bufferize { device = None; _ } -> true
  | _ -> false

(* A stored value of no device, an arange or a table of constants, lives where
   the kernel that reads it computes: on the device of the nearest node above it
   that has one. A sink over several devices, each kernel on one, gives each
   device that reads the value a stage of its own: [exp2 (arange 4)] read by
   sums on CPU and CPU:1 is stored once on each. A value read by no node with a
   device, or by a store to a disk alone, lives on the sink's. On a sink of
   one device both rules place it there, and the walk that finds the readers
   is all the rule adds. *)
let fix_deviceless tsink =
  (* The nodes of no device that reach a stage of no device, and the devices
     of the nodes that read them. *)
  let pending = Tbl.create 16 and readers = ref [] in
  List.iter
    (fun u ->
      match device u with
      | None ->
          if deviceless_stage u || List.exists (Tbl.mem pending) (src u) then
            Tbl.replace pending u ()
      | Some d ->
          if
            List.exists (Tbl.mem pending) (src u)
            && not (List.exists (String.equal d) !readers)
          then readers := d :: !readers)
    (toposort ~calls:Skip tsink);
  let on_sink u =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(device tsink) u
      (After_sources pm_fix_deviceless)
  in
  (* A reader on a disk computes nothing: what it stores is computed where
     the sink computes. *)
  let reader d = not (is_disk_device d) in
  let elsewhere d = reader d && Some d <> device tsink in
  if not (List.exists elsewhere !readers) then on_sink tsink
  else
    let placed = Hashtbl.create 4 in
    let place d u =
      let memo =
        match Hashtbl.find_opt placed d with
        | Some m -> m
        | None ->
            let m = Tbl.create 16 in
            Hashtbl.replace placed d m;
            m
      in
      match Tbl.find_opt memo u with
      | Some p -> p
      | None ->
          let p =
            graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Some d) u
              (After_sources pm_fix_deviceless)
          in
          Tbl.replace memo u p;
          p
    in
    let pm =
      Pattern_matcher.v (fun () ->
          [
            rule_ctx (Upat.v ~op:Op.Set.all ~name:"x" ()) (fun () m ->
                let x = m "x" in
                match device x with
                | Some d when reader d && List.exists (Tbl.mem pending) (src x)
                  ->
                    let src =
                      List.map
                        (fun s -> if Tbl.mem pending s then place d s else s)
                        (src x)
                    in
                    Some (replace x ~src)
                | _ -> None);
          ])
    in
    on_sink
      (graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() tsink
         (After_sources pm))

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
  graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() (Ops.sink axes_out)
    (After_sources
       (Pattern_matcher.concat
          Symbolic.[ symbolic; pm_simplify_valid; pm_drop_and_clauses ]))

let pad_valid = Pattern_matcher.concat Symbolic.[ symbolic; pm_simplify_valid ]

let apply_movement_op in_shape m rngs =
  match m with
  | Shrink b ->
      List.map2
        (fun a (off, _) ->
          if Sint.equal off (Int 0) then a else add a (sint off))
        rngs b
  | Permute p -> List.map (List.nth rngs) (Helpers.argsort p)
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
            valid
              (sub r (sint off))
              (graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() inside
                 (After_sources pad_valid)))
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
                ~arg:
                  (Range { axis_type = Axis_type.Placeholder; axis_id = [ i ] })
            ))
          (Nodes.to_list (ranges sink))
      in
      let reshaped =
        apply_reshape in_shape out_shape
          (substitute ~calls:Skip ~pass:Fixed_point sink sub_array)
      in
      src
        (substitute ~calls:Skip ~pass:Fixed_point reshaped
           (List.map (fun (r, p) -> (p, r)) sub_array))

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
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
          (valid (List.hd local) minimum_valid)
          (After_sources Shape.symbolic))
      locals axes
  else begin
    Tbl.replace rctx.realize_map x (Some (List.init (List.length axes) Fun.id));
    new_ranges rctx (List.take (List.length axes) (shape x))
  end

(* Stores that share their loops

   Stored values of one shape and device share their loops when the nodes each
   computes before it reads storage or another stored value share a computed
   node, the two read no computed node through different views, as two slices
   of one reduction are, neither reads what the other writes, and neither
   waits for a write over storage the other reads as it was. Each then indexes the shared nodes
   alike, which are computed once, and the stores end as one kernel that writes
   them all: a double word's high and low parts, the two ends of one chain, are
   one kernel. A reduction of one of them alone keeps them apart, since a
   kernel of two reductions can lose the optimisations each would take.

   Two groups join only when neither reads the other through the stores of a
   third: where [b] reads [c] and [d] reads [a], the groups {a, b} and {c, d}
   would each run before and after the other. *)

let computing = Op.Set.union Op.Set.elementwise (ops [ Op.Reduce ])
let computed u = Op.Set.mem (op u) computing

(* Whether [s] computes [u] in its own kernel, as its value or a node that is
   neither stored nor always contiguous. *)
let fused rctx s u =
  u == nth s 1
  || not (Tbl.mem rctx.realize_map u || Op.Set.mem (op u) always_contiguous)

(* The nodes [s]'s value reaches before storage or a stored value, each after
   its sources. *)
let reached rctx s =
  let value = nth s 1 in
  if Tbl.mem rctx.realize_map value then []
  else toposort ~calls:Skip ~gate:(fused rctx s) value

(* The computed nodes of {!reached}. *)
let region topo = List.filter computed topo

(* The views of computed nodes among [topo], the nodes a store reaches
   ({!reached}): each movement over a computed node that [s] computes, by the
   computed node behind it. *)
let views rctx s topo =
  let movement u = Op.Set.mem (op u) Op.Set.movement in
  let rec behind u = if movement u then behind (nth u 0) else u in
  let by = Tbl.create 4 in
  List.iter
    (fun u ->
      if movement u then
        let c = behind (nth u 0) in
        if computed c && fused rctx s c then
          Tbl.replace by c (u :: Option.value (Tbl.find_opt by c) ~default:[]))
    topo;
  by

(* Whether the store [b], of backward slice [slice], is read through anything:
   its value or where it stores, which may be storage another store wrote. *)
let reads slice b = Nodes.mem b slice || Nodes.mem (base (nth b 0)) slice

(* Whether a store of backward slice [sa] waits for a write over storage that
   one of [sb] reads in a state before it. One kernel of both would run
   neither before nor after the write, as for [v + a] and [v * v + a'], where
   [a'] is [a] once assigned. *)
let waits_over sa sb =
  Nodes.fold
    (fun w found ->
      found
      || op w = Op.After
         && (not (Nodes.mem w sb))
         && Nodes.mem (base (nth w 0)) sb)
    sa false

(* The checks of a pair of stores, for tests. *)
let checks = Atomic.make 0
let store_checks () = Atomic.get checks

let joinable regions ~crossed slice a b =
  Atomic.incr checks;
  let reductions s =
    List.filter (fun u -> op u = Op.Reduce) (Tbl.find regions s)
  in
  let within s t =
    List.for_all (fun r -> List.memq r (Tbl.find regions t)) (reductions s)
  in
  let apart () =
    let sa = slice a and sb = slice b in
    reads sa b || reads sb a || waits_over sa sb || waits_over sb sa
  in
  List.equal Sint.equal (shape (nth a 1)) (shape (nth b 1))
  && Option.equal String.equal (device (nth a 0)) (device (nth b 0))
  && base (nth a 0) != base (nth b 0)
  && within a b && within b a
  && (not (crossed a b))
  && not (apart ())

let share_loops rctx topo =
  (* A store of no loop ends nothing, so it stays alone. *)
  let looped u = List.exists (fun s -> Sint.(resolve (s <> Int 1))) (shape u) in
  let stores =
    List.filter
      (fun u -> op u = Op.Store && Tbl.mem rctx.realize_map u && looped u)
      topo
  in
  (* A region holds each node once, so a store enters each node's list once. *)
  let regions = Tbl.create 16 and by_node = Tbl.create 64 in
  let views_of = Tbl.create 16 in
  List.iter
    (fun s ->
      let topo = reached rctx s in
      let r = region topo in
      Tbl.replace regions s r;
      Tbl.replace views_of s (lazy (views rctx s topo));
      List.iter
        (fun u ->
          let ss = Option.value (Tbl.find_opt by_node u) ~default:[] in
          Tbl.replace by_node u (s :: ss))
        r)
    stores;
  (* Whether [a] and [b] read one computed node through different views, as
     two slices of it are: indexed alike, they would compute none of it
     once. *)
  let crossed a b =
    let va = Lazy.force (Tbl.find views_of a)
    and vb = Lazy.force (Tbl.find views_of b) in
    let same us us' =
      List.for_all (fun u -> List.memq u us') us
      && List.for_all (fun u -> List.memq u us) us'
    in
    Tbl.fold
      (fun c us found ->
        found
        ||
        match Tbl.find_opt vb c with
        | Some us' -> not (same us us')
        | None -> false)
      va false
  in
  (* Each store's backward slice, made once. *)
  let slices = Tbl.create 16 in
  let slice s =
    match Tbl.find_opt slices s with
    | Some sl -> sl
    | None ->
        let sl = backward_slice ~calls:Skip s in
        Tbl.replace slices s sl;
        sl
  in
  (* The stores each store reads through anything. *)
  let reads_of = Tbl.create 16 in
  let read s =
    match Tbl.find_opt reads_of s with
    | Some r -> r
    | None ->
        let sl = slice s in
        let r = List.filter (fun t -> Nodes.mem t sl) stores in
        Tbl.replace reads_of s r;
        r
  in
  (* A group's members, in the order of [stores]: two groups merge by the
     stores' positions there. *)
  let position = Tbl.create 16 in
  List.iteri (fun i s -> Tbl.replace position s i) stores;
  let rec merge a b =
    match (a, b) with
    | [], l | l, [] -> l
    | x :: a', y :: b' ->
        if Tbl.find position x < Tbl.find position y then x :: merge a' b
        else y :: merge a b'
  in
  (* Groups by their first store, each store's group by its first store. *)
  let groups = Tbl.create 16 and first = Tbl.create 16 in
  List.iter
    (fun s ->
      Tbl.replace groups s [ s ];
      Tbl.replace first s s)
    stores;
  (* Whether a store of the group of [fa] reads one of the group of [fb],
     through anything and the stores of other groups. *)
  let reaches fa fb =
    let seen = Tbl.create 8 in
    let rec go = function
      | [] -> false
      | f :: _ when f == fb -> true
      | f :: rest when Tbl.mem seen f -> go rest
      | f :: rest ->
          Tbl.replace seen f ();
          let next =
            List.concat_map
              (fun s -> List.map (Tbl.find first) (read s))
              (Tbl.find groups f)
          in
          go (List.rev_append next rest)
    in
    go [ fa ]
  in
  (* Pairs of groups that cannot join, by their first stores' positions [i <
     j] as [i * n + j]. A group only grows and keeps its first store, or stops
     being a group, so a pair that failed fails again: two stores are checked
     once, not at each node they share. *)
  let apart = Hashtbl.create 16 and n = List.length stores in
  let join a b =
    let fa = Tbl.find first a and fb = Tbl.find first b in
    let i = Tbl.find position fa and j = Tbl.find position fb in
    let pair = (min i j * n) + max i j in
    if fa != fb && not (Hashtbl.mem apart pair) then
      let ga = Tbl.find groups fa and gb = Tbl.find groups fb in
      if
        List.for_all
          (fun s -> List.for_all (joinable regions ~crossed slice s) gb)
          ga
        && not (reaches fa fb || reaches fb fa)
      then begin
        let members = merge ga gb in
        let f = List.hd members in
        Tbl.remove groups fa;
        Tbl.remove groups fb;
        Tbl.replace groups f members;
        List.iter (fun s -> Tbl.replace first s f) members
      end
      else Hashtbl.replace apart pair ()
  in
  List.iter
    (fun u ->
      match Tbl.find_opt by_node u with
      | Some (s :: rest) -> List.iter (join s) rest
      | _ -> ())
    topo;
  Tbl.iter
    (fun _ members ->
      if List.length members > 1 then
        List.iter (fun s -> Tbl.replace rctx.shares s members) members)
    groups

(* A realized node's loops: new ones, or its group's. *)
let loops_of rctx x =
  match Tbl.find_opt rctx.shares x with
  | None -> new_ranges rctx (shape x)
  | Some members -> (
      let f = List.hd members in
      match Tbl.find_opt rctx.loops f with
      | Some r -> r
      | None ->
          let r = new_ranges rctx (shape x) in
          Tbl.replace rctx.loops f r;
          r)

(* Reductions that share their loops

   A reduction reduces its source over ranges of its own. Reductions whose
   outputs are indexed by the same ranges and that reduce axes of the same sizes
   take the same ranges instead when neither reads the other: their sources are
   then indexed alike, so the nodes they share are computed once, and their
   accumulators add in one loop. Reverse mode makes such reductions: the
   cotangent of a value that operations broadcast sums one reduction per
   operation, each over the same cotangent computation.

   The walk meets a reduction before those it reads, so a reduction joins the
   first group of its key whose members do not reach it. A group walks what its
   members reach only once a second reduction asks to join it. *)

let reaches g x =
  let unseen u = not (Tbl.mem g.reached u) in
  List.iter
    (fun m ->
      List.iter
        (fun u -> Tbl.replace g.reached u ())
        (toposort ~calls:Skip ~gate:unseen (nth m 0)))
    g.pending;
  g.pending <- [];
  Tbl.mem g.reached x

let reduce_loops rctx x sizes outer =
  let fits g =
    List.equal Sint.equal g.sizes sizes
    && List.equal Ops.equal g.outer outer
    && not (reaches g x)
  in
  match List.find_opt fits rctx.siblings with
  | Some g ->
      g.pending <- x :: g.pending;
      g.reduced
  | None ->
      let reduced = new_ranges rctx sizes in
      let g =
        { sizes; outer; reduced; reached = Tbl.create 64; pending = [ x ] }
      in
      rctx.siblings <- g :: rctx.siblings;
      reduced

(* Kernels are internal, and an after carries no ranges, as a sink does not. *)
let no_ranges = ops Op.[ Call; After ]

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
  (* A gather's source sees all its ranges, and its index the leading ones,
     which the index shapes. *)
  let consumer_rngs =
    List.concat_map
      (fun c ->
        match Tbl.find_opt rctx.range_map c with
        | None -> []
        | Some (rngs, _) when is_gather c ->
            List.concat
              (List.mapi
                 (fun j s ->
                   if s != x then []
                   else if j = 0 then [ rngs ]
                   else [ List.take (ndim x) rngs ])
                 (src c))
        | Some (rngs, _) -> [ broadcast_rngs c x rngs ])
      consumers
  in
  let out_rngs =
    if Tbl.mem rctx.realize_map x then begin
      ending := [];
      Tbl.replace rctx.realize_map x (Some (List.init (ndim x) Fun.id));
      Some (loops_of rctx x)
    end
    else
      match consumer_rngs with
      | [] -> None
      | [ rngs ] -> Some rngs
      | _ ->
          let rngs = merge_consumer_rngs rctx x consumer_rngs in
          (* Stored for consumers that read it at different indices, a value
             they broadcast is read where it is broadcast. *)
          if Tbl.mem rctx.realize_map x && broadcast_ending_ranges <> [] then
            Tbl.replace rctx.broadcast x ();
          Some rngs
  in
  Option.iter
    (fun out_rngs ->
      let out_rngs =
        if
          (not (List.is_empty !ending))
          && computed x
        then begin
          ending := [];
          (* A value of no axes that computes a reduction is stored too, as one
             element: inlined where it is broadcast, each element of the
             consumer would compute the reduction whole. Without a reduction it
             is scalar arithmetic, which may be an offset. *)
          if
            List.is_empty out_rngs
            && not (op_in_backward_slice_with_self ~calls:Skip x [ Op.Reduce ])
          then out_rngs
          else begin
            Tbl.replace rctx.realize_map x
              (Some (List.init (List.length out_rngs) Fun.id));
            Tbl.replace rctx.broadcast x ();
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
          (* an index whose scalars and output ranges address every axis of its
             source: the scalars select the leading axes *)
        else if
          scalar_index x
          && List.length (src x) - 1 + List.length out_rngs = ndim (nth x 0)
        then List.tl (src x) @ out_rngs
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
        | Reduce { num_axes; _ } ->
            reduce_loops rctx x (List.take num_axes (shape (nth x 0))) out_rngs
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
  let tsink_toposort = toposort ~calls:Skip ~gate:gate_kernel_sink tsink in
  let rctx =
    {
      realize_map = Tbl.create 64;
      non_removable = Tbl.create 8;
      broadcast = Tbl.create 8;
      range_map = Tbl.create 256;
      range_idx = 0;
      shares = Tbl.create 8;
      loops = Tbl.create 8;
      ends = Tbl.create 8;
      siblings = [];
    }
  in
  ignore
    (graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:rctx tsink
       (After_sources pm_generate_realize_map));
  share_loops rctx tsink_toposort;
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
        (data_srcs c))
    tsink_toposort;
  let ending_ranges = Tbl.create 256 in
  List.iter
    (fun x ->
      if not (Op.Set.mem (op x) no_ranges) then
        assign_ranges rctx ~debug ~consumer_map ~ending_ranges x)
    (List.rev tsink_toposort);
  let tsink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:rctx tsink
      (Before_sources pm_apply_rangeify)
  in
  fix_deviceless tsink
