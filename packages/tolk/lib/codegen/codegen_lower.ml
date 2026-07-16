(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/__init__.py::full_rewrite_to_sink to tolk_uop.
   Runs every pass from "postopt symbolic" through "number params". *)

open Tolk_uop
module U = Uop
module PM = Upat.Pattern_matcher

(* Helpers *)

let prod = List.fold_left ( * ) 1

(* Index-dtype integer constant, used for lane and shape positions. *)
let int_ n = U.const_int n

let shape_arg dims =
  match dims with [ dim ] -> dim | dims -> U.stack dims

let shape_arg_of_ints dims = shape_arg (List.map U.const_int dims)

let shape_opt = U.shape_opt

let dim_is_one dim = match U.const_int_value dim with Some 1 -> true | _ -> false

let argsort order =
  order |> List.mapi (fun i v -> (v, i)) |> List.sort compare |> List.map snd

let all_same = function
  | [] -> true
  | first :: rest -> List.for_all (List.equal U.equal first) rest

let align_left rank shape =
  List.init (rank - List.length shape) (fun _ -> U.const_int 1) @ shape

(* Reshape [src] to [shape], skipping the reshape when the shape is unchanged. *)
let reshape_to src shape =
  match shape_opt src with
  | Some s when List.equal U.equal s shape -> src
  | _ -> U.reshape ~src ~shape:(shape_arg shape)

(* Broadcast [src] to [new_shape], stretching size-one axes. [Uop.expand] only
   prepends leading axes, so the axes that must stretch — the newly-prepended
   leading axes and each own size-one axis whose target differs — are squeezed
   out, re-introduced by a single leading EXPAND, and permuted back into place. *)
let broadcast_to src new_shape =
  match shape_opt src with
  | None -> src
  | Some s ->
      if List.equal U.equal s new_shape then src
      else
        let m = List.length s and rank = List.length new_shape in
        let n_left = rank - m in
        let s_arr = Array.of_list s and t_arr = Array.of_list new_shape in
        let axes = List.init m Fun.id in
        let expand_at =
          List.filter
            (fun i -> dim_is_one s_arr.(i) && not (dim_is_one t_arr.(n_left + i)))
            axes
        in
        let kept = List.filter (fun i -> not (List.mem i expand_at)) axes in
        let squeezed = reshape_to src (List.map (fun i -> s_arr.(i)) kept) in
        let prepend =
          List.init n_left (fun i -> t_arr.(i))
          @ List.map (fun i -> t_arr.(n_left + i)) expand_at
        in
        let expanded = U.expand ~src:squeezed ~dims:(shape_arg prepend) in
        let idx_in lst x =
          let rec go k = function
            | [] -> raise Not_found
            | y :: _ when y = x -> k
            | _ :: tl -> go (k + 1) tl
          in
          go 0 lst
        in
        let n_expand = List.length expand_at in
        let order =
          List.init n_left Fun.id
          @ List.map
              (fun i ->
                n_left
                + (if List.mem i expand_at then idx_in expand_at i
                   else n_expand + idx_in kept i))
              axes
        in
        if order = List.init rank Fun.id then expanded
        else U.permute ~src:expanded ~order

(* Value-lane INDEX selecting lane [i] of a stacked value. *)
let lane src i = U.index ~ptr:src ~idxs:[ int_ i ] ()

(* Env-derived decomposition toggles, read once at module init. *)
let debug = Helpers.getenv "DEBUG" 0
let disable_fast_idiv = Helpers.getenv "DISABLE_FAST_IDIV" 1 <> 0
let transcendental_env = Helpers.getenv "TRANSCENDENTAL" 1
let spec_enabled () = Helpers.getenv "SPEC" 0 <> 0

(* Matcher aliases. [sym]/[symbolic]/[symbolic_simple] each already append
   the movement-op cleanup, so it is never re-appended on top of them. *)
let sym = Symbolic.sym
let symbolic = Symbolic.symbolic
let symbolic_simple = Symbolic.symbolic_simple
let mop_cleanup = Movement.mop_cleanup

(* Movement-op / index pushing rule shared with the scheduler. *)
let pm_mops = Rangeify.movement_ops

(* pm_no_index: index dtype only exists before the renderer; concretise the
   remaining index-typed ALU/CONST to int32, and drop index casts. *)
let pm_no_index =
  let open Upat in
  PM.make
    [
      ops ~name:"x" ~dtype:Dtype.index (Ops.Const :: Ops.Group.alu)
      => (fun bs -> Some (U.replace (bs $ "x") ~dtype:Dtype.int32 ()));
      op ~name:"c" ~dtype:Dtype.index ~src:[ var "x" ] Ops.Cast
      => (fun bs -> Some (U.cast ~src:(bs $ "x") ~dtype:Dtype.int32));
    ]

(* expander (expand_rewrite) *)

let build_range_map sink =
  let ctx = Hashtbl.create 8 in
  U.toposort sink
  |> List.iter (fun node ->
         match U.as_range node with
         | Some { axis; kind = Axis_type.Unroll | Axis_type.Upcast; _ } ->
             if not (Hashtbl.mem ctx axis) then
               Hashtbl.add ctx axis (Hashtbl.length ctx)
         | Some _ | None -> ());
  ctx

(* Front-permuted reduce with [num_axes] leading horizontal axes plus the loop
   [ranges]: a reduce over the ranges whose reduce-arg carries [num_axes]. *)
let reduce_with_num_axes ~src ~ranges ~op ~num_axes ~dtype =
  U.replace
    (U.reduce ~src ~ranges ~op ~dtype)
    ~arg:(U.Arg.Reduce_arg { op; num_axes })
    ()

let expand_reduce node =
  match U.as_reduce node with
  | Some { src; ranges; op; num_axes = 0 } ->
      let range_srcs = ref [] and new_axes = ref [] in
      List.iter
        (fun u ->
          if U.op u = Ops.Range then range_srcs := u :: !range_srcs
          else
            U.shape u
            |> List.iteri (fun i s -> if U.vmax s > 1 then new_axes := i :: !new_axes))
        ranges;
      let new_axes = List.rev !new_axes in
      if new_axes = [] then None
      else
        let src_shape = U.shape src in
        let rank = List.length src_shape in
        let perm =
          new_axes
          @ (List.init rank Fun.id |> List.filter (fun i -> not (List.mem i new_axes)))
        in
        let out_shape =
          List.mapi
            (fun i s -> if List.mem i new_axes then U.const_int 1 else s)
            src_shape
        in
        Some
          (U.reshape
             ~src:
               (reduce_with_num_axes ~src:(U.permute ~src ~order:perm)
                  ~ranges:(List.rev !range_srcs) ~op
                  ~num_axes:(List.length new_axes) ~dtype:(U.dtype node))
             ~shape:(shape_arg out_shape))
  | Some _ | None -> None

let contract_axis range_map u axes =
  let permute_tail = List.map (fun (rn, _) -> Hashtbl.find range_map rn) axes in
  let shape = U.shape u in
  let permute_head =
    List.init (List.length shape) Fun.id
    |> List.filter (fun i -> not (List.mem i permute_tail))
  in
  let out = U.permute ~src:u ~order:(permute_head @ permute_tail) in
  let out_shape = U.max_shape out in
  let head_len = List.length permute_head in
  let head_shape = List.filteri (fun i _ -> i < head_len) out_shape in
  let tail = prod out_shape / prod head_shape in
  U.reshape ~src:out ~shape:(shape_arg_of_ints (head_shape @ [ tail ]))

let unroll_axis range_map u axes =
  let permute_tail = List.map (fun (rn, _) -> Hashtbl.find range_map rn) axes in
  let shape = U.shape u in
  let prefix =
    match List.rev shape with [] -> [] | _last :: rest -> List.rev rest
  in
  let out =
    U.reshape ~src:u
      ~shape:(shape_arg (prefix @ List.map (fun (_, size) -> U.const_int size) axes))
  in
  let out_shape = U.shape out in
  let permute_head =
    List.init (List.length out_shape) Fun.id
    |> List.filter (fun i -> not (List.mem i permute_tail))
  in
  U.permute ~src:out ~order:(argsort (permute_head @ permute_tail))

let expand_wmma range_map node =
  match U.as_wmma node with
  | Some { a; b; c; info } when U.node_tag node = Some "1" ->
      let in0, in1, out0 = info.upcast_axes in
      let wmma =
        U.replace node
          ~src:[| contract_axis range_map a in0; contract_axis range_map b in1; c |]
          ~node_tag:None ()
      in
      Some (unroll_axis range_map wmma out0)
  | Some _ | None -> None

let expander2 range_map =
  let open Upat in
  PM.make
    [
      op ~name:"r" Ops.Reduce => (fun bs -> expand_reduce (bs $ "r"));
      ( op ~name:"r" Ops.Range
      => fun bs ->
           let r = bs $ "r" in
           match U.as_range r with
           | Some { axis; _ } when Hashtbl.mem range_map axis ->
               let idx = Hashtbl.find range_map axis in
               let n = U.vmax r + 1 in
               let dtype = U.dtype r in
               let dims =
                 List.init (Hashtbl.length range_map) (fun i ->
                     U.const_int (if i = idx then n else 1))
               in
               Some
                 (U.reshape
                    ~src:(U.stack ~dtype (List.init n (fun i -> U.const (Const.int dtype i))))
                    ~shape:(shape_arg dims))
           | Some _ | None -> None );
      op ~name:"u" Ops.Wmma => (fun bs -> expand_wmma range_map (bs $ "u"));
    ]

(* unbroadcast *)

(* Fold [ADD(Wmma, x)] into the Wmma accumulator, pushing an intervening
   permute or reshape to the other side of the add. *)
let wmma_add =
  let open Upat in
  PM.make
    [
      ( alu [ op ~name:"wmma" Ops.Wmma; var "add" ] Ops.Add => fun bs ->
        let wmma = bs $ "wmma" and add = bs $ "add" in
        match U.as_wmma wmma with
        | Some v -> Some (U.replace wmma ~src:[| v.a; v.b; U.O.(v.c + add) |] ())
        | None -> None );
      ( alu
          [ op ~name:"permute" ~src:[ op ~name:"wmma" Ops.Wmma ] Ops.Permute; var "add" ]
          Ops.Add
      => fun bs ->
        let permute = bs $ "permute" and wmma = bs $ "wmma" and add = bs $ "add" in
        match U.marg permute with
        | U.Marg_permute order ->
            let pushed = U.O.(wmma + U.permute ~src:add ~order:(argsort order)) in
            Some (U.permute ~src:pushed ~order)
        | _ -> None );
      ( alu
          [
            op ~name:"permute"
              ~src:[ op ~name:"reshape" ~src:[ op ~name:"wmma" Ops.Wmma; any ] Ops.Reshape ]
              Ops.Permute;
            var "add";
          ]
          Ops.Add
      => fun bs ->
        let permute = bs $ "permute"
        and reshape = bs $ "reshape"
        and wmma = bs $ "wmma"
        and add = bs $ "add" in
        match U.marg permute with
        | U.Marg_permute order ->
            let rearranged =
              U.reshape
                ~src:(U.permute ~src:add ~order:(argsort order))
                ~shape:(shape_arg (U.shape wmma))
            in
            let pushed = U.O.(wmma + rearranged) in
            Some
              (U.permute
                 ~src:(U.reshape ~src:pushed ~shape:(shape_arg (U.shape reshape)))
                 ~order)
        | _ -> None );
    ]

(* Broadcast every operand of a binary / ternary / store node to a common
   shape. STORE's destination is an INDEX and carries a shape like any other
   source, so it broadcasts through the same uniform path. *)
let broadcast_binary node =
  let srcs = Array.to_list (U.src node) in
  let shapes = List.map shape_opt srcs in
  if List.exists Option.is_none shapes || all_same (List.filter_map Fun.id shapes)
  then None
  else
    let shapes = List.map Option.get shapes in
    let target = U.broadcast_shape shapes in
    let rank = List.length target in
    let src' =
      List.map2
        (fun u s -> broadcast_to (reshape_to u (align_left rank s)) target)
        srcs shapes
    in
    if List.for_all2 U.equal srcs src' then None
    else Some (U.replace node ~src:(Array.of_list src') ())

(* Broadcast then devectorise WMMA operands (all but the lane axis). *)
let broadcast_and_devec_wmma node =
  match U.as_wmma node with
  | None -> None
  | Some _ -> (
      let srcs = Array.to_list (U.src node) in
      let prefix_of u =
        match List.rev (U.shape u) with [] -> [] | _lane :: pre -> List.rev pre
      in
      let prefixes = List.map prefix_of srcs in
      if all_same prefixes then None
      else
        let target = U.broadcast_shape prefixes in
        let rank = List.length target in
        let normalized =
          List.map
            (fun u ->
              match List.rev (U.shape u) with
              | [] -> u
              | lane :: _ ->
                  let pre = prefix_of u in
                  broadcast_to
                    (reshape_to u (align_left rank pre @ [ lane ]))
                    (target @ [ lane ]))
            srcs
        in
        let rec product = function
          | [] -> [ [] ]
          | dim :: dims ->
              let tail = product dims in
              List.concat
                (List.init (U.vmax dim) (fun i ->
                     List.map (fun idx -> int_ i :: idx) tail))
        in
        let lanes =
          List.map
            (fun idxs ->
              U.replace node
                ~src:(Array.of_list (List.map (fun u -> U.index ~ptr:u ~idxs ()) normalized))
                ())
            (product target)
        in
        Some (U.reshape ~src:(U.stack lanes) ~shape:(shape_arg (U.shape node))))

let unbroadcast =
  let open Upat in
  PM.compose
    [
      wmma_add;
      PM.make
        [
          ops ~name:"x" (Ops.Store :: Ops.Group.broadcastable)
          => (fun bs -> broadcast_binary (bs $ "x"));
          op ~name:"b" Ops.Wmma => (fun bs -> broadcast_and_devec_wmma (bs $ "b"));
        ];
    ]

(* devectorizer2 *)

(* Scalarise a shaped node lane by lane, then re-stack. *)
let do_devectorize node =
  match U.shape_opt node with
  | None | Some [] -> None
  | Some _ ->
      let src_shapes = Array.to_list (U.src node) |> List.map U.shape_opt in
      if List.exists Option.is_none src_shapes
         || not (all_same (List.filter_map Fun.id src_shapes))
      then None
      else
        let shape = U.max_shape node in
        let rec product = function
          | [] -> [ [] ]
          | n :: ns ->
              List.concat_map
                (fun i -> List.map (fun tail -> int_ i :: tail) (product ns))
                (List.init n Fun.id)
        in
        let lanes =
          product shape
          |> List.map (fun idxs ->
                 U.replace node
                   ~src:(Array.map (fun s -> U.index ~ptr:s ~idxs ()) (U.src node))
                   ())
        in
        if U.op node = Ops.Store then Some (U.group lanes)
        else Some (U.reshape ~src:(U.stack lanes) ~shape:(shape_arg (U.shape node)))

(* Unpack WMMA: replace each non-STACK source with an explicit STACK of its
   scalar elements. *)
let do_stack_wmma node =
  let srcs = U.src node in
  if Array.for_all (fun x -> U.op x = Ops.Stack || U.op x = Ops.Wmma) srcs then None
  else begin
    assert (List.length (U.shape node) = 1);
    let stacked b =
      if U.op b = Ops.Stack then b
      else U.stack (List.init (prod (U.max_shape b)) (lane b))
    in
    Some (U.replace node ~src:(Array.map stacked srcs) ())
  end

(* Elementwise-only devectorizer (the "add images" pass). *)
let ew_devectorizer =
  let open Upat in
  PM.make
    [ ops ~name:"b" Ops.Group.elementwise => (fun bs -> do_devectorize (bs $ "b")) ]

(* Local devectorizer rules; the driver prefixes movement-op cleanup and the
   movement-op pushing rule ahead of these. *)
let devectorizer2 =
  let open Upat in
  PM.make
    [
      (* unpack broadcasting *)
      ops ~name:"b" (Ops.Group.elementwise @ [ Ops.Load; Ops.Store ])
      => (fun bs -> do_devectorize (bs $ "b"));
          (* INDEX without src is nothing *)
          op ~name:"x" ~src:[ var "x0" ] Ops.Index => (fun bs -> Some (bs $ "x0"));
          (* unpack WMMA *)
          op ~name:"u" Ops.Wmma => (fun bs -> do_stack_wmma (bs $ "u"));
          (* stacked INDEX is many INDEX *)
          ( op ~name:"idx"
              ~src:[ ops ~name:"b" [ Ops.Param; Ops.Buffer ]; op ~name:"s" Ops.Stack ]
              Ops.Index
          => fun bs ->
            let b = bs $ "b" and s = bs $ "s" in
            Some (U.stack (Array.to_list (U.src s) |> List.map (fun u -> U.index ~ptr:b ~idxs:[ u ] ()))) );
          (* INDEX into RESHAPE moves the RESHAPE *)
          ( op ~name:"idx"
              ~src:[ ops ~name:"b" [ Ops.Param; Ops.Buffer ]; op ~name:"s" Ops.Reshape ]
              Ops.Index
          => fun bs ->
            let b = bs $ "b" and s = bs $ "s" in
            match U.src s with
            | [| inner; shape |] -> Some (U.reshape ~src:(U.index ~ptr:b ~idxs:[ inner ] ()) ~shape)
            | _ -> None );
          (* RESHAPE a void is removed (hack for AFTER) *)
          ( op ~name:"x" ~dtype:Dtype.void Ops.Reshape
          => fun bs -> Some (U.src (bs $ "x")).(0) );
          (* reshape of a single-element shaped value to scalar is an index *)
          ( op ~name:"x" Ops.Reshape => fun bs ->
            let x = bs $ "x" in
            match U.src x with
            | [| src; _ |] when U.shape x = [] && U.max_shape src = [ 1 ] ->
                Some (lane src 0)
            | _ -> None );
          (* EXPAND on scalar -> STACK *)
          ( op ~name:"out" ~src:[ var "x"; any ] Ops.Expand => fun bs ->
            let x = bs $ "x" and out = bs $ "out" in
            let n = prod (U.max_shape out) in
            if U.shape x = [] && U.max_shape out = [ n ] then
              Some (U.stack (List.init n (fun _ -> x)))
            else None );
          (* INDEX on INDEX is INDEX *)
          ( op ~name:"idx2" ~src:[ op ~name:"idx1" ~allow_any_len:true Ops.Index ]
              ~allow_any_len:true Ops.Index
          => fun bs ->
            let idx1 = bs $ "idx1" and idx2 = bs $ "idx2" in
            let tail u = Array.sub (U.src u) 1 (Array.length (U.src u) - 1) |> Array.to_list in
            Some (U.index ~ptr:(U.src idx1).(0) ~idxs:(tail idx1 @ tail idx2) ()) );
        ]

(* remove reduces *)

let identity_element op dtype =
  match op with
  | Ops.Add -> Const.zero dtype
  | Ops.Mul -> Const.one dtype
  | Ops.Max -> Const.min_value dtype
  | _ -> invalid_arg "identity_element: unsupported reduce op"

let reduce_fold op = function
  | [] -> invalid_arg "reduce_fold: empty list"
  | first :: rest ->
      List.fold_left (fun a x -> U.alu_binary ~op ~lhs:a ~rhs:x) first rest

(* A storage placeholder. GLOBAL uses a PARAM, LOCAL and REG use a BUFFER; the
   flat storage carries [prod shape] elements and a multi-dim view is
   reintroduced by reshape. *)
let placeholder ~shape ~dtype ~slot ?(addrspace = Dtype.Global) () =
  let flat = shape_arg_of_ints [ prod shape ] in
  let base =
    match addrspace with
    | Dtype.Global -> U.param ~slot ~dtype ~shape:flat ~addrspace ()
    | Dtype.Local | Dtype.Reg -> U.buffer ~slot ~dtype ~shape:flat ~addrspace ()
    | Dtype.Alu -> invalid_arg "placeholder: alu address space"
  in
  if List.length shape > 1 then U.reshape ~src:base ~shape:(shape_arg_of_ints shape)
  else base

let placeholder_like node ~slot ~addrspace =
  placeholder ~shape:(U.max_shard_shape node) ~dtype:(U.dtype node) ~slot ~addrspace ()

(* fix group for reduce: split grouped reduces into a local buffer written by the
   non-grouped reduces, then a final reduce over the group loops. *)
let range_kind_is kind r =
  match U.as_range r with Some v -> Axis_type.equal v.kind kind | None -> false

let clone_group_reduce_range r =
  match U.as_range r with
  | Some v ->
      U.range ~size:v.size ~axis:(v.axis + 100) ~kind:Axis_type.Reduce ~sub:v.sub
        ~dtype:(U.dtype r) ~parents:v.parents ()
  | None -> invalid_arg "clone_group_reduce_range: expected RANGE"

let fix_group_for_reduce_rule node =
  match U.as_reduce node with
  | None -> None
  | Some v ->
      let group_reduce, reduce_ranges =
        List.partition (range_kind_is Axis_type.Group_reduce) v.ranges
      in
      if group_reduce = [] then None
      else
        let upstream_locals =
          U.toposort node |> List.filter (range_kind_is Axis_type.Local)
        in
        let partial = U.replace node ~src:(Array.of_list (v.src :: reduce_ranges)) () in
        let reduce_loop = List.map clone_group_reduce_range group_reduce in
        let opts : U.stage_opts =
          { device = None; addrspace = Dtype.Local; removable = false }
        in
        let local =
          U.stage ~src:partial ~ranges:(upstream_locals @ group_reduce) ~opts
        in
        let indexed = U.index ~ptr:local ~idxs:(upstream_locals @ reduce_loop) () in
        Some
          (reduce_with_num_axes ~src:indexed ~ranges:reduce_loop ~op:v.op ~num_axes:0
             ~dtype:(U.dtype node))

let fix_group_for_reduce =
  let open Upat in
  op ~name:"reduce" Ops.Reduce => fun bs -> fix_group_for_reduce_rule (bs $ "reduce")

type reduce_ctx = { mutable acc_num : int; acc_slots : int U.Ref_tbl.t }

(* Precompute a deterministic accumulator slot per reduce, numbered in the
   reference graph-rewrite visitation order so rendered accumulator buffer
   names (acc0, acc1, ...) stay byte-identical regardless of local rewrite
   order. *)
let reduce_slots_in_tinygrad_order root =
  let replaced = U.Ref_tbl.create 128 in
  let on_stack = U.Ref_tbl.create 128 in
  let waitlist = U.Ref_tbl.create 32 in
  let order = ref [] in
  let stack = ref [ (root, 0) ] in
  U.Ref_tbl.replace on_stack root ();
  let push_wait dep item =
    let items =
      match U.Ref_tbl.find_opt waitlist dep with
      | Some items -> item :: items
      | None -> [ item ]
    in
    U.Ref_tbl.replace waitlist dep items
  in
  let release_waiters n =
    match U.Ref_tbl.find_opt waitlist n with
    | None -> ()
    | Some items ->
        U.Ref_tbl.remove waitlist n;
        stack := items @ !stack
  in
  let rec first_missing srcs i =
    if i = Array.length srcs then None
    else if U.Ref_tbl.mem replaced srcs.(i) then first_missing srcs (i + 1)
    else Some srcs.(i)
  in
  while !stack <> [] do
    match !stack with
    | [] -> ()
    | (node, stage) :: rest ->
        stack := rest;
        if not (U.Ref_tbl.mem replaced node) then
          if stage = 0 then begin
            stack := (node, 1) :: !stack;
            let srcs = U.src node in
            let enter =
              match U.op node with Ops.Call | Ops.Function -> false | _ -> true
            in
            for i = Array.length srcs - 1 downto 0 do
              let child = srcs.(i) in
              if (i <> 0 || enter) && not (U.Ref_tbl.mem on_stack child) then begin
                stack := (child, 0) :: !stack;
                U.Ref_tbl.replace on_stack child ()
              end
            done
          end
          else
            let srcs = U.src node in
            match first_missing srcs 0 with
            | Some dep -> push_wait dep (node, 1)
            | None ->
                (match U.as_reduce node with
                | Some { ranges = _ :: _; _ } -> order := node :: !order
                | Some { ranges = []; _ } | None -> ());
                U.Ref_tbl.replace replaced node ();
                release_waiters node
  done;
  let slots = U.Ref_tbl.create 16 in
  List.rev !order |> List.iteri (fun slot node -> U.Ref_tbl.replace slots node slot);
  slots

(* Ranges live in the reduce body but not in the reduce loops or already ended:
   these must be sequenced before the accumulator init. *)
let reduce_input_ranges src reduce_range =
  let rec ended_ranges u =
    let children = U.src u in
    match U.op u with
    | Ops.After ->
        let ret = ref [] in
        for i = 1 to Array.length children - 1 do
          ret := ended_ranges children.(i) @ !ret
        done;
        !ret
    | _ ->
        let start =
          match U.op u with
          | Ops.Stage | Ops.Reduce | Ops.End | Ops.Call | Ops.Function | Ops.Copy ->
              Some 1
          | Ops.Slice -> Some 2
          | Ops.Wmma -> Some 3
          | Ops.Linear -> Some 0
          | _ -> None
        in
        (match start with
        | None -> []
        | Some k ->
            let ret = ref [] in
            for i = k to Array.length children - 1 do
              ret := children.(i) :: !ret
            done;
            !ret)
  in
  let topo = U.toposort src in
  let ended = U.Ref_tbl.create 16 in
  List.iter
    (fun n -> List.iter (fun r -> U.Ref_tbl.replace ended r ()) (ended_ranges n))
    topo;
  let reduce_set = U.Ref_tbl.create 8 in
  List.iter (fun r -> U.Ref_tbl.replace reduce_set r ()) reduce_range;
  List.filter
    (fun n ->
      U.op n = Ops.Range
      && (not (U.Ref_tbl.mem reduce_set n))
      && not (U.Ref_tbl.mem ended n))
    topo

let reduce_ranges_to_acc ctx node =
  match U.as_reduce node with
  | Some { src; ranges = _ :: _ as reduce_range; op; num_axes } ->
      let dtype = U.dtype node in
      let slot =
        match U.Ref_tbl.find_opt ctx.acc_slots node with
        | Some slot ->
            ctx.acc_num <- max ctx.acc_num (slot + 1);
            slot
        | None ->
            let slot = ctx.acc_num in
            ctx.acc_num <- ctx.acc_num + 1;
            slot
      in
      let acc = placeholder_like node ~slot ~addrspace:Dtype.Reg in
      let input_ranges = reduce_input_ranges src reduce_range in
      let acc_init =
        U.store ~dst:(U.after ~src:acc ~deps:input_ranges)
          ~value:(U.const (identity_element op dtype)) ()
      in
      let acc_initted = U.after ~src:acc ~deps:(acc_init :: reduce_range) in
      let inp =
        if num_axes <> 0 then
          reduce_with_num_axes ~src ~ranges:[] ~op ~num_axes ~dtype
        else src
      in
      let acc_out =
        U.with_tag "mergeable"
          (U.end_
             ~value:
               (U.store ~dst:acc_initted
                  ~value:(U.alu_binary ~op ~lhs:acc_initted ~rhs:inp) ())
             ~ranges:reduce_range)
      in
      Some (U.after ~src:acc ~deps:[ acc_out ])
  | _ -> None

let expand_horizontal_reduce node =
  match U.as_reduce node with
  | Some { src; ranges = []; op; num_axes } when num_axes > 0 ->
      let mshape = U.max_shape src in
      let rec product = function
        | 0 -> [ [] ]
        | n ->
            let dim = List.nth mshape (num_axes - n) in
            List.concat
              (List.init dim (fun i -> List.map (fun idx -> int_ i :: idx) (product (n - 1))))
      in
      let vals = List.map (fun idxs -> U.index ~ptr:src ~idxs ()) (product num_axes) in
      Some (reduce_fold op vals)
  | Some _ | None -> None

(* Merge [End] nodes sharing the same ranges and nesting scope (created by
   [reduce_ranges_to_acc], carrying the [mergeable] tag). Ends at different
   nesting depths get cloned ranges so each range maps to one end. *)
let merge_reduce_ends_rule node =
  match U.op node with
  | Ops.Sink ->
      let topo = U.toposort node in
      let range_groups = ref [] in
      let same_ranges a b = List.length a = List.length b && List.for_all2 U.equal a b in
      let add_end ranges e =
        match
          List.find_opt (fun (ranges', _) -> same_ranges ranges ranges') !range_groups
        with
        | Some (_, ends) -> ends := e :: !ends
        | None -> range_groups := !range_groups @ [ (ranges, ref [ e ]) ]
      in
      List.iter
        (fun n ->
          match U.as_end n with
          | Some { ranges; _ } when U.node_tag n = Some "mergeable" -> add_end ranges n
          | _ -> ())
        (U.backward_slice node);
      let next_axis =
        List.fold_left
          (fun acc n ->
            match U.as_range n with Some { axis; _ } -> max acc (axis + 1) | None -> acc)
          0 topo
        |> ref
      in
      let clone_ranges ranges =
        let base = !next_axis in
        next_axis := !next_axis + List.length ranges;
        List.mapi
          (fun j r ->
            match U.as_range r with
            | Some { size; parents; sub; kind; _ } ->
                U.range ~size ~axis:(base + j) ~kind ~sub ~dtype:(U.dtype r) ~parents ()
            | None -> r)
          ranges
      in
      let mappings =
        List.fold_left
          (fun acc (ranges, ends_ref) ->
            let ends = List.rev !ends_ref in
            if List.length ends <= 1 then acc
            else
              let scope_equal a b =
                let mem x xs = List.exists (U.equal x) xs in
                List.length a = List.length b && List.for_all (fun x -> mem x b) a
              in
              let groups = ref [] in
              List.iter
                (fun e ->
                  let scope = U.ranges e in
                  match
                    List.find_opt (fun (scope', _) -> scope_equal scope scope') !groups
                  with
                  | Some (_, group) -> group := e :: !group
                  | None -> groups := !groups @ [ (scope, ref [ e ]) ])
                ends;
              let groups = List.map (fun (_, group) -> List.rev !group) !groups in
              let _, acc =
                List.fold_left
                  (fun (group_idx, acc) group ->
                    let ranges', mapped_group =
                      if group_idx = 0 then (ranges, group)
                      else
                        let ranges' = clone_ranges ranges in
                        let subs = List.combine ranges ranges' in
                        (ranges', List.map (fun e -> U.substitute subs e) group)
                    in
                    let merged =
                      match mapped_group with
                      | [ e ] -> e
                      | _ ->
                          let stores =
                            List.map
                              (fun e ->
                                match U.as_end e with
                                | Some { value; _ } -> value
                                | None -> assert false)
                              mapped_group
                          in
                          U.end_ ~value:(U.group stores) ~ranges:ranges'
                    in
                    let acc =
                      List.fold_left
                        (fun acc old ->
                          if
                            old == merged
                            || List.exists (( == ) old) (U.backward_slice merged)
                          then acc
                          else (old, merged) :: acc)
                        acc group
                    in
                    (group_idx + 1, acc))
                  (0, acc) groups
              in
              acc)
          [] !range_groups
      in
      (match mappings with [] -> None | _ -> Some (U.substitute mappings node))
  | _ -> None

let merge_reduce_ends =
  let open Upat in
  op ~name:"sink" Ops.Sink => fun bs -> merge_reduce_ends_rule (bs $ "sink")

(* GROUP/SINK cleanup: a single-source GROUP is its child, and SINK/GROUP nodes
   absorb NOOP/STACK/SINK/GROUP children. Appended to the reduce lowering at the
   upstream position; ownership mirrors the symbolic layer's import of it. *)
let clean_up_group_sink node =
  let remove_like u =
    match U.op u with
    | Ops.Noop | Ops.Stack | Ops.Sink | Ops.Group -> true
    | _ -> false
  in
  match U.op node, U.src node with
  | Ops.Group, [| x |] -> Some x
  | (Ops.Sink | Ops.Group), srcs when Array.exists remove_like srcs ->
      Some
        (U.replace node
           ~src:
             (Array.of_list
                (List.concat_map
                   (fun u ->
                     if remove_like u then Array.to_list (U.src u) else [ u ])
                   (Array.to_list srcs)))
           ())
  | _ -> None

let pm_clean_up_group_sink =
  let open Upat in
  PM.make
    [
      ops ~name:"n" [ Ops.Sink; Ops.Group ]
      => (fun bs -> clean_up_group_sink (bs $ "n"));
    ]

let pm_reduce_local ctx =
  let open Upat in
  PM.compose
    [
      wmma_add;
      PM.make
        [
          fix_group_for_reduce;
          op ~name:"r" ~allow_any_len:true Ops.Reduce
          => (fun bs -> reduce_ranges_to_acc ctx (bs $ "r"));
          op ~name:"r" Ops.Reduce => (fun bs -> expand_horizontal_reduce (bs $ "r"));
          merge_reduce_ends;
        ];
      pm_clean_up_group_sink;
    ]

let pm_reduce root =
  let ctx = { acc_num = 0; acc_slots = reduce_slots_in_tinygrad_order root } in
  U.graph_rewrite ~name:"remove reduces"
    (U.first_match [ PM.rewrite mop_cleanup; PM.rewrite (pm_reduce_local ctx) ])
    root

(* add loads *)

let maybe_load src =
  match U.addrspace src with
  | Some (Dtype.Global | Dtype.Local | Dtype.Reg) -> U.load ~src ()
  | Some Dtype.Alu | None -> src

let pm_add_loads =
  let open Upat in
  PM.make
    [
      ops ~name:"x" (Ops.Group.elementwise @ [ Ops.Reduce; Ops.Wmma; Ops.Stack ])
      => (fun bs ->
           let x = bs $ "x" in
           Some (U.replace x ~src:(Array.map maybe_load (U.src x)) ()));
      ( op ~name:"x" Ops.Store => fun bs ->
        let x = bs $ "x" in
        match U.as_store x with
        | Some { dst; value; gate } ->
            let value' = maybe_load value in
            if U.equal value value' then None
            else Some (U.store ~dst ~value:value' ?gate ())
        | None -> None );
    ]

(* add local buffers *)

let add_local_buffer_rule counter node =
  match U.as_stage node with
  | None -> None
  | Some { src; ranges; opts } ->
      let slot = !counter in
      incr counter;
      let buf =
        placeholder ~shape:(U.max_shape node) ~dtype:(U.dtype node) ~slot
          ~addrspace:opts.addrspace ()
      in
      let store = U.store ~dst:(U.index ~ptr:buf ~idxs:ranges ()) ~value:src () in
      Some
        (U.after ~src:buf
           ~deps:[ U.barrier ~srcs:[ U.end_ ~value:store ~ranges ] () ])

let pm_add_local_buffers counter root =
  U.graph_rewrite ~name:"add local buffers"
    (U.first_match [ add_local_buffer_rule counter; pm_mops ])
    root

(* cast float alu operands *)

let pm_cast_float_alu =
  let open Upat in
  PM.make
    [
      ( ops ~name:"u"
          ~src:[ var "x" ]
          [ Ops.Sin; Ops.Log2; Ops.Exp2; Ops.Sqrt; Ops.Reciprocal ]
      => fun bs ->
        let u = bs $ "u" and x = bs $ "x" in
        if Dtype.equal (U.dtype x) (U.dtype u) then None
        else Some (U.replace u ~src:[| U.cast ~src:x ~dtype:(U.dtype u) |] ()) );
    ]

(* number params *)

let number_params sink =
  let next_slot =
    U.toposort sink
    |> List.fold_left
         (fun acc node ->
           match U.as_param node with
           | Some { param = { slot; _ }; _ } when slot >= 0 -> acc + 1
           | _ -> acc)
         0
    |> ref
  in
  let rewrite_param node =
    match U.as_param node with
    | Some { param; _ } when param.slot = -1 ->
        let slot = !next_slot in
        incr next_slot;
        Some (U.replace node ~arg:(U.Arg.Param_arg { param with slot }) ())
    | _ -> None
  in
  U.graph_rewrite ~name:"number params with -1" ~walk:true rewrite_param sink

(* Stamp renderer capabilities with env-derived decomposition toggles. *)
let supported_ops_of (ren : Renderer.t) : Decomp_op.supported_ops =
  let ir = Renderer.supported_ops ren in
  {
    ir with
    is_metal = Renderer.name ren = "metal";
    supports_dtype = Renderer.supports_dtype ren;
    disable_fast_idiv;
    force_transcendental = transcendental_env >= 2;
  }

(* Lower an optimized kernel AST to a form ready for linearization. Mirrors
   [full_rewrite_to_sink] after the [apply_opts] call. *)
let lower (ren : Renderer.t) (sink : U.t) : U.t =
  let rewrite ?name ?bottom_up rule = U.graph_rewrite ?name ?bottom_up rule in
  let pm pm' = PM.rewrite pm' in

  (* postopt symbolic: [sym + pm_move_where_on_load + pm_flatten_range]. *)
  let sink =
    rewrite ~name:"postopt symbolic"
      (U.first_match
         [ pm PM.(sym ++ Symbolic.pm_move_where_on_load); Simplify.flatten_range ])
      sink
  in

  (* expander: [expander2 = expand rules + pm_flatten_range + mop_cleanup]. *)
  let range_map = build_range_map sink in
  let sink =
    rewrite ~name:"expander"
      (U.first_match
         [ pm (expander2 range_map); Simplify.flatten_range; pm mop_cleanup ])
      sink
  in

  (* remove reduces: [mop_cleanup + pm_reduce_local]. *)
  let sink = pm_reduce sink in

  (* add local buffers: [pm_add_local_buffers = add_local_buffer + pm_mops]. *)
  let sink = pm_add_local_buffers (ref 0) sink in

  (* add gpu dims: [pm_add_gpudims]. *)
  let sink = Gpudims.pm_add_gpudims ren sink in

  (* unbroadcast / add loads: [symbolic_simple + unbroadcast + pm_add_loads]. *)
  let sink =
    rewrite ~name:"unbroadcast / add loads"
      (pm PM.(symbolic_simple ++ unbroadcast ++ pm_add_loads))
      sink
  in

  (* devectorize:
     [symbolic_simple + (mop_cleanup + pm_mops + devectorizer2 rules)]. *)
  let sink =
    rewrite ~name:"devectorize2"
      (U.first_match
         [ pm symbolic_simple; pm mop_cleanup; pm_mops; pm devectorizer2 ])
      sink
  in

  (* simplify load/store indexing: [indexing_simplify]. *)
  let sink =
    rewrite ~name:"simplify load/store indexing" (pm Coalese.indexing_simplify) sink
  in

  (* some coalesing misses without this: [sym]. *)
  let sink = rewrite ~name:"early symbolic" (pm sym) sink in

  (* do memory coalesing (late). *)
  let sink = Coalese.memory_coalesing ren sink in

  (* add images: [symbolic_simple + ew_devectorizer + pm_simplify_add_image]. *)
  let sink =
    rewrite ~name:"add images" ~bottom_up:true
      (pm PM.(symbolic_simple ++ ew_devectorizer ++ Coalese.pm_simplify_add_image ren))
      sink
  in

  (* extra symbolic before decomp: [sym]. *)
  let sink = rewrite ~name:"extra symbolic" (pm sym) sink in

  (* lower index dtype: [pm_lower_index_dtype + indexing_simplify]. *)
  let sink =
    rewrite ~name:"lower all index dtypes"
      (U.first_match
         [ pm Symbolic.pm_lower_index_dtype; pm Coalese.indexing_simplify ])
      sink
  in

  (* final symbolic before decomp: [symbolic]. *)
  let sink = rewrite ~name:"final symbolic" (pm symbolic) sink in

  (* cast float alu operands: [pm_cast_float_alu]. *)
  let sink = rewrite ~name:"cast float alu operands" (pm pm_cast_float_alu) sink in

  (* early decompositions: [symbolic_simple + get_simplifying_rewrite_patterns]. *)
  let ops = supported_ops_of ren in
  let pm_decomp =
    U.first_match
      [ pm symbolic_simple; Decomp_op.get_simplifying_rewrite_patterns ops ]
  in
  let sink = rewrite ~name:"early decompositions" pm_decomp sink in

  (* decomp dtypes: [pm_dtype_decomps]. *)
  let sink = Decomp_dtype.do_dtype_decomps ren sink in

  (* late decompositions:
     [pm_decomp + get_late_rewrite_patterns + get_transcendental_patterns]. *)
  let pm_decomp =
    U.first_match
      [
        pm_decomp;
        Decomp_op.get_late_rewrite_patterns ops;
        Decomp_transcendental.get_transcendental_patterns ops;
      ]
  in
  let sink = rewrite ~name:"late decompositions" pm_decomp sink in

  (* move gates from index: [pm_move_gates_from_index]. *)
  let sink = Gater.pm_move_gates_from_index sink in

  (* final rewrite:
     [pm_decomp + extra_matcher + pm_split_ends + pm_no_index + pm_remove_invalid]. *)
  let extra_matcher =
    match Renderer.extra_matcher ren with None -> fun _ -> None | Some m -> m
  in
  let sink =
    rewrite ~name:"final rewrite"
      (U.first_match
         [
           pm_decomp;
           extra_matcher;
           Linearizer.do_split_ends;
           pm pm_no_index;
           pm Symbolic.pm_remove_invalid;
         ])
      sink
  in

  (* add control flow: [pm_add_control_flow], bottom-up. *)
  let sink = Linearizer.pm_add_control_flow sink in

  (* put unnumbered variable PARAMs in slots. *)
  let sink = number_params sink in

  if spec_enabled () then Spec.type_verify Spec.program_spec sink;
  if debug >= 6 then print_string (Render.uops_to_string ~label:"lower" sink);
  sink
