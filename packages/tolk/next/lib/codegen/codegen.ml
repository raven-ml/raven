(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let pm = Pattern_matcher.v
let ( ++ ) = Pattern_matcher.append
let lift = Pattern_matcher.with_ctx
let ops l = Op.Set.of_list l
let setting = Helpers.Context_var.value
let is o u = Op.equal (op u) o
let upto n = List.init n Fun.id
let srcs u = List.tl (src u)
let without_last l = List.rev (List.tl (List.rev l))
let last l = List.hd (List.rev l)
let index_ints u idx = index u (List.map int idx)

let reduce_arg u =
  match arg u with
  | Reduce { op; num_axes } -> (op, num_axes)
  | _ -> assert false

let others dims n = List.filter (fun i -> not (List.mem i dims)) (upto n)

(* The tuples of [List.map upto sizes], the last position varying fastest. *)
let rec product = function
  | [] -> [ [] ]
  | n :: sizes ->
      let tails = product sizes in
      List.concat_map (fun i -> List.map (List.cons i) tails) (upto n)

let int_shape s =
  List.map
    (function
      | Int n -> n | Sym _ -> invalid_arg "a symbolic axis has no lanes")
    s

let pm_number_params =
  pm
    [
      rule_ctx (Upat.op Op.Param ~name:"x") (fun ctx m ->
          match arg (m "x") with
          | Param p when p.slot = -1 ->
              incr ctx;
              Some (replace (m "x") ~arg:(Param { p with slot = !ctx - 1 }))
          | _ -> None);
    ]

let build_range_map sink =
  let ctx = Hashtbl.create 8 in
  List.iter
    (fun x ->
      if is Op.Range x && List.mem (axis_type x) Axis_type.[ Unroll; Upcast ]
      then Hashtbl.replace ctx (axis_id x) (Hashtbl.length ctx))
    (toposort sink);
  ctx

let expand_reduce r =
  let range_srcs, others_srcs = List.partition (is Op.Range) (srcs r) in
  let wide u =
    List.concat
      (List.mapi
         (fun i s -> if Sint.(resolve (s > Int 1)) then [ i ] else [])
         (shape u))
  in
  match List.concat_map wide others_srcs with
  | [] -> None
  | new_axes ->
      let op, num_axes = reduce_arg r and x = nth r 0 in
      assert (num_axes = 0);
      (* permute so new_axes come to front, then reduce *)
      let perm = new_axes @ others new_axes (ndim x) in
      let out_shape =
        List.mapi
          (fun i s -> if List.mem i new_axes then Int 1 else s)
          (shape x)
      in
      let num_axes = List.length new_axes in
      let arg = Reduce { op; num_axes } in
      Some
        (reshape
           (v Op.Reduce ~src:(permute x perm :: range_srcs) ~arg)
           out_shape)

let contract_axis u dims =
  flatten ~start:(-List.length dims) (permute u (others dims (ndim u) @ dims))

let unroll_axis u dims sizes =
  let out = unflatten u (-1) (List.map (fun n -> Int n) sizes) in
  permute out (Helpers.argsort (others dims (ndim out) @ dims))

let expand_wmma ctx u =
  match arg u with
  | Wmma ({ upcast_axes = Some (a0, a1, a2); _ } as w) ->
      let axes l = List.map (fun (rn, _) -> Hashtbl.find ctx rn) l in
      let wmma =
        replace u
          ~src:
            [
              contract_axis (nth u 0) (axes a0);
              contract_axis (nth u 1) (axes a1);
              nth u 2;
            ]
          ~arg:(Wmma { w with upcast_axes = None })
      in
      Some (unroll_axis wmma (axes a2) (List.map snd a2))
  | _ -> None

let expand_range ctx r =
  Hashtbl.find_opt ctx (axis_id r)
  |> Option.map (fun axis ->
      let n = Dtype.Value.to_int (vmax r) + 1 in
      let c =
        consts ~dtype:(dtype r) (List.map (fun i -> `Int (Z.of_int i)) (upto n))
      in
      reshape c
        (List.map
           (fun i -> Int (if i = axis then n else 1))
           (upto (Hashtbl.length ctx))))

let expander =
  pm
    [
      rule (Upat.op Op.Reduce ~name:"r") (fun m -> expand_reduce (m "r"));
      rule_ctx (Upat.op Op.Range ~name:"r") (fun ctx m ->
          expand_range ctx (m "r"));
      rule_ctx (Upat.op Op.Wmma ~name:"u") (fun ctx m ->
          expand_wmma ctx (m "u"));
    ]
  ++ lift Simplify.pm_flatten_range
  ++ lift Movement.mop_cleanup

(* Broadcasting and devectorizing *)

let expand_broadcast x =
  match List.map shape_opt (src x) with
  | shapes when List.mem None shapes -> None
  | shapes ->
      let shapes = List.filter_map Fun.id shapes in
      if Helpers.all_same (List.equal Sint.equal) shapes then None
      else
        let s = broadcast_shape shapes in
        Some (replace x ~src:(List.map (fun u -> expand u s) (src x)))

let broadcast_and_devec_wmma b =
  let shapes = List.map (fun u -> without_last (shape u)) (src b) in
  if List.for_all (function [] -> true | _ -> false) shapes then None
  else
    let s = broadcast_shape shapes in
    let expanded =
      List.map (fun u -> expand u (s @ [ last (shape u) ])) (src b)
    in
    let lane idx =
      replace b ~src:(List.map (fun x -> index_ints x idx) expanded)
    in
    let lanes = product (int_shape (without_last (shape b))) in
    Some (reshape (stack (List.map lane lanes)) (shape b))

(* A sum's identity is the zero accumulator a WMMA starts from: the running sum
   replaces it, so no addition of +0. stays in the loop (D24). *)
let wmma_accumulate wmma add =
  let acc = nth wmma 2 in
  let lane s = if is Op.Cast s then nth s 0 else s in
  let zero s =
    is Op.Const s
    &&
    match value s with
    | #Dtype.value as v -> Dtype.Value.(v = of_int 0)
    | `Invalid -> false
  in
  let is_zero =
    is Op.Stack acc && List.for_all (fun s -> zero (lane s)) (src acc)
  in
  let acc = if is_zero then add else O.(acc + add) in
  v Op.Wmma ~arg:(arg wmma) ~src:[ nth wmma 0; nth wmma 1; acc ]

let pm_wmma_add =
  let wmma = Upat.op Op.Wmma ~name:"wmma" in
  let plus p = Upat.add p (Upat.var "add") in
  let permute_of p = Upat.op Op.Permute ~src:[ p ] ~name:"permute" in
  let order m =
    match marg (m "permute") with Permute o -> o | _ -> assert false
  in
  pm
    [
      rule (plus wmma) (fun m -> Some (wmma_accumulate (m "wmma") (m "add")));
      (* push permute/reshape to the other side of the add *)
      rule
        (plus (permute_of wmma))
        (fun m ->
          let o = order m in
          Some (permute O.(m "wmma" + permute (m "add") (Helpers.argsort o)) o));
      rule
        (plus
           (permute_of
              (Upat.op Op.Reshape ~src:[ wmma; Upat.wild ] ~name:"reshape")))
        (fun m ->
          let o = order m and w = m "wmma" in
          let add = reshape (permute (m "add") (Helpers.argsort o)) (shape w) in
          Some (permute (reshape O.(w + add) (shape (m "reshape"))) o));
    ]

let pm_expand_broadcast =
  pm_wmma_add
  ++ pm
       [
         rule
           (Upat.v
              ~op:
                (Op.Set.union Op.Set.binary
                   (Op.Set.union Op.Set.ternary (ops [ Op.Store ])))
              ~name:"x" ())
           (fun m -> expand_broadcast (m "x"));
         rule (Upat.op Op.Wmma ~name:"b") (fun m ->
             broadcast_and_devec_wmma (m "b"));
       ]

let do_devectorize b =
  let s = shape b in
  let invalid x = is_invalid (base x) in
  (* A scalar value stands for every lane: a fold may drop a source's width
     after broadcasting was unpacked (D59). *)
  let scalar x = shape_opt x = Some [] && addrspace x = Some Dtype.Alu in
  if
    s = []
    || not
         (List.for_all
            (fun x ->
              List.equal Sint.equal (shape x) s || invalid x || scalar x)
            (src b))
  then None
  else
    let lane idx =
      replace b
        ~src:
          (List.map
             (fun x ->
               if invalid x then base x
               else if scalar x then x
               else index_ints x idx)
             (src b))
    in
    let lanes = List.map lane (product (int_shape s)) in
    Some (if is Op.Store b then group lanes else reshape (stack lanes) s)

let do_stack_wmma u =
  if List.for_all (fun x -> is Op.Stack x || is Op.Wmma x) (src u) then None
  else begin
    assert (ndim u = 1);
    let unpack b =
      if is Op.Stack b then b
      else
        stack
          (List.map (index_ints b)
             (List.map (fun i -> [ i ]) (upto (max_numel b))))
    in
    Some (replace u ~src:(List.map unpack (src u)))
  end

let storage = ops [ Op.Param; Op.Buffer; Op.Alloc ]

let devectorizer2 =
  Prepare.pm_mops
  ++ pm
       [
         (* unpack broadcasting *)
         rule
           (Upat.v
              ~op:(Op.Set.union Op.Set.elementwise (ops [ Op.Load; Op.Store ]))
              ~name:"b" ())
           (fun m -> do_devectorize (m "b"));
         (* INDEX without src is nothing *)
         rule (Upat.op Op.Index ~src:[ Upat.var "x" ]) (fun m -> Some (m "x"));
         (* a scalar value stands for every lane: its lane is itself, since a
            fold may drop a value's width, as a stack of invalids folds to one
            (D53) *)
         rule
           (Upat.op Op.Index ~src:[ Upat.var "x"; Upat.cvar "c" ])
           (fun m ->
             let x = m "x" in
             match (shape_opt x, addrspace x) with
             | Some [], Some Dtype.Alu -> Some x
             | _ -> None);
         (* unpack WMMA *)
         rule (Upat.op Op.Wmma ~name:"u") (fun m -> do_stack_wmma (m "u"));
         (* stacked INDEX is many INDEX *)
         rule
           (Upat.op Op.Index ~name:"x"
              ~src:
                [ Upat.v ~op:storage ~name:"b" (); Upat.op Op.Stack ~name:"s" ])
           (fun m ->
             Some
               (stack
                  (List.map
                     (fun u -> replace (m "x") ~src:[ m "b"; u ])
                     (src (m "s")))));
         (* INDEX into RESHAPE moves the RESHAPE *)
         rule
           (Upat.op Op.Index
              ~src:
                [
                  Upat.v ~op:storage ~name:"b" (); Upat.op Op.Reshape ~name:"s";
                ])
           (fun m ->
             Some (reshape (index (m "b") [ nth (m "s") 0 ]) (shape (m "s"))));
         (* RESHAPE a void is removed (hack for AFTER) *)
         rule (Upat.op Op.Reshape ~dtype:[ Dtype.Void ] ~name:"x") (fun m ->
             Some (nth (m "x") 0));
         (* reshape of a single element shaped value to scalar is an index *)
         rule (Upat.op Op.Reshape ~name:"x") (fun m ->
             let x = m "x" in
             match (marg x, shape (nth x 0)) with
             | Reshape [], [ Int 1 ] -> Some (index_ints (nth x 0) [ 0 ])
             | _ -> None);
         (* EXPAND on scalar -> nested STACKs with the same shape *)
         rule
           (Upat.op Op.Expand ~src:[ Upat.var "x"; Upat.wild ] ~name:"out")
           (fun m ->
             let x = m "x" and s = shape (m "out") in
             let sizes =
               List.filter_map (function Int n -> Some n | _ -> None) s
             in
             match shape x with
             | []
               when List.length sizes = List.length s && not (List.mem 0 sizes)
               ->
                 let broadcast x n = stack (List.init n (fun _ -> x)) in
                 Some (List.fold_left broadcast x (List.rev sizes))
             | _ -> None);
       ]

(* Reductions *)

let fix_group_for_reduce x =
  let threads = Axis_type.[ Warp; Local ] in
  let thread u = is Op.Range u && List.mem (axis_type u) threads in
  match List.partition thread (srcs x) with
  | [], _ -> None
  | reduce_gfr, reduce_r ->
      (* NOTE: if there's other locals here, we need them in the buffer too *)
      let upstream_locals =
        List.filter
          (fun u -> List.mem (axis_type u) threads)
          (Nodes.to_list (ranges x))
      in
      (* do only the non grouped reduces early *)
      let ret = replace x ~src:(nth x 0 :: reduce_r) in
      let reduce_loop =
        List.map
          (fun r ->
            match axis_id r with
            | a :: rest ->
                replace r
                  ~arg:
                    (Range { axis_id = (a + 100) :: rest; axis_type = Reduce })
            | [] -> assert false)
          reduce_gfr
      in
      let opts = { device = None; addrspace = Dtype.Local; removable = true } in
      let buf =
        index
          (bufferize ~opts ret (upstream_locals @ reduce_gfr))
          (upstream_locals @ reduce_loop)
      in
      (* do the final reduce (if/barrier are added in gpudims step) *)
      (* NOTE: we remove all horizontal reduces here, they remain in the first reduce *)
      Some (reduce buf (fst (reduce_arg x)) reduce_loop)

let mergeable = Tag.String "mergeable"

(* An association list keeps the order its keys were first added in, as the
   Python dictionaries it stands for do. *)
let add_to key equal x l =
  if List.exists (fun (k, _) -> equal k key) l then
    List.map (fun (k, xs) -> if equal k key then (k, xs @ [ x ]) else (k, xs)) l
  else l @ [ (key, [ x ]) ]

let same_nodes s0 s1 =
  Nodes.cardinal s0 = Nodes.cardinal s1
  && Nodes.fold (fun u ok -> ok && Nodes.mem u s1) s0 true

let merge_reduce_ends sink =
  (* merge ENDs that share the same range and nesting context (only those created by reduce_to_acc) *)
  (* ENDs at different nesting depths get cloned RANGEs so each RANGE maps to one END *)
  let slice = Nodes.to_list (backward_slice sink) in
  let range_to_ends =
    List.fold_left
      (fun acc u ->
        if is Op.End u && Option.equal Tag.equal (tag u) (Some mergeable) then
          add_to (srcs u) (List.equal ( == )) u acc
        else acc)
      [] slice
  in
  let subs = ref [] in
  let next_axis =
    ref
      (1
      + List.fold_left
          (fun m u -> if is Op.Range u then max m (List.hd (axis_id u)) else m)
          (-1) slice)
  in
  List.iter
    (fun (r, ends) ->
      if List.length ends > 1 then
        let by_ctx =
          List.fold_left
            (fun acc e -> add_to (ranges e) same_nodes e acc)
            [] ends
        in
        List.iteri
          (fun i (_, ends) ->
            let tr =
              if i = 0 then r
              else
                List.mapi
                  (fun j rr ->
                    let axis_id = (!next_axis + j) :: List.tl (axis_id rr) in
                    replace rr
                      ~arg:(Range { axis_id; axis_type = axis_type rr }))
                  r
            in
            if i > 0 then next_axis := !next_axis + List.length r;
            let mapped =
              if i = 0 then ends
              else List.map (fun e -> substitute e (List.combine r tr)) ends
            in
            let merged =
              match mapped with
              | [ e ] -> e
              | _ -> end_ (group (List.map (fun e -> nth e 0) mapped)) tr
            in
            List.iter (fun e -> subs := (e, merged) :: !subs) ends)
          by_ctx)
    range_to_ends;
  match !subs with [] -> None | subs -> Some (substitute sink (List.rev subs))

let reduce_ranges_to_acc slots r =
  let op, num_axes = reduce_arg r and x = nth r 0 and rngs = srcs r in
  let acc = alloc_like ~slot:(slots ()) ~addrspace:Dtype.Reg r in
  let input_ranges =
    List.filter (fun u -> not (List.memq u rngs)) (Nodes.to_list (ranges x))
  in
  let acc_init =
    store (after acc input_ranges) (const (identity_element op (dtype r)))
  in
  let acc_initted = after acc (acc_init :: rngs) in
  let inp = if num_axes > 0 then v Op.Reduce ~src:[ x ] ~arg:(arg r) else x in
  let acc_out = store acc_initted (alu acc_initted op [ inp ]) in
  Some (after acc [ rtag ~tag:mergeable (end_ acc_out rngs) ])

let expand_horizontal_reduce r =
  let op, num_axes = reduce_arg r and inp = nth r 0 in
  let sizes = List.filteri (fun a _ -> a < num_axes) (max_shape inp) in
  match List.map (index_ints inp) (product sizes) with
  | v0 :: vals -> Some (List.fold_left (fun x y -> alu x op [ y ]) v0 vals)
  | [] -> assert false

(* an Invalid in a REDUCE source is that reduce's identity *)
let pm_reduce_identity =
  pm
    [
      rule
        (Upat.reduce ~allow_any_len:true ~name:"red" Symbolic.invalid_gate [])
        (fun m ->
          let red = m "red" and x = m "x" in
          let id =
            const_like x (identity_element (fst (reduce_arg red)) (dtype red))
          in
          Some (replace red ~src:(where (m "cond") x id :: srcs red)));
    ]

let pm_reduce_local =
  lift pm_wmma_add
  ++ pm
       [
         (* fix group for reduce *)
         rule (Upat.op Op.Reduce ~name:"x") (fun m ->
             fix_group_for_reduce (m "x"));
         (* remove reduces *)
         rule_ctx
           (Upat.op Op.Reduce ~src:[ Upat.wild; Upat.wild ] ~allow_any_len:true
              ~name:"r")
           (fun slots m -> reduce_ranges_to_acc slots (m "r"));
         rule (Upat.op Op.Reduce ~src:[ Upat.wild ] ~name:"r") (fun m ->
             expand_horizontal_reduce (m "r"));
         rule (Upat.op Op.Sink ~name:"sink") (fun m ->
             merge_reduce_ends (m "sink"));
       ]
  ++ lift Symbolic.pm_clean_up_group_sink

(* Loads and local buffers *)

let is_shape_changing_bitcast u =
  is Op.Bitcast u && not (List.equal Sint.equal (shape u) (shape (nth u 0)))

let maybe_load u =
  match addrspace u with
  | Some (Dtype.Global | Dtype.Local | Dtype.Reg) -> load u []
  | _ -> u

let pm_add_loads =
  pm
    [
      rule
        (Upat.v
           ~op:
             (Op.Set.union Op.Set.elementwise
                (ops [ Op.Reduce; Op.Wmma; Op.Stack ]))
           ~name:"x" ())
        (fun m ->
          let x = m "x" in
          if is_shape_changing_bitcast x then None
          else Some (replace x ~src:(List.map maybe_load (src x))));
      rule (Upat.op Op.Store ~name:"x") (fun m ->
          match src (m "x") with
          | p :: x :: rest ->
              Some (replace (m "x") ~src:(p :: maybe_load x :: rest))
          | _ -> None);
    ]

let add_local_buffer slots x =
  let addrspace =
    match arg x with Bufferize o -> o.addrspace | _ -> assert false
  in
  let buf =
    alloc ~slot:(slots ()) ~addrspace
      (List.map (fun n -> Int n) (max_shape x))
      (dtype x)
  in
  Some (after buf [ end_ (store (index buf (srcs x)) (nth x 0)) (srcs x) ])

let pm_add_local_buffers =
  pm
    [
      rule_ctx (Upat.op Op.Stage ~name:"x") (fun slots m ->
          add_local_buffer slots (m "x"));
    ]
  ++ lift Prepare.pm_mops

(* float ALUs need a float operand *)
(* make that cast explicit before the decomps, which expand SIN/LOG2/EXP2 into
   float polynomials and assert a float operand *)
let pm_cast_float_alu =
  pm
    [
      rule
        (Upat.v
           ~op:(ops [ Op.Sin; Op.Log2; Op.Exp2; Op.Sqrt; Op.Reciprocal ])
           ~src:[ Upat.var "x" ]
           ~name:"u" ())
        (fun m ->
          let u = m "u" and x = m "x" in
          if Dtype.equal (dtype x) (dtype u) then None
          else Some (replace u ~src:[ cast x (dtype u) ]));
    ]

(* Barriers *)

let is_local_store x = is Op.Store x && addrspace x = Some Dtype.Local

let add_raw_barrier after =
  (* loads from a LOCAL buffer that depend (via AFTER) on stores to LOCAL memory
     need a workgroup barrier *)
  if addrspace after <> Some Dtype.Local then None
  else
    (* one toposort over all the deps *)
    let deps =
      toposort ~gate:(fun x -> not (is Op.Barrier x)) (sink (srcs after))
    in
    if not (List.exists is_local_store deps) then None
    else Some (Ops.after (nth after 0) [ v Op.Barrier ~src:(srcs after) ])

let add_war_barrier end_ =
  (* a LOCAL buffer stored and loaded in the same loop needs a barrier at the
     end of the loop body *)
  let rngs =
    List.filter
      (fun r ->
        List.mem (axis_type r) Axis_type.[ Reduce; Weak; Loop ]
        && Dtype.Value.(vmax r > of_int 0))
      (ended_ranges end_)
  in
  let body = nth end_ 0 in
  if rngs = [] || is Op.Barrier body then None
  else
    let sl = Nodes.to_list (backward_slice_with_self body) in
    (* only stores that are inside this loop body (not in the backward slice
       through AFTER chains from other loops) *)
    let store_bufs =
      List.filter_map
        (fun x ->
          if
            is_local_store x
            && List.exists (fun r -> Nodes.mem r (ranges x)) rngs
          then Some (buf_uop x)
          else None)
        sl
    in
    (* a load whose buffer matches a local store's buffer is necessarily a local
       load *)
    if
      not
        (List.exists
           (fun x -> is Op.Load x && List.memq (buf_uop (nth x 0)) store_bufs)
           sl)
    then None
    else Some (replace end_ ~src:(v Op.Barrier ~src:[ body ] :: srcs end_))

let pm_implicit_barriers =
  pm
    [
      rule (Upat.op Op.After ~name:"after") (fun m ->
          add_raw_barrier (m "after"));
      rule
        (Upat.v ~op:(ops [ Op.End; Op.Backedge ]) ~name:"end" ())
        (fun m -> add_war_barrier (m "end"));
    ]

(* Lowering *)

let rewrite ?bottom_up ?enter_calls ?walk ?(ctx = ()) m sink =
  graph_rewrite ?bottom_up ?enter_calls ?walk ~ctx sink m

let kernel_info u =
  match arg u with
  | Kernel k -> k
  | _ -> invalid_arg "a kernel's sink needs kernel information"

let check_spec spec sink =
  if setting Helpers.spec <> 0 then Spec.type_verify spec sink

let apply_opts ?beam sink ren =
  let k = kernel_info sink in
  let search s =
    match beam with
    | Some beam -> beam k.beam s
    | None ->
        invalid_arg
          (Printf.sprintf
             "kernel %s asks for a beam search of width %d, and no search is \
              given"
             k.name k.beam)
  in
  let beam = if k.beam >= 1 then Some search else None in
  Postrange.apply_opts ?beam ~hand_coded:Heuristic.hand_coded_optimizations sink
    ren

(* Read at each failure, unlike the settings, which are read once. *)
let dbgtv () =
  match Sys.getenv_opt "DBGTV" with Some v -> v <> "" | None -> false

let full_rewrite_to_sink ?(optimize = true) ?beam ast ren =
  check_spec Spec.tensor ast;
  (* resolve UNSHARDs (multi-device UNSHARDs are already resolved by the
     scheduler; this handles in-kernel shards, e.g. fragments) *)
  let sink = rewrite Multi.multi_pm ast in
  (* preprocess *)
  let sink = rewrite ~bottom_up:true Prepare.pm_mops sink in
  let sink =
    if not optimize then sink
    else
      (* collapse loads reduce (indexing by a tensor) *)
      let sink = rewrite Simplify.pm_load_collapse sink in
      let sink =
        graph_rewrite ~ctx:(Tbl.create 8) sink
          (Simplify.pm_split_ranges ++ lift Simplify.pm_flatten_range)
      in
      (* symbolic (NOTE: this is a requirement for pm_simplify_ranges to be
         correct) *)
      let sink = rewrite (Symbolic.sym ++ Simplify.pm_flatten_range) sink in
      let sink =
        graph_rewrite ~ctx:(Tbl.create 8) sink
          (lift Simplify.pm_flatten_range ++ Simplify.pm_simplify_ranges)
      in
      (* do postrange optimization, BEAM or hand_coded_optimizations *)
      apply_opts ?beam sink ren
  in
  (* reduce_unparented: a REDUCE whose src folded to a CONST (e.g. x*0) has no
     parented ranges, collapse it before the expander *)
  let sink =
    rewrite
      (Pattern_matcher.concat
         [
           Symbolic.sym;
           Symbolic.pm_move_where_on_load;
           Simplify.pm_flatten_range;
           Simplify.pm_reduce_unparented;
           pm_reduce_identity;
         ])
      sink
  in
  let sink = graph_rewrite ~ctx:(build_range_map sink) sink expander in
  let slots =
    let next =
      List.fold_left
        (fun n u ->
          match (op u, arg u) with
          | (Op.Buffer | Op.Alloc), Param p -> max n (p.slot + 1)
          | _ -> n)
        0 (toposort sink)
      |> ref
    in
    fun () ->
      let slot = !next in
      incr next;
      slot
  in
  let sink =
    graph_rewrite ~ctx:slots sink (lift Movement.mop_cleanup ++ pm_reduce_local)
  in
  let sink = graph_rewrite ~ctx:slots sink pm_add_local_buffers in
  (* add gpu dims (late). this works after devectorize, but it's faster here *)
  let sink = graph_rewrite ~ctx:ren sink Gpudims.pm_add_gpudims in
  (* optimizations are done, now we lower to actual code *)
  let sink =
    rewrite
      (Pattern_matcher.concat
         [ Symbolic.symbolic_simple; pm_expand_broadcast; pm_add_loads ])
      sink
  in
  let sink =
    rewrite
      (Pattern_matcher.concat
         [ Symbolic.symbolic_simple; devectorizer2; Coalesce.indexing_simplify ])
      sink
  in
  (* some coalescing misses without this *)
  let sink = rewrite Symbolic.sym sink in
  (* do memory coalescing (late) *)
  let sink = Coalesce.memory_coalescing sink ren in
  let sink = rewrite ~bottom_up:true Symbolic.symbolic_simple sink in
  (* extra symbolic before decomp. crashes without this? *)
  (* NOTE: also run indexing_simplify here, while the index is still weakint
     and (x+y)*c -> x*c+y*c applies *)
  (* commit widths minted in this fixpoint before lowering inspects INDEX
     shapes *)
  let sink =
    rewrite
      (Pattern_matcher.concat
         [ Symbolic.sym; Coalesce.indexing_simplify; Uop_weak.pm_commit_weak ])
      sink
  in
  (* the boundary: required compute dtypes settle here; derivable const edges
     may stay bare *)
  (* NOTE: we need indexing_simplify to remove the cast to long using the
     Invalid *)
  (* NOTE: symbolic must NOT be composed here -- pm_data_invalid pushes the
     weak result CAST into a gated WHERE, remaking the weak node, and it
     cycles *)
  let sink =
    rewrite ~enter_calls:true
      (Uop_weak.pm_lower_weak ++ Coalesce.indexing_simplify)
      sink
  in
  (* final symbolic before decomp *)
  let sink = rewrite Symbolic.symbolic sink in
  let sink = rewrite pm_cast_float_alu sink in
  (* floordiv+mod / dtype decomp (early) *)
  let supported_ops = ops (List.map fst ren.Renderer.code_for_op) in
  let pm_decomp =
    Symbolic.symbolic_simple ++ Decomp_op.simplifying_patterns supported_ops
  in
  let sink = rewrite pm_decomp sink in
  (* late decomps + move gates from unrenderable INVALID where *)
  let sink =
    graph_rewrite ~ctx:(Decomp_dtype.ctx ren) sink
      (Decomp_dtype.pm_dtype_decomps ++ lift Uop_weak.pm_commit_weak)
  in
  let pm_decomp =
    Pattern_matcher.concat
      [
        lift pm_decomp;
        Decomp_op.late_patterns
          ~disable_fast_idiv:(setting Helpers.disable_fast_idiv)
          supported_ops;
        lift
          (Transcendental.patterns
             ~force:(setting Helpers.transcendental >= 2)
             supported_ops);
      ]
  in
  let sink = graph_rewrite ~ctx:ren sink pm_decomp in
  let sink = rewrite Gater.pm_move_gates_from_index sink in
  (* final rules for the renderer (without sym) *)
  let pm_final_rewrite =
    Pattern_matcher.concat
      [
        lift Uop_weak.pm_commit_weak;
        pm_decomp;
        lift ren.extra_matcher;
        lift Linearizer.pm_split_ends;
        lift Symbolic.pm_remove_invalid;
      ]
  in
  let sink = graph_rewrite ~ctx:ren sink pm_final_rewrite in
  (* commit every const still bare so no renderer reads one *)
  let sink = rewrite Uop_weak.pm_cast_const sink in
  (* add implicit barriers (stores/loads through LOCAL memory ordered by AFTER
     or across loop iterations need workgroup barriers) *)
  let sink = rewrite pm_implicit_barriers sink in
  (* this was the linearizer *)
  let sink =
    graph_rewrite ~bottom_up:true
      ~ctx:(Linearizer.cfg_context sink)
      sink Linearizer.pm_add_control_flow
  in
  (* put unnumbered variable PARAMs in slots *)
  let num_params =
    List.length
      (List.filter
         (fun x ->
           match (op x, arg x) with
           | Op.Param, Param p -> p.slot <> -1
           | _ -> false)
         (toposort sink))
  in
  let sink =
    graph_rewrite ~walk:true ~ctx:(ref num_params) sink pm_number_params
  in
  if setting Helpers.spec <> 0 then (
    try Spec.type_verify Spec.program sink
    with Invalid_argument _ as e when dbgtv () ->
      Format.printf "%a@." Render.pp_uops (toposort sink);
      raise e);
  sink

(* Linearizing *)

(* inject IF/ENDIF. only needed if device doesn't support gated stores *)
let pm_linearize_cleanups =
  Pattern_matcher.fold
    [
      (* if statements are not allowed in graph *)
      rule
        (Upat.v ~op:(ops [ Op.If; Op.Endif ]) ())
        (fun _ -> invalid_arg "if not allowed in graph");
      (* gated STORE becomes IF-STORE-ENDIF. this is the only use of IF-ENDIF *)
      rule
        (Upat.op Op.Store ~name:"u"
           ~src:
             [
               Upat.or_casted (Upat.v ~op:(ops [ Op.Index; Op.Shrink ]) ());
               Upat.wild;
               Upat.var ~dtype:[ Dtype.Bool ] "gate";
             ])
        (fun m ->
          let u = m "u" in
          let st = replace u ~src:[ nth u 0; nth u 1 ] in
          let mif = v Op.If ~src:[ m "gate"; nth u 0 ] in
          Some (st, [ mif; st; v Op.Endif ~src:[ mif ] ]));
    ]

let pm_alloc_to_buf =
  Pattern_matcher.fold
    [
      rule (Upat.op Op.Alloc ~name:"x") (fun m ->
          let buf = replace (m "x") ~op:Op.Buffer in
          Some (buf, [ buf ]));
    ]

(* requires lst be toposorted. like graph rewrite, but for lines *)
let line_rewrite lst m ctx =
  let replaced = Tbl.create 64 in
  let find x = Option.value (Tbl.find_opt replaced x) ~default:x in
  List.concat_map
    (fun u ->
      let nu = replace u ~src:(List.map find (src u)) in
      let ret, lines =
        Option.value (Pattern_matcher.rewrite m ctx nu) ~default:(nu, [ nu ])
      in
      Tbl.replace replaced u ret;
      lines)
    lst

(* Programs *)

let pp_opts ppf = function
  | [ o ] -> Format.fprintf ppf "(%a,)" Opt.pp o
  | opts ->
      Format.fprintf ppf "(%a)"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
           Opt.pp)
        opts

let do_linearize prg sink =
  let k = kernel_info sink in
  if setting Helpers.debug >= 3 && k.applied_opts <> [] then
    Format.printf "%-25s opts: %a@." (function_name k) pp_opts k.applied_opts;
  let lst =
    line_rewrite
      (Linearizer.linearize sink)
      (pm_linearize_cleanups ++ pm_alloc_to_buf)
      ()
  in
  replace prg ~src:[ last lst; v Op.Linear ~src:lst ]

let do_estimates prg sink lin =
  let k = kernel_info sink in
  if Option.is_some k.estimates then None
  else
    let estimates =
      Renderer.Estimates.of_uops ~ignore_indexing:true (src lin)
    in
    let sink =
      replace sink ~arg:(Kernel { k with estimates = Some estimates })
    in
    Some (replace prg ~src:(sink :: srcs prg))

let do_render (ren : Renderer.t) prg lin =
  let source = ren.render (src lin) in
  Some (replace prg ~src:(src prg @ [ v Op.Source ~arg:(String source) ]))

(* nx.device calls a host program as [void f(void **b, const int64_t *v)], its
   buffers in the order of its globals and its variables in order. The kernel is
   renamed [NAME_], and [NAME] becomes an entry of that form that passes each of
   the kernel's parameters on, which C converts to the parameter's type. The
   source the program keeps is the renderer's. *)
let host_entry prg lin source =
  let info = match arg prg with Program info -> info | _ -> assert false in
  let name = function_name (kernel_info (nth prg 0)) in
  let pass u =
    let index x xs = Option.get (List.find_index x xs) in
    match arg u with
    | Param _ when addrspace u = Some Dtype.Alu ->
        Printf.sprintf "v[%d]" (index (equal u) info.vars)
    | Param p -> Printf.sprintf "b[%d]" (index (Int.equal p.slot) info.globals)
    | _ -> assert false
  in
  let params = List.filter (fun u -> op u = Op.Param) (src lin) in
  let abi = if Sys.win32 then "__attribute__((ms_abi)) " else "" in
  String.concat "\n"
    [
      Printf.sprintf "#define %s %s_" name name;
      source;
      Printf.sprintf "#undef %s" name;
      Printf.sprintf "%svoid %s(void **b, const long long *v) { %s_(%s); }" abi
        name name
        (String.concat ", " (List.map pass params));
    ]

let do_compile (ren : Renderer.t) prg source =
  let source = match arg source with String s -> s | _ -> assert false in
  if setting Helpers.debug >= 4 then print_endline source;
  let source =
    if ren.target.device = "CPU" then host_entry prg (nth prg 1) source
    else source
  in
  let lib = Renderer.Compiler.compile_cached ren.compiler source in
  if setting Helpers.debug >= 7 then
    Renderer.Compiler.disassemble ren.compiler lib;
  Some (replace prg ~src:(src prg @ [ v Op.Binary ~arg:(Bytes lib) ]))

let pm_to_program =
  let program srcs = Upat.op Op.Program ~src:srcs ~name:"prg" in
  let sink = Upat.op Op.Sink ~name:"sink"
  and lin = Upat.op Op.Linear ~name:"lin" in
  pm
    [
      rule (program [ sink ]) (fun m ->
          Some (do_linearize (m "prg") (m "sink")));
      rule
        (program [ sink; lin ])
        (fun m -> do_estimates (m "prg") (m "sink") (m "lin"));
      rule_ctx
        (program [ Upat.wild; lin ])
        (fun ren m -> do_render ren (m "prg") (m "lin"));
      rule_ctx
        (program
           [ Upat.wild; Upat.op Op.Linear; Upat.op Op.Source ~name:"source" ])
        (fun ren m -> do_compile ren (m "prg") (m "source"));
    ]

let do_to_program ?beam ast (ren : Renderer.t) =
  let prg =
    match (op ast, arg ast) with
    | Op.Program, _ -> ast
    | Op.Sink, Kernel _ ->
        let optimize = Option.is_none (tag ast) in
        let full_sink = full_rewrite_to_sink ~optimize ?beam ast ren in
        let info = program_info_of_sink ~target:ren.target full_sink in
        v Op.Program ~src:[ full_sink ] ~arg:(Program info)
    | Op.Sink, _ ->
        invalid_arg "to_program needs a sink with kernel information"
    | o, _ ->
        invalid_arg (Format.asprintf "can't call to_program on %a" Op.pp o)
  in
  let prg =
    match arg prg with
    | Program _ -> prg
    | _ ->
        replace prg
          ~arg:(Program (program_info_of_sink ~target:ren.target (nth prg 0)))
  in
  graph_rewrite ~ctx:ren prg pm_to_program

(* config affects generated programs and cache keys *)
let to_program_key ast (ren : Renderer.t) =
  let open Helpers in
  ( key ast,
    ren.name,
    ren.target,
    setting noopt,
    setting emulated_dtypes,
    setting use_tc,
    setting disable_fast_idiv,
    setting transcendental,
    setting allow_tf32,
    setting default_float,
    setting default_int,
    setting tc_select,
    setting tc_opt,
    setting tc_min_globals )

(* Each kernel's program is made once: a domain that asks for one being made
   waits for it, holding the entry's lock, rather than making it again. *)
type entry = { lock : Mutex.t; mutable prg : Ops.t option }

let to_program_cache = Hashtbl.create 64
let to_program_lock = Mutex.create ()

let to_program ?beam ast ren =
  let key = to_program_key ast ren in
  let entry =
    Mutex.protect to_program_lock (fun () ->
        match Hashtbl.find_opt to_program_cache key with
        | Some entry -> entry
        | None ->
            let entry = { lock = Mutex.create (); prg = None } in
            Hashtbl.replace to_program_cache key entry;
            entry)
  in
  Mutex.protect entry.lock (fun () ->
      match entry.prg with
      | Some prg -> prg
      | None ->
          let prg = do_to_program ?beam ast ren in
          entry.prg <- Some prg;
          prg)
