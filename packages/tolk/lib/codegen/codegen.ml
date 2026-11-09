(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let rule = Pattern_matcher.rule
let pm = Pattern_matcher.v
let ( ++ ) = Pattern_matcher.append
let lift = Pattern_matcher.with_ctx
let ops l = Op.Set.of_list l
let setting = Setting.value
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

(* After the parameters, one slot per name. *)
let pm_number_params =
  pm
    (fun () -> [
      rule_ctx (Upat.op Op.Param ~name:"x") (fun (first, names) m ->
          match arg (m "x") with
          | Param p when p.slot = -1 ->
              let i =
                match Hashtbl.find_opt names p.name with
                | Some i -> i
                | None ->
                    let i = Hashtbl.length names in
                    Hashtbl.add names p.name i;
                    i
              in
              Some (replace (m "x") ~arg:(Param { p with slot = first + i }))
          | _ -> None);
    ])

let build_range_map sink =
  let ctx = Hashtbl.create 8 in
  List.iter
    (fun x ->
      if is Op.Range x && axis_type x = Axis_type.Upcast then
        Hashtbl.replace ctx (axis_type x, axis_id x) (Hashtbl.length ctx))
    (toposort ~calls:Skip sink);
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
  Hashtbl.find_opt ctx (axis_type r, axis_id r)
  |> Option.map (fun axis ->
      let n = Dtype.Value.to_int (vmax r) + 1 in
      let c =
        consts ~dtype:(dtype r) (List.map (fun i -> `Int (Bigint.of_int i)) (upto n))
      in
      reshape c
        (List.map
           (fun i -> Int (if i = axis then n else 1))
           (upto (Hashtbl.length ctx))))

let expander =
  pm
    (fun () -> [
      rule (Upat.op Op.Reduce ~name:"r") (fun m -> expand_reduce (m "r"));
      rule_ctx (Upat.op Op.Range ~name:"r") (fun ctx m ->
          expand_range ctx (m "r"));
      rule_ctx (Upat.op Op.Wmma ~name:"u") (fun ctx m ->
          expand_wmma ctx (m "u"));
    ])
  ++ lift Simplify.pm_flatten_range
  ++ lift Shape.mop_cleanup

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
   replaces it, so no addition of +0. stays in the loop. A core of another
   accumulator, which only a hand-built core has, keeps it. *)
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
  if not (is Op.Stack acc && List.for_all (fun s -> zero (lane s)) (src acc))
  then None
  else Some (v Op.Wmma ~arg:(arg wmma) ~src:[ nth wmma 0; nth wmma 1; add ])

(* A running sum [add] added to a core's value [x], which the expander leaves
   behind reshapes and permutes of the core's lanes, enters the core as its
   accumulator: [m w + add = m w'], [w'] the core of accumulator [m^-1 add].
   Anything else stays an addition. *)
let rec into_core x add =
  match op x with
  | Op.Wmma -> wmma_accumulate x add
  | Op.Reshape ->
      let s = nth x 0 in
      into_core s (reshape add (shape s))
      |> Option.map (fun c -> reshape c (shape x))
  | Op.Permute ->
      let o = match marg x with Permute o -> o | _ -> assert false in
      into_core (nth x 0) (permute add (Helpers.argsort o))
      |> Option.map (fun c -> permute c o)
  | _ -> None

(* [x] at position [k] of its axis [a], kept at size 1, computed by a core of
   that position alone: a core's batch axes are independent cores, so taking
   one position of the core's sources takes that core. [None] unless [a]
   reaches a batch axis of a core whole through [x]'s reshapes and permutes. *)
let rec core_at x a k =
  let one s i = List.mapi (fun j d -> if j = i then Int 1 else d) s in
  if List.nth (max_shape x) a = 1 then Some x
  else
    match op x with
    | Op.Wmma ->
        let nb = ndim x - 1 in
        let pick s =
          let j = a - (nb - (ndim s - 1)) in
          if j < 0 || Sint.equal (List.nth (shape s) j) (Int 1) then s
          else
            reshape
              (index (permute s (j :: others [ j ] (ndim s))) [ int k ])
              (one (shape s) j)
        in
        if a >= nb then None
        else Some (replace x ~src:(List.map pick (src x)))
    | Op.Reshape -> (
        (* The source's axis of [a]'s size after as many elements as [a]. *)
        let s = nth x 0 and out = max_shape x in
        let below l i =
          List.fold_left ( * ) 1 (List.filteri (fun j _ -> j < i) l)
        in
        let at i d =
          d = List.nth out a && below (max_shape s) i = below out a
        in
        match List.find_index Fun.id (List.mapi at (max_shape s)) with
        | Some i ->
            core_at s i k |> Option.map (fun c -> reshape c (one (shape x) a))
        | None -> None)
    | Op.Permute ->
        let o = match marg x with Permute o -> o | _ -> assert false in
        core_at (nth x 0) (List.nth o a) k |> Option.map (fun c -> permute c o)
    | _ -> None

let pm_expand_broadcast =
  pm
    (fun () -> [
      rule
        (Upat.v
           ~op:
             (Op.Set.union Op.Set.binary
                (Op.Set.union Op.Set.ternary (ops [ Op.Store ])))
           ~name:"x" ())
        (fun m -> expand_broadcast (m "x"));
      rule (Upat.op Op.Wmma ~name:"b") (fun m ->
          broadcast_and_devec_wmma (m "b"));
    ])

(* The scalar every lane of [x] holds, if [x] expands one. *)
let splat x =
  if is Op.Expand x && shape (nth x 0) = [] then Some (nth x 0) else None

(* The lanes [x] holds in the flat order of its shape, if it is a view of values
   that move into a vector as a literal, memory or registers do: constants,
   loads of memory, lanes of vectors and elementwise operations, whose lanes are
   their own, through reshapes, expansions, which repeat their source's lanes,
   permutations and stacks. *)
let rec lanes_of x =
  let known = function Int _ -> true | Sym _ -> false in
  let memory l = is Op.Load l && addrspace (nth l 0) <> Some Dtype.Reg in
  if shape x = [] then
    if is_literal x || memory x || is Op.Index x then Some [ x ] else None
  else if is Op.Reshape x then lanes_of (nth x 0)
  else if is Op.Stack x then
    let lanes = List.map lanes_of (src x) in
    if List.for_all Option.is_some lanes then
      Some (List.concat_map Option.get lanes)
    else None
  else if is Op.Expand x then
    match (lanes_of (nth x 0), marg x) with
    | Some l, Expand front when List.for_all known front ->
        let copies = Helpers.prod (int_shape front) in
        Some (List.concat (List.init copies (fun _ -> l)))
    | _ -> None
  else if is Op.Permute x then
    match (lanes_of (nth x 0), marg x) with
    | Some l, Permute order ->
        let lanes = Array.of_list l and sizes = int_shape (shape (nth x 0)) in
        let stride i = Helpers.prod (List.filteri (fun j _ -> j > i) sizes) in
        let at index =
          List.fold_left2 (fun at axis i -> at + (stride axis * i)) 0 order index
        in
        Some (List.map (fun i -> lanes.(at i)) (product (int_shape (shape x))))
    | _ -> None
  else
    match shape x with
    | [ Int m ] when Op.Set.mem (op x) Op.Set.elementwise ->
        Some (List.init m (fun k -> index_ints x [ k ]))
    | _ -> None

(* The vector [x] is of [n] lanes in the flat order of its shape, if it holds
   them in order, through reshapes: an elementwise operation of [n] lanes, or a
   stack of [n] lanes of values that move into a vector ([lanes_of]). A load
   fills every lane that reads it, and a lane of a computed vector any lanes
   that read it, through a broadcast or a permutation: the vector is put
   together by a shuffle. A stack of lanes computed apart would be put
   together lane by lane. *)
let in_order n x =
  let rec base x = if is Op.Reshape x then base (nth x 0) else x in
  let x = base x in
  if
    List.equal Sint.equal (shape x) [ Int n ]
    && Op.Set.mem (op x) Op.Set.elementwise
  then Some x
  else
    match lanes_of x with
    | Some l when List.compare_length_with l n = 0 ->
        if is Op.Stack x && List.equal ( == ) l (src x) then Some x
        else Some (stack l)
    | _ -> None

(* The [w] lanes from lane [a] of [x], an operand of a vector operation: a
   vector of lanes in order, or a scalar every lane reads, which reads them
   all. *)
let lanes_at x a w =
  if shape x = [] || is_invalid x then x
  else
    let lane k =
      if is Op.Stack x then nth x (a + k) else index_ints x [ a + k ]
    in
    stack (List.init w lane)

(* The widths of the vectors [n] lanes split into, more than [lanes]: vectors
   of [lanes] and a last of the rest. A rest of one lane would be computed
   apart, which made Clang's loop strength reduction take seconds: the last
   [lanes + 1] lanes are two vectors instead. *)
let split_widths lanes n =
  let k = n / lanes and r = n mod lanes in
  let whole k = List.init k (fun _ -> lanes) in
  if r = 1 then whole (k - 1) @ [ (lanes / 2) + 1; lanes - (lanes / 2) ]
  else if r = 0 then whole k
  else whole k @ [ r ]

(* The operation [b] of [n] lanes on the operands [src], in the vectors of
   [split_widths lanes n], as the stack of their lanes. *)
let split_lanes lanes b src n =
  let chunk (a, chunks) w =
    let c = replace b ~src:(List.map (fun x -> lanes_at x a w) src) in
    (a + w, List.rev_append (List.init w (fun k -> index_ints c [ k ])) chunks)
  in
  stack (List.rev (snd (List.fold_left chunk (0, []) (split_widths lanes n))))

(* An elementwise operation of a renderer that computes on vectors keeps its
   lanes whole, on one axis, the flat order of its shape: each source is a
   vector of those lanes in order, or a scalar every lane reads. Where every
   source is a scalar, the operation is computed once and expanded. An
   operation of more lanes than the renderer's [lanes] computes them in vectors
   of [lanes]: Clang's time grows faster than the lanes of the vectors it puts
   together. *)
let vector_operation lanes b s =
  let n = Helpers.prod (int_shape s) in
  let operand x =
    if is_invalid (base x) then base x
    else
      match (splat x, in_order n x) with
      | Some c, _ -> c
      | None, Some v -> v
      | None, None -> invalid_arg "a vector operand of lanes out of order"
  in
  let src = List.map operand (src b) in
  let scalar x = shape x = [] in
  if n > lanes && not (List.for_all scalar src) then
    let split = split_lanes lanes b src n in
    Some (if List.compare_length_with s 1 = 0 then split else reshape split s)
  else if List.equal ( == ) src (Ops.src b) then None
  else if List.for_all scalar src then Some (mop (replace b ~src) (Expand s))
  else
    let flat = replace b ~src in
    Some (if List.compare_length_with s 1 = 0 then flat else reshape flat s)

(* Whether an operation of [n] lanes on the sources [srcs], each a scalar every
   lane reads or a vector of the lanes in order, computes each lane once. A lane
   of a vector operation is its own, and a lane of a stack is its element: where
   a load fills several lanes and every other source repeats them too, the
   lanes repeat, and computed apart each is computed once. *)
let distinct_lanes n srcs =
  let vectors =
    List.filter_map
      (fun x -> if Option.is_some (splat x) then None else in_order n x)
      srcs
  in
  List.exists (fun v -> not (is Op.Stack v)) vectors
  ||
  let lane i = List.map (fun v -> nth v i) vectors in
  let lanes = Array.init n lane in
  let rec distinct i j =
    if i >= n then true
    else if j >= n then distinct (i + 1) (i + 2)
    else if List.equal ( == ) lanes.(i) lanes.(j) then false
    else distinct i (j + 1)
  in
  distinct 0 1

(* The data types whose elementwise operations [ren] computes on vectors: those
   it computes natively, if it computes on vectors at all. Weak types are those
   of addresses, which are computed lane by lane. *)
let vector_dtypes (ren : Renderer.t) =
  if ren.vector_lanes <= 1 then []
  else
    let emulated = Decomp_dtype.emulates ren in
    List.filter (fun dt -> not (emulated dt)) (Renderer.supported_dtypes ren)

(* Elementwise operations, loads and stores of vectors compute each lane apart,
   except an elementwise operation on the [vectors] types of several distinct
   lanes whose sources hold them in order ([in_order]): a source of lanes
   computed apart would be put together lane by lane, and lanes that repeat
   are computed once each apart. Such an operation computes vectors of at most
   [lanes] lanes ({!vector_operation}). *)
let do_devectorize (vectors, lanes) b =
  let s = shape b in
  let invalid x = is_invalid (base x) in
  let on_vectors x = Dtype.mem (dtype x) vectors in
  let in_lanes x =
    Option.is_some (splat x)
    || (on_vectors x && Option.is_some (in_order (Helpers.prod (int_shape s)) x))
  in
  if
    s = []
    || not
         (List.for_all
            (fun x -> List.equal Sint.equal (shape x) s || invalid x)
            (src b))
  then None
  else if
    Op.Set.mem (op b) Op.Set.elementwise
    && Helpers.prod (int_shape s) > 1
    && on_vectors b
    && List.for_all (fun x -> invalid x || in_lanes x) (src b)
    && distinct_lanes
         (Helpers.prod (int_shape s))
         (List.filter (fun x -> not (invalid x)) (src b))
  then vector_operation lanes b s
  else
    let lane idx =
      replace b
        ~src:
          (List.map
             (fun x -> if invalid x then base x else index_ints x idx)
             (src b))
    in
    let lanes = List.map lane (product (int_shape s)) in
    Some (if is Op.Store b then group lanes else reshape (stack lanes) s)

(* An address is computed lane by lane: an access reads or writes each lane at
   its own address, and merging them needs each address as a sum. A lane of an
   elementwise operation on vectors, in an address, is the operation on its
   sources' lanes. *)
let address_lanes e =
  let memo = Tbl.create 16 in
  let rec go u =
    match Tbl.find_opt memo u with
    | Some r -> r
    | None ->
        let r =
          match src u with
          | [ v; k ]
            when is Op.Index u && is Op.Const k
                 && Op.Set.mem (op v) Op.Set.elementwise
                 && shape v <> [] ->
              let lane x =
                if is_invalid (base x) then base x
                else if shape x = [] then x
                else index x [ k ]
              in
              go (replace v ~src:(List.map lane (src v)))
          | srcs when Op.Set.mem (op u) Op.Set.elementwise ->
              let srcs' = List.map go srcs in
              if List.equal ( == ) srcs srcs' then u else replace u ~src:srcs'
          | _ -> u
        in
        Tbl.replace memo u r;
        r
  in
  let e' = go e in
  if e' == e then None else Some e'

let is_shape_changing_bitcast u =
  is Op.Bitcast u && not (List.equal Sint.equal (shape u) (shape (nth u 0)))

(* A tensor core reads each operand as the stack of its lanes. A bitcast that
   changes the lane count stays whole: no lane of it is a lane of its source,
   so its stack of lanes is only itself again. *)
let do_stack_wmma u =
  let whole x = is Op.Stack x || is_shape_changing_bitcast x in
  if List.for_all (fun x -> whole x || is Op.Wmma x) (src u) then None
  else begin
    assert (ndim u = 1);
    let unpack b =
      if whole b then b
      else
        stack
          (List.map (index_ints b)
             (List.map (fun i -> [ i ]) (upto (max_numel b))))
    in
    Some (replace u ~src:(List.map unpack (src u)))
  end


(* The boolean a constant is, if it is one. *)
let boolean u =
  let c = if is Op.Cast u then nth u 0 else u in
  if not (is Op.Const c) then None
  else match value c with `Bool b -> Some b | _ -> None

(* Constants in vectors. A scalar expanded to a vector, which [const_like] makes
   of the vectors the passes after devectorize compute on, is read as the
   scalar: by an elementwise operation, which reads it in every lane, and at any
   lane. A selection by a vector of constant conditions picks each lane. *)
let pm_vector_constants =
  pm (fun () ->
      [
        rule (Upat.op ~name:"x" Op.Where) (fun m ->
            match src (m "x") with
            | [ c; a; b ] when is Op.Stack c -> (
                match List.map boolean (src c) with
                | picks when List.for_all Option.is_some picks ->
                    let lane x i =
                      if shape x = [] then x else index_ints x [ i ]
                    in
                    Some
                      (stack
                         (List.mapi
                            (fun i pick ->
                              lane (if pick = Some true then a else b) i)
                            picks))
                | _ -> None)
            | _ -> None);
        rule (Upat.v ~op:Op.Set.elementwise ~name:"x" ()) (fun m ->
            let x = m "x" in
            if not (List.exists (fun s -> Option.is_some (splat s)) (src x))
            then None
            else
              let src =
                List.map (fun s -> Option.value (splat s) ~default:s) (src x)
              in
              let scalar = replace x ~src in
              if shape scalar = [] then Some (mop scalar (Expand (shape x)))
              else Some scalar);
        rule (Upat.op Op.Index ~name:"x") (fun m ->
            match src (m "x") with [ e; _ ] -> splat e | _ -> None);
      ])

let devectorizer2 =
  lift Prepare.pm_mops
  ++ pm
       (fun () -> [
         (* unpack broadcasting *)
         rule_ctx
           (Upat.v
              ~op:(Op.Set.union Op.Set.elementwise (ops [ Op.Load; Op.Store ]))
              ~name:"b" ())
           (fun ctx m -> do_devectorize ctx (m "b"));
         (* an index of storage by a vector indexes each lane apart, at an
            address computed lane by lane *)
         rule
           (Upat.op Op.Index
              ~src:
                [
                  Upat.v ~op:Op.Set.defines ~name:"b" ();
                  Upat.v ~op:Op.Set.elementwise ~name:"i" ();
                ])
           (fun m ->
             match shape (m "i") with
             | [ Int n ] ->
                 let lane k =
                   let l = index_ints (m "i") [ k ] in
                   Option.value (address_lanes l) ~default:l
                 in
                 Some (index (m "b") [ stack (List.init n lane) ])
             | _ -> None);
         rule_ctx
           (Upat.op Op.Index
              ~src:[ Upat.v ~op:Op.Set.defines ~name:"b" (); Upat.var "i" ])
           (fun (vectors, _) m ->
             if vectors = [] || shape (m "i") <> [] then None
             else
               Option.map
                 (fun i -> index (m "b") [ i ])
                 (address_lanes (m "i")));
         (* INDEX without src is nothing *)
         rule (Upat.op Op.Index ~src:[ Upat.var "x" ]) (fun m -> Some (m "x"));
         (* unpack WMMA *)
         rule (Upat.op Op.Wmma ~name:"u") (fun m -> do_stack_wmma (m "u"));
         (* reshape of a single element shaped value to scalar is an index *)
         rule (Upat.op Op.Reshape ~name:"x") (fun m ->
             let x = m "x" in
             match (marg x, shape (nth x 0)) with
             | Reshape [], [ Int 1 ] -> Some (index_ints (nth x 0) [ 0 ])
             | _ -> None);
         (* a scalar index into a STACK selects its lanes after the STACK and
            the index, renderers only extract lanes at const indices *)
         rule
           (Upat.op Op.Index
              ~src:[ Upat.op Op.Stack ~name:"a"; Upat.var "i" ])
           (fun m ->
             let a = m "a" and i = m "i" in
             if shape i = [] && ndim a = 1 then
               Some (after (Indexing.stack_select i (src a)) [ a; i ])
             else None);
       ])

(* Reductions *)

let fix_group_for_reduce x =
  let threads = Axis_type.[ Warp; Local ] in
  let thread u = is Op.Range u && List.mem (axis_type u) threads in
  match List.partition thread (srcs x) with
  | [], _ -> None
  | reduce_gfr, reduce_r ->
      (* do only the non grouped reduces early *)
      let ret = replace x ~src:(nth x 0 :: reduce_r) in
      let reduce_loop =
        List.map
          (fun r ->
            match axis_id r with
            | a :: rest ->
                replace r
                  ~arg:(Range { axis_type = Weak; axis_id = (a + 100) :: rest })
            | [] -> assert false)
          reduce_gfr
      in
      let opts = { device = None; addrspace = Dtype.Local; keep = Removable } in
      let buf = index (bufferize ~opts ret reduce_gfr) reduce_loop in
      (* do the final reduce (if/barrier are added in gpudims step) *)
      (* NOTE: we remove all horizontal reduces here, they remain in the first reduce *)
      Some (reduce buf (fst (reduce_arg x)) reduce_loop)

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
  (* merge ENDs that share the same range and nesting context *)
  (* An END that closes a range an earlier END closes, at another nesting depth
     or along other ranges, gets cloned RANGEs so each RANGE maps to one END.
     Reductions that share a loop can stop closing the same ranges: a sum over
     [i] and [k] of a source that does not read [i] closes [k] alone, where a
     maximum beside it closes [k] within [i]'s loop. *)
  let slice = Nodes.to_list (backward_slice ~calls:Skip sink) in
  let range_to_ends =
    List.fold_left
      (fun acc u ->
        if is Op.End u then add_to (srcs u) (List.equal ( == )) u acc
        else acc)
      [] slice
  in
  let subs = ref [] and claimed = Tbl.create 16 in
  let next_axis =
    ref
      (1
      + List.fold_left
          (fun m u -> if is Op.Range u then max m (List.hd (axis_id u)) else m)
          (-1) slice)
  in
  let clone rr =
    if not (Tbl.mem claimed rr) then rr
    else
      let axis_id = !next_axis :: List.tl (axis_id rr) in
      incr next_axis;
      replace rr ~arg:(Range { axis_type = axis_type rr; axis_id })
  in
  (* Ends of one context merge unless one reaches another: a copy that reads
     what another copy of the same range stored runs after it, in a loop of
     its own, and merged it would contain itself. *)
  let slices = Tbl.create 8 in
  let reaches e x =
    let s =
      match Tbl.find_opt slices e with
      | Some s -> s
      | None ->
          let s = backward_slice ~calls:Skip e in
          Tbl.replace slices e s;
          s
    in
    x != e && Nodes.mem x s
  in
  let apart e l = List.for_all (fun x -> not (reaches e x || reaches x e)) l in
  let layers ends =
    let rec place e = function
      | [] -> [ [ e ] ]
      | l :: ls -> if apart e l then (l @ [ e ]) :: ls else l :: place e ls
    in
    List.fold_left (fun ls e -> place e ls) [] ends
  in
  (* A clone renames the ranges of its own loop only: the other ends that close
     any of [r], along other ranges too, stand aside as holes while it is
     renamed. An end inside its loop closes none of [r]. *)
  let all_ends = List.filter (is Op.End) slice in
  let renamed ends r tr e =
    let closes x = List.exists (fun rr -> List.memq rr (srcs x)) r in
    let others =
      List.filter (fun x -> closes x && not (List.memq x ends)) all_ends
    in
    let holes =
      List.mapi
        (fun i x -> (x, range ~axis_type:Placeholder (Int 1) [ -1; i ]))
        others
    in
    let sub e pairs = substitute ~calls:Skip ~pass:Fixed_point e pairs in
    sub
      (sub e (holes @ List.combine r tr))
      (List.map (fun (x, h) -> (h, x)) holes)
  in
  List.iter
    (fun (r, ends) ->
      let by_ctx =
        List.fold_left (fun acc e -> add_to (ranges e) same_nodes e acc) [] ends
      in
      List.iter
        (fun ends ->
          let tr = List.map clone r in
          List.iter (fun rr -> Tbl.replace claimed rr ()) tr;
          let mapped =
            if List.equal ( == ) r tr then ends
            else List.map (renamed ends r tr) ends
          in
          match mapped with
          | [ e ] when e == List.hd ends -> ()
          | _ ->
              let merged =
                match mapped with
                | [ e ] -> e
                | _ -> end_ (group (List.map (fun e -> nth e 0) mapped)) tr
              in
              List.iter (fun e -> subs := (e, merged) :: !subs) ends)
        (List.concat_map (fun (_, ends) -> layers ends) by_ctx))
    range_to_ends;
  match !subs with
  | [] -> None
  | subs -> Some (substitute ~calls:Skip ~pass:Fixed_point sink (List.rev subs))

(* A sum adds each product of its source into its running sum as one
   multiply-add, rounded once, where the renderer [ren] writes one for its type,
   in the order it adds them unfused. [fuses ren r x] is [true] iff the
   reduce [r] of [x] adds products that way. Its additions are marked by the
   argument [fusable] while the products are indexed down to their operands,
   and [pm_fuse_products] fuses them. A reduce of unrolled lanes beside upcast ones sums a view of the
   products, a permute that brings the unrolled axes first, so the product is
   looked for under [x]'s movements. *)
let fuses (ren : Renderer.t) r x =
  let op, _ = reduce_arg r and dt = dtype r in
  Op.equal op Op.Add
  && is Op.Mul (base x)
  && List.mem_assoc Op.Mulacc ren.code_for_op
  && Dtype.mem dt Dtype.[ Float32; Float64 ]
  && ren.native dt

let fusable = String "fusable"
let mark_fusable u = replace u ~arg:fusable

(* A reduce of [r]'s source over its first [num_axes] axes, in place, and
   over the ranges after its source, into an accumulator. *)
let lower_reduce (slots, ren) r =
  let op, num_axes = reduce_arg r and inp = nth r 0 and rngs = srcs r in
  let fused = fuses ren r inp in
  let add x y =
    let u = alu x op [ y ] in
    if fused then mark_fusable u else u
  in
  let ret =
    if num_axes = 0 then inp
    else
      let sizes = List.filteri (fun a _ -> a < num_axes) (max_shape inp) in
      match List.map (index_ints inp) (product sizes) with
      | v0 :: vals -> List.fold_left add v0 vals
      | [] -> assert false
  in
  (* A sum of cores' values adds each core's term in order, the running sum
     entering each core as its accumulator. *)
  let rest = List.filteri (fun a _ -> a >= num_axes) (shape inp) in
  let term idx =
    List.fold_left
      (fun t (a, k) -> Option.bind t (fun t -> core_at t a k))
      (Some inp)
      (List.mapi (fun a k -> (a, k)) idx)
    |> Option.map (fun t -> reshape t rest)
  in
  let into acc idx =
    Option.bind acc (fun acc ->
        Option.bind (term idx) (fun t ->
            if equal_shape (broadcast_shape [ rest; shape acc ]) rest then
              into_core t (expand acc rest)
            else None))
  in
  let folded acc =
    if not (Op.equal op Op.Add) then None
    else
      let sizes = List.filteri (fun a _ -> a < num_axes) (max_shape inp) in
      match product sizes with
      | [] -> None
      | i0 :: is -> (
          match acc with
          | Some acc -> List.fold_left into (Some acc) (i0 :: is)
          | None -> List.fold_left into (term i0) is)
  in
  if rngs = [] then
    if num_axes = 0 then Some ret
    else Some (Option.value (folded None) ~default:ret)
  else
    let acc = alloc_like ~slot:(slots ()) ~addrspace:Dtype.Reg r in
    let input_ranges =
      List.filter (fun u -> not (List.memq u rngs)) (Nodes.to_list (ranges inp))
    in
    let acc_init =
      store (after acc input_ranges) (const (identity_element op (dtype r)))
    in
    let acc_initted = after acc (acc_init :: rngs) in
    let update =
      match folded (Some acc_initted) with
      | Some u -> u
      | None ->
          if num_axes = 0 then add acc_initted ret
          else alu acc_initted op [ ret ]
    in
    Some (load (after acc [ end_ (store acc_initted update) rngs ]) [])

let pm_reduce_local =
  pm
    (fun () -> [
      (* fix group for reduce *)
      rule (Upat.op Op.Reduce ~name:"x") (fun m ->
          fix_group_for_reduce (m "x"));
      (* remove reduces *)
      rule_ctx (Upat.op Op.Reduce ~name:"r") (fun ctx m ->
          lower_reduce ctx (m "r"));
      rule (Upat.op Op.Sink ~name:"sink") (fun m ->
          merge_reduce_ends (m "sink"));
    ])
  ++ lift Symbolic.pm_clean_up_group_sink

(* Loads and local buffers *)

let in_memory u =
  match addrspace u with
  | Some (Dtype.Global | Dtype.Local | Dtype.Reg) -> true
  | _ -> false

let maybe_load u = if in_memory u then load u [] else u

let pm_add_loads =
  pm
    (fun () -> [
      rule
        (Upat.v
           ~op:
             (Op.Set.union Op.Set.elementwise
                (ops [ Op.Wmma; Op.Stack ]))
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
    ])

(* A bitcast of memory that changes the shape reinterprets the pointer; of a
   value, it is expanded here, since a hand-written kernel reaches codegen
   without the scheduler's expansion. *)
let pm_expand_value_bitcast =
  pm (fun () -> [
      rule (Upat.op Op.Bitcast ~name:"bc") (fun m ->
          let bc = m "bc" in
          if in_memory bc then None else Prepare.expand_bitcast bc);
    ])

(* A local buffer is shared by the threads of a group: each of the hardware
   indices it is computed under, which its own indices leave out, gives it a
   dimension of its own. *)
let add_local_buffer slots x =
  let addrspace =
    match arg x with Bufferize o -> o.addrspace | _ -> assert false
  in
  let locals =
    if addrspace <> Dtype.Local then []
    else
      List.filter
        (fun u -> List.mem (axis_type u) Axis_type.[ Warp; Local ])
        (Nodes.to_list (ranges x))
  in
  let extent u = Int (Dtype.Value.to_int (vmax u) + 1) in
  let buf =
    alloc ~slot:(slots ()) ~addrspace
      (List.map extent locals @ List.map (fun n -> Int n) (max_shape x))
      (dtype x)
  in
  let stored =
    end_ (store (index buf (locals @ srcs x)) (nth x 0)) (srcs x)
  in
  Some (index (after buf [ stored ]) locals)

let pm_add_local_buffers =
  pm
    (fun () -> [
      rule_ctx (Upat.op Op.Stage ~name:"x") (fun slots m ->
          add_local_buffer slots (m "x"));
    ])
  ++ lift Prepare.pm_mops

(* float ALUs need a float operand *)
(* make that cast explicit before the decomps, which expand SIN/LOG2/EXP2 into
   float polynomials and assert a float operand *)
let pm_cast_float_alu =
  pm
    (fun () -> [
      rule
        (Upat.v
           ~op:(ops [ Op.Sin; Op.Log2; Op.Exp2; Op.Sqrt; Op.Recip ])
           ~src:[ Upat.var "x" ]
           ~name:"u" ())
        (fun m ->
          let u = m "u" and x = m "x" in
          if Dtype.equal (dtype x) (dtype u) then None
          else Some (replace u ~src:[ cast x (dtype u) ]));
    ])

(* The operands of the product [p] is, if it is one: a product, a lane of a
   product of vectors, or a vector of lanes of one product, its operands
   gathered as the lanes are. *)
let product_operands p =
  let lane l =
    match src l with
    | [ m; k ] when is Op.Index l && is Op.Mul m && shape m <> [] -> Some (m, k)
    | _ -> None
  in
  (* Lane [k] of [x]: a stack's element, through reshapes, at a constant
     lane. *)
  let rec elements x = if is Op.Reshape x then elements (nth x 0) else x in
  let lane_of x k =
    let e = elements x in
    if
      is Op.Stack e && is_literal k
      && List.for_all (fun l -> shape l = []) (src e)
    then nth e (Dtype.Value.to_int (vmax k))
    else index x [ k ]
  in
  let at ks x = if shape x = [] then x else stack (List.map (lane_of x) ks) in
  if is Op.Mul p then Some (nth p 0, nth p 1)
  else
    match (lane p, List.filter_map lane (src p)) with
    | Some (m, k), _ ->
        let at x = if shape x = [] then x else lane_of x k in
        Some (at (nth m 0), at (nth m 1))
    | None, ((m, _) :: _ as lanes)
      when is Op.Stack p
           && List.compare_lengths lanes (src p) = 0
           && List.for_all (fun (m', _) -> m' == m) lanes ->
        let ks = List.map snd lanes in
        Some (at ks (nth m 0), at ks (nth m 1))
    | _ -> None

(* The first term [s] of a sum of a product's lanes, when it is one lane, as
   the product of its operands' lanes: every other lane fuses into a
   multiply-add, so the product of vectors is then read by no lane. *)
let first_lane s =
  if not (is Op.Index s && shape s = []) then s
  else
    match product_operands s with
    | Some (a, b) -> alu a Op.Mul [ b ]
    | None -> s

(* A sum's addition of a product is one multiply-add: the product is the
   second operand, the running sum the first. An addition that adds no product
   keeps its two roundings. *)
let pm_fuse_products =
  pm (fun () ->
      [
        rule (Upat.op ~name:"s" Op.Add) (fun m ->
            let s = m "s" in
            if not (equal_arg (arg s) fusable) then None
            else
              match src s with
              | [ sum; p ] -> (
                  match product_operands p with
                  | Some (a, b) ->
                      Some (alu a Op.Mulacc [ b; first_lane sum ])
                  | None -> Some (replace s ~arg:No_arg))
              | _ -> Some (replace s ~arg:No_arg));
      ])

(* A multiply-add of a float narrower than a float32, rounded once, where the
   renderer's does not round once ({!Renderer.t.fma}). Targets compute such a
   float in a float32, where its multiply-add rounds once, and then round to
   the narrow float: two roundings, which differ from one where the exact value
   lies off a midpoint of the narrow float by less than a float32 resolves. So
   the product is computed exactly in a float32, and the sum rounded to odd
   there, which the cast then rounds as the exact value (Boldo and Melquiond,
   "Emulation of FMA and correctly-rounded sums: proved algorithms using
   rounding to odd", IEEE Trans. Computers 57(4), 2008, Theorem 3, for a float
   two bits more precise and at least two binades wider at the bottom).

   Rounded to odd is rounded toward zero with the last bit set if inexact
   (Section 2.B). The sum [r] rounded to nearest and its exact error [e]
   (Knuth's TwoSum) give it: [r] steps one value toward zero where [e] has the
   other sign, then its last bit is set where [e] is not zero. A NaN or an
   infinity makes [e] NaN, which no comparison holds, and [r] passes.

   A product of halves or 8-bit floats has at most 22 bits and lies within a
   float32's normal range. A bfloat16 has a float32's range, so three guards
   keep its product exact or its rounding right. From [2^127] the product and
   the addend are halved, exactly, and the result doubled back: a sum that
   overflowed only through the product's float32 is finite. A product past a
   float32's range even halved is held at its greatest value, whose sum with
   any addend overflows as the exact one does. A product that underflows
   beside a zero addend is the result alone, so that it keeps its sign, which
   [r = p + 0] loses. A product inexact in a float32's subnormals lies below
   [2^-134], further from a bfloat16 midpoint than its error. Checked against
   the exact sum rounded once on every pair of bfloat16s, NaNs and infinities
   included, beside twenty addends chosen at the edges of the range, and on
   4e9 random and cancelling triples.

   A target may flush a float32's subnormals to zero, Metal among them (MSL
   4.1, Section 8.1). On a lane whose operands, product, sum and result are
   normal, a flush could change the result only where [r] is a midpoint of the
   narrow float: elsewhere [odd] is [r] or its odd neighbour, which round as
   [r] does. There [e] is the addend or zero, and every value TwoSum computes
   is zero or normal. Halving an addend below [2^-125] would make it
   subnormal, so it is kept whole: beside a product of at least [2^127] it
   moves only the direction of the rounding, which its sign keeps. Halves and
   8-bit floats compute in multiples of [2^-48], never subnormal. Checked with
   every float32 operation flushing, on the lanes this holds: every pair of
   bfloat16s beside seven addends, five of the least normal binade, and 4e9
   random and cancelling triples. *)
let narrow_mulacc x =
  let dt = dtype x in
  let a, b, c =
    match src x with
    | [ a; b; c ] -> (cast a Float32, cast b Float32, cast c Float32)
    | _ -> invalid_arg "a multiply-add takes three operands"
  in
  let open O in
  let k v = const_like a (`Float v) in
  let zero = k 0. in
  let bfloat16 = Dtype.equal dt Dtype.Bfloat16 in
  let big =
    let p = a * b in
    lnot ((p < k 0x1p127) land (k (-0x1p127) < p))
  in
  let half = where big (k 0.5) (k 1.) in
  let p = if bfloat16 then a * half * b else a * b in
  let p =
    if not bfloat16 then p
    else
      let greatest = 0x1.fffffep127 in
      let infinite v = (k greatest < v) lor (v < k (-.greatest)) in
      let held = infinite p land lnot (infinite a lor infinite b) in
      where held (where (p < zero) (k (-.greatest)) (k greatest)) p
  in
  let c =
    if not bfloat16 then c
    else
      let whole = (k (-0x1p-125) < c) land (c < k 0x1p-125) in
      where whole c (c * half)
  in
  let r = p + c in
  let d = r - p in
  let e = p - (r - d) + (c - d) in
  let r, e =
    if not bfloat16 then (r, e)
    else
      let alone = lnot (c <> zero) land (a <> zero) land (b <> zero) in
      (where alone p r, where alone zero e)
  in
  let below = e < zero and above = zero < e in
  let toward_zero = (below land (zero < r)) lor (above land (r < zero)) in
  let bits = bitcast r Uint32 in
  let bits = where toward_zero (bits - const_like bits (`Int Bigint.one)) bits in
  let odd = bitcast (bits lor cast (below lor above) Uint32) Float32 in
  cast (if bfloat16 then where big (odd * k 2.) odd else odd) dt

(* The narrow floats whose multiply-add a float32 computes, rounded there and
   then to the type, rounds once. A product of e4m3s has at most 8 bits and is
   exact in a float32, so the float32 sum [S] is rounded once; where it is
   inexact, [|S| >= 2^6] (every sum is a multiple of 2^-18), and for [S] to land
   on a midpoint [M <= 464] of the type within half a float32 spacing, the
   addend would have to cancel the product's distance from [M], at least
   [2^(e(M) - 8)], to [2^(e(M) - 24)], which a value of 4 bits cannot. Checked
   on every finite triple of each, 16.2 and 16.6 million, the bound past which
   the type overflows included. *)
let float32_rounds_once = Dtype.[ Fp8e4m3; Fp8e4m3fnuz ]

let narrow_floats =
  List.filter (fun dt -> Dtype.itemsize dt < 4) Dtype.floats

let pm_narrow_mulacc =
  pm (fun () ->
      [
        rule_ctx
          (Upat.v ~op:(ops [ Op.Mulacc ]) ~dtype:narrow_floats ~name:"x" ())
          (fun (ren : Renderer.t) m ->
            let x = m "x" in
            let dt = dtype x in
            if ren.fma dt || Dtype.mem dt float32_rounds_once then None
            else Some (narrow_mulacc x));
      ])

(* Barriers *)

let is_local_store x = is Op.Store x && addrspace (nth x 0) = Some Dtype.Local

let is_local_load x = is Op.Load x && addrspace (nth x 0) = Some Dtype.Local

(* A call's body is lowered alone, so its barriers order only its own accesses:
   to order those it makes through its LOCAL arguments against its caller's,
   the caller counts the call as an access of each of them. *)
let local_args x =
  List.filter_map
    (fun a ->
      if addrspace a = Some Dtype.Local then Some (buf_uop a) else None)
    (src_without_body x)

let body_has f x =
  is Op.Call x && List.exists f (toposort ~calls:Enter (Ops.body x))

(* The LOCAL buffers [x] writes: a local store's, or a call's LOCAL arguments'
   where its body, or a body that body calls, stores to LOCAL memory. *)
let local_writes x =
  if is_local_store x then [ buf_uop x ]
  else if body_has is_local_store x then local_args x
  else []

(* The buffers [x] reads: a load's, or a call's LOCAL arguments' where its body
   loads from LOCAL memory. *)
let reads x =
  if is Op.Load x then [ buf_uop (nth x 0) ]
  else if body_has is_local_load x then local_args x
  else []

let writes_local x = local_writes x <> []

let add_raw_barrier after =
  (* loads from a LOCAL buffer that depend (via AFTER) on stores to LOCAL memory
     need a workgroup barrier *)
  if addrspace after <> Some Dtype.Local then None
  else
    (* one toposort over all the deps *)
    let deps =
      toposort ~calls:Skip
        ~gate:(fun x -> not (is Op.Barrier x))
        (sink (srcs after))
    in
    if not (List.exists writes_local deps) then None
    else Some (Ops.after (nth after 0) [ v Op.Barrier ~src:(srcs after) ])

let add_war_barrier end_ =
  (* a LOCAL buffer stored and loaded in the same loop needs a barrier at the
     end of the loop body *)
  let rngs =
    List.filter
      (fun r ->
        List.mem (axis_type r) Axis_type.[ Weak; Loop ]
        && Dtype.Value.(vmax r > of_int 0))
      (ended_ranges end_)
  in
  let body = nth end_ 0 in
  if rngs = [] || is Op.Barrier body then None
  else
    let sl = Nodes.to_list (backward_slice_with_self ~calls:Skip body) in
    (* only stores that are inside this loop body (not in the backward slice
       through AFTER chains from other loops) *)
    let store_bufs =
      List.concat_map
        (fun x ->
          if List.exists (fun r -> Nodes.mem r (ranges x)) rngs then
            local_writes x
          else [])
        sl
    in
    (* a load whose buffer matches a local store's buffer is necessarily a local
       load *)
    let read_back x =
      List.exists (fun b -> List.memq b store_bufs) (reads x)
    in
    if not (List.exists read_back sl) then None
    else Some (replace end_ ~src:(v Op.Barrier ~src:[ body ] :: srcs end_))

let pm_implicit_barriers =
  pm
    (fun () -> [
      rule (Upat.op Op.After ~name:"after") (fun m ->
          add_raw_barrier (m "after"));
      rule
        (Upat.v ~op:(ops [ Op.End; Op.Backedge ]) ~name:"end" ())
        (fun m -> add_war_barrier (m "end"));
    ])

(* Lowering *)

let rewrite ?(ctx = ()) m sink =
  graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx sink (After_sources m)

let kernel_info u =
  match arg u with
  | Kernel k -> k
  | _ -> invalid_arg "a kernel's sink needs kernel information"

let check_spec spec sink =
  if setting Setting.spec <> 0 then Spec.type_verify ~calls:Skip spec sink

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

(* Host programs in blocks *)

(* The memory a store writes, as opposed to a register or local memory: a
   parameter of the global address space. *)
let writes_memory st =
  let at = nth st 0 in
  let at = if is Op.Cast at then nth at 0 else at in
  (is Op.Index at || is Op.Shrink at)
  && addrspace (buf_uop (nth at 0)) = Some Dtype.Global

let int_vmax u = match vmax u with `Int z -> Bigint.to_int z | _ -> 0

(* A host program's launch runs blocks of one loop on the host's cores at once
   (rig's host code). A loop qualifies when it is an output loop no optimisation
   claimed (Weak), whose bound the launch computes, as those a GPU makes global
   are, and no store to memory, call or code misses it ({!Postrange.misses}):
   each block runs every effect of the kernel, so one that missed the loop
   would run once per block. A weak loop's steps are independent, so a block
   computes exactly what the whole loop would for its indices. A serial loop
   (Loop), whose iterations may read each other's writes, never qualifies. The
   largest qualifying loop splits. It runs [block_lo + r] for [r] below
   [block_hi - block_lo], two parameters the host entry sets for each block
   ([host_entry]); the passes before kept the loop's own bounds, so this is
   done once the kernel is final. The entry knows the two by their slots, the
   first past those already taken, which the numbering of the kernel's other
   variables keeps; their names, fresh among the kernel's variables, only name
   them in its source. *)
let split_blocks sink =
  let all = Nodes.to_list (backward_slice_with_self ~calls:Skip sink) in
  let slots, names =
    List.fold_left
      (fun (slots, names) u ->
        match arg u with
        | Param p when op u = Op.Param ->
            let slots = if p.slot >= 0 then max slots (p.slot + 1) else slots in
            (slots, Option.fold ~none:names ~some:(fun n -> n :: names) p.name)
        | _ -> (slots, names))
      (0, []) all
  in
  let rec fresh ?(i = 0) base =
    let name = if i = 0 then base else base ^ "_" ^ string_of_int i in
    if List.mem name names then fresh ~i:(i + 1) base else name
  in
  let effects =
    List.filter
      (fun u ->
        (is Op.Store u && writes_memory u) || is Op.Call u)
      all
  in
  let qualifies r =
    is Op.Range r
    && axis_type r = Axis_type.Weak
    && known_at_launch r
    && int_vmax (nth r 0) >= 2
    && not (List.exists (Postrange.misses r) effects)
  in
  let larger a b = if int_vmax (nth b 0) > int_vmax (nth a 0) then b else a in
  match (effects, List.filter qualifies all) with
  | [], _ | _, [] -> sink
  | _, r :: rs ->
      let r = List.fold_left larger r rs in
      let n = nth r 0 and dt = dtype r in
      let bound slot name =
        param
          ~vmin_vmax:(`Int Bigint.zero, vmax n)
          ~name:(fresh name) ~addrspace:Dtype.Alu slot dt
      in
      let lo = bound slots "block_lo" and hi = bound (slots + 1) "block_hi" in
      let r' = replace r ~src:[ alu hi Op.Sub [ lo ] ] in
      (* The loop's uses read [block_lo + r'], and its ends end [r']. *)
      let ends =
        pm (fun () ->
            [
              rule (Upat.op ~name:"e" Op.End) (fun m ->
                  let e = m "e" in
                  if List.memq r (src e) then
                    let ended x = if x == r then r' else x in
                    Some (replace e ~src:(List.map ended (src e)))
                  else None);
            ])
      in
      let sink =
        substitute ~extra_pm:ends ~calls:Skip ~pass:Fixed_point sink
          [ (r, alu lo Op.Add [ r' ]) ]
      in
      let k = match arg sink with Kernel k -> k | _ -> assert false in
      let split = { iterations = ssimplify n; lo = slots; hi = slots + 1 } in
      replace sink ~arg:(Kernel { k with split = Some split })

(* Uniform gates

   A reduce whose result is read only where a gate holds need not run where it
   fails: [where (g, Σ, e)] reads [Σ] only where [g] holds, and a store through
   [where (g, i, Invalid)] writes it only there. When some clauses of [g] are
   the same for every thread of a workgroup ({!Uniform.varying}) and known
   outside the reduce, the reduce's outermost serial loop runs only where they
   hold: a workgroup with nothing to compute skips the loop and its loads, and
   the value it then holds, the reduce's identity, is never read. This runs
   after the optimisations, which decide what a workgroup holds: a gate of
   blocks of 64 rows is uniform over a workgroup of 64 rows, not of 128.

   out[i] = where (fl[i / 64] <> 0 && i % 64 < m, Σ_ρ<n a·b, Invalid)

   becomes, [fl[i / 64] <> 0] uniform over a workgroup of 64 rows,

   out[i] = where (..., Σ_ρ<where (fl[i / 64] <> 0, n, 0) a·b, Invalid) *)

(* Where a reduce's result is read: where [Holds] clauses all hold, or where the
   uniform [Fails] gate does not. *)
type read_where = Holds of t list | Fails of t

let rec clauses g =
  if is Op.And g && Dtype.equal (dtype g) Dtype.Bool then
    clauses (nth g 0) @ clauses (nth g 1)
  else [ g ]

(* The consumers of each node of [sink], each once. *)
let consumers sink =
  let tbl = Tbl.create 256 in
  List.iter
    (fun u ->
      List.iter
        (fun s ->
          let us = Option.value ~default:[] (Tbl.find_opt tbl s) in
          if not (List.memq u us) then Tbl.replace tbl s (u :: us))
        (src u))
    (toposort ~calls:Skip sink);
  tbl

(* The nodes from [u] down to the reduce it computes from, through casts and
   arithmetic whose other sources read no reduce. *)
let rec reduce_chain u =
  if is Op.Reduce u then Some [ u ]
  else if not (is Op.Cast u || Op.Set.mem (op u) Op.Set.alu) then None
  else
    let reads x = op_in_backward_slice_with_self ~calls:Skip x [ Op.Reduce ] in
    match List.filter reads (src u) with
    | [ x ] -> Option.map (List.cons u) (reduce_chain x)
    | _ -> None

let serial_range q =
  is Op.Range q
  && List.mem (axis_type q) Axis_type.[ Weak; Loop ]
  && not (Dtype.equal (dtype q) Dtype.Void)

(* Whether the loop [rho] of the reduce [r] is [r]'s alone: each node of [r]
   that computes from it is read inside [r] only, so that a new bound inside [r]
   leaves no node of the old loop elsewhere. *)
let own_loop read_by r rho =
  let inside = backward_slice_with_self ~calls:Skip r in
  let from_rho = Tbl.create 16 in
  List.for_all
    (fun u ->
      let reads = u == rho || List.exists (Tbl.mem from_rho) (src u) in
      if reads then Tbl.replace from_rho u ();
      (not reads) || u == r
      || List.for_all (fun c -> Nodes.mem c inside) (read_by u))
    (toposort ~calls:Skip r)

(* [bounded (varying, users) read v] is [v], read once, computed by casts and
   arithmetic from one reduce that only it reads, with the reduce's outermost
   serial loop bounded to run only where [read] says, through those of its
   conditions that are uniform and that read only ranges the reduce runs inside.
   It is [None] when [v] is not such a value, when no condition is, or when the
   loop has no bound or is not the reduce's alone, as when another reduce runs
   over it. *)
let bounded (varying, users) read v =
  let ( let* ) = Option.bind in
  let* path = reduce_chain v in
  let r = List.nth path (List.length path - 1) in
  let* rho =
    List.nth_opt
      (List.sort compare_range_arg (List.filter serial_range (srcs r)))
      0
  in
  let users = Lazy.force users in
  let read_by x = Option.value ~default:[] (Tbl.find_opt users x) in
  (* each node of the path read by the one above it alone, [v] once *)
  let rec once = function
    | u :: (x :: _ as rest) -> (
        match read_by x with [ c ] -> c == u && once rest | _ -> false)
    | _ -> true
  in
  let* () =
    if
      List.compare_length_with (read_by v) 1 = 0
      && once path && own_loop read_by r rho
    then Some ()
    else None
  in
  let varying = Lazy.force varying in
  let outside = ranges r in
  let known q =
    (not (varying q))
    && Nodes.fold (fun x ok -> ok && Nodes.mem x outside) (ranges q) true
  in
  let n = nth rho 0 in
  let zero = const_like n (`Int Bigint.zero) in
  let* trips =
    match read with
    | Holds cs -> (
        match List.filter known cs with
        | [] -> None
        | c :: cs ->
            Some (where (List.fold_left (fun a q -> O.(a land q)) c cs) n zero))
    | Fails g -> if known g then Some (where g zero n) else None
  in
  let rho' = replace rho ~src:[ trips ] in
  Some (substitute ~calls:Skip ~pass:Fixed_point v [ (rho, rho') ])

(* Each arm of a selection is read only where the selection takes it, so each is
   bounded alone; a store through an invalid index writes only where its gate
   holds. *)
let pm_lift_uniform_gates =
  pm (fun () ->
      [
        rule_ctx (Upat.op Op.Where ~name:"w") (fun ctx m ->
            let w = m "w" in
            let g = nth w 0 and x = nth w 1 and e = nth w 2 in
            if x == e then None
            else
              match
                (bounded ctx (Holds (clauses g)) x, bounded ctx (Fails g) e)
              with
              | None, None -> None
              | x', e' ->
                  let x = Option.value ~default:x x'
                  and e = Option.value ~default:e e' in
                  Some (replace w ~src:[ g; x; e ]));
        rule_ctx
          (Upat.op Op.Store ~name:"st" ~allow_any_len:true
             ~src:Upat.[ op Op.Index ~src:[ wild; var "w" ]; var "v" ])
          (fun ctx m ->
            let w = m "w" and st = m "st" in
            if not (is Op.Where w && is_invalid (nth w 2)) then None
            else
              Option.map
                (fun v ->
                  replace st
                    ~src:
                      (List.mapi (fun i s -> if i = 1 then v else s) (src st)))
                (bounded ctx (Holds (clauses (nth w 0))) (m "v")));
      ])

let lift_uniform_gates sink =
  graph_rewrite ~calls:Skip ~pass:Once
    ~ctx:(lazy (Uniform.varying sink), lazy (consumers sink))
    sink (After_sources pm_lift_uniform_gates)

let full_rewrite_to_sink ?(optimize = true) ?beam ast ren =
  if setting Setting.debug >= 5 then
    print_endline (Render.render_uir (toposort ~calls:Enter ast));
  check_spec Spec.tensor ast;
  (* preprocess *)
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() ast
      (Before_sources
         (Pattern_matcher.append Prepare.pm_mops pm_expand_value_bitcast))
  in
  (* effects keep their ranges until the expander *)
  let sink = Simplify.pin_effects sink in
  let sink =
    if not optimize then sink
    else
      (* collapse loads reduce (indexing by a tensor) *)
      let sink = rewrite Simplify.pm_load_collapse sink in
      let sink =
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Tbl.create 8) sink
          (After_sources
             (Simplify.pm_split_ranges ++ lift Simplify.pm_flatten_range))
      in
      (* symbolic (NOTE: this is a requirement for pm_simplify_ranges to be
         correct) *)
      let sink = rewrite (Symbolic.sym ++ Simplify.pm_flatten_range) sink in
      let sink =
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Tbl.create 8) sink
          (After_sources
             (lift Simplify.pm_flatten_range ++ Simplify.pm_simplify_ranges))
      in
      (* do postrange optimization, BEAM or hand_coded_optimizations *)
      Stage.decodes ren (Stage.operands ren (apply_opts ?beam sink ren))
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
           Simplify.pm_unpin_effects;
         ])
      sink
  in
  (* a reduce whose result a workgroup-uniform gate selects or stores runs its
     outer serial loop only where the gate holds *)
  let sink = lift_uniform_gates sink in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(build_range_map sink) sink
      (After_sources expander)
  in
  let slots =
    let next =
      List.fold_left
        (fun n u ->
          match (op u, arg u) with
          | (Op.Buffer | Op.Alloc), Param p -> max n (p.slot + 1)
          | _ -> n)
        0
        (toposort ~calls:Skip sink)
      |> ref
    in
    fun () ->
      let slot = !next in
      incr next;
      slot
  in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(slots, ren) sink
      (After_sources (lift Shape.mop_cleanup ++ pm_reduce_local))
  in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:slots sink
      (After_sources pm_add_local_buffers)
  in
  (* group GPU dimensions early so their index arithmetic goes through normal
     lowering *)
  let sink =
    graph_rewrite ~calls:Skip ~pass:Once ~ctx:ren sink
      (After_sources Gpudims.pm_group_gpudims)
  in
  (* optimizations are done, now we lower to actual code *)
  let sink =
    rewrite
      (Pattern_matcher.concat
         [ Shape.symbolic_simple; pm_expand_broadcast; pm_add_loads ])
      sink
  in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point
      ~ctx:(vector_dtypes ren, ren.vector_lanes)
      sink
      (After_sources
         (Pattern_matcher.concat
            [
              lift Shape.symbolic_simple;
              devectorizer2;
              lift Coalesce.indexing_simplify;
            ]))
  in
  (* some coalescing misses without this *)
  let sink = rewrite Symbolic.sym sink in
  (* do memory coalescing (late) *)
  let sink = Coalesce.memory_coalescing sink ren in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
      (Before_sources Shape.symbolic_simple)
  in
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
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
      (After_sources (Uop_weak.pm_lower_weak ++ Coalesce.indexing_simplify))
  in
  (* final symbolic before decomp *)
  let sink = rewrite Shape.symbolic sink in
  let sink = rewrite pm_cast_float_alu sink in
  let sink = rewrite pm_fuse_products sink in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:ren sink
      (After_sources pm_narrow_mulacc)
  in
  (* floordiv+mod / dtype decomp (early) *)
  (* a multiply-add is a sum's alone, which pm_reduce_local makes: a product
     and a sum elsewhere round apart *)
  let supported_ops =
    ops
      (List.filter
         (fun o -> not (Op.equal o Op.Mulacc))
         (List.map fst ren.Renderer.code_for_op))
  in
  let pm_decomp =
    Shape.symbolic_simple ++ Decomp_op.simplifying_patterns supported_ops
  in
  let sink = rewrite pm_decomp sink in
  (* late decomps + move gates from unrenderable INVALID where *)
  let pm_decomp =
    Pattern_matcher.concat
      [
        lift pm_decomp;
        Decomp_op.late_patterns
          ~disable_fast_idiv:(setting Setting.disable_fast_idiv)
          supported_ops;
        lift
          (Transcendental.patterns
             ~force:(setting Setting.transcendental >= 2)
             supported_ops);
      ]
  in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:ren sink
      (After_sources pm_decomp)
  in
  (* the transcendental decompositions can make types the target lacks, as the
     uint64 of Payne-Hanek's reduction *)
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Decomp_dtype.ctx ren) sink
      (After_sources
         (Decomp_dtype.pm_dtype_decomps ++ lift Uop_weak.pm_commit_weak))
  in
  let sink = rewrite Gater.pm_move_gates_from_index sink in
  (* final rules for the renderer (without sym) *)
  let pm_final_rewrite =
    Pattern_matcher.concat
      [
        lift Uop_weak.pm_commit_weak;
        pm_decomp;
        lift ren.extra_matcher;
        lift pm_vector_constants;
        lift Linearizer.pm_split_ends;
        lift Shape.pm_remove_invalid;
      ]
  in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:ren sink
      (After_sources pm_final_rewrite)
  in
  (* commit every const still bare so no renderer reads one *)
  let sink =
    graph_rewrite ~calls:Skip ~pass:Once ~ctx:() sink
      (After_sources Uop_weak.pm_cast_const)
  in
  let sink =
    if optimize && ren.target.device = "CPU" then split_blocks sink else sink
  in
  (* add implicit barriers (stores/loads through LOCAL memory ordered by AFTER
     or across loop iterations need workgroup barriers) *)
  let sink = rewrite pm_implicit_barriers sink in
  (* every barrier exists and every range keeps its role: a barrier or a
     matrix multiply-accumulate in a loop whose trips differ between threads
     hangs the workgroup or reads memory not yet written *)
  Uniform.check sink;
  (* hardware ranges are no longer loops; preserve their already lowered
     bounds *)
  let sink = rewrite Gpudims.pm_range_to_special sink in
  (* this was the linearizer *)
  let sink =
    graph_rewrite ~calls:Skip ~pass:Fixed_point
      ~ctx:(Linearizer.cfg_context sink)
      sink (Before_sources Linearizer.pm_add_control_flow)
  in
  (* put the variables in slots, past every slot a PARAM takes: the slots can
     skip some, and a split's bounds take the first two past the others *)
  let num_params =
    List.fold_left
      (fun n x ->
        match (op x, arg x) with
        | Op.Param, Param p when p.slot <> -1 -> max n (p.slot + 1)
        | _ -> n)
      0
      (toposort ~calls:Skip sink)
  in
  let sink =
    graph_rewrite ~calls:Skip ~pass:Once
      ~ctx:(num_params, Hashtbl.create 8)
      sink (After_sources pm_number_params)
  in
  if setting Setting.spec <> 0 then (
    try Spec.type_verify ~calls:Skip Spec.program sink
    with Invalid_argument _ as e when Setting.value Setting.dbgtv <> "" ->
      print_endline (Render.render_uir (toposort ~calls:Enter sink));
      raise e);
  sink

(* Linearizing *)

(* inject IF/ENDIF. only needed if device doesn't support gated stores *)
let pm_linearize_cleanups =
  Pattern_matcher.fold
    (fun () -> [
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
    ])

let pm_alloc_to_buf =
  Pattern_matcher.fold
    (fun () -> [
      rule (Upat.op Op.Alloc ~name:"x") (fun m ->
          let buf = replace (m "x") ~op:Op.Buffer in
          Some (buf, [ buf ]));
    ])

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

(* A call of a sink or of a program inside a kernel runs its body as code of
   the kernel's program: the body, its own calls lowered first, becomes the
   list of its lines, a sink's lowered alone and unoptimised, which the
   renderer writes as a function of the kernel's program. The function is
   named as its call, numbered where other bodies of the kernel take that
   name: [names] holds each name's bodies in the order the kernel meets them.
   A program already compiled to a binary has no instructions to inline, and
   is refused. *)
type lower_ctx = { ren : Renderer.t; names : (string, t list) Hashtbl.t }

let lower_calls = ref (Pattern_matcher.v (fun () -> []))

let rec index_of b i = function
  | [] -> None
  | x :: _ when x == b -> Some i
  | _ :: rest -> index_of b (i + 1) rest

let numbered ctx name b =
  let bodies = Option.value (Hashtbl.find_opt ctx.names name) ~default:[] in
  match index_of b 0 bodies with
  | Some 0 -> name
  | Some i -> Printf.sprintf "%s_%d" name i
  | None ->
      Hashtbl.replace ctx.names name (bodies @ [ b ]);
      if bodies = [] then name
      else Printf.sprintf "%s_%d" name (List.length bodies)

let lower_call ctx call b =
  if op b = Op.Program && List.exists (is Op.Binary) (src b) then
    invalid_arg "a call inside a kernel of a compiled program"
  else if
    op b = Op.Program && ctx.ren.packed_args && (kernel_info (nth b 0)).kernel
  then
    invalid_arg
      "a call inside a kernel of a program linearized for a launch, whose \
       arguments are packed (kernel_info.kernel)"
  else
    let name =
      match arg call with
      | Call { name = Some n; _ } ->
          numbered ctx (Helpers.to_function_name n) b
      | _ -> invalid_arg "a call inside a kernel needs a name"
    in
    let b =
      graph_rewrite ~calls:Skip ~pass:Once ~ctx b (After_sources !lower_calls)
    in
    let lines =
      if op b = Op.Program then src (nth b 1)
      else
        Linearizer.linearize (full_rewrite_to_sink ~optimize:false b ctx.ren)
    in
    let lines =
      line_rewrite lines (pm_linearize_cleanups ++ pm_alloc_to_buf) ()
    in
    let body = v Op.Linear ~src:lines ~arg:(String name) in
    Some (replace call ~src:(body :: src_without_body call))

let pm_lower_calls =
  pm (fun () ->
      [
        rule_ctx
          (Upat.op Op.Call ~name:"call" ~allow_any_len:true
             ~src:[ Upat.v ~op:(ops [ Op.Sink; Op.Program ]) ~name:"b" () ])
          (fun ctx m -> lower_call ctx (m "call") (m "b"));
      ])

(* A body's calls are lowered by the same rule; set once, when the module is
   initialised, and only read after. *)
let () = lower_calls := pm_lower_calls

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
  if setting Setting.debug >= 3 && k.applied_opts <> [] then
    Format.printf "%-25s opts: %a@." (function_name k) pp_opts k.applied_opts;
  let lst =
    line_rewrite
      (Linearizer.linearize sink)
      (pm_linearize_cleanups ++ pm_alloc_to_buf)
      ()
  in
  replace prg ~src:[ last lst; v Op.Linear ~src:lst ]

(* A split kernel's estimates count all its blocks: its loop's variables span
   the whole loop. The counts are in the index's type, which a whole loop's
   count outgrows past 2^31 operations, so a count that reads no other variable
   is evaluated exactly, a constant node past an [int]. *)
let whole_loop (k : kernel_info) lin (e : estimates) =
  match k.split with
  | None -> e
  | Some s ->
      let var slot =
        List.find
          (fun u ->
            is_variable u
            && match arg u with Param p -> p.slot = slot | _ -> false)
          (src lin)
      in
      let lo = var s.lo and hi = var s.hi in
      let n = sint_to_uop ~dtype:(dtype hi) s.iterations in
      let bounds_only u =
        List.for_all
          (fun v -> (not (is_variable v)) || v == lo || v == hi)
          (toposort ~calls:Skip u)
      in
      let fill = function
        | Int _ as i -> i
        | Sym u as count -> (
            match s.iterations with
            | Int whole when bounds_only u ->
                let bound v () = if expr v = expr hi then whole else 0 in
                let z = sym_compile_z count bound () in
                if Bigint.fits_int z then Int (Bigint.to_int z)
                else Sym (const (`Int z))
            | _ ->
                ssimplify
                  (substitute ~calls:Skip ~pass:Fixed_point u
                     [ (lo, int ~dtype:(dtype lo) 0); (hi, n) ]))
      in
      { ops = fill e.ops; lds = fill e.lds; mem = fill e.mem }

(* [sink] with the estimates of its linear order [lin], unless it has some. *)
let with_estimates sink lin =
  let k = kernel_info sink in
  if Option.is_some k.estimates then None
  else
    let estimates =
      whole_loop k lin
        (Renderer.Estimates.of_uops ~ignore_indexing:true (src lin))
    in
    Some (replace sink ~arg:(Kernel { k with estimates = Some estimates }))

(* A kernel that takes its arguments as bytes reads them from one parameter,
   [args], where each lies at its offset ({!Device.Tiny_elf.iter_sig}): a scalar
   is the value loaded there, and a buffer the address loaded there, read as the
   buffer ({!Op.Refparam}). Its estimates are taken before, while its loads
   read its buffers. *)
let pack_args sink params =
  let args =
    Shape.param ~shape:[ Int (8 * List.length params) ] ~name:"args" 0 Uint8
  in
  let load (offset, dt) =
    let at = offset / Dtype.itemsize dt in
    let offset = cconst Int32 (`Int (Bigint.of_int at)) in
    Ops.load (index (bitcast args dt) [ offset ]) []
  in
  let read p v =
    if addrspace p = Some Dtype.Alu then (p, v)
    else (p, Ops.v Op.Refparam ~src:[ nth p 0; v ] ~arg:(arg p))
  in
  let loads = List.map load (Device.Tiny_elf.layout params) in
  substitute ~calls:Skip ~pass:Fixed_point sink (List.map2 read params loads)

let do_render (ren : Renderer.t) prg lin =
  let source = ren.render (src lin) in
  Some (replace prg ~src:(src prg @ [ v Op.Source ~arg:(String source) ]))

(* The host calls a program as [void f(const void *p, long long first, long long
   end)] (rig's host code), where [p] points to the bytes of the kernel's
   arguments, each at its offset ({!Device.Tiny_elf.iter_sig}), which the
   kernel itself takes. The call runs the iterations [first] to [end - 1], at
   most the loop's last: the launch's threads number the iterations. The kernel
   is renamed [NAME_], and [NAME] becomes the entry that passes [p] on, or for a
   split program a copy of the bytes whose bounds are [first] and [end], [end]
   at most the iterations of the whole loop, which the launch passes as its
   bound [hi]. The copies are inline, where Clang compiles a copy of many bytes
   as a call of [memcpy], which a remote Windows host's object, compiled for no
   system, would call by the wrong convention. On x86_64 Windows, whose object
   calls by Windows' convention, the entry is declared System V's, by which rig
   calls it (rig's host code). The source the program keeps is the
   renderer's. *)
let host_entry ~arch prg source =
  let k = kernel_info (nth prg 0) in
  let name = function_name k in
  let body =
    match k.split with
    | None -> Printf.sprintf "%s_((unsigned char *)p);" name
    | Some s ->
        let params = kernel_params prg in
        let offsets = Device.Tiny_elf.layout params in
        let size =
          List.fold_left
            (fun m (o, dt) -> max m (o + Dtype.itemsize dt))
            0 offsets
        in
        let field slot =
          let i =
            Option.get
              (List.find_index (fun p -> (param_arg_of p).slot = slot) params)
          in
          let offset, dt = List.nth offsets i in
          let ty =
            match Dtype.itemsize dt with
            | 4 -> "int"
            | 8 -> "long long"
            | n -> invalid_arg (Printf.sprintf "a split bound of %d bytes" n)
          in
          (offset, ty)
        in
        let (lo, lo_ty), (hi, hi_ty) = (field s.lo, field s.hi) in
        Printf.sprintf
          "unsigned char a[%d] __attribute__((aligned(8))); \
           __builtin_memcpy_inline(a, p, sizeof a); %s hi; \
           __builtin_memcpy_inline(&hi, a + %d, sizeof hi); if (end > hi) \
           end = hi; if (first < end) { %s lo = first; hi = end; \
           __builtin_memcpy_inline(a + %d, &lo, sizeof lo); \
           __builtin_memcpy_inline(a + %d, &hi, sizeof hi); %s_(a); }"
          size hi_ty hi lo_ty lo hi name
  in
  String.concat "\n"
    [
      Printf.sprintf "#define %s %s_" name name;
      source;
      Printf.sprintf "#undef %s" name;
      Printf.sprintf
        "%svoid %s(const void *p, long long first, long long end) { %s }"
        (if Compiler_cpu.win64 arch then "__attribute__((sysv_abi)) "
         else "")
        name body;
    ]

let text source = match arg source with String s -> s | _ -> assert false

(* The text [ren]'s compiler compiles for [prg]'s rendered [source]: on the
   host, with the entry rig calls. *)
let compiled_text (ren : Renderer.t) prg source =
  if ren.target.device = "CPU" then
    host_entry ~arch:ren.target.arch prg (text source)
  else text source

(* [compiled_text], printing the program's source from DEBUG 4. *)
let compiled_source ren prg source =
  if setting Setting.debug >= 4 then print_endline (text source);
  compiled_text ren prg source

let with_binary (ren : Renderer.t) prg lib =
  if setting Setting.debug >= 7 then
    Renderer.Compiler.disassemble ren.compiler lib;
  replace prg ~src:(src prg @ [ v Op.Binary ~arg:(Bytes lib) ])

let do_compile ((ren : Renderer.t), compile) prg source =
  let lib = compile ren.compiler (compiled_source ren prg source) in
  Some (with_binary ren prg lib)

let program_with srcs = Upat.op Op.Program ~src:srcs ~name:"prg"

let pm_linearize =
  let sink = Upat.op Op.Sink ~name:"sink" in
  pm
    (fun () -> [
      rule (program_with [ sink ]) (fun m ->
          Some (do_linearize (m "prg") (m "sink")));
    ])

(* Its context is the renderer and how a source is compiled. *)
let pm_compile =
  let lin = Upat.op Op.Linear ~name:"lin" in
  pm
    (fun () -> [
      rule_ctx
        (program_with [ Upat.wild; lin ])
        (fun (ren, _) m -> do_render ren (m "prg") (m "lin"));
      rule_ctx
        (program_with
           [ Upat.wild; Upat.op Op.Linear; Upat.op Op.Source ~name:"source" ])
        (fun ctx m -> do_compile ctx (m "prg") (m "source"));
    ])

let kernel_of ast =
  match (op ast, arg ast) with
  | Op.Sink, Kernel k -> k
  | _ -> invalid_arg "a program is made of a sink with kernel information"

(* Launches

   A launch's members run as one program. Each member is lowered as a kernel
   of its own: its optimisations, its uniform gates and its launch
   dimensions, checked ({!Uniform.check}) while its hardware indices are
   still indices. Its hardware indices then become parameters, and on the
   host its split loop's bounds stay parameters, its named variables too,
   and its instructions become a body the launch calls, alike members one
   body.

   On a GPU, the launch's workgroups are its members' laid end to end, and
   its threads the most any member has. Member [i] runs inside a loop of one
   trip where the workgroup index [g] falls in its workgroups, and its
   threads, none past its own:

     g:  0 .. nb_0 - 1 | nb_0 .. nb_0 + nb_1 - 1 | ...
         member 0       member 1
     for r < (o_i <= g < o_i + nb_i && l < nt_i) { member_i (g - o_i + r, l) }

   and is passed its workgroup and thread indices decomposed by its own
   sizes, the first axis fastest, as a GPU numbers threads, so that its warps
   hold the threads they hold in its own launch. The loop's bound reads only
   the workgroup index where a member holds a barrier: such a member shares
   its launch only with members of its kernel ({!Schedule}), whose threads
   are as many as the launch's. The trip [r], passed as an argument, keeps
   each call in its loop. On the host, the launch's split iterations are its
   members' laid end to end, and a block [[lo, hi)] runs, of each member it
   meets, the iterations it overlaps. A body is out of line: inlined, a
   compiler compiles it again at each call. Each member is a call of its
   own, since a call's storage argument is a parameter, which no selection
   picks: a launch's source grows with its members. *)

type member = {
  m_body : t;  (* the member's instructions, as a body *)
  m_slots : int;  (* the slots its body takes *)
  m_vars : (int * t) list;  (* each named variable's slot, and the variable *)
  m_specials : (char * int * int * int * Dtype.t) list;
      (* each hardware index: its kind, axis, size, slot and type *)
  m_split : (split * Dtype.t) option;
  m_estimates : estimates;
}

(* The member of the kernel [b], lowered once for every call of it. *)
let lower_member ctx b ~nargs =
  let b =
    graph_rewrite ~calls:Skip ~pass:Once ~ctx b (After_sources pm_lower_calls)
  in
  let optimize = not (kernel_of b).optimized in
  let low = full_rewrite_to_sink ~optimize b ctx.ren in
  let nodes = toposort ~calls:Skip low in
  let params =
    List.filter_map
      (fun u ->
        match (op u, arg u) with Op.Param, Param p -> Some (u, p) | _ -> None)
      nodes
  in
  let split = (kernel_info low).split in
  let m_split =
    Option.map
      (fun (s : split) ->
        let dt =
          List.find_map
            (fun (u, (p : param_arg)) ->
              if p.slot = s.lo then Some (dtype u) else None)
            params
        in
        (s, Option.value dt ~default:Dtype.Int32))
      split
  in
  let slots =
    List.fold_left
      (fun n (_, (p : param_arg)) -> max n (p.slot + 1))
      nargs params
  in
  let is_bound (p : param_arg) =
    match split with Some s -> p.slot = s.lo || p.slot = s.hi | None -> false
  in
  (* A named variable is passed as the launch's, which a run binds. *)
  let vars =
    List.filter_map
      (fun (u, (p : param_arg)) ->
        if p.name <> None && addrspace u = Some Dtype.Alu && not (is_bound p)
        then Some (p.slot, replace u ~arg:(Param { p with slot = -1 }))
        else None)
      params
  in
  let next = ref slots in
  let specials =
    List.filter_map
      (fun u ->
        if not (is Op.Special u) then None
        else
          let name = match arg u with String s -> s | _ -> assert false in
          if not (is_literal (nth u 0)) then
            invalid_arg "a launch's member has a symbolic launch dimension";
          let slot = !next in
          incr next;
          let p =
            param ~vmin_vmax:(vmin u, vmax u) ~addrspace:Dtype.Alu slot
              (dtype u)
          in
          Some
            ( (u, p),
              ( name.[0],
                Char.code name.[String.length name - 1] - Char.code '0',
                int_vmax u + 1,
                slot,
                dtype u ) ))
      nodes
  in
  let subs, m_specials = List.split specials in
  let low =
    if subs = [] then low else substitute ~calls:Skip ~pass:Once low subs
  in
  let lines =
    line_rewrite (Linearizer.linearize low)
      (pm_linearize_cleanups ++ pm_alloc_to_buf)
      ()
  in
  let name = numbered ctx "member" low in
  let m_body = v Op.Linear ~src:lines ~arg:(String name) in
  let m_estimates =
    whole_loop (kernel_info low) m_body
      (Renderer.Estimates.of_uops ~ignore_indexing:true lines)
  in
  { m_body; m_slots = !next; m_vars = vars; m_specials; m_split; m_estimates }

(* What a call [c] of the member [m] passes, by the member's slots: its
   arguments, then the variables the launch binds. *)
let member_args m c =
  let a = Array.make m.m_slots (int 0) in
  List.iteri (fun i x -> a.(i) <- x) (src_without_body c);
  List.iter (fun (slot, x) -> a.(slot) <- x) m.m_vars;
  a

let sizes kind m =
  let a = Array.make 3 1 in
  List.iter (fun (k, ax, n, _, _) -> if k = kind then a.(ax) <- n) m.m_specials;
  Array.to_list a

let prod = List.fold_left ( * ) 1

(* The index on axis [ax] of sizes [ns] of the linear index [i], the first
   axis fastest. *)
let decomposed i ns ax =
  let open O in
  let below = prod (List.filteri (fun j _ -> Stdlib.(j < ax)) ns) in
  let i = if below = 1 then i else i // int below in
  let rest = prod (List.filteri (fun j _ -> Stdlib.(j >= ax)) ns) in
  if Stdlib.(List.nth ns ax = rest) then i else i % int (List.nth ns ax)

(* A sum of counts; a symbolic count is left out. *)
let add_count a b =
  match (a, b) with
  | Int x, Int y -> Int (x + y)
  | Int x, Sym _ | Sym _, Int x -> Int x
  | Sym _, Sym _ -> Int 0

let launcher ren ast =
  let ctx = { ren; names = Hashtbl.create 8 } in
  (* Members of one kernel are lowered once. *)
  let lowered = Tbl.create 8 in
  let ms =
    List.map
      (fun c ->
        let b = body c and nargs = List.length (src_without_body c) in
        let m =
          match Tbl.find_opt lowered b with
          | Some (n, m) when n = nargs -> m
          | _ ->
              let m = lower_member ctx b ~nargs in
              Tbl.replace lowered b (nargs, m);
              m
        in
        (m, member_args m c))
      (src ast)
  in
  (* The host splits a launch into blocks by its cost: its members'. *)
  let estimates =
    List.fold_left
      (fun (e : estimates) (m, _) ->
        {
          ops = add_count e.ops m.m_estimates.ops;
          lds = add_count e.lds m.m_estimates.lds;
          mem = add_count e.mem m.m_estimates.mem;
        })
      { ops = Int 0; lds = Int 0; mem = Int 0 }
      ms
  in
  let nlaunch =
    List.fold_left
      (fun n c ->
        List.fold_left
          (fun n a ->
            match (op a, arg a) with
            | Op.Param, Param p -> max n (p.slot + 1)
            | _ -> n)
          n (src_without_body c))
      0 (src ast)
  in
  let call_of (m, args) r filled =
    let a = Array.copy args in
    List.iter (fun (slot, x) -> a.(slot) <- x) filled;
    call ~precompile:true ~name:"member" m.m_body
      (Array.to_list a @ [ cast r Dtype.Int32 ])
  in
  let info = kernel_info ast in
  let open O in
  if ren.has_local then begin
    let nb m = prod (sizes 'g' m) and nt m = prod (sizes 'l' m) in
    let lt = List.fold_left (fun a (m, _) -> max a (nt m)) 1 ms in
    let total = List.fold_left (fun a (m, _) -> Stdlib.(a + nb m)) 0 ms in
    let g = range ~axis_type:Global (Int total) [ 0 ] in
    let l = range ~axis_type:Local (Int lt) [ 1 ] in
    let _, ends =
      List.fold_left
        (fun (o, ends) ((m, _) as ma) ->
          let i = List.length ends in
          let n = nb m and t = nt m in
          let inside = (g >= int o) land (g < int Stdlib.(o + n)) in
          let inside =
            if Stdlib.(t < lt) then inside land (l < int t) else inside
          in
          let r =
            range ~axis_type:Loop
              (Sym (where inside (int 1) (int 0)))
              [ Stdlib.(i + 2) ]
          in
          let blk = g - int o + r in
          let gs = sizes 'g' m and ls = sizes 'l' m in
          let filled =
            List.map
              (fun (k, ax, _, slot, dt) ->
                let x =
                  if k = 'g' then decomposed blk gs ax else decomposed l ls ax
                in
                (slot, cast x dt))
              m.m_specials
          in
          (Stdlib.(o + n), end_ (call_of ma r filled) [ r ] :: ends))
        (0, []) ms
    in
    sink
      ~kernel:{ info with optimized = true; estimates = Some estimates }
      [ end_ (group (List.rev ends)) [ g; l ] ]
  end
  else begin
    let iters m =
      match m.m_split with
      | Some ({ iterations = Int n; _ }, _) -> n
      | Some _ -> invalid_arg "a launch's member has a symbolic split"
      | None -> 1
    in
    let total = List.fold_left (fun a (m, _) -> Stdlib.(a + iters m)) 0 ms in
    let bound slot name =
      param
        ~vmin_vmax:(`Int Bigint.zero, `Int (Bigint.of_int total))
        ~name ~addrspace:Dtype.Alu slot Dtype.Int32
    in
    (* The bounds' names are fresh among the variables the members read. *)
    let names =
      List.concat_map
        (fun (m, _) ->
          List.filter_map
            (fun (_, v) -> (param_arg_of v).name)
            m.m_vars)
        ms
    in
    let rec fresh ?(i = 0) base =
      let name =
        if Stdlib.(i = 0) then base else base ^ "_" ^ string_of_int i
      in
      if List.mem name names then fresh ~i:Stdlib.(i + 1) base else name
    in
    let lo = bound nlaunch (fresh "block_lo")
    and hi = bound Stdlib.(nlaunch + 1) (fresh "block_hi") in
    let _, ends =
      List.fold_left
        (fun (o, ends) ((m, _) as ma) ->
          let i = List.length ends in
          let e = int Stdlib.(o + iters m) in
          let a = where (lo < int o) (int o) lo in
          let z = where (hi < e) hi e in
          let r =
            range ~axis_type:Loop
              (Sym (where (a < z) (int 1) (int 0)))
              [ Stdlib.(i + 1) ]
          in
          let c =
            match m.m_split with
            | Some (s, dt) ->
                call_of ma r
                  [
                    (s.lo, cast (a - int o + r) dt);
                    (s.hi, cast (z - int o + r) dt);
                  ]
            | None -> call_of ma r []
          in
          (Stdlib.(o + iters m), end_ c [ r ] :: ends))
        (0, []) ms
    in
    sink
      ~kernel:
        {
          info with
          optimized = true;
          estimates = Some estimates;
          split =
            Some
              {
                iterations = Int total;
                lo = nlaunch;
                hi = Stdlib.(nlaunch + 1);
              };
        }
      [ group (List.rev ends) ]
  end

let linearized ?beam ast (ren : Renderer.t) =
  let ast = if Shape.is_launch ast then launcher ren ast else ast in
  let optimize = not (kernel_of ast).optimized in
  let ast =
    graph_rewrite ~calls:Skip ~pass:Once
      ~ctx:{ ren; names = Hashtbl.create 8 }
      ast (After_sources pm_lower_calls)
  in
  let full_sink = full_rewrite_to_sink ~optimize ?beam ast ren in
  let info = program_info_of_sink ~target:ren.target full_sink in
  let full_sink =
    let lin = v Op.Linear ~src:(Linearizer.linearize full_sink) in
    Option.value (with_estimates full_sink lin) ~default:full_sink
  in
  let full_sink =
    if ren.packed_args && (kernel_info full_sink).kernel then
      pack_args full_sink info.params
    else full_sink
  in
  graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
    (v Op.Program ~src:[ full_sink ] ~arg:(Program info))
    (After_sources pm_linearize)

let linearize ast ren = linearized ast ren

let compiled compile_source prg (ren : Renderer.t) =
  match src prg with
  | [ sink; lin ]
    when op prg = Op.Program && op sink = Op.Sink && op lin = Op.Linear ->
      graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(ren, compile_source) prg
        (After_sources pm_compile)
  | _ -> invalid_arg "compile needs a program linearize made"

let compile prg ren = compiled Renderer.Compiler.compile prg ren

(* A kernel's key, read from [kernel], the kernel the program is made from: a
   kernel's parameters name no device, so kernels on every device of a target
   are one kernel and one program, which a link binds to its call's buffers.
   They are kept with the kernel while it lives, since a kernel asks for its
   program several times; [key] holds its last key and the [suffix] that key
   ends with. *)
type params = {
  free : string;
  key : (string * string) option Atomic.t;
  kernel : Ops.t;
}

module Kernels = Ephemeron.K1.Make (struct
  type t = Ops.t

  let equal = Ops.equal
  let hash = Ops.hash
end)

let kernels = Kernels.create 64
let kernels_lock = Mutex.create ()

let params_of ast =
  (* The schedule's symbolic rules order the operands of a commutative index
     operation by its ranges' numbers in the whole graph, before each kernel
     renumbers its own. Ordered again by the kernel's numbers, kernels that
     differ in that order alone are one kernel, keyed and made alike: the key
     reads the kernel the program is made from, so one key is one source. *)
  let ast =
    graph_rewrite ~calls:Skip ~pass:Once ~ctx:() ast
      (After_sources Shape.commutative)
  in
  { free = key ast; key = Atomic.make None; kernel = ast }

let params ast =
  match Mutex.protect kernels_lock (fun () -> Kernels.find_opt kernels ast) with
  | Some p -> p
  | None ->
      let p = params_of ast in
      Mutex.protect kernels_lock (fun () -> Kernels.replace kernels ast p);
      p

(* The key of a program, in memory and on disk: the kernel, the renderer and
   its target, the table
   of its compiler's binaries, which names the compiler and its options, every
   setting and variable that shapes what compilation makes
   ([Setting.shaping]), and the digest of this library's sources, of which a
   program is a function. The part past the kernel is kept while the renderer
   and the settings' list stay the same, as the list does while no setting
   changes. *)
let suffixes = Atomic.make None

let suffix (ren : Renderer.t) =
  let shaping = Setting.shaping () in
  match Atomic.get suffixes with
  | Some (r, s, text) when r == ren && s == shaping -> text
  | Some _ | None ->
      let text =
        String.concat "\n"
          ([
             ren.name;
             Format.asprintf "%a" Helpers.Target.pp ren.target;
             Option.value (Renderer.Compiler.cachekey ren.compiler) ~default:"";
           ]
          @ List.map (fun (k, v) -> k ^ "=" ^ v) shaping)
      in
      Atomic.set suffixes (Some (ren, shaping, text));
      text

let keyed p ren =
  let text = suffix ren in
  match Atomic.get p.key with
  | Some (t, key) when t == text -> key
  | Some _ | None ->
      let key = String.concat "\n" [ Source_digest.digest; p.free; text ] in
      Atomic.set p.key (Some (text, key));
      key

let program_key ast ren = keyed (params ast) ren

(* A program is kept on disk up to its source, only while its compiler's
   binaries are, under a table that names the compiler: its binary is the
   compiler's entry for that source, so each binary is stored once. A kernel
   that asks for a beam search is not kept, since its program is what the
   search found. *)
let kept (k : kernel_info) (ren : Renderer.t) =
  Setting.value Setting.ccache
  && Option.is_some (Renderer.Compiler.cachekey ren.compiler)
  && k.beam = 0

let rendered prg = op prg = Op.Program && List.length (src prg) = 3

let programs_made = Atomic.make 0
let programs () = Atomic.get programs_made

(* Raised by a making that only reads ([only_read]) and finds no program or
   no binary kept on disk. *)
exception Unkept

(* A program up to its source, made or read back from disk. [only_read]
   raises [Unkept] where it would render. *)
let rendered_program ?beam ~only_read ~key ~kernel ast (ren : Renderer.t) =
  let render () =
    if only_read then raise_notrace Unkept;
    let prg = linearized ?beam ast ren in
    Option.get (do_render ren prg (nth prg 1))
  in
  if not (kept kernel ren) then render ()
  else fst (Graph.cached ~table:"to_program" ~key ~valid:rendered render)

(* A program read back is compiled as a made one is, so it shows its source
   from DEBUG 4 and its instructions from 7. [only_read] makes the program
   from what the disk keeps alone, and raises [Unkept] for want of it, before
   it shows anything. *)
let made_program ?beam ~only_read ~key ~kernel ast (ren : Renderer.t) =
  let prg = rendered_program ?beam ~only_read ~key ~kernel ast ren in
  let compile =
    if not only_read then Renderer.Compiler.compile_cached
    else
      let source = compiled_text ren prg (nth prg 2) in
      match Renderer.Compiler.kept ren.compiler source with
      | Some lib -> fun _ _ -> lib
      | None -> raise_notrace Unkept
  in
  let prg = Option.get (do_compile (ren, compile) prg (nth prg 2)) in
  Atomic.incr programs_made;
  prg

(* Each kernel's program is made once: a domain that asks for one being made
   waits for it rather than making it again. *)
let to_program_cache = (Helpers.Memo.create () : Ops.t Helpers.Memo.t)

let to_program ?beam ast ren =
  let kernel = kernel_of ast in
  let p = params ast in
  let key = keyed p ren in
  Helpers.Memo.get to_program_cache key (fun () ->
      made_program ?beam ~only_read:false ~key ~kernel p.kernel ren)

let read ast ren =
  let kernel = kernel_of ast in
  let p = params ast in
  let key = keyed p ren in
  match
    Helpers.Memo.try_get to_program_cache key (fun () ->
        made_program ~only_read:true ~key ~kernel p.kernel ren)
  with
  | m -> m
  | exception Unkept -> None

(* A kernel's program up to its source, to be compiled. *)
type pending = { ren : Renderer.t; key : string; prg : Ops.t; source : string }

let to_programs ?beam kernels =
  let seen = Hashtbl.create 64 in
  let fresh (ast, ren) =
    let key = program_key ast ren in
    let fresh = (not (Hashtbl.mem seen key)) && Option.is_none (read ast ren) in
    Hashtbl.replace seen key ();
    fresh
  in
  let render (ast, (ren : Renderer.t)) =
    let kernel = kernel_of ast in
    let key = keyed (params ast) ren in
    let prg = rendered_program ?beam ~only_read:false ~key ~kernel ast ren in
    let source = compiled_source ren prg (nth prg 2) in
    { ren; key; prg; source }
  in
  let kernels = List.filter fresh kernels in
  (* A beam search times its candidates, which concurrent compiles would
     disturb: kernels then make their programs one at a time, in order. *)
  if List.exists (fun (ast, _) -> (kernel_of ast).beam > 0) kernels then
    List.iter (fun (ast, ren) -> ignore (to_program ?beam ast ren)) kernels
  else
  (* Each of up to [PARALLEL] workers takes kernels to render until none is
     left, then compiles its sources, each compiler's in one call, which
     starts its toolchain once for them. A worker that starts once the
     kernels are taken does nothing. *)
  let kernels = Array.of_list kernels and next = Atomic.make 0 in
  let compile mine =
    let compilers =
      List.fold_left
        (fun cs k ->
          if List.memq k.ren.compiler cs then cs else k.ren.compiler :: cs)
        [] mine
    in
    List.concat_map
      (fun c ->
        let ks = List.filter (fun k -> k.ren.compiler == c) mine in
        List.combine ks
          (Renderer.Compiler.compile_all c (List.map (fun k -> k.source) ks)))
      (List.rev compilers)
  in
  let work _ =
    let rec take mine =
      let i = Atomic.fetch_and_add next 1 in
      if i >= Array.length kernels then List.rev mine
      else take (render kernels.(i) :: mine)
    in
    compile (take [])
  in
  let keep (k, lib) =
    let made () =
      let prg = with_binary k.ren k.prg lib in
      Atomic.incr programs_made;
      prg
    in
    ignore (Helpers.Memo.get to_program_cache k.key made)
  in
  let workers =
    min (max 1 (Setting.value Setting.parallel)) (Array.length kernels)
  in
  List.init workers Fun.id |> Worker.map work |> List.concat |> List.iter keep
