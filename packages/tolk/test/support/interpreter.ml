open Tolk

let name u =
  match (Ops.op u, Ops.arg u) with
  | Param, Param { name = Some name; vmin_vmax = Some _; _ } -> Some name
  | Special, String name -> Some name
  | Range, _ -> Some ("r" ^ Ops.range_str u)
  | _ -> None

let is_invalid = function `Invalid -> true | #Dtype.value -> false
let zero dt = Dtype.const dt (`Int Bigint.zero)
let weak u = List.mem (Ops.dtype u) Dtype.weaks

(* [held ~check dt v] is [v] as [dt] holds it: converted, then wrapped to [dt]'s
   width, after [check dt] has seen the exact value. A NaN converted to a float
   keeps its sign and payload as far as [dt] does, as compiled code keeps them;
   [Dtype.const] would make it the one canonical NaN. *)
let held ~check dt (v : Dtype.value) : Dtype.const =
  match v with
  | `Float x when Float.is_nan x && Dtype.is_float dt ->
      (Dtype.truncate dt v :> Dtype.const)
  | _ -> (
      match Dtype.const dt v with
      | #Dtype.value as v ->
          check dt v;
          (Dtype.truncate dt v :> Dtype.const)
      | `Invalid -> `Invalid)

(* [of_stacks u] is [true] iff [u] is a stack, or an elementwise operation, a
   cast or a bit reinterpretation of one; [lane k u] is the node that computes
   its lane [k]. *)
let rec of_stacks u =
  match Ops.op u with
  | Stack -> true
  | op when Op.Set.mem op Op.Set.alu || op = Cast || op = Bitcast ->
      List.exists of_stacks (Ops.src u)
  | _ -> false

and lane k u =
  match Ops.op u with
  | Stack -> (
      match List.nth_opt (Ops.src u) k with
      | Some x when k >= 0 -> x
      | _ ->
          invalid_arg
            (Printf.sprintf "lane %d is outside a stack of %d" k
               (List.length (Ops.src u))))
  | _ ->
      Ops.replace u
        ~src:
          (List.map (fun s -> if of_stacks s then lane k s else s) (Ops.src u))

(* Programs *)

(* A graph is evaluated as a program: its nodes numbered once, each with the
   numbers of its sources and what its evaluation needs beyond its node, so that
   evaluating reads arrays, not tables. Named leaves read an environment of
   slots, one per name. Each state of the environment has a generation, and a
   node's value is kept with the generation it was computed in: a value is
   reused only under the binding it was computed under. A node whose sources
   reach no named leaf (not [bound]) has one value per evaluation, kept with the
   evaluation's first generation, its epoch. A lane of stacks adds the nodes it
   makes when it is first read. *)
type program = {
  ids : int Ops.Tbl.t;
  slots : (string, int) Hashtbl.t;
  lanes : (int * int, int) Hashtbl.t;
  root : Ops.t;
  mutable vars : (string * Dtype.value) list;
  mutable params : (int * Dtype.value) list;
  mutable buffers : (int * Dtype.value array) list;
  mutable nodes : Ops.t array;
  mutable srcs : int array array;
  mutable leaves : int array;
  mutable commits : Dtype.t option array array;
  mutable stacks : bool array;
  mutable selects : bool array;
  mutable bound : bool array;
  mutable values : Dtype.const array;
  mutable stamps : int array;
  mutable count : int;
  mutable env : Dtype.value option array;
  mutable epoch : int;
  mutable generation : int;
  mutable generations : int;
}

let grown a n x =
  let b = Array.make n x in
  Array.blit a 0 b 0 (Array.length a);
  b

let grow p =
  let n = 2 * Array.length p.nodes in
  p.nodes <- grown p.nodes n p.nodes.(0);
  p.srcs <- grown p.srcs n [||];
  p.leaves <- grown p.leaves n (-1);
  p.commits <- grown p.commits n [||];
  p.stacks <- grown p.stacks n false;
  p.selects <- grown p.selects n false;
  p.bound <- grown p.bound n false;
  p.values <- grown p.values n `Invalid;
  p.stamps <- grown p.stamps n (-1)

(* [leaf_slot p u] is the slot of [u]'s name, bound as [vars] binds it. *)
let leaf_slot p u =
  match name u with
  | None -> -1
  | Some n -> (
      match Hashtbl.find_opt p.slots n with
      | Some slot -> slot
      | None ->
          let slot = Hashtbl.length p.slots in
          Hashtbl.replace p.slots n slot;
          if slot = Array.length p.env then p.env <- grown p.env (2 * slot) None;
          p.env.(slot) <- List.assoc_opt n p.vars;
          slot)

(* The type each operand of [u] commits to as compiled code reads it: a weak
   integer operand of an operation on a committed integer type is committed to
   that type, as Uop_weak commits it, and wraps. A selection's condition is no
   operand. *)
let commits u =
  let src = Ops.src u in
  let operands =
    match (Ops.op u, src) with Op.Where, _ :: tl -> tl | _ -> src
  in
  let peer =
    List.find_map
      (fun s ->
        let dt = Ops.dtype s in
        if Dtype.is_int dt && not (weak s) then Some dt else None)
      operands
  in
  Array.of_list
    (List.map
       (fun s ->
         match peer with
         | Some dt when weak s && List.memq s operands -> Some dt
         | _ -> None)
       src)

let rec add p u =
  match Ops.Tbl.find_opt p.ids u with
  | Some i -> i
  | None ->
      let srcs = Array.of_list (List.map (add p) (Ops.src u)) in
      let i = p.count in
      if i = Array.length p.nodes then grow p;
      p.nodes.(i) <- u;
      p.srcs.(i) <- srcs;
      p.leaves.(i) <- leaf_slot p u;
      p.commits.(i) <- commits u;
      p.stacks.(i) <-
        (match Ops.op u with
        | Stack -> true
        | op when Op.Set.mem op Op.Set.alu || op = Cast || op = Bitcast ->
            Array.exists (fun s -> p.stacks.(s)) srcs
        | _ -> false);
      p.bound.(i) <-
        p.leaves.(i) >= 0 || Array.exists (fun s -> p.bound.(s)) srcs;
      p.selects.(i) <-
        (match (Ops.op u, Ops.src u) with
        | Where, _ :: operands -> not (List.exists weak operands)
        | _ -> false);
      p.count <- i + 1;
      Ops.Tbl.replace p.ids u i;
      i

let make u =
  let nodes = Ops.toposort u in
  let n = List.length nodes + 16 in
  let p =
    {
      ids = Ops.Tbl.create n;
      slots = Hashtbl.create 16;
      lanes = Hashtbl.create 16;
      root = u;
      vars = [];
      params = [];
      buffers = [];
      nodes = Array.make n u;
      srcs = Array.make n [||];
      leaves = Array.make n (-1);
      commits = Array.make n [||];
      stacks = Array.make n false;
      selects = Array.make n false;
      bound = Array.make n false;
      values = Array.make n `Invalid;
      stamps = Array.make n (-1);
      count = 0;
      env = Array.make 16 None;
      epoch = 0;
      generation = 0;
      generations = 0;
    }
  in
  List.iter (fun u -> ignore (add p u)) nodes;
  p

(* The program of the graph last evaluated on this domain, kept: a graph
   evaluated at many bindings in turn, as a law over its inputs evaluates it, is
   numbered once. *)
let last = Domain.DLS.new_key (fun () -> None)

(* [program ~vars ~params ~buffers u] is the program of [u] with its leaves
   bound by [vars], in a generation of its own. *)
let program ~vars ~params ~buffers u =
  let p =
    match Domain.DLS.get last with
    | Some p when p.root == u -> p
    | _ ->
        let p = make u in
        Domain.DLS.set last (Some p);
        p
  in
  p.vars <- vars;
  p.params <- params;
  p.buffers <- buffers;
  Hashtbl.iter (fun n slot -> p.env.(slot) <- List.assoc_opt n vars) p.slots;
  p.generations <- p.generations + 1;
  p.epoch <- p.generations;
  p.generation <- p.generations;
  p

let lane_of p vector k =
  match Hashtbl.find_opt p.lanes (vector, k) with
  | Some i -> i
  | None ->
      let i = add p (lane k p.nodes.(vector)) in
      Hashtbl.replace p.lanes (vector, k) i;
      i

let leaf p i =
  let u = p.nodes.(i) in
  match (p.env.(p.leaves.(i)), Ops.arg u) with
  | Some v, _ -> v
  | None, Param { bound = Some v; _ } -> v
  | None, _ ->
      invalid_arg
        (Printf.sprintf "%s has no value"
           (Option.value (name u) ~default:"a leaf"))

(* [node ~check p i operands] computes the node [i] from the values of its
   sources as compiled code reads them, [check] seeing each value that an
   operation, a cast or a commitment gives a type before it is wrapped to it. *)
let node ~check p i operands =
  let u = p.nodes.(i) in
  match (Ops.op u, Ops.arg u, operands) with
  | Const, Const c, [] -> c
  | (Param | Range | Special), _, _ when p.leaves.(i) >= 0 ->
      (leaf p i :> Dtype.const)
  | Param, Param { slot; size = None; _ }, [] -> (
      match List.assoc_opt slot p.params with
      | Some v -> (v :> Dtype.const)
      | None -> invalid_arg (Printf.sprintf "parameter %d has no value" slot))
  | Load, _, v :: _ -> v
  | Where, _, `Invalid :: _ -> `Invalid
  | Where, _, [ `Bool c; x; y ] when p.selects.(i) -> if c then x else y
  | op, _, src when op <> Op.Where && List.exists is_invalid src -> `Invalid
  | Cast, _, [ v ] when Dtype.equal (Ops.dtype (Ops.nth u 0)) (Ops.dtype u) -> v
  | Cast, _, [ (#Dtype.value as v) ] -> held ~check (Ops.dtype u) v
  | Bitcast, _, [ (#Dtype.value as v) ] ->
      (Dtype.bitcast (Ops.dtype (Ops.nth u 0)) (Ops.dtype u) v :> Dtype.const)
  | op, _, src when Op.Set.mem op Op.Set.alu -> (
      let dt = Ops.dtype u in
      match Ops.exec_alu ~truncate_output:false op dt src with
      | #Dtype.value as v -> held ~check dt v
      | `Invalid -> `Invalid)
  | op, _, _ -> invalid_arg (Format.asprintf "cannot evaluate %a" Op.pp op)

let storage_slot u =
  match Ops.arg u with
  | Param { slot; size = Some _; _ } -> slot
  | _ -> invalid_arg "cannot access a node that is not storage"

(* [over ~check p ranges f acc] folds [f] over the bindings of the environment
   extended by each value of [ranges], the last varying fastest, each range's
   end evaluated under the binding of those before it. Each binding is a
   generation of its own; the environment and its generation are restored
   after. *)
let rec over :
    'a.
    check:(Dtype.t -> Dtype.value -> unit) ->
    program ->
    int list ->
    ('a -> 'a) ->
    'a ->
    'a =
 fun ~check p ranges f acc ->
  match ranges with
  | [] -> f acc
  | r :: rs -> (
      if Ops.op p.nodes.(r) <> Op.Range then
        invalid_arg "cannot evaluate a loop over a node that is not a range";
      match value ~check p p.srcs.(r).(0) with
      | `Int n ->
          let slot = p.leaves.(r) in
          let saved = p.env.(slot) and generation = p.generation in
          let acc = ref acc in
          for k = 0 to Bigint.to_int n - 1 do
            p.env.(slot) <- Some (`Int (Bigint.of_int k));
            p.generations <- p.generations + 1;
            p.generation <- p.generations;
            acc := over ~check p rs f !acc
          done;
          p.env.(slot) <- saved;
          p.generation <- generation;
          !acc
      | _ -> invalid_arg "a range's end is not an integer")

(* A reduction and an index evaluate their sources themselves: a reduction at
   each value of its ranges, an index without reading its storage. *)
and value ~check p i =
  let generation = if p.bound.(i) then p.generation else p.epoch in
  if p.stamps.(i) = generation then p.values.(i)
  else
    let v =
      match Ops.op p.nodes.(i) with
      | Reduce -> reduction ~check p i
      | Index -> read ~check p i
      | Load when Array.length p.srcs.(i) = 3 -> gated_load ~check p i
      | _ -> node ~check p i (operands ~check p i 0)
    in
    p.values.(i) <- v;
    p.stamps.(i) <- generation;
    v

(* [operands ~check p i j] is the values of the sources of [i] from the [j]th,
   each committed to the type [commits] gives it. *)
and operands ~check p i j =
  let src = p.srcs.(i) in
  if j = Array.length src then []
  else
    let v =
      match (p.commits.(i).(j), value ~check p src.(j)) with
      | Some dt, (`Int _ as v) -> held ~check dt v
      | _, v -> v
    in
    v :: operands ~check p i (j + 1)

(* The index of a gated load is read only where its gate holds. *)
and gated_load ~check p i =
  match p.srcs.(i) with
  | [| index; alternative; gate |] -> (
      match value ~check p gate with
      | `Bool true -> value ~check p index
      | `Bool false -> value ~check p alternative
      | `Invalid -> `Invalid
      | _ -> invalid_arg "a load's gate is not a boolean")
  | _ -> invalid_arg "a gated load has an index, an alternative and a gate"

and reduction ~check p i =
  let red = p.nodes.(i) in
  let op =
    match Ops.arg red with
    | Reduce { op; num_axes = 0 } -> op
    | _ -> invalid_arg "cannot evaluate a reduction of axes"
  in
  let dt = Ops.dtype red in
  let body, ranges =
    match Array.to_list p.srcs.(i) with
    | v :: rs -> (v, rs)
    | [] -> invalid_arg "a reduction has no value"
  in
  let combine (acc : Dtype.const) =
    match (acc, value ~check p body) with
    | (#Dtype.value as acc), (#Dtype.value as v) -> (
        match Ops.exec_alu ~truncate_output:false op dt [ acc; v ] with
        | #Dtype.value as v -> held ~check dt v
        | `Invalid -> `Invalid)
    | _ -> `Invalid
  in
  over ~check p ranges combine (Ops.identity_element op dt)

and read ~check p i =
  match p.srcs.(i) with
  | [| vector; ix |] when p.stacks.(vector) -> (
      match value ~check p ix with
      | `Int k -> value ~check p (lane_of p vector (Bigint.to_int k))
      | `Invalid -> `Invalid
      | _ -> invalid_arg "a lane is not an integer")
  | [| storage; ix |] -> (
      (* A lane of a vector load reads its element past the vector's offset. *)
      let storage, offset, length =
        match (Ops.op p.nodes.(storage), p.srcs.(storage)) with
        | Load, [| vector |] when Ops.op p.nodes.(vector) = Op.Shrink -> (
            match p.srcs.(vector) with
            | [| storage; offset; length |] ->
                (storage, value ~check p offset, Some p.nodes.(length))
            | _ ->
                invalid_arg
                  "a vector load has a storage, an offset and a length")
        | _ -> (storage, `Int Bigint.zero, None)
      in
      let slot = storage_slot p.nodes.(storage) in
      let elements =
        match List.assoc_opt slot p.buffers with
        | Some elements -> elements
        | None -> invalid_arg (Printf.sprintf "buffer %d has no elements" slot)
      in
      (* Compiled code gates a read at an invalid index, which then reads
         zero. *)
      match (offset, value ~check p ix) with
      | `Invalid, _ | _, `Invalid -> zero (Ops.dtype p.nodes.(i))
      | `Int o, `Int k -> (
          (match length with
          | Some n when not Bigint.(geq k zero && lt k (Ops.to_z n)) ->
              invalid_arg
                (Format.asprintf "lane %a is outside a vector of %a"
                   Bigint.pp_print k Bigint.pp_print (Ops.to_z n))
          | _ -> ());
          match Bigint.add o k with
          | e when Bigint.(geq e zero && lt e (of_int (Array.length elements)))
            ->
              (elements.(Bigint.to_int e) :> Dtype.const)
          | e ->
              invalid_arg
                (Format.asprintf "index %a is outside buffer %d" Bigint.pp_print
                   e slot))
      | _ -> invalid_arg "an index is not an integer")
  | _ -> invalid_arg "cannot evaluate an index by more than one index"

let fold ~check ~vars ~params ~buffers u =
  let p = program ~vars ~params ~buffers u in
  value ~check p (Ops.Tbl.find p.ids u)

let eval ?(vars = []) ?(params = []) ?(buffers = []) u =
  fold ~check:(fun _ _ -> ()) ~vars ~params ~buffers u

exception Overflow

let overflows ?(vars = []) ?(params = []) ?(buffers = []) u =
  let check dt (v : Dtype.value) =
    match v with
    | `Int z when Dtype.is_int dt && dt <> Dtype.Weak_int ->
        let lo, hi = Dtypes.int_bounds dt in
        if Bigint.lt z lo || Bigint.gt z hi then raise Overflow
    | _ -> ()
  in
  match fold ~check ~vars ~params ~buffers u with
  | _ -> false
  | exception Overflow -> true

(* Kernels *)

let compare_write (s0, i0, v0) (s1, i1, v1) =
  match Int.compare s0 s1 with
  | 0 -> (
      match Int.compare i0 i1 with 0 -> Dtype.Value.compare v0 v1 | c -> c)
  | c -> c

let writes ?(vars = []) ?(params = []) ?(buffers = []) u =
  let check _ _ = () in
  let p = program ~vars ~params ~buffers u in
  let value = value ~check p and id = Ops.Tbl.find p.ids in
  let store s =
    let dst, value_node, gate =
      match Ops.src s with
      | [ dst; value ] -> (dst, value, None)
      | [ dst; value; gate ] -> (dst, value, Some (id gate))
      | _ -> invalid_arg "a store has a destination, a value and a gate"
    in
    (* A store through a vector writes each lane of a stack past the offset. *)
    let slot, index, lanes =
      match (Ops.op dst, Ops.src dst) with
      | Index, [ storage; index ] ->
          (storage_slot storage, id index, [ id value_node ])
      | Shrink, [ storage; offset; length ]
        when Ops.op value_node = Op.Stack
             && Bigint.equal (Ops.to_z length)
                  (Bigint.of_int (List.length (Ops.src value_node))) ->
          (storage_slot storage, id offset, List.map id (Ops.src value_node))
      | Shrink, _ ->
          invalid_arg "a store through a vector stores a stack of its length"
      | _ -> invalid_arg "cannot evaluate a store through more than one index"
    in
    let inside = Ops.ranges s in
    let ranges =
      List.filter_map
        (fun n ->
          if Ops.op n = Op.Range && Ops.Nodes.mem n inside then Some (id n)
          else None)
        (Ops.toposort s)
    in
    let write acc =
      let opened =
        match gate with
        | None -> true
        | Some g -> Dtype.equal_const (value g) (`Bool true)
      in
      match value index with
      | `Int i when opened ->
          let write (k, acc) lane =
            match value lane with
            | #Dtype.value as v -> (k + 1, (slot, Bigint.to_int i + k, v) :: acc)
            | `Invalid -> (k + 1, acc)
          in
          snd (List.fold_left write (0, acc) lanes)
      | _ -> acc
    in
    over ~check p ranges write []
  in
  let stores =
    List.filter
      (fun n -> Ops.op n = Op.Store)
      (Array.to_list (Array.sub p.nodes 0 p.count))
  in
  List.sort_uniq compare_write (List.concat_map store stores)
