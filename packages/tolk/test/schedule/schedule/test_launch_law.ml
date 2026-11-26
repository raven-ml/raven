(* The law of launches: a schedule of random programs of views of buffers, each
   interpreted element by element, leaves in memory what its graph states,
   whichever order each launch runs its members in, and no two members of a
   launch write one element.

   A program is a list of writes of float32 buffers, each a store through a view
   of a buffer of a value read through views of buffers (the graph's states, so
   a later write does not change an earlier value): runs, strided columns,
   reversed runs, runs from a bound variable, bits read through a bitcast,
   stores of one write marked disjoint where they are made so, and stores of one
   write that may overlap, landing in the order listed.

   Its oracle (spec) applies the graph's writes one by one: each write computes
   its stores from the states named, then lands them, a store listed twice at
   its first place. The schedule is interpreted, and compiled and run on the
   host and on each GPU present. *)

open Windtrap
open Tolk

let dt = Dtype.Float32
let int_ i = Ops.Int i
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Programs *)

type view =
  | Whole
  | Run of int * int
  | Rows of int * int * int * int
  | Cols of int * int * int * int
  | Rev of int * int
  | Bound of int * int
  | Bit of int * int
  | Bytes of int * int

type op = Add of float | Mul of float | Max of float

(* [At (b, k, v)] reads buffer [b] through [v], in the state [k] writes before
   its latest, or in its first state where it has fewer. *)
type read = At of int * int * view

(* What a store writes: 3.5, or the sum of its reads with each op then applied
   in turn. *)
type value = Const | Sum of read list * op list

(* [Store (b, v, x)] stores [x] into buffer [b] through [v]. *)
type store = Store of int * view * value

type step =
  | Assign of store
  | Assemble of store list (* into one buffer, marked disjoint *)
  | Crossed of store list * store list (* into two buffers, marked disjoint *)
  | Overlap of store list (* into one buffer, unmarked, may overlap *)

(* Buffers of [sizes] elements, each written whole with 3.5, then [steps]. *)
type program = { sizes : int array; steps : step list }

(* Printed as OCaml, so that a counterexample can be kept as an example. *)

let pp_view ppf = function
  | Whole -> Format.fprintf ppf "Whole"
  | Run (a, b) -> Format.fprintf ppf "Run (%d, %d)" a b
  | Rows (h, w, a, b) -> Format.fprintf ppf "Rows (%d, %d, %d, %d)" h w a b
  | Cols (h, w, a, b) -> Format.fprintf ppf "Cols (%d, %d, %d, %d)" h w a b
  | Rev (a, b) -> Format.fprintf ppf "Rev (%d, %d)" a b
  | Bound (a, b) -> Format.fprintf ppf "Bound (%d, %d)" a b
  | Bit (a, b) -> Format.fprintf ppf "Bit (%d, %d)" a b
  | Bytes (a, b) -> Format.fprintf ppf "Bytes (%d, %d)" a b

let pp_list pp ppf = function
  | [] -> Format.fprintf ppf "[]"
  | l ->
      let sep ppf () = Format.fprintf ppf ";@ " in
      Format.fprintf ppf "@[<hov 2>[ %a ]@]"
        (Format.pp_print_list ~pp_sep:sep pp)
        l

let pp_op ppf = function
  | Add c -> Format.fprintf ppf "Add %F" c
  | Mul c -> Format.fprintf ppf "Mul %F" c
  | Max c -> Format.fprintf ppf "Max %F" c

let pp_read ppf (At (b, k, v)) =
  Format.fprintf ppf "At (%d, %d, %a)" b k pp_view v

let pp_store ppf (Store (b, v, x)) =
  match x with
  | Const -> Format.fprintf ppf "Store (%d, %a, Const)" b pp_view v
  | Sum (reads, ops) ->
      Format.fprintf ppf "@[<hov 2>Store (%d, %a,@ Sum (%a,@ %a))@]" b pp_view v
        (pp_list pp_read) reads (pp_list pp_op) ops

(* A store a line, unless all fit on one. *)
let pp_stores ppf ss =
  let sep ppf () = Format.fprintf ppf ";@ " in
  Format.fprintf ppf "@[<hv 2>[ %a ]@]"
    (Format.pp_print_list ~pp_sep:sep pp_store)
    ss

let pp_step ppf = function
  | Assign s -> Format.fprintf ppf "Assign (%a)" pp_store s
  | Assemble ss -> Format.fprintf ppf "Assemble %a" pp_stores ss
  | Overlap ss -> Format.fprintf ppf "Overlap %a" pp_stores ss
  | Crossed (a, b) ->
      Format.fprintf ppf "@[<hv 9>Crossed (%a,@ %a)@]" pp_stores a pp_stores b

(* A step a line. *)
let pp_program ppf p =
  let sep ppf () = Format.fprintf ppf ";@," in
  Format.fprintf ppf
    "@[<v 2>{ sizes = [| %s |];@ steps =@;<1 2>@[<v 2>[ %a ]@] }@]"
    (String.concat "; " (Array.to_list (Array.map string_of_int p.sizes)))
    (Format.pp_print_list ~pp_sep:sep pp_step)
    p.steps

let dims_of n = function
  | Whole -> [ n ]
  | Run (a, b) | Rev (a, b) | Bound (a, b) | Bit (a, b) | Bytes (a, b) ->
      [ b - a ]
  | Rows (_, w, a, b) -> [ b - a; w ]
  | Cols (h, _, a, b) -> [ h; b - a ]

let numel n v = List.fold_left ( * ) 1 (dims_of n v)

(* The view [v] of the state [st] of a buffer of [n] elements. *)
let apply_view st n v =
  match v with
  | Whole -> st
  | Run (a, b) -> Shape.shrink st [ Some (int_ a, int_ b) ]
  | Rev (a, b) -> Shape.flip (Shape.shrink st [ Some (int_ a, int_ b) ]) [ 0 ]
  | Bound (a, b) ->
      let start =
        Shape.bind
          (Ops.variable
             ("start" ^ string_of_int (b - a))
             (`Int Bigint.zero)
             (`Int (Bigint.of_int n)))
          (`Int (Bigint.of_int a))
      in
      let len = Ops.int (b - a) in
      Shape.shrink st [ Some (Ops.Sym start, Ops.Sym Ops.O.(start + len)) ]
  | Rows (h, w, a, b) ->
      Shape.shrink
        (Shape.reshape st [ int_ h; int_ w ])
        [ Some (int_ a, int_ b); None ]
  | Cols (h, w, a, b) ->
      Shape.shrink
        (Shape.reshape st [ int_ h; int_ w ])
        [ None; Some (int_ a, int_ b) ]
  | Bit (a, b) ->
      Ops.bitcast
        (Ops.bitcast (Shape.shrink st [ Some (int_ a, int_ b) ]) Dtype.Int32)
        Dtype.Float32
  | Bytes (a, b) ->
      Ops.bitcast
        (Shape.shrink
           (Ops.bitcast st Dtype.Uint8)
           [ Some (int_ (a * 4), int_ (b * 4)) ])
        Dtype.Float32

let indices n v =
  match v with
  | Whole -> Array.init n Fun.id
  | Run (a, b) | Bound (a, b) | Bit (a, b) | Bytes (a, b) ->
      Array.init (b - a) (fun i -> a + i)
  | Rev (a, b) -> Array.init (b - a) (fun i -> b - 1 - i)
  | Rows (_, w, a, b) ->
      Array.init ((b - a) * w) (fun k -> ((a + (k / w)) * w) + (k mod w))
  | Cols (h, w, a, b) ->
      Array.init
        (h * (b - a))
        (fun k -> (k / (b - a) * w) + a + (k mod (b - a)))

let dst (Store (b, _, _)) = b

(* The program's writes: each buffer whole first, so that any later read of it
   holds what the program wrote, then its steps. *)
let writes p =
  List.init (Array.length p.sizes) (fun b -> Assign (Store (b, Whole, Const)))
  @ p.steps

let stores p =
  List.concat_map
    (function
      | Assign s -> [ s ]
      | Assemble ss | Overlap ss -> ss
      | Crossed (a, b) -> a @ b)
    p.steps

(* Generators: a seed's random choices *)

let pick r l = List.nth l (Random.State.int r (List.length l))
let int_in r lo hi = lo + Random.State.int r (hi - lo + 1)

let factorings n =
  List.filter_map
    (fun h -> if n mod h = 0 then Some (h, n / h) else None)
    (List.init n (fun i -> i + 1))

let gen_view r n =
  let run () =
    let a = int_in r 0 (n - 1) in
    Run (a, int_in r (a + 1) n)
  in
  match Random.State.int r 8 with
  | 0 -> Whole
  | 1 | 2 -> run ()
  | 3 ->
      let a = int_in r 0 (n - 1) in
      Rev (a, int_in r (a + 1) n)
  | 4 ->
      let a = int_in r 0 (n - 1) in
      let b = int_in r (a + 1) n in
      if Random.State.bool r then Bit (a, b) else Bytes (a, b)
  | k -> (
      match List.filter (fun (h, w) -> h > 1 && w > 1) (factorings n) with
      | [] -> run ()
      | fs ->
          let h, w = pick r fs in
          if k = 5 then
            let a = int_in r 0 (h - 1) in
            Rows (h, w, a, int_in r (a + 1) h)
          else
            let a = int_in r 0 (w - 1) in
            Cols (h, w, a, int_in r (a + 1) w))

(* A view of a buffer of [n] elements holding [m] of them, if one exists. *)
let view_of_numel r n m =
  if m > n then None
  else if m = n && Random.State.int r 4 = 0 then Some Whole
  else
    let a = int_in r 0 (n - m) in
    Some
      (match Random.State.int r 5 with
      | 0 -> Rev (a, a + m)
      | 1 -> Bit (a, a + m)
      | 2 -> Bytes (a, a + m)
      | _ -> Run (a, a + m))

let op_of = function 0, c -> Add c | 1, c -> Mul c | _, c -> Max c

(* The reads of a store of [m] elements: any buffer but [excluding], the targets
   of a write marked disjoint, whose promise stops its stores reading the
   elements another writes, through a view of the buffer's latest state or one
   up to two writes older. *)
let gen_value r sizes ~excluding m =
  if Random.State.int r 7 = 0 then Const
  else
    let usable =
      List.filter
        (fun b -> not (List.mem b excluding))
        (List.init (Array.length sizes) Fun.id)
    in
    let reads =
      List.filter_map
        (fun _ ->
          if usable = [] then None
          else
            let b = pick r usable in
            Option.map
              (fun v -> At (b, int_in r 0 2, v))
              (view_of_numel r sizes.(b) m))
        (List.init (int_in r 1 2) Fun.id)
    in
    if reads = [] then Const
    else
      Sum
        ( reads,
          List.init (int_in r 0 2) (fun _ ->
              op_of (Random.State.int r 3, Float.of_int (int_in r 2 7))) )

let gen_store r sizes b =
  let view = gen_view r sizes.(b) in
  let v = gen_value r sizes ~excluding:[] (numel sizes.(b) view) in
  Store (b, view, v)

let gen_assembly r sizes b =
  let n = sizes.(b) in
  let pairs l =
    let rec go = function a :: (c :: _ as t) -> (a, c) :: go t | _ -> [] in
    go l
  in
  let cuts n k =
    List.sort_uniq compare
      (0 :: n :: List.init (k - 1) (fun _ -> int_in r 1 (n - 1)))
  in
  let fs = List.filter (fun (h, w) -> h > 1 && w > 1) (factorings n) in
  let views =
    if n < 2 then []
    else if fs = [] || Random.State.int r 3 = 0 then
      let bound = Random.State.int r 3 = 0 in
      List.map
        (fun (a, c) -> if bound then Bound (a, c) else Run (a, c))
        (pairs (cuts n (int_in r 2 4)))
    else
      let h, w = pick r fs in
      if Random.State.bool r then
        List.map
          (fun (a, c) -> Rows (h, w, a, c))
          (pairs (cuts h (int_in r 2 4)))
      else
        List.map
          (fun (a, c) -> Cols (h, w, a, c))
          (pairs (cuts w (int_in r 2 4)))
  in
  if List.compare_length_with views 2 < 0 then Assign (gen_store r sizes b)
  else
    Assemble
      (List.map
         (fun view ->
           let v = gen_value r sizes ~excluding:[ b ] (numel n view) in
           Store (b, view, v))
         views)

(* Two columns of two equal buffers, stored crosswise from shared values. *)
let gen_crossed r sizes ~first_target =
  let buffers = List.init (Array.length sizes) Fun.id in
  let targets = List.filter (fun b -> b >= first_target) buffers in
  let same =
    List.concat_map
      (fun i ->
        List.filter_map
          (fun j ->
            if i < j && sizes.(i) = sizes.(j) && sizes.(i) mod 2 = 0 then
              Some (i, j)
            else None)
          targets)
      targets
  in
  if same = [] then None
  else
    let i, j = pick r same in
    let n = sizes.(i) in
    let h = n / 2 in
    let view c = Cols (h, 2, c, c + 1) in
    let shared () = gen_value r sizes ~excluding:[ i; j ] h in
    let v0 = shared () and v1 = shared () in
    Some
      (Crossed
         ( [ Store (i, view 0, v0); Store (i, view 1, v1) ],
           if Random.State.bool r then
             [ Store (j, view 0, v1); Store (j, view 1, v0) ]
           else [ Store (j, view 0, v0); Store (j, view 1, v1) ] ))

let gen_overlap r sizes b =
  let n = sizes.(b) in
  let fs = List.filter (fun (h, w) -> h > 1 && w > 1) (factorings n) in
  let view () =
    if fs = [] || Random.State.int r 3 = 0 then
      let a = int_in r 0 (n - 1) in
      Run (a, int_in r (a + 1) n)
    else
      let h, w = pick r fs in
      if Random.State.bool r then
        let a = int_in r 0 (h - 1) in
        Rows (h, w, a, int_in r (a + 1) h)
      else
        let a = int_in r 0 (w - 1) in
        Cols (h, w, a, int_in r (a + 1) w)
  in
  Overlap
    (List.init (int_in r 2 3) (fun _ ->
         let view = view () in
         let v = gen_value r sizes ~excluding:[] (numel n view) in
         Store (b, view, v)))

let gen_program r =
  let n_bufs = int_in r 3 6 in
  let sizes = Array.init n_bufs (fun _ -> pick r [ 1; 4; 6; 8; 12; 16 ]) in
  let first_target = n_bufs / 2 in
  let steps =
    List.init (int_in r 2 8) (fun _ ->
        let b = int_in r first_target (n_bufs - 1) in
        match Random.State.int r 8 with
        | 0 | 1 | 2 -> Assign (gen_store r sizes b)
        | 3 | 4 -> gen_assembly r sizes b
        | 5 -> gen_overlap r sizes b
        | _ -> (
            match gen_crossed r sizes ~first_target with
            | Some s -> s
            | None -> gen_assembly r sizes b))
  in
  { sizes; steps }

let program_of_seed seed = gen_program (Random.State.make [| seed |])

let programs =
  Gen.with_pp pp_program (Gen.map program_of_seed (Gen.int_range 1 1_000_000))

(* The graph *)

let build ?slots ?(device = "CPU") p =
  (* Storage takes a slot from the counter: a slot given by hand could be one a
     schedule's own storage takes, and hash-consing would make them one. [slots]
     are those of an earlier build, which are the counter's. *)
  let slot i = Option.map (fun s -> s.(i)) slots in
  let base =
    Array.init (Array.length p.sizes) (fun i ->
        Ops.new_buffer ?slot:(slot i) device p.sizes.(i) dt)
  in
  let states = Array.map (fun b -> [ b ]) base in
  let latest b = List.hd states.(b) in
  let state_at b k =
    let st = states.(b) in
    List.nth st (min k (List.length st - 1))
  in
  let value x dims =
    let broadcast c =
      Shape.expand
        (Shape.reshape (Ops.cast c dt) (List.map (fun _ -> int_ 1) dims))
        (List.map int_ dims)
    in
    match x with
    | Const -> broadcast (Ops.float 3.5)
    | Sum (reads, ops) ->
        let read (At (b, k, v)) =
          Shape.reshape
            (apply_view (state_at b k) p.sizes.(b) v)
            (List.map int_ dims)
        in
        let sum =
          match List.map read reads with
          | [] -> invalid_arg "a sum of no reads"
          | r :: rs -> List.fold_left Ops.add r rs
        in
        List.fold_left
          (fun x -> function
            | Add c -> Ops.add x (broadcast (Ops.float c))
            | Mul c -> Ops.mul x (broadcast (Ops.float c))
            | Max c -> Ops.maximum x (broadcast (Ops.float c)))
          sum ops
  in
  let store_node (Store (b, v, x)) =
    Ops.store
      (apply_view (latest b) p.sizes.(b) v)
      (Ops.cast (value x (dims_of p.sizes.(b) v)) dt)
  in
  let write ?(after = Ops.after) b stores =
    states.(b) <- after (latest b) stores :: states.(b)
  in
  List.iter
    (function
      | Assign s -> write (dst s) [ store_node s ]
      | Overlap ss -> write (dst (List.hd ss)) (List.map store_node ss)
      | Assemble ss ->
          write ~after:Ops.after_disjoint
            (dst (List.hd ss))
            (List.map store_node ss)
      | Crossed (a, b) ->
          let sa = List.map store_node a and sb = List.map store_node b in
          write ~after:Ops.after_disjoint (dst (List.hd a)) sa;
          write ~after:Ops.after_disjoint (dst (List.hd b)) sb)
    (writes p);
  (* Listed by buffer: the listing is part of the program, so it may not depend
     on node ids, which depend on the nodes that outlived earlier programs. *)
  (Ops.sink (Array.to_list (Array.map List.hd states)), base)

(* The spec: each write's values computed from the state before it. *)

let initial s j = f32 (float_of_int (((j * 7) + (s * 13) + 3) mod 11) /. 4.)

let spec p =
  let memory =
    Array.init (Array.length p.sizes) (fun s ->
        Array.init p.sizes.(s) (initial s))
  in
  (* Each buffer's states, the latest first. *)
  let states = Array.map (fun m -> [ Array.copy m ]) memory in
  let older b k = min k (List.length states.(b) - 1) in
  let read (At (b, k, v)) =
    let m = List.nth states.(b) (older b k) in
    Array.map (fun i -> m.(i)) (indices p.sizes.(b) v)
  in
  let value (Store (b, v, x)) =
    match x with
    | Const -> Array.make (numel p.sizes.(b) v) (f32 3.5)
    | Sum (reads, ops) ->
        let sum =
          match List.map read reads with
          | [] -> invalid_arg "a sum of no reads"
          | r :: rs ->
              List.fold_left
                (fun a b -> Array.mapi (fun i x -> f32 (x +. b.(i))) a)
                r rs
        in
        Array.map
          (fun x ->
            List.fold_left
              (fun x -> function
                | Add c -> f32 (x +. c)
                | Mul c -> f32 (x *. c)
                | Max c -> Float.max x c)
              x ops)
          sum
  in
  (* A store listed twice in one write is one store, which lands at its first
     place: views of the same elements of the latest state, of one shape, and
     the same reads of the same states make the same node. *)
  let same (Store (a, va, xa)) (Store (b, vb, xb)) =
    let named (At (b, k, v)) =
      ( b,
        older b k,
        indices p.sizes.(b) v,
        match v with Bit _ | Bytes _ -> true | _ -> false )
    in
    let n = p.sizes.(a) in
    a = b
    && dims_of n va = dims_of n vb
    && indices n va = indices n vb
    &&
    match (xa, xb) with
    | Const, Const -> true
    | Sum (ra, oa), Sum (rb, ob) ->
        oa = ob && List.map named ra = List.map named rb
    | _ -> false
  in
  let rec once = function
    | [] -> []
    | s :: rest -> s :: once (List.filter (fun t -> not (same s t)) rest)
  in
  let apply ss =
    let vs = List.map (fun s -> (s, value s)) (once ss) in
    List.iter
      (fun (Store (b, v, _), x) ->
        Array.iteri (fun k i -> memory.(b).(i) <- x.(k)) (indices p.sizes.(b) v))
      vs;
    List.iter
      (fun b -> states.(b) <- Array.copy memory.(b) :: states.(b))
      (List.sort_uniq Int.compare (List.map dst ss))
  in
  List.iter
    (function
      | Assign s -> apply [ s ]
      | Assemble ss | Overlap ss -> apply ss
      | Crossed (a, b) -> apply (a @ b))
    (writes p);
  memory

(* Interpreting a schedule: memory keyed by storage node. A launch runs its
   members forwards, each element a member writes noted with the member, then
   backwards from the memory before it; both leave the same memory. *)

type place = { node : Ops.t; off : int; len : int }

let sint_value ~vars = function
  | Ops.Int i -> i
  | Sym s -> (
      match Interpreter.eval ~vars s with
      | `Int z -> Bigint.to_int z
      | `Float x -> int_of_float x
      | _ -> failwith "a view of a non-integer offset")

let rec place_of ~vars u =
  match (Ops.op u, Ops.arg u) with
  | Op.After, _ | Op.Bitcast, _ -> place_of ~vars (Ops.nth u 0)
  | Op.Shrink, _ ->
      Option.map
        (fun pl ->
          match Shape.marg u with
          | Shrink [ (o, n) ] ->
              {
                pl with
                off = pl.off + sint_value ~vars o;
                len = sint_value ~vars n;
              }
          | _ -> failwith "a view of rank above 1 as a call's argument")
        (place_of ~vars (Ops.nth u 0))
  | (Param | Buffer | Alloc), Param p when p.addrspace <> Dtype.Alu ->
      Some { node = u; off = 0; len = Shape.max_numel u }
  | _ -> None

let store_through_two = "cannot evaluate a store through more than one index"
let is_kernel_call u = Ops.op u = Call && Ops.op (Ops.nth u 0) = Sink
let is_launch u = Ops.op u = Call && Shape.is_launch (Ops.nth u 0)
let count_launches linear = List.length (List.filter is_launch (Ops.src linear))

let to_float : Dtype.value -> float = function
  | `Float x -> x
  | _ -> invalid_arg "a float32 element of another type"

(* Memory holds float32: a store of a bitcast's integer leaves its bits. *)
let as_float : Dtype.value -> float = function
  | `Int z -> Int32.float_of_bits (Int32.of_int (Bigint.to_int z))
  | v -> to_float v

let contents = list (pair Uops.uop (array float_exact))

let run_linear ~base ~vars linear =
  let memory = Ops.Tbl.create 16 in
  let values_of pl =
    match Ops.Tbl.find_opt memory pl.node with
    | Some v -> v
    | None ->
        let v =
          match Array.find_index (fun b -> b == pl.node) base with
          | Some s -> Array.init (Shape.max_numel pl.node) (initial s)
          | None -> Array.make (Shape.max_numel pl.node) Float.nan
        in
        Ops.Tbl.replace memory pl.node v;
        v
  in
  let writes_of call =
    if not (is_kernel_call call) then failwith "a call of no kernel";
    let args = List.tl (Ops.src call) in
    let buffers, params =
      List.fold_left
        (fun (bs, ps) (k, a) ->
          match place_of ~vars a with
          | Some pl ->
              let elements = Array.sub (values_of pl) pl.off pl.len in
              ((k, Array.map (fun x -> `Float x) elements) :: bs, ps)
          | None -> (
              match Ops.arg a with
              | Param { bound = Some v; _ } -> (bs, (k, v) :: ps)
              | Param { name = Some n; _ } when List.mem_assoc n vars ->
                  (bs, (k, List.assoc n vars) :: ps)
              | _ -> (
                  match Shape.ssimplify a with
                  | Int n -> (bs, (k, Dtype.Value.of_int n) :: ps)
                  | Sym _ -> failwith "an argument of no value")))
        ([], [])
        (List.mapi (fun k a -> (k, a)) args)
    in
    List.map
      (fun (k, i, v) ->
        match place_of ~vars (List.nth args k) with
        | Some pl -> (pl, i, v)
        | None -> failwith "a write through no storage")
      (Interpreter.writes ~vars ~params ~buffers (Ops.nth call 0))
  in
  let apply (pl, i, v) = (values_of pl).(pl.off + i) <- as_float v in
  let snapshot () =
    List.sort
      (fun (a, _) (b, _) -> Ops.compare a b)
      (Ops.Tbl.fold (fun k v acc -> (k, Array.copy v) :: acc) memory [])
  in
  let restore saved =
    Ops.Tbl.reset memory;
    List.iter (fun (k, v) -> Ops.Tbl.replace memory k (Array.copy v)) saved
  in
  let launches = ref 0 in
  let run_launch members =
    incr launches;
    let saved = snapshot () in
    let writer = Ops.Tbl.create 8 in
    let note m (pl, i, _) =
      let elements =
        match Ops.Tbl.find_opt writer pl.node with
        | Some h -> h
        | None ->
            let h = Hashtbl.create 16 in
            Ops.Tbl.replace writer pl.node h;
            h
      in
      let e = pl.off + i in
      (match Hashtbl.find_opt elements e with
      | Some m' when m' <> m ->
          failf "members %d and %d of a launch write element %d of@.%a" m' m e
            Ops.pp pl.node
      | _ -> ());
      Hashtbl.replace elements e m
    in
    List.iteri
      (fun m call ->
        let ws = writes_of call in
        List.iter (note m) ws;
        List.iter apply ws)
      members;
    let forwards = snapshot () in
    restore saved;
    List.iter (fun call -> List.iter apply (writes_of call)) (List.rev members);
    equal ~msg:"memory after the members run backwards" contents forwards
      (snapshot ())
  in
  List.iter
    (fun e ->
      if is_launch e then run_launch (Shape.launch_members e)
      else List.iter apply (writes_of e))
    (Ops.src linear);
  (memory, !launches)

(* Schedules made under [f] are kept neither in memory nor on disk. *)
let uncached f = Setting.context [ B (Setting.scache, 0) ] f

(* A refusal to verify a graph of views states nothing of the schedule. *)
let verification_failed m =
  String.starts_with ~prefix:"UOp verification failed" m

let cover_steps p =
  let has f = List.exists f p.steps in
  cover "an assembly" (has (function Assemble _ -> true | _ -> false));
  cover "a crossed write" (has (function Crossed _ -> true | _ -> false));
  cover "an unmarked overlap" (has (function Overlap _ -> true | _ -> false))

(* Uncached: its programs' schedules, kept in memory, would hold the heap that a
   later law's collections walk. *)
let order_independent p =
  uncached @@ fun () ->
  cover_steps p;
  let big, base = build p in
  let linear, vars =
    match
      Schedule.create_linear_with_vars ~cores:Cores.host ~capturing:true big
    with
    | r -> r
    | exception Invalid_argument m when verification_failed m -> reject ()
  in
  let vars = List.map (fun (n, v) -> (n, `Int (Bigint.of_int v))) vars in
  (* A store through a bytes bitcast writes through two indices, which the
     interpreter does not read. *)
  let memory, launches =
    match run_linear ~base ~vars linear with
    | r -> r
    | exception Invalid_argument m when m = store_through_two -> reject ()
  in
  cover "a launch" (launches > 0);
  let expected = spec p in
  Array.iteri
    (fun s b ->
      let got =
        match Ops.Tbl.find_opt memory b with
        | Some v -> v
        | None -> Array.init p.sizes.(s) (initial s)
      in
      equal
        ~msg:(Printf.sprintf "buffer %d" s)
        (array float_exact) expected.(s) got)
    base

(* Process history: nodes are interned weakly, so a node that outlived an
   earlier build of a program keeps its id and one collected is made anew,
   younger, and a node another program built first is reused, with what was
   computed of it there. Node ids order nodes by that history alone, and no
   schedule may read it. *)

(* A schedule printed with the storage it made numbered by first use, above
   [slots], the program's own: a made slot is a process-wide counter's. The
   program's keep theirs, so that a swap of two of its calls shows. *)
let schedule_text ~slots (linear, vars) =
  let made u =
    match Ops.op u with
    | Alloc -> true
    | Buffer -> not (Array.mem (Ops.param_arg_of u).slot slots)
    | _ -> false
  in
  let storage = List.filter made (Ops.toposort ~calls:Enter linear) in
  let above = 1 + Array.fold_left max (-1) slots in
  let renumber i u =
    let p = Ops.param_arg_of u in
    (u, Ops.replace u ~arg:(Param { p with slot = above + i }))
  in
  let renumbered =
    Ops.substitute ~calls:Enter ~pass:Once linear (List.mapi renumber storage)
  in
  Format.asprintf "%a@.%s" Ops.pp renumbered
    (String.concat ","
       (List.map (fun (n, v) -> Printf.sprintf "%s=%d" n v) vars))

(* Whether each pair of [nodes] is ordered as [Ops.compare] orders them. *)
let id_order nodes =
  List.concat_map (fun a -> List.map (Ops.compare a) nodes) nodes

(* [rest] seeds what else the process does: which nodes outlive the first build,
   and which programs are scheduled before the second. *)
let same_program_same_schedule (p, rest) =
  let r = Random.State.make [| rest |] in
  let schedule big =
    match Schedule.create_linear_with_vars ~cores:Cores.host big with
    | s -> s
    | exception Invalid_argument m when verification_failed m -> reject ()
  in
  uncached @@ fun () ->
  let others = List.init (Random.State.int r 3) (fun _ -> gen_program r) in
  cover "another program scheduled in between" (others <> []);
  (* In a function of its own, so that only [kept] outlives it. *)
  let first () =
    let big, base = build p in
    let slots = Array.map (fun b -> (Ops.param_arg_of b).slot) base in
    let kept = List.filter (fun _ -> Random.State.bool r) (Ops.src big) in
    (schedule_text ~slots (schedule big), slots, kept, id_order (Ops.src big))
  in
  let text, slots, kept, order = first () in
  Gc.full_major ();
  (* A drawn part of each other program and of its schedule is kept. *)
  let keep q =
    let big, _ = build q in
    let scheduled =
      match Schedule.create_linear_with_vars ~cores:Cores.host big with
      | linear, _ -> Ops.toposort ~calls:Enter linear
      | exception Invalid_argument m when verification_failed m -> []
    in
    List.filter (fun _ -> Random.State.bool r) (Ops.src big @ scheduled)
  in
  let others_kept = List.concat_map keep others in
  Gc.full_major ();
  let big, _ = build ~slots p in
  cover "a node older in one build, younger in the other"
    (id_order (Ops.src big) <> order);
  let second = schedule big in
  cover "a launch" (count_launches (fst second) > 0);
  let shared = Ops.Tbl.create 64 in
  List.iter (fun u -> Ops.Tbl.replace shared u ()) others_kept;
  cover "a node of the schedule kept from another program's"
    (List.exists (Ops.Tbl.mem shared) (Ops.toposort ~calls:Enter (fst second)));
  equal
    ~msg:(Format.asprintf "after: %a" (pp_list pp_program) others)
    string text
    (schedule_text ~slots second);
  ignore (Sys.opaque_identity (kept, others_kept))

(* A store of one element and a store of many into other storage run as one
   launch whichever is listed first: the one element joins the many's, or the
   many the latest launch, the one's. The order the nodes were built in does not
   change it. *)
let listing_decides () =
  let launches ~built_first ~listed_first =
    uncached @@ fun () ->
    let store n =
      let b = Ops.new_buffer "CPU" n dt in
      let one = Shape.reshape (Ops.cast (Ops.float 3.5) dt) [ int_ 1 ] in
      Ops.after b [ Ops.store b (Shape.expand one [ int_ n ]) ]
    in
    let many, single =
      if built_first = `Many then
        let m = store 4 in
        (m, store 1)
      else
        let s = store 1 in
        (store 4, s)
    in
    let listing =
      if listed_first = `Many then [ many; single ] else [ single; many ]
    in
    let linear, _ =
      Schedule.create_linear_with_vars ~cores:Cores.host ~capturing:true
        (Ops.sink listing)
    in
    count_launches linear
  in
  let launches ~listed_first =
    let a = launches ~built_first:`Many ~listed_first in
    Gc.full_major ();
    equal int ~msg:"built in the other order" a
      (launches ~built_first:`One ~listed_first);
    a
  in
  equal int ~msg:"the many first" 1 (launches ~listed_first:`Many);
  equal int ~msg:"the one first" 1 (launches ~listed_first:`One)

(* The mark: an assembly of columns, which no byte range separates, runs as one
   launch only because it is marked; unmarked, it runs apart. *)

let cat ~marked =
  let device = "CPU" in
  let out = Ops.new_buffer ~slot:0 device 16 dt in
  let columns = Shape.reshape out [ int_ 4; int_ 4 ] in
  let piece c slot =
    let b = Ops.new_buffer ~slot device 4 dt in
    Ops.store
      (Shape.shrink columns [ None; Some (int_ c, int_ (c + 1)) ])
      (Shape.reshape b [ int_ 4; int_ 1 ])
  in
  let stores = List.init 4 (fun c -> piece c (c + 1)) in
  Ops.sink [ (if marked then Ops.after_disjoint else Ops.after) out stores ]

let cat_launches ~marked =
  count_launches
    (fst
       (Schedule.create_linear_with_vars ~cores:Cores.host ~capturing:true
          (cat ~marked)))

let marked_columns_launch () =
  equal int ~msg:"marked" 1 (cat_launches ~marked:true);
  equal int ~msg:"unmarked" 0 (cat_launches ~marked:false)

(* Two calls of functions compiled on their own write one buffer in turn, each
   through a write marked disjoint: the promise covers the stores of one write,
   so the kernels of the second call join no launch of the first's. *)
let calls_in_turn () =
  let device = "CPU" in
  let x = Ops.new_buffer ~slot:0 device 4 dt in
  let filled c =
    let p = Shape.param ~shape:[ int_ 4 ] ~device 0 dt in
    let at a b = Shape.shrink p [ Some (int_ a, int_ b) ] in
    let value n =
      Shape.expand
        (Shape.reshape (Ops.cast (Ops.float c) dt) [ int_ 1 ])
        [ int_ n ]
    in
    Ops.sink
      [
        Ops.after_disjoint p
          [ Ops.store (at 0 2) (value 2); Ops.store (at 2 4) (value 2) ];
      ]
  in
  let first = Ops.call ~precompile:true (filled 1.) [ x ] in
  let x1 = Ops.after x [ first ] in
  let second = Ops.call ~precompile:true (filled 2.) [ x1 ] in
  let big = Ops.sink [ Ops.after x1 [ second ] ] in
  let linear, _ =
    Schedule.create_linear_with_vars ~cores:Cores.host ~capturing:true big
  in
  equal int ~msg:"a launch a call" 2 (count_launches linear)

(* A write of no stores states nothing of the stores before it. Marking it
   leaves the state as it is, so an unmarked write under it gains no mark. *)
let no_stores_no_mark () =
  let x = Ops.new_buffer ~slot:0 "CPU" 4 dt in
  let written = Ops.after x [ Ops.store x (Ops.float 1.) ] in
  equal Uops.uop written (Ops.after_disjoint written [])

(* The mark's tag reaches the launches through a schedule read back from the
   disk: a child made it, and another reads it. Each child prints the bodies it
   scheduled anew ({!Schedule.schedules}), none where it read them back, and the
   launches of its schedule. *)

let launches_of_cat () =
  let launches = cat_launches ~marked:true in
  Printf.printf "%d %d\n" (Schedule.schedules ()) launches

let launches_read db =
  match Disk_cache.child ~cachedb:db "launches" with
  | Error err -> failf "the child failed: %s" err
  | Ok out -> Scanf.sscanf out "%d %d" (fun made launches -> (made, launches))

let disk_keeps_mark () =
  let db = Disk_cache.fresh () in
  let made, launches = launches_read db in
  at_least int ~msg:"bodies made" ~than:1 made;
  equal int ~msg:"launches made" 1 launches;
  equal (pair int int) ~msg:"read back" (0, 1) (launches_read db)

(* The law compiled: the schedule of a program whose buffers live on the host or
   on [kind]'s GPU, compiled for that device and run there, leaves what its
   graph states. *)

let device kind = if kind = "CPU" then Rig.host else Gpus.gpu kind

let runs_on kind p =
  let d = device kind in
  let through f = List.exists (fun (Store (_, v, _)) -> f v) (stores p) in
  cover_steps p;
  cover "a store through bytes"
    (through (function Bytes _ -> true | _ -> false));
  cover "a store from a bound start"
    (through (function Bound _ -> true | _ -> false));
  let big, base = build ~device:kind p in
  let linear, vars =
    Schedule.create_linear_with_vars ~cores:Cores.host ~capturing:true big
  in
  cover "a launch" (count_launches linear > 0);
  let compiled =
    Realize.compile_linear ~targets:(fun _ -> Run.target d) linear
  in
  let buffers =
    Array.mapi
      (fun s n ->
        Run.buffer d dt (Array.init n (fun j -> `Float (initial s j))))
      p.sizes
  in
  let bound = Array.to_list (Array.map2 (fun b x -> (b, x)) base buffers) in
  Tolk_engine.run ~vars
    (Tolk_engine.link ~devices:(fun _ -> d) ~bound compiled)
    [||];
  let expected = spec p in
  Array.iteri
    (fun s x ->
      equal
        ~msg:(Printf.sprintf "buffer %d" s)
        (array float_exact) expected.(s)
        (Array.map to_float (Run.values dt x)))
    buffers

(* Examples *)

(* Programs whose schedules raised internal errors before effects were ordered
   by the states they name, as the generator drew them from the seeds named. *)
let effects_misordered =
  [
    (* seed 323170 *)
    { sizes = [| 4; 1; 8; 16; 1 |];
      steps =
        [ Assemble [ Store (3, Cols (2, 8, 0, 3), Const);
                     Store (3, Cols (2, 8, 3, 8), Const) ];
          Assemble [ Store (3, Cols (4, 4, 0, 1),
                       Sum ([ At (0, 2, Run (0, 4)) ], []));
                     Store (3, Cols (4, 4, 1, 3),
                       Sum ([ At (2, 2, Bit (0, 8)) ], [ Add 5.; Mul 5. ]));
                     Store (3, Cols (4, 4, 3, 4), Const) ];
          Assign (Store (2, Cols (2, 4, 2, 3),
                    Sum ([ At (3, 1, Bytes (8, 10)) ], [ Max 3.; Max 3. ])));
          Overlap [ Store (2, Run (6, 7), Sum ([ At (1, 0, Run (0, 1)) ],
                      [ Max 6.; Add 3. ]));
                    Store (2, Rows (4, 2, 1, 3), Const) ];
          Assign (Store (4, Run (0, 1), Sum ([ At (4, 0, Bit (0, 1)) ], [])));
          Assemble [ Store (2, Rows (4, 2, 0, 1), Const);
                     Store (2, Rows (4, 2, 1, 3), Const);
                     Store (2, Rows (4, 2, 3, 4),
                       Sum ([ At (0, 2, Run (0, 2)); At (3, 0, Run (11, 13)) ],
                       [ Add 7. ])) ];
          Overlap [ Store (2, Rows (4, 2, 2, 4),
                      Sum ([ At (2, 1, Bit (0, 4)) ], [ Add 6. ]));
                    Store (2, Rows (4, 2, 3, 4), Const);
                    Store (2, Cols (4, 2, 1, 2),
                      Sum ([ At (3, 2, Bytes (7, 11)) ], [ Add 7.; Add 3. ])) ];
          Assemble [ Store (3, Run (0, 1), Sum ([ At (2, 2, Bytes (6, 7)) ],
                       []));
                     Store (3, Run (1, 12), Const);
                     Store (3, Run (12, 15), Const);
                     Store (3, Run (15, 16),
                       Sum ([ At (1, 1, Bytes (0, 1)); At (2, 0, Rev (7, 8)) ],
                       [ Mul 6.; Mul 5. ])) ] ] };
    (* seed 267777 *)
    { sizes = [| 8; 16; 4; 12 |];
      steps =
        [ Assemble [ Store (2, Cols (2, 2, 0, 1),
                       Sum ([ At (1, 2, Rev (8, 10)) ], []));
                     Store (2, Cols (2, 2, 1, 2),
                       Sum ([ At (1, 1, Rev (0, 2)) ], [])) ];
          Assign (Store (2, Cols (2, 2, 1, 2),
                    Sum ([ At (3, 0, Run (6, 8)); At (2, 1, Bytes (0, 2)) ],
                    [])));
          Assign (Store (3, Run (2, 7), Sum ([ At (0, 0, Bytes (3, 8)) ],
                    [ Mul 7. ])));
          Overlap [ Store (3, Run (4, 9), Sum ([ At (3, 1, Rev (6, 11)) ], []));
                    Store (3, Cols (3, 4, 3, 4), Const) ];
          Assign (Store (2, Rows (2, 2, 0, 2),
                    Sum ([ At (0, 1, Bytes (3, 7)) ], [])));
          Overlap [ Store (3, Run (10, 11), Sum ([ At (0, 2, Run (1, 2)) ],
                      [ Add 7. ]));
                    Store (3, Rows (4, 3, 1, 4), Const) ];
          Assign (Store (3, Run (3, 11), Const));
          Assemble [ Store (3, Cols (6, 2, 0, 1),
                       Sum ([ At (0, 1, Rev (1, 7)) ], [ Max 3. ]));
                     Store (3, Cols (6, 2, 1, 2),
                       Sum ([ At (0, 0, Bit (1, 7)) ], [ Add 3. ])) ] ] };
    (* seed 160398 *)
    { sizes = [| 4; 6; 4; 12 |];
      steps =
        [ Overlap [ Store (3, Rows (6, 2, 2, 3),
                      Sum ([ At (3, 1, Bit (10, 12)) ], [ Mul 7.; Mul 7. ]));
                    Store (3, Rows (4, 3, 1, 4), Const) ];
          Assemble [ Store (3, Rows (3, 4, 0, 1),
                       Sum ([ At (0, 1, Bit (0, 4)) ], [ Add 2. ]));
                     Store (3, Rows (3, 4, 1, 3), Const) ];
          Overlap [ Store (2, Rows (2, 2, 1, 2),
                      Sum ([ At (3, 0, Rev (5, 7)) ], [ Mul 3. ]));
                    Store (2, Cols (2, 2, 1, 2), Const) ];
          Assign (Store (3, Rev (9, 10), Sum ([ At (1, 0, Run (0, 1)) ], [])));
          Assemble [ Store (2, Rows (2, 2, 0, 1),
                       Sum ([ At (1, 1, Run (4, 6)); At (3, 1, Bytes (4, 6)) ],
                       [ Add 3. ]));
                     Store (2, Rows (2, 2, 1, 2),
                       Sum ([ At (0, 2, Run (0, 2)) ], [])) ];
          Assign (Store (2, Run (0, 1),
                    Sum ([ At (0, 2, Rev (2, 3)); At (2, 1, Rev (2, 3)) ], []))) ] };
    (* seed 401830 *)
    { sizes = [| 6; 8; 1; 1 |];
      steps =
        [ Assign (Store (2, Whole, Sum ([ At (3, 1, Bytes (0, 1)) ],
                    [ Max 4. ])));
          Assign (Store (2, Run (0, 1),
                    Sum ([ At (3, 2, Whole); At (3, 0, Whole) ], [])));
          Overlap [ Store (3, Run (0, 1), Const);
                    Store (3, Run (0, 1), Sum ([ At (1, 0, Bytes (3, 4)) ],
                      [ Max 5.; Mul 2. ]));
                    Store (3, Run (0, 1), Const) ] ] };
    (* seed 879636 *)
    { sizes = [| 6; 16; 1; 16 |];
      steps =
        [ Assign (Store (3, Rows (4, 4, 1, 3), Sum ([ At (3, 2, Run (8, 16)) ],
                    [ Max 5. ])));
          Assign (Store (3, Run (13, 16), Const));
          Overlap [ Store (2, Run (0, 1), Sum ([ At (2, 1, Whole) ], []));
                    Store (2, Run (0, 1),
                      Sum ([ At (1, 1, Bit (7, 8)); At (1, 2, Rev (8, 9)) ],
                      [ Mul 6.; Mul 3. ]));
                    Store (2, Run (0, 1), Sum ([ At (2, 2, Run (0, 1)) ], [])) ] ] };
  ]
[@@ocamlformat "disable"]

(* A program of every write the compiled law covers, and a launch. *)
let every_write =
  [
    (* seed 118508 *)
    { sizes = [| 6; 1; 4; 4 |];
      steps =
        [ Assign (Store (2, Rev (2, 4), Const));
          Assemble [ Store (2, Bound (0, 2), Const);
                     Store (2, Bound (2, 4), Const) ];
          Assign (Store (2, Bytes (3, 4),
                    Sum ([ At (3, 0, Bytes (1, 2)); At (0, 1, Bytes (5, 6)) ],
                    [ Mul 3. ])));
          Assign (Store (2, Cols (2, 2, 1, 2), Sum ([ At (0, 0, Bit (3, 5)) ],
                    [ Max 5. ])));
          Crossed ([ Store (2, Cols (2, 2, 0, 1), Const);
                     Store (2, Cols (2, 2, 1, 2), Const) ],
                   [ Store (3, Cols (2, 2, 0, 1), Const);
                     Store (3, Cols (2, 2, 1, 2), Const) ]);
          Overlap [ Store (3, Rows (2, 2, 0, 2),
                      Sum ([ At (3, 0, Run (0, 4)) ], []));
                    Store (3, Run (3, 4),
                      Sum ([ At (3, 1, Run (3, 4)); At (2, 2, Bit (0, 1)) ],
                      [ Max 7. ])) ] ] };
  ]
[@@ocamlformat "disable"]

(* Over 1 s: a case compiles about nine fresh programs, 4.4 ms each on the host,
   the cost of tolk's compile. *)
let compiled kind =
  let where = if kind = "CPU" then "the host" else kind in
  prop ~count:120 ~examples:every_write
    ("on " ^ where ^ ", a schedule's launches write what its graph states")
    programs (runs_on kind)

let () =
  Disk_cache.play [ ("launches", launches_of_cat) ];
  exit
  @@ run "Tolk.Launch_law"
       [
         group "create_linear_with_vars"
           [
             (* Over 1 s: each case schedules its program two to four times, at
                tolk's 2.7 ms a program, and collects the heap twice. *)
             prop
               "a schedule does not depend on what the process built before it"
               (Gen.pair programs (Gen.int_range 1 1_000_000))
               same_program_same_schedule;
             group "launches"
               ([
                  (* Over 1 s: a case spends about 2.7 ms scheduling its
                     program, tolk's cost per program (get_kernel_graph 1.4 ms),
                     spread over the rewrite engine. *)
                  prop ~count:1500 ~examples:effects_misordered
                    "a schedule's launches write what its graph states, in \
                     either order of their members"
                    programs order_independent;
                  test
                    "a one-element store and a store of many run as one \
                     launch, whichever is listed or built first"
                    listing_decides;
                  test
                    "an assembly's columns run as one launch only where their \
                     write is marked disjoint"
                    marked_columns_launch;
                  test "marked writes of two calls run in turn" calls_in_turn;
                  test "a marked write of no stores marks nothing"
                    no_stores_no_mark;
                  test "a schedule read back keeps a marked write's launch"
                    disk_keeps_mark;
                ]
               @ List.map compiled ("CPU" :: Gpus.kinds));
           ];
       ]
