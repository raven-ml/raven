(* Shape.mop_cleanup: the merges and removals of movements and indexing that
   shape.mli states, as their goldens record them, and the law that a cleanup
   keeps the elements a chain of movements denotes. *)

open Windtrap
open Tolk

let var = Simplification.var
let int = Ops.int
let ints = List.map (fun n : Ops.sint -> Int n)
let storage shape = Shape.param ~shape:(ints shape) 0 Float32

let cleanup u =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
    (After_sources Shape.mop_cleanup)

let index u idxs = Ops.v Op.Index ~src:(u :: idxs)
let element u i = index u [ int i ]

(* Movements as nodes of one movement each, which [Shape.shrink] and the other
   builders would merge or skip. *)
let shrink u bounds =
  Shape.mop u (Shrink (List.map (fun (s, n) -> (Ops.Int s, Ops.Int n)) bounds))

let reshape u shape = Shape.mop u (Reshape (ints shape))
let permute u order = Shape.mop u (Permute order)

(* A golden holds a graph and its cleanup. *)
let cleans name u =
  Golden.graph (name ^ ".golden") (fun () -> Ops.sink [ u; cleanup u ])

let shrinks =
  group "merges a shrink of a shrink as its goldens state"
    [
      cleans "merge_two_shrinks"
        (shrink
           (shrink (storage [ 8; 10 ]) [ (1, 6); (2, 7) ])
           [ (2, 2); (3, 4) ]);
      cleans "merge_three_shrinks"
        (shrink
           (shrink (shrink (storage [ 16 ]) [ (1, 12) ]) [ (2, 8) ])
           [ (3, 4) ]);
      Golden.graph "merge_shrinks_of_symbolic_starts.golden" (fun () ->
          let start = Ops.Sym (var "o" 0 4) and size = Ops.Sym (var "n" 1 3) in
          let inner = Shape.mop (storage [ 16 ]) (Shrink [ (start, Int 8) ]) in
          let u = Shape.mop inner (Shrink [ (Int 2, size) ]) in
          Ops.sink [ u; cleanup u ]);
    ]

let reshapes =
  group
    "merges a reshape of a reshape, and removes a reshape to its source's \
     shape, as its goldens state"
    [
      cleans "merge_two_reshapes"
        (reshape (reshape (storage [ 8; 10 ]) [ 80 ]) [ 4; 20 ]);
      cleans "merge_reshapes_back_to_the_source_shape"
        (reshape (reshape (storage [ 8; 10 ]) [ 80 ]) [ 8; 10 ]);
      cleans "remove_a_reshape_to_the_source_shape"
        (reshape (permute (storage [ 8; 10 ]) [ 1; 0 ]) [ 10; 8 ]);
    ]

let permutes =
  group "composes a permutation of a permutation as its goldens state"
    [
      cleans "merge_two_permutes"
        (permute (permute (storage [ 2; 3; 4 ]) [ 1; 2; 0 ]) [ 0; 2; 1 ]);
      cleans "merge_inverse_permutes_into_the_source"
        (permute (permute (storage [ 2; 3; 4 ]) [ 1; 2; 0 ]) [ 2; 0; 1 ]);
      cleans "remove_the_identity_permute" (permute (storage [ 2; 3 ]) [ 0; 1 ]);
      cleans "keep_a_permute_that_moves_an_axis"
        (permute (storage [ 2; 3 ]) [ 1; 0 ]);
    ]

let stacks =
  let x = storage [ 3 ] in
  let stack_of order = Ops.v Op.Stack ~src:(List.map (element x) order) in
  group
    "removes a stack of a node's elements in order and of its shape, as its \
     goldens state"
    [
      cleans "stack_the_elements_of_a_node_in_order" (stack_of [ 0; 1; 2 ]);
      cleans "keep_a_stack_of_elements_out_of_order" (stack_of [ 1; 0; 2 ]);
      cleans "keep_a_stack_of_some_elements" (stack_of [ 0; 1 ]);
    ]

let stack_indexing =
  group "indexes a stack by a constant as its source, as its goldens state"
    [
      cleans "index_a_stack_by_a_constant"
        (index (Shape.stack [ var "a" 0 9; var "b" 0 9 ]) [ int 1 ]);
      cleans "index_a_stack_by_a_constant_and_further_indices"
        (index
           (Shape.stack
              [ storage [ 4 ]; Shape.param ~shape:(ints [ 4 ]) 1 Float32 ])
           [ int 1; var "j" 0 3 ]);
    ]

let index_indexing =
  let i = var "i" 0 3 and j = var "j" 0 4 in
  group "merges an index of an index by scalars as its golden states"
    [
      cleans "index_an_index_by_scalars"
        (index (index (storage [ 4; 5 ]) [ i ]) [ j ]);
    ]

let shaped_indexing =
  let i = var "i" 0 3 and j = var "j" 0 4 in
  let table = Shape.param ~shape:(ints [ 4; 5 ]) 1 Int32 in
  group
    "indexes storage indexed by a shaped index by the element it selects, as \
     its goldens state"
    [
      cleans "index_an_index_of_a_shaped_index"
        (index (index (storage [ 20 ]) [ table ]) [ i; j ]);
      cleans "keep_an_index_of_a_shaped_index_by_fewer_indices"
        (index (index (storage [ 20 ]) [ table ]) [ i ]);
    ]

(* The elements a chain denotes *)

(* The elements a chain of movements over storage denotes: its shape, and the
   position in the storage of each of its elements, in row-major order. *)
let rec elements u =
  let sizes u =
    List.map
      (function Ops.Int n -> n | Sym _ -> invalid_arg "a symbolic size")
      (Shape.shape u)
  in
  let unravel shape flat =
    List.fold_right
      (fun d (idx, rest) -> ((rest mod d) :: idx, rest / d))
      shape ([], flat)
    |> fst
  in
  let ravel shape idx =
    List.fold_left2 (fun acc d k -> (acc * d) + k) 0 shape idx
  in
  let gather u source =
    let shape = sizes u in
    let n = List.fold_left ( * ) 1 shape in
    (shape, Array.init n (fun flat -> source (unravel shape flat)))
  in
  match (Ops.op u, Ops.src u) with
  | Op.Param, _ -> ([ Shape.max_numel u ], Array.init (Shape.max_numel u) Fun.id)
  | Op.Reshape, s :: _ -> (sizes u, snd (elements s))
  | Op.Shrink, s :: _ ->
      let shape, e = elements s in
      let starts =
        match Shape.marg u with
        | Shrink bounds ->
            List.map
              (function
                | Ops.Int n, _ -> n | _ -> invalid_arg "a symbolic start")
              bounds
        | _ -> assert false
      in
      gather u (fun idx -> e.(ravel shape (List.map2 ( + ) idx starts)))
  | Op.Permute, s :: _ ->
      let shape, e = elements s in
      let order =
        match Shape.marg u with Permute order -> order | _ -> assert false
      in
      gather u (fun idx ->
          let src = Array.make (List.length shape) 0 in
          List.iteri (fun k axis -> src.(axis) <- List.nth idx k) order;
          e.(ravel shape (Array.to_list src)))
  | o, _ -> invalid_arg (Format.asprintf "not a movement: %a" Op.pp o)

(* A chain of movements from storage of [start], drawn as the seeds that pick
   each step on the shape it meets, so that a failing chain shrinks. *)
type step = Shrink of int | Reshape of int | Permute of int
type chain = { start : int list; steps : step list }

let pp_chain ppf c =
  let pp_step ppf = function
    | Shrink s -> Format.fprintf ppf "shrink %d" s
    | Reshape s -> Format.fprintf ppf "reshape %d" s
    | Permute s -> Format.fprintf ppf "permute %d" s
  in
  Format.fprintf ppf "[%a] then %a"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       Format.pp_print_int)
    c.start
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ", ")
       pp_step)
    c.steps

let rec permutations = function
  | [] -> [ [] ]
  | l ->
      List.concat_map
        (fun x ->
          List.map (fun p -> x :: p) (permutations (List.filter (( <> ) x) l)))
        l

let apply u s =
  let shape =
    List.map (function Ops.Int n -> n | Sym _ -> assert false) (Shape.shape u)
  in
  match s with
  | Shrink seed ->
      (* an axis of no element keeps its empty whole *)
      let bound k d =
        if d = 0 then (0, 0)
        else
          let h = Hashtbl.hash (seed, k) in
          let start = h mod d in
          (start, 1 + (h / d mod (d - start)))
      in
      shrink u (List.mapi bound shape)
  | Reshape seed ->
      let n = List.fold_left ( * ) 1 shape in
      let splits =
        List.filter_map
          (fun f -> if n mod f = 0 then Some [ f; n / f ] else None)
          [ 2; 3; 4; 6 ]
      in
      let merged =
        match shape with d0 :: d1 :: rest -> [ (d0 * d1) :: rest ] | _ -> []
      in
      let shapes = [ n ] :: (splits @ merged) in
      reshape u (List.nth shapes (seed mod List.length shapes))
  | Permute seed ->
      let orders = permutations (List.init (List.length shape) Fun.id) in
      permute u (List.nth orders (seed mod List.length orders))

let chain c = List.fold_left apply (storage c.start) c.steps

(* Starting shapes, with an axis of no element and axes of one. *)
let starts = [ [ 2; 3; 4 ]; [ 6 ]; [ 0; 3 ]; [ 1; 4 ]; [ 2; 1; 3 ]; [ 1 ] ]

let gen_chain =
  let open Gen in
  let seed = int_range 0 1000 in
  let step =
    frequency
      [
        (2, map (fun s -> Shrink s) seed);
        (1, map (fun s -> Reshape s) seed);
        (2, map (fun s -> Permute s) seed);
      ]
  in
  with_pp pp_chain
    (let+ start = of_list starts and+ steps = list ~size:(int_range 1 5) step in
     { start; steps })

(* Whether some node of [u] is [outer] of [inner]. *)
let has outer inner u =
  List.exists
    (fun n -> Ops.op n = outer && Ops.op (Ops.nth n 0) = inner)
    (Ops.toposort ~calls:Skip u)

let reshapes_to_its_source u =
  List.exists
    (fun n ->
      Ops.op n = Op.Reshape
      && List.equal Shape.Sint.equal (Shape.shape n) (Shape.shape (Ops.nth n 0)))
    (Ops.toposort ~calls:Skip u)

(* One chain for each rule and each starting shape the law covers. *)
let named_chains =
  [
    { start = [ 2; 3; 4 ]; steps = [ Shrink 1; Shrink 2 ] };
    { start = [ 2; 3; 4 ]; steps = [ Reshape 0; Reshape 1 ] };
    { start = [ 2; 3; 4 ]; steps = [ Permute 3; Permute 4 ] };
    { start = [ 6 ]; steps = [ Reshape 0 ] };
    { start = [ 0; 3 ]; steps = [ Permute 1; Shrink 5 ] };
    { start = [ 1; 4 ]; steps = [ Shrink 7; Reshape 2 ] };
  ]

let keeps_the_elements c =
  let u = chain c in
  let cleaned = cleanup u in
  cover "a shrink of a shrink" (has Op.Shrink Op.Shrink u);
  cover "a reshape of a reshape" (has Op.Reshape Op.Reshape u);
  cover "a permutation of a permutation" (has Op.Permute Op.Permute u);
  cover "a reshape to its source's shape" (reshapes_to_its_source u);
  cover "an axis of no element" (List.mem 0 c.start);
  cover "an axis of one element" (List.mem 1 c.start);
  equal
    (pair (list Windtrap.int) (array Windtrap.int))
    (elements u) (elements cleaned)

let mop_cleanup =
  group "mop_cleanup"
    [
      shrinks;
      reshapes;
      permutes;
      stacks;
      stack_indexing;
      index_indexing;
      shaped_indexing;
      prop "keeps the elements a chain of movements denotes"
        ~examples:named_chains gen_chain keeps_the_elements;
    ]
