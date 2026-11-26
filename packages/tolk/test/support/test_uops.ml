open Windtrap
open Tolk

let cpu = "CPU"
let int32 n = Ops.int ~dtype:Int32 n

(* uop *)

let uop =
  group "uop"
    [
      test "compares nodes by their structure" (fun () ->
          equal Uops.uop Ops.O.(int32 2 + int32 1) Ops.O.(int32 2 + int32 1);
          not_equal Uops.uop Ops.O.(int32 2 + int32 2) Ops.O.(int32 2 * int32 2));
      prop "orders nodes as Ops.compare does"
        Gen.(pair (int_range (-3) 3) (int_range (-3) 3))
        (fun (a, b) ->
          let a = Ops.sink [ int32 a ] and b = Ops.sink [ int32 b ] in
          let order = Option.get (Testable.compare Uops.uop) in
          equal int (Ops.compare a b) (order a b));
      test "prints the graph under a node in the graph format" (fun () ->
          let u = Ops.sink [ Ops.O.(int32 2 + int32 2) ] in
          equal text
            (String.trim (Graph.to_string u))
            (Testable.to_string Uops.uop u));
    ]

(* numbered_like *)

type kind = Buffer | Alloc

let kind =
  Testable.make ~equal:( = ) ~pp:(fun ppf k ->
      Format.pp_print_string ppf
        (match k with Buffer -> "Buffer" | Alloc -> "Alloc"))

let storage_of kind slot =
  match kind with
  | Buffer -> Ops.new_buffer ~slot cpu 4 Dtype.Float32
  | Alloc -> Shape.alloc ~slot ~device:cpu [ Int 4 ] Float32

(* [graph places slots] is a sink of storage, the [i]th of the kind [places.(i)]
   names in [slots.(i)], at the top or, where [places.(i)] says so, in the body
   of a call. *)
let graph places slots =
  let nodes =
    List.map2
      (fun (kind, in_body) slot -> (in_body, storage_of kind slot))
      places slots
  in
  let at body =
    List.filter_map (fun (b, n) -> if b = body then Some n else None) nodes
  in
  match at true with
  | [] -> Ops.sink (at false)
  | inner -> Ops.sink (at false @ [ Ops.call (Ops.sink inner) [] ])

(* The kind and slot of each storage of [u], in visit order. *)
let storage u =
  List.filter_map
    (fun n ->
      match Ops.op n with
      | Op.Buffer -> Some (Buffer, (Ops.param_arg_of n).slot)
      | Alloc -> Some (Alloc, (Ops.param_arg_of n).slot)
      | _ -> None)
    (Ops.toposort ~calls:Enter u)

(* One graph's places and two numberings of them, each of distinct slots, drawn
   from a few so that the numberings often share slots. *)
let numberings =
  Gen.(
    let* n = int_range 0 4 in
    let place = pair (map (fun b -> if b then Buffer else Alloc) bool) bool in
    let slots =
      map (List.filteri (fun i _ -> i < n)) (permutation [ 0; 1; 2; 3; 4 ])
    in
    triple (list ~size:(constant n) place) slots slots)

(* Whether some slot of [like] is [given]'s at another place of the same kind,
   and whether two such slots swap. *)
let crossings places like given =
  let n = List.length places in
  let crosses i j =
    i <> j
    && fst (List.nth places i) = fst (List.nth places j)
    && List.nth like i = List.nth given j
  in
  let pairs =
    List.concat_map
      (fun i -> List.init n (fun j -> (i, j)))
      (List.init n Fun.id)
  in
  ( List.exists (fun (i, j) -> crosses i j) pairs,
    List.exists (fun (i, j) -> crosses i j && crosses j i) pairs )

let renumbers (places, like, given) =
  let crosses, swaps = crossings places like given in
  cover "a slot of one graph is another's in the other" crosses;
  cover "two slots swap" swaps;
  cover "storage in a call body" (List.exists snd places);
  cover "both kinds"
    (List.mem Buffer (List.map fst places)
    && List.mem Alloc (List.map fst places));
  equal
    (list (pair kind int))
    (storage (graph places like))
    (storage (Uops.numbered_like (graph places like) (graph places given)))

let numbered_like =
  let two_buffers = [ (Buffer, false); (Buffer, false) ] in
  group "numbered_like"
    [
      prop
        "puts storage in the slots of like's of the same kind, in visit order"
        ~examples:
          [
            (* A swap and a chain, which a substitution to a fixed point cycled
               on and numbered wrong. *)
            (two_buffers, [ 4; 3 ], [ 3; 4 ]);
            (two_buffers, [ 4; 5 ], [ 3; 4 ]);
            ([ (Alloc, true); (Buffer, false) ], [ 1; 2 ], [ 3; 4 ]);
          ]
        numberings renumbers;
      cases "is u where the graphs make different numbers of storage of a kind"
        ~id:fst
        [
          ( "one more buffer",
            [ (Buffer, false); (Buffer, false); (Buffer, false) ] );
          ("as many in all, of other kinds", [ (Buffer, false); (Alloc, false) ]);
        ]
        (fun (_, places) ->
          let u = graph two_buffers [ 7; 8 ] in
          let like = graph places (List.init (List.length places) Fun.id) in
          equal Uops.uop u (Uops.numbered_like like u));
    ]

(* binaries_as_sources *)

(* [program source binary] is a compiled program of the text [source] whose
   binary is [binary], if any. *)
let program ?binary source =
  Ops.v Program
    ~src:
      ([
         Ops.sink ~kernel:(Ops.kernel_info ()) [];
         Ops.v Linear;
         Ops.v Source ~arg:(String source);
       ]
      @ Option.to_list
          (Option.map (fun b -> Ops.v Binary ~arg:(Bytes b)) binary))

let binaries u =
  List.map Uops.binary_of
    (List.filter (fun n -> Ops.op n = Program) (Ops.toposort ~calls:Enter u))

let binaries_as_sources =
  group "binaries_as_sources"
    [
      test "gives each program its source's bytes, call bodies included"
        (fun () ->
          let u =
            Ops.sink
              [
                program "a" ~binary:"\x7fELF a";
                Ops.call (Ops.sink [ program "b" ~binary:"\x7fELF b" ]) [];
              ]
          in
          equal (list string) [ "a"; "b" ]
            (binaries (Uops.binaries_as_sources u)));
      (* Programs of one compiled text share one binary node. *)
      test "gives two programs of one binary each its own source" (fun () ->
          let u =
            Ops.sink
              [ program "a" ~binary:"\x7fELF"; program "b" ~binary:"\x7fELF" ]
          in
          equal (list string) [ "a"; "b" ]
            (binaries (Uops.binaries_as_sources u)));
      test "keeps a program of no binary" (fun () ->
          let u = Ops.sink [ program "a" ] in
          equal Uops.uop u (Uops.binaries_as_sources u));
    ]

(* Compiled programs *)

let binary_of =
  group "binary_of"
    [
      test "is the bytes of the program's last source" (fun () ->
          equal string "\x7fELF\x00lib"
            (Uops.binary_of (program "a" ~binary:"\x7fELF\x00lib")));
      test "raises Invalid_argument for a program of no binary" (fun () ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Uops.binary_of (program "a")));
    ]

let () =
  exit (run "Uops" [ uop; numbered_like; binaries_as_sources; binary_of ])
