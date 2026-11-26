open Windtrap
open Tolk
open Dtypes

let targets =
  [
    ("cuda_sm75", Tc.cuda_sm75);
    ("cuda_sm80", Tc.cuda_sm80);
    ("cuda_sm89", Tc.cuda_sm89);
    ("amd_rdna3", Tc.amd_rdna3);
    ("amd_rdna4", Tc.amd_rdna4);
    ("amd_cdna3", Tc.amd_cdna3);
    ("amd_cdna4", Tc.amd_cdna4);
    ("metal", Tc.metal);
  ]

(* Every core, named by its target and position: [metal[0]]. *)
let named_cores =
  List.concat_map
    (fun (target, cores) ->
      List.mapi (fun i tc -> (Printf.sprintf "%s[%d]" target i, tc)) cores)
    targets

(* Every core and its transpose, [metal[0]] and [metal[0] transposed]. *)
let cores_and_transposes =
  List.concat_map
    (fun (name, tc) -> [ (name, tc); (name ^ " transposed", Tc.transpose tc) ])
    named_cores

let core = Testable.make ~pp:Tc.pp ~equal:Tc.equal

(* A bit as the cells write it: its dimension in lower case and its index. *)
let bit_to_string = function
  | Tc.M i -> Printf.sprintf "m%d" i
  | Tc.N i -> Printf.sprintf "n%d" i
  | Tc.K i -> Printf.sprintf "k%d" i

let pp_bit ppf b = Format.pp_print_string ppf (bit_to_string b)
let bit = Testable.make ~pp:pp_bit ~equal:Tc.equal_bit
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f
let is_k = function Tc.K _ -> true | M _ | N _ -> false
let bits (f : Tc.fragment) = f.lanes @ f.elements
let fragment lanes elements = { Tc.lanes; elements }

(* Cells: bits separated by spaces, [m0 k1]; a fragment as its lanes and its
   elements separated by a slash, [m0 k1/k0]. *)

let bit_of_string s =
  let index = int_of_string (String.sub s 1 (String.length s - 1)) in
  match s.[0] with
  | 'm' -> Tc.M index
  | 'n' -> Tc.N index
  | 'k' -> Tc.K index
  | _ -> invalid_arg s

let bits_of_cell s =
  String.split_on_char ' ' s
  |> List.filter (( <> ) "")
  |> List.map bit_of_string

let fragment_of_cell s =
  match String.split_on_char '/' s with
  | [ lanes; elements ] -> fragment (bits_of_cell lanes) (bits_of_cell elements)
  | _ -> invalid_arg s

let bits_text bits = String.concat " " (List.map bit_to_string bits)

let fragment_text (f : Tc.fragment) =
  bits_text f.lanes ^ "/" ^ bits_text f.elements

let relabel_text pairs =
  String.concat " "
    (List.map (fun (c, y) -> bit_to_string c ^ ">" ^ bit_to_string y) pairs)

let coords_text lanes =
  Array.to_list lanes
  |> List.map (fun elements ->
      Array.to_list elements
      |> List.map (fun (a, b) -> Printf.sprintf "%d,%d" a b)
      |> String.concat " ")
  |> String.concat " | "

(* [core_of_cell "metal[2]"] is the third core of [Tc.metal]. *)
let core_of_cell s =
  Scanf.sscanf s "%[a-z0-9_][%d]" (fun target i ->
      List.nth (List.assoc target targets) i)

(* [recorded claim column check] is the test [claim] that [check] holds of each
   core and its cell in [column] of tensor_cores.golden. *)
let recorded claim column check =
  group claim
    [
      Golden.cases "tensor_cores.golden" (fun cell ->
          check (core_of_cell (cell "core")) (cell column));
    ]

let each_core name law =
  cases name ~id:fst cores_and_transposes (fun (_, tc) -> law tc)

(* equal_bit *)

let equal_bits =
  let all =
    List.concat_map (fun i -> [ Tc.M i; Tc.N i; Tc.K i ]) [ 0; 1; -1; max_int ]
  in
  let pairs =
    List.concat_map (fun b0 -> List.map (fun b1 -> (b0, b1)) all) all
  in
  group "equal_bit"
    [
      cases "is true iff the bits are the same bit"
        ~id:(fun (b0, b1) -> bit_to_string b0 ^ " " ^ bit_to_string b1)
        pairs
        (fun (b0, b1) ->
          if b0 = b1 then equal bit b0 b1 else not_equal bit b0 b1);
    ]

(* fragment *)

(* [count rows cols coords] is how many times each cell of a [rows] by [cols]
   tile appears in [coords]. *)
let count rows cols coords =
  let seen = Array.make_matrix rows cols 0 in
  Array.iter
    (Array.iter (fun (r, c) ->
         if r < 0 || r >= rows || c < 0 || c >= cols then
           failf "(%d, %d) is outside the %d x %d tile" r c rows cols;
         seen.(r).(c) <- seen.(r).(c) + 1))
    coords;
  Array.to_list seen |> List.concat_map Array.to_list |> List.sort_uniq compare

(* The lane bits of [fragment] of a dimension outside [dims]. *)
let broadcast dims (fragment : Tc.fragment) =
  List.length
    (List.filter
       (fun b ->
         match (b : Tc.bit) with
         | M _ -> not (String.contains dims 'm')
         | N _ -> not (String.contains dims 'n')
         | K _ -> not (String.contains dims 'k'))
       fragment.lanes)

let fragments =
  group "fragment"
    [
      each_core "C's fragment holds each element of D once" (fun tc ->
          let n, m, _ = Tc.dims tc in
          let _, _, c = Tc.frag_coords tc in
          equal (list int) [ 1 ] (count m n c));
      each_core "A's fragment holds each element once per broadcast lane"
        (fun tc ->
          let _, m, k = Tc.dims tc in
          let a, _, _ = Tc.frag_coords tc in
          equal (list int) [ 1 lsl broadcast "mk" tc.frag_a ] (count m k a));
      each_core "B's fragment holds each element once per broadcast lane"
        (fun tc ->
          let n, _, k = Tc.dims tc in
          let _, b, _ = Tc.frag_coords tc in
          equal (list int) [ 1 lsl broadcast "kn" tc.frag_b ] (count k n b));
      each_core "a lane holds as many elements as its fragment has element bits"
        (fun tc ->
          let a, b, c = Tc.frag_coords tc in
          List.iter
            (fun (coords, (fragment : Tc.fragment)) ->
              equal int (Tc.threads tc) (Array.length coords);
              Array.iter
                (fun elements ->
                  equal int
                    (1 lsl List.length fragment.elements)
                    (Array.length elements))
                coords)
            [ (a, tc.frag_a); (b, tc.frag_b); (c, tc.frag_c) ]);
    ]

(* v *)

(* The N, M and K bits of the tile of [frag_a] and [frag_c], as axis_coords
   states it: each dimension up to its highest bit they name. *)
let tile frag_a frag_c =
  let used = bits frag_a @ bits frag_c in
  let upto make =
    let top =
      List.fold_left
        (fun top b ->
          match (make 0, b) with
          | Tc.N _, Tc.N i | M _, M i | K _, K i -> max top i
          | _ -> top)
        (-1) used
    in
    List.init (top + 1) make
  in
  (upto (fun i -> Tc.N i), upto (fun i -> Tc.M i), upto (fun i -> Tc.K i))

(* The conditions of [Tc.v] that [frag_a], [frag_b] and [frag_c] fail, each with
   the fragment whose condition it is. C's lanes are the warp's, so a lane count
   other than C's is A's or B's condition, and the order of K is B's. *)
let faults frag_a frag_b frag_c =
  let ns, ms, ks = tile frag_a frag_c in
  let mem b bs = List.exists (Tc.equal_bit b) bs in
  let check name (f : Tc.fragment) own =
    let all = bits f in
    [
      ("as many lanes as C", List.compare_lengths f.lanes frag_c.Tc.lanes <> 0);
      ( "distinct bits",
        List.length (List.sort_uniq compare all) < List.length all );
      ( "elements of its own dimensions",
        List.exists (fun b -> not (mem b own)) f.elements );
      ( "every bit of its own dimensions",
        List.exists (fun b -> not (mem b all)) own );
      ( "other bits M or N of the tile",
        List.exists (fun b -> not (mem b own || mem b (ms @ ns))) all );
    ]
    |> List.filter_map (fun (condition, fails) ->
        if fails then Some (condition, name) else None)
  in
  let k_order (f : Tc.fragment) = List.filter is_k (f.elements @ f.lanes) in
  check "frag_c" frag_c (ms @ ns)
  @ check "frag_a" frag_a (ms @ ks)
  @ check "frag_b" frag_b (ks @ ns)
  @
  if List.equal Tc.equal_bit (k_order frag_a) (k_order frag_b) then []
  else [ ("K in A's order", "frag_b") ]

(* The field the refusal names, as tc.mli states: the first of C, A and B that
   fails one of its conditions, the order of K last. *)
let at_fault faults =
  match faults with [] -> None | (_, field) :: _ -> Some field

let conditions =
  [
    "as many lanes as C";
    "distinct bits";
    "elements of its own dimensions";
    "every bit of its own dimensions";
    "other bits M or N of the tile";
    "K in A's order";
  ]

type change =
  | Kept
  | Lane_dropped
  | Bit_twice
  | Element_of_another_dimension
  | Element_dropped
  | Lane_outside_the_tile
  | K_exchanged_in_b

let pp_change ppf c =
  Format.pp_print_string ppf
    (match c with
    | Kept -> "kept"
    | Lane_dropped -> "a lane dropped"
    | Bit_twice -> "a bit held twice"
    | Element_of_another_dimension -> "an element of another dimension"
    | Element_dropped -> "an element dropped"
    | Lane_outside_the_tile -> "a lane outside the tile's M and N bits"
    | K_exchanged_in_b -> "two of B's K bits exchanged")

type field = Frag_a | Frag_b | Frag_c

(* A core's fragments, kept or with [field]'s changed by [change] at its bit
   [pos]. *)
let changed (tc : Tc.t) change field pos =
  let at l = pos mod List.length l in
  let nth l = List.nth l (at l) in
  let without l = List.filteri (fun i _ -> i <> at l) l in
  let replace l b = List.mapi (fun i x -> if i = at l then b else x) l in
  let alien, outside =
    match field with
    | Frag_a -> (Tc.N 0, Tc.M (-1))
    | Frag_b -> (M 0, M 31)
    | Frag_c -> (K 0, K 0)
  in
  let change_one (f : Tc.fragment) =
    match change with
    | Kept | K_exchanged_in_b -> f
    | Lane_dropped -> { f with lanes = without f.lanes }
    | Bit_twice -> { f with elements = f.elements @ [ nth (bits f) ] }
    | Element_of_another_dimension ->
        { f with elements = f.elements @ [ alien ] }
    | Element_dropped when f.elements <> [] ->
        { f with elements = without f.elements }
    | Element_dropped -> { f with lanes = without f.lanes }
    | Lane_outside_the_tile -> { f with lanes = replace f.lanes outside }
  in
  let exchange (f : Tc.fragment) =
    let ks = List.filter is_k (bits f) in
    let k0 = nth ks and k1 = nth (List.tl ks @ [ List.hd ks ]) in
    let swap b =
      if Tc.equal_bit b k0 then k1 else if Tc.equal_bit b k1 then k0 else b
    in
    fragment (List.map swap f.lanes) (List.map swap f.elements)
  in
  let a, b, c = (tc.frag_a, tc.frag_b, tc.frag_c) in
  match (change, field) with
  | K_exchanged_in_b, _ -> (a, exchange b, c)
  | _, Frag_a -> (change_one a, b, c)
  | _, Frag_b -> (a, change_one b, c)
  | _, Frag_c -> (a, b, change_one c)

let drawn_fragments =
  let open Gen in
  with_pp
    (fun ppf (a, b, c) ->
      Format.fprintf ppf "frag_a=%s frag_b=%s frag_c=%s" (fragment_text a)
        (fragment_text b) (fragment_text c))
    (let+ tc = of_list ~pp:Tc.pp (List.map snd named_cores)
     and+ change =
       frequency
         (List.map
            (fun c -> ((if c = Kept then 2 else 1), constant ~pp:pp_change c))
            [
              Kept;
              Lane_dropped;
              Bit_twice;
              Element_of_another_dimension;
              Element_dropped;
              Lane_outside_the_tile;
              K_exchanged_in_b;
            ])
     and+ field = of_list [ Frag_a; Frag_b; Frag_c ]
     and+ pos = nat in
     changed tc change field pos)

let make (frag_a, frag_b, frag_c) =
  Tc.v ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float32 ~frag_a ~frag_b ~frag_c

(* The field a refusal names: the text before its first colon. *)
let named msg =
  match String.index_opt msg ':' with
  | Some i -> String.sub msg 0 i
  | None -> msg

let construction =
  group "v"
    [
      (* Each change but [Kept] is drawn 1 time in 8. Every draw of
         [Lane_dropped], [Bit_twice], [Element_of_another_dimension],
         [Lane_outside_the_tile] and [K_exchanged_in_b] fails, in turn, the
         lanes, distinct bits, own elements, tile and K order conditions, so a
         run of 200 misses one with probability (7/8)^200, 3e-12. A dropped lane
         or element fails "every bit" unless it was the highest of its
         dimension, in about a third of the cases. Each field is named in at
         least a sixth of them. *)
      prop
        "raises iff a condition fails, naming the first fragment whose \
         condition fails"
        ~count:200 drawn_fragments (fun ((a, b, c) as frags) ->
          let faults = faults a b c in
          List.iter
            (fun condition -> cover condition (List.mem_assoc condition faults))
            conditions;
          List.iter
            (fun field ->
              cover ("names " ^ field) (at_fault faults = Some field))
            [ "frag_a"; "frag_b"; "frag_c" ];
          cover "accepted" (faults = []);
          match make frags with
          | _ -> equal (option string) None (at_fault faults)
          | exception Invalid_argument msg ->
              equal (option string) (at_fault faults) (Some (named msg)));
      group "accepts and refuses fragments as its golden states"
        [
          Golden.cases ~key:[ "case" ] "refused.golden" (fun cell ->
              let frags =
                ( fragment_of_cell (cell "frag_a"),
                  fragment_of_cell (cell "frag_b"),
                  fragment_of_cell (cell "frag_c") )
              in
              match cell "accepted" with
              | "True" -> ignore (make frags)
              | _ -> rejects (fun () -> make frags));
        ];
    ]

(* The tile *)

let tiles =
  [
    group "dims"
      [
        recorded "is the tile its golden row states" "dims" (fun tc s ->
            let n, m, k = Tc.dims tc in
            equal string s (Printf.sprintf "%d %d %d" n m k));
      ];
    group "threads"
      [
        each_core "is 2 to the number of lanes" (fun tc ->
            equal int (1 lsl List.length tc.frag_c.lanes) (Tc.threads tc));
        recorded "is the number its golden row states" "threads" (fun tc s ->
            equal int (int_of_string s) (Tc.threads tc));
      ];
    group "axis_coords"
      [
        recorded "is the tile's bits as its golden row states" "axis_coords"
          (fun tc s -> equal string s (bits_text (Tc.axis_coords tc)));
      ];
    group "base_upcast_axes"
      [
        each_core
          "is the bits of C's elements, then the K bits, most significant first"
          (fun tc ->
            equal (list bit)
              (List.rev tc.frag_c.elements
              @ List.rev (List.filter is_k (Tc.axis_coords tc)))
              (Tc.base_upcast_axes tc));
        recorded "is the bits its golden row states" "base_upcast_axes"
          (fun tc s -> equal string s (bits_text (Tc.base_upcast_axes tc)));
      ];
    group "relabel"
      [
        each_core
          "pairs the lane bits with C's in order, and the element bits with \
           the first of base_upcast_axes from the last" (fun tc ->
            let expected (f : Tc.fragment) =
              let slots =
                List.filteri
                  (fun i _ -> i < List.length f.elements)
                  (Tc.base_upcast_axes tc)
              in
              List.combine (bits f) (tc.frag_c.lanes @ List.rev slots)
            in
            let a, b = Tc.relabel tc in
            equal (list (pair bit bit)) ~msg:"A" (expected tc.frag_a) a;
            equal (list (pair bit bit)) ~msg:"B" (expected tc.frag_b) b);
        recorded "pairs A's bits as its golden row states" "relabel_a"
          (fun tc s -> equal string s (relabel_text (fst (Tc.relabel tc))));
        recorded "pairs B's bits as its golden row states" "relabel_b"
          (fun tc s -> equal string s (relabel_text (snd (Tc.relabel tc))));
      ];
    group "frag_coords"
      [
        group "places each element as its golden states"
          [
            Golden.cases ~key:[ "core"; "operand" ] "frag_coords.golden"
              (fun cell ->
                let a, b, c = Tc.frag_coords (core_of_cell (cell "core")) in
                let coords =
                  match cell "operand" with
                  | "a" -> a
                  | "b" -> b
                  | "c" -> c
                  | s -> invalid_arg s
                in
                equal text (cell "coords") (coords_text coords));
          ];
      ];
  ]

(* equal and pp *)

let metal = List.hd Tc.metal

(* [like ...] is [metal] with the fields given. *)
let like ?(dtype_in = metal.dtype_in) ?(dtype_out = metal.dtype_out)
    ?(frag_a = metal.frag_a) ?(frag_b = metal.frag_b) ?(frag_c = metal.frag_c)
    () =
  Tc.v ~dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c

(* The equality the .mli states: the same types and fragments. *)
let same (tc0 : Tc.t) (tc1 : Tc.t) =
  let fragments (f0 : Tc.fragment) (f1 : Tc.fragment) =
    List.equal Tc.equal_bit f0.lanes f1.lanes
    && List.equal Tc.equal_bit f0.elements f1.elements
  in
  Dtype.equal tc0.dtype_in tc1.dtype_in
  && Dtype.equal tc0.dtype_out tc1.dtype_out
  && fragments tc0.frag_a tc1.frag_a
  && fragments tc0.frag_b tc1.frag_b
  && fragments tc0.frag_c tc1.frag_c

let equality =
  let pairs =
    List.concat_map
      (fun x -> List.map (fun y -> (x, y)) named_cores)
      named_cores
  in
  group "equal"
    [
      cases "is true iff the cores have the same types and fragments"
        ~id:(fun ((n0, _), (n1, _)) -> n0 ^ " " ^ n1)
        pairs
        (fun ((_, tc0), (_, tc1)) ->
          if same tc0 tc1 then equal core tc0 tc1 else not_equal core tc0 tc1);
      cases "is false of cores that differ in one field" ~id:fst
        [
          ("dtype_in", like ~dtype_in:Dtype.Bfloat16 ());
          ("dtype_out", like ~dtype_out:Dtype.Float16 ());
          ( "frag_a",
            like ~frag_a:(fragment Tc.[ K 1; M 1; M 0; K 2; M 2 ] [ K 0 ]) () );
          ( "frag_b",
            like ~frag_b:(fragment Tc.[ N 0; K 0; K 1; N 2; K 2 ] [ N 1 ]) () );
          ( "frag_c",
            like ~frag_c:(fragment Tc.[ N 0; M 0; M 1; N 2; M 2 ] [ N 1 ]) () );
        ]
        (fun (_, other) -> not_equal core metal other);
    ]

let printing =
  group "pp"
    [
      recorded "prints a core as its golden row states" "repr" (fun tc repr ->
          equal string repr (Format.asprintf "%a" Tc.pp tc));
    ]

(* factor and takes *)

let value dtype =
  Ops.load (Ops.index (Shape.param ~shape:[ Int 16 ] 0 dtype) [ Ops.int 0 ]) []

let widened dtype = Ops.cast (value dtype) Dtype.Float32

(* The first shipped core with these types. *)
let typed dtype_in dtype_out =
  List.find
    (fun (tc : Tc.t) ->
      Dtype.equal tc.dtype_in dtype_in && Dtype.equal tc.dtype_out dtype_out)
    (List.map snd named_cores)

let half_float = typed Float16 Float32
let half_half = typed Float16 Float16
let bf16_float = typed Bfloat16 Float32
let float_float = typed Float32 Float32
let char_int = typed Int8 Int32

let factors =
  group "factor"
    [
      cases "is what a core multiplies for a factor, or None"
        ~id:(fun (row, _, _, _) -> row)
        [
          ( "a half on a half core",
            half_float,
            value Float16,
            Some (value Float16) );
          ( "a float32 on a float32 core",
            float_float,
            value Float32,
            Some (value Float32) );
          ("an int8 on an int8 core", char_int, value Int8, Some (value Int8));
          ( "a half widened, on a half core adding float32",
            half_float,
            widened Float16,
            Some (value Float16) );
          ( "a bfloat16 widened, on a bfloat16 core adding float32",
            bf16_float,
            widened Bfloat16,
            Some (value Bfloat16) );
          ( "a half widened, on a half core adding halves",
            half_half,
            widened Float16,
            None );
          ( "a bfloat16 widened, on a half core",
            half_float,
            widened Bfloat16,
            None );
          ( "an int8 widened, on a bfloat16 core adding float32",
            bf16_float,
            widened Int8,
            Some (Ops.cast (value Int8) Bfloat16) );
          ( "a uint8 widened, on a half core adding float32",
            half_float,
            widened Uint8,
            Some (Ops.cast (value Uint8) Float16) );
          ( "an int32 widened, on a bfloat16 core, which holds it inexactly",
            bf16_float,
            widened Int32,
            None );
          ( "an int8 widened, on a half core adding halves",
            half_half,
            widened Int8,
            None );
          ("an int8 widened, on an int8 core", char_int, widened Int8, None);
          ("a float32 on a half core", half_float, value Float32, None);
        ]
        (fun (_, tc, u, expected) ->
          equal (option Uops.uop) expected (Tc.factor tc u));
    ]

type product = { tc : Tc.t; acc : Dtype.t; a : Ops.t; b : Ops.t }

let product =
  Testable.make
    ~pp:(fun ppf p ->
      Format.fprintf ppf "%a into %a, acc %a, %a, %a" Dtype.pp p.tc.dtype_in
        Dtype.pp p.tc.dtype_out Dtype.pp p.acc Ops.pp p.a Ops.pp p.b)
    ~equal:( == )

let takes p = Tc.takes p.tc ~acc:p.acc p.a p.b

let taking =
  group "takes"
    [
      cases "is true iff acc is the output type and the core multiplies both"
        ~id:(fun (row, _, _) -> row)
        [
          ( "both at the input type",
            {
              tc = half_float;
              acc = Float32;
              a = value Float16;
              b = value Float16;
            },
            true );
          ( "one widened",
            {
              tc = half_float;
              acc = Float32;
              a = value Float16;
              b = widened Float16;
            },
            true );
          ( "a code in bfloat16",
            {
              tc = bf16_float;
              acc = Float32;
              a = widened Int8;
              b = value Bfloat16;
            },
            true );
          ( "halves into halves",
            {
              tc = half_half;
              acc = Float16;
              a = value Float16;
              b = value Float16;
            },
            true );
          ( "another accumulator",
            {
              tc = half_float;
              acc = Float16;
              a = value Float16;
              b = value Float16;
            },
            false );
          ( "a wider accumulator",
            {
              tc = half_half;
              acc = Float32;
              a = value Float16;
              b = value Float16;
            },
            false );
          ( "a second factor it does not multiply",
            {
              tc = half_float;
              acc = Float32;
              a = value Float16;
              b = widened Int32;
            },
            false );
          ( "a first factor it does not multiply",
            {
              tc = half_float;
              acc = Float32;
              a = value Float32;
              b = value Float16;
            },
            false );
        ]
        (fun (_, p, verdict) ->
          if verdict then satisfies ~claim:"takes the product" product takes p
          else
            satisfies ~claim:"refuses the product" product
              (fun p -> not (takes p))
              p);
    ]

(* chain *)

let operand_name = function Tc.A -> "A" | Tc.B -> "B"

let chain_t =
  Testable.make
    ~pp:(fun ppf (c : Tc.chain) ->
      Format.fprintf ppf "{ k_bits = [%s]; same_lines = %b }"
        (String.concat "; " (List.map string_of_int c.k_bits))
        c.same_lines)
    ~equal:( = )

let chains k_bits = Some { Tc.k_bits; same_lines = true }

(* Each core chained into itself, the attention kernel's case, as chain's rules
   give it from the fragments: CUDA's accumulator is A, two of its column tiles
   one A; Metal's either operand; RDNA 4's B, with K's bits 2 and 3 swapped;
   RDNA 3's neither. *)
let chain_cases =
  [
    ("cuda_sm80[1]", Tc.A, chains [ 0; 1; 2 ]);
    ("cuda_sm80[1]", Tc.B, None);
    ("cuda_sm75[0]", Tc.A, chains [ 0; 1; 2 ]);
    ("cuda_sm80[5]", Tc.A, chains [ 2; 0; 1 ]);
    ("cuda_sm89[6]", Tc.A, chains [ 0; 2; 3 ]);
    ("metal[0]", Tc.A, chains [ 0; 1; 2 ]);
    ("metal[0]", Tc.B, chains [ 0; 1; 2 ]);
    ("amd_rdna4[2]", Tc.A, None);
    ("amd_rdna4[2]", Tc.B, chains [ 0; 1; 3; 2 ]);
    ("amd_rdna3[2]", Tc.A, None);
    ("amd_rdna3[2]", Tc.B, None);
    ("amd_cdna3[3]", Tc.A, None);
    ("amd_cdna3[3]", Tc.B, chains [ 0; 1; 2; 3 ]);
  ]

(* The coordinates' bits [k_bits] of [k], gathered from bit [0]. *)
let gather k_bits k =
  List.fold_left
    (fun (acc, i) j -> (acc lor (((k lsr j) land 1) lsl i), i + 1))
    (0, 0) k_bits
  |> fst

(* [reads_own x y op k_bits] is whether each element of [y]'s operand [op] is an
   element of [x]'s accumulator in the same lane, at the coordinates [k_bits]
   maps its [K] to. A search tries many [k_bits] on one pair, so the coordinates
   are made once. *)
let reads_own x y op =
  let fa, fb, _ = Tc.frag_coords y and _, _, xc = Tc.frag_coords x in
  let operand = match op with Tc.A -> fa | Tc.B -> fb in
  fun k_bits ->
    let own held (r, col) =
      let want =
        match op with
        | Tc.A -> (r, gather k_bits col)
        | Tc.B -> (gather k_bits r, col)
      in
      Array.mem want held
    in
    Array.for_all2
      (fun elements held -> Array.for_all (own held) elements)
      operand xc

(* Whether each lane of [x]'s and [y]'s accumulators holds the same lines of the
   dimension [op] keeps. *)
let same_lines x y op =
  let _, _, xc = Tc.frag_coords x and _, _, yc = Tc.frag_coords y in
  let line (r, col) = if op = Tc.A then r else col in
  let lines coords =
    Array.map
      (fun elements ->
        List.sort_uniq compare (Array.to_list (Array.map line elements)))
      coords
  in
  lines xc = lines yc

let rec log2 n = if n <= 1 then 0 else 1 + log2 (n / 2)

(* Every injective map of [m] bits into the bits [avail], as the list of each
   bit's image. *)
let rec injections m avail =
  if m = 0 then [ [] ]
  else
    List.concat_map
      (fun j ->
        List.map
          (fun rest -> j :: rest)
          (injections (m - 1) (List.filter (( <> ) j) avail)))
      avail

(* Why [x]'s accumulator is or is not [y]'s operand [op], by chain's conditions,
   and the first permutation of [K] found that reads it in place, trying every
   map of the moved bits. *)
let search x y op =
  let nx, mx, _ = Tc.dims x and ny, my, ky = Tc.dims y in
  let kept_x, kept_y, moved =
    match op with Tc.A -> (mx, my, nx) | Tc.B -> (nx, ny, mx)
  in
  if Tc.threads x <> Tc.threads y then ("threads differ", None)
  else if kept_x <> kept_y then ("kept dimensions differ", None)
  else if moved > ky then ("the other dimension is larger than K", None)
  else
    match
      List.find_opt (reads_own x y op)
        (injections (log2 moved) (List.init (log2 ky) Fun.id))
    with
    | None -> ("a lane needs another lane's element", None)
    | found -> ("chains", found)

(* Every pair of cores of a target, and each operand. *)
let pairs cores =
  List.concat_map
    (fun x -> List.concat_map (fun y -> [ (x, y, Tc.A); (x, y, Tc.B) ]) cores)
    cores

(* Every pair of shipped cores, and each operand, named. *)
let all_pairs =
  List.concat_map
    (fun (nx, x) ->
      List.concat_map
        (fun (ny, y) ->
          List.map
            (fun op ->
              ( Printf.sprintf "%s into %s as %s" nx ny (operand_name op),
                (x, y, op) ))
            [ Tc.A; Tc.B ])
        named_cores)
    named_cores

let pair_name x y op =
  Format.asprintf "%a into %a as %s" Tc.pp x Tc.pp y (operand_name op)

(* A core whose B takes C's elements with M1 in its own place: M0 must go past
   it. *)
let crossed =
  Tc.v ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float32
    ~frag_a:(fragment Tc.[ M 0; K 0 ] Tc.[ K 1; K 2; M 1; M 2 ])
    ~frag_b:(fragment Tc.[ N 0; K 0 ] Tc.[ K 1; K 2 ])
    ~frag_c:(fragment Tc.[ N 0; M 2 ] Tc.[ M 0; M 1 ])

(* A core whose accumulator's elements, M1 then M0, both leave their own indices
   to B's lanes, K1 and K0: as B, M1 takes K2, the least index left free, and M0
   then K3. *)
let reordered =
  Tc.v ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float32
    ~frag_a:(fragment Tc.[ M 0; K 0; K 1 ] Tc.[ K 2; K 3; M 1; M 2; M 3 ])
    ~frag_b:(fragment Tc.[ N 0; K 0; K 1 ] Tc.[ K 2; K 3 ])
    ~frag_c:(fragment Tc.[ N 0; M 2; M 3 ] Tc.[ M 1; M 0 ])

let chaining =
  group "chain"
    [
      cases "chains a core's accumulator into itself as its fragments allow"
        ~id:(fun (name, op, _) -> name ^ " as " ^ operand_name op)
        chain_cases
        (fun (name, op, expected) ->
          let tc = core_of_cell name in
          equal (option chain_t) expected (Tc.chain tc tc op));
      test "keeps an element bit's own index before another bit takes it"
        (fun () ->
          equal (option chain_t)
            (Some { Tc.k_bits = [ 2; 1; 0 ]; same_lines = true })
            (Tc.chain crossed crossed Tc.B));
      test
        "takes, for each other bit in the order of the accumulator's elements, \
         the least index left free" (fun () ->
          let chained = Tc.chain reordered reordered Tc.B in
          equal (option chain_t)
            (Some { Tc.k_bits = [ 3; 2; 0; 1 ]; same_lines = true })
            chained;
          satisfies ~claim:"reads the accumulator in its own lanes" (list int)
            (reads_own reordered reordered Tc.B)
            (Option.get chained).k_bits);
      prop
        "chains iff some permutation of K reads the accumulator in place, and \
         the chain reads it so"
        ~count:(List.length all_pairs)
        (Gen.of_list
           ~pp:(fun ppf (name, _) -> Format.pp_print_string ppf name)
           all_pairs)
        (fun (_, (x, y, op)) ->
          let reason, found = search x y op in
          List.iter
            (fun r -> cover r (reason = r))
            [
              "threads differ";
              "kept dimensions differ";
              "the other dimension is larger than K";
              "a lane needs another lane's element";
              "chains";
            ];
          match Tc.chain x y op with
          | None -> equal (option (list int)) None found
          | Some c ->
              satisfies ~claim:"reads the accumulator in its own lanes"
                (list int) (reads_own x y op) c.k_bits;
              satisfies ~claim:"names each bit of K once" (list int)
                (fun ks ->
                  List.compare_lengths (List.sort_uniq compare ks) ks = 0)
                c.k_bits;
              equal chain_t { c with same_lines = same_lines x y op } c);
    ]

(* transpose *)

let swap (r, c) = (c, r)

let transposition =
  group "transpose"
    [
      cases "is an involution" ~id:fst named_cores (fun (_, tc) ->
          Law.involutive core Tc.transpose tc);
      cases "keeps the types and exchanges N and M" ~id:fst named_cores
        (fun (_, tc) ->
          let t = Tc.transpose tc in
          let n, m, k = Tc.dims tc in
          equal dtype tc.dtype_in t.dtype_in;
          equal dtype tc.dtype_out t.dtype_out;
          equal (triple int int int) (m, n, k) (Tc.dims t));
      cases "gives a lane the core's B as its A, A as its B and C transposed"
        ~id:fst named_cores (fun (_, tc) ->
          let a, b, c = Tc.frag_coords tc in
          let a', b', c' = Tc.frag_coords (Tc.transpose tc) in
          let flip = Array.map (Array.map swap) in
          equal (array (array (pair int int))) ~msg:"A" (flip b) a';
          equal (array (array (pair int int))) ~msg:"B" (flip a) b';
          equal (array (array (pair int int))) ~msg:"C" (flip c) c');
      cases "chains as A the accumulators that chain as B" ~id:fst targets
        (fun (_, cores) ->
          List.iter
            (fun (x, y, op) ->
              let other = match op with Tc.A -> Tc.B | Tc.B -> Tc.A in
              equal ~msg:(pair_name x y op) (option chain_t) (Tc.chain x y op)
                (Tc.chain (Tc.transpose x) (Tc.transpose y) other))
            (pairs cores));
    ]

(* Targets *)

(* The cores tensor_cores.golden records for [target], in order, made with
   [Tc.v] from their cells. *)
let recorded_cores target =
  Golden.rows "tensor_cores.golden"
  |> List.filter_map (fun cell ->
      if String.starts_with ~prefix:(target ^ "[") (cell "core") then
        Some
          (Tc.v
             ~dtype_in:(dtype_of_cell (cell "dtype_in"))
             ~dtype_out:(dtype_of_cell (cell "dtype_out"))
             ~frag_a:(fragment_of_cell (cell "frag_a"))
             ~frag_b:(fragment_of_cell (cell "frag_b"))
             ~frag_c:(fragment_of_cell (cell "frag_c")))
      else None)

let cuda_cores cell =
  let arch = Scanf.sscanf (cell "arch") "%S" Fun.id in
  match cell "cores" with
  | "none" -> equal (list core) [] (Tc.cuda arch)
  | c when String.starts_with ~prefix:"raises" c ->
      rejects (fun () -> Tc.cuda arch)
  | target -> equal (list core) (List.assoc target targets) (Tc.cuda arch)

let listed =
  [
    group "targets"
      [
        cases "a target lists its cores in order, as their golden rows state"
          ~id:fst targets (fun (target, cores) ->
            equal (list core) (recorded_cores target) cores);
        test "cuda_sm80 holds every core of cuda_sm75" (fun () ->
            List.iter (fun c -> mem core c Tc.cuda_sm80) Tc.cuda_sm75);
        test "cuda_sm89 is cuda_sm80 then two cores of 8-bit floats" (fun () ->
            let n = List.length Tc.cuda_sm80 in
            equal (list core) Tc.cuda_sm80
              (List.filteri (fun i _ -> i < n) Tc.cuda_sm89);
            equal (list dtype)
              [ Dtype.Fp8e4m3; Dtype.Fp8e5m2 ]
              (List.filteri (fun i _ -> i >= n) Tc.cuda_sm89
              |> List.map (fun (tc : Tc.t) -> tc.dtype_in)));
        test "amd_rdna4 has two cores of 8-bit floats into floats" (fun () ->
            let fp8 =
              List.filter
                (fun (tc : Tc.t) -> List.mem tc.dtype_in Dtype.fp8s)
                Tc.amd_rdna4
            in
            equal
              (list (pair dtype dtype))
              Dtype.[ (Fp8e4m3, Float32); (Fp8e5m2, Float32) ]
              (List.map (fun (tc : Tc.t) -> (tc.dtype_in, tc.dtype_out)) fp8));
      ];
    group "cuda"
      [
        group "gives an architecture's cores as its golden states"
          [ Golden.cases "cuda.golden" cuda_cores ];
      ];
    group "amd"
      [
        group "gives an architecture's cores as its golden states"
          [
            Golden.cases "amd.golden" (fun cell ->
                let arch = Scanf.sscanf (cell "arch") "%S" Fun.id in
                equal (list core)
                  (List.assoc (cell "cores") targets)
                  (Tc.amd arch));
          ];
      ];
  ]

(* pm_wmma_fp8 *)

(* A product of [lanes] lanes of [dtype] into [acc]. *)
let product_of ?(acc = Dtype.Float32) dtype lanes =
  let operand slot =
    Shape.stack
      (List.init lanes (fun i ->
           Ops.load
             (Ops.index
                (Shape.param ~shape:[ Int lanes ] slot dtype)
                [ Ops.int i ])
             []))
  in
  let a = operand 0 and b = operand 1 in
  ( a,
    b,
    Ops.wmma a b ~acc:(Ops.float ~dtype:acc 0.) ~dims:(16, 16, 16) ~threads:32
  )

let as_words =
  group "pm_wmma_fp8"
    [
      cases "makes a product of 8 lanes of 8-bit floats a product of their bits"
        ~id:(fun dt -> Format.asprintf "%a" Dtype.pp dt)
        Dtype.fp8s
        (fun dt ->
          let a, b, product = product_of dt 8 in
          let rewritten =
            Ops.Pattern_matcher.rewrite (Tc.pm_wmma_fp8 Dtype.Uint64) () product
          in
          equal
            (option (list Uops.uop))
            (Some
               [ Ops.bitcast a Uint64; Ops.bitcast b Uint64; Ops.nth product 2 ])
            (Option.map Ops.src rewritten));
      cases "leaves another product alone"
        ~id:(fun (dt, lanes, acc) ->
          Format.asprintf "%d lanes of %a into %a" lanes Dtype.pp dt Dtype.pp
            acc)
        [
          (Dtype.Fp8e4m3, 16, Dtype.Float32);
          (Dtype.Fp8e4m3, 4, Dtype.Float32);
          (Dtype.Float16, 8, Dtype.Float32);
          (Dtype.Fp8e4m3, 8, Dtype.Float16);
        ]
        (fun (dt, lanes, acc) ->
          let _, _, product = product_of ~acc dt lanes in
          equal (option Uops.uop) None
            (Ops.Pattern_matcher.rewrite
               (Tc.pm_wmma_fp8 Dtype.Uint64)
               () product));
    ]

let () =
  exit
    (run "Tolk.Tc"
       ([ equal_bits; fragments; construction ]
       @ tiles
       @ [ equality; printing; factors; taking; chaining; transposition ]
       @ listed @ [ as_words ]))
