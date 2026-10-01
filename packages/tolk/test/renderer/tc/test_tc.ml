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

let any_core = Gen.of_list ~pp:Tc.pp (List.map snd named_cores)
let core = Testable.make ~pp:Tc.pp ~equal:Tc.equal
let bit = Testable.make ~pp:Tc.pp_bit ~equal:Tc.equal_bit
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

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
  | [ lanes; elements ] ->
      { Tc.lanes = bits_of_cell lanes; elements = bits_of_cell elements }
  | _ -> invalid_arg s

let bits_text bits =
  String.concat " " (List.map (Format.asprintf "%a" Tc.pp_bit) bits)

let fragment_text f = bits_text f.Tc.lanes ^ "/" ^ bits_text f.elements

let relabel_text pairs =
  String.concat " "
    (List.map
       (fun (c, y) -> Format.asprintf "%a>%a" Tc.pp_bit c Tc.pp_bit y)
       pairs)

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

let per_core column check =
  group column
    [
      Golden.cases "tensor_cores.golden" (fun cell ->
          check (core_of_cell (cell "core")) (cell column));
    ]

(* Bits *)

let bits =
  group "bits"
    [
      cases ~name:fst "pp_bit writes the dimension in lower case and the index"
        [ ("m0", Tc.M 0); ("n2", Tc.N 2); ("k6", Tc.K 6); ("m12", Tc.M 12) ]
        (fun (text, b) -> equal string text (Format.asprintf "%a" Tc.pp_bit b));
      test "bits of different dimensions differ" (fun () ->
          not_equal bit (Tc.M 0) (Tc.N 0);
          not_equal bit (Tc.N 1) (Tc.K 1));
      test "bits of one dimension differ by their index" (fun () ->
          not_equal bit (Tc.K 0) (Tc.K 1));
      prop "equal_bit is an equivalence"
        (let one =
           Gen.map
             (fun (d, i) -> [| Tc.M i; Tc.N i; Tc.K i |].(d))
             (Gen.pair (Gen.int_range 0 2) (Gen.int_range 0 3))
         in
         Gen.with_pp
           (fun ppf (a, b) ->
             Format.fprintf ppf "%a, %a" Tc.pp_bit a Tc.pp_bit b)
           (Gen.pair one one))
        (Law.equivalence bit);
    ]

(* Targets *)

let names_of_rows target =
  Golden.rows "tensor_cores.golden"
  |> List.filter_map (fun cell ->
      let name = cell "core" in
      if String.starts_with ~prefix:(target ^ "[") name then Some (cell "repr")
      else None)

let listed =
  group "targets"
    [
      cases ~name:fst "a target lists tinygrad's cores, in order" targets
        (fun (target, cores) ->
          equal (list string) (names_of_rows target)
            (List.map (Format.asprintf "%a" Tc.pp) cores));
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
      Golden.cases "cuda.golden" (fun cell ->
          let arch = Scanf.sscanf (cell "arch") "%S" Fun.id in
          match cell "cores" with
          | "none" -> equal (list core) [] (Tc.cuda arch)
          | c when String.starts_with ~prefix:"raises" c ->
              rejects (fun () -> Tc.cuda arch)
          | target ->
              equal (list core) (List.assoc target targets) (Tc.cuda arch));
      Golden.cases "amd.golden" (fun cell ->
          let arch = Scanf.sscanf (cell "arch") "%S" Fun.id in
          equal (list core) (List.assoc (cell "cores") targets) (Tc.amd arch));
    ]

(* Cores *)

let described =
  group "tensor cores"
    [
      per_core "repr" (fun tc repr ->
          equal string repr (Format.asprintf "%a" Tc.pp tc));
      per_core "dtype_in" (fun tc s ->
          equal dtype (dtype_of_cell s) tc.dtype_in);
      per_core "dtype_out" (fun tc s ->
          equal dtype (dtype_of_cell s) tc.dtype_out);
      per_core "frag_a" (fun tc s -> equal string s (fragment_text tc.frag_a));
      per_core "frag_b" (fun tc s -> equal string s (fragment_text tc.frag_b));
      per_core "frag_c" (fun tc s -> equal string s (fragment_text tc.frag_c));
      per_core "dims" (fun tc s ->
          let n, m, k = Tc.dims tc in
          equal string s (Printf.sprintf "%d %d %d" n m k));
      per_core "threads" (fun tc s ->
          equal int (int_of_string s) (Tc.threads tc));
      per_core "axis_coords" (fun tc s ->
          equal string s (bits_text (Tc.axis_coords tc)));
      per_core "base_upcast_axes" (fun tc s ->
          equal string s (bits_text (Tc.base_upcast_axes tc)));
      per_core "relabel_a" (fun tc s ->
          equal string s (relabel_text (fst (Tc.relabel tc))));
      per_core "relabel_b" (fun tc s ->
          equal string s (relabel_text (snd (Tc.relabel tc))));
      group "frag_coords"
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
    ]

(* Laws *)

let each_core name law =
  cases ~name:fst name named_cores (fun (_, tc) -> law tc)

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

let foreign dims fragment =
  List.length
    (List.filter
       (fun b ->
         match (b : Tc.bit) with
         | M _ -> not (String.contains dims 'm')
         | N _ -> not (String.contains dims 'n')
         | K _ -> not (String.contains dims 'k'))
       fragment.Tc.lanes)

let laws =
  group "laws"
    [
      each_core "C's fragment holds each element of D once" (fun tc ->
          let n, m, _ = Tc.dims tc in
          let _, _, c = Tc.frag_coords tc in
          equal (list int) [ 1 ] (count m n c));
      each_core "A's fragment holds each element once per broadcast lane"
        (fun tc ->
          let _, m, k = Tc.dims tc in
          let a, _, _ = Tc.frag_coords tc in
          equal (list int) [ 1 lsl foreign "mk" tc.frag_a ] (count m k a));
      each_core "B's fragment holds each element once per broadcast lane"
        (fun tc ->
          let n, _, k = Tc.dims tc in
          let _, b, _ = Tc.frag_coords tc in
          equal (list int) [ 1 lsl foreign "kn" tc.frag_b ] (count k n b));
      each_core "a lane holds as many elements as its fragment has element bits"
        (fun tc ->
          let a, b, c = Tc.frag_coords tc in
          List.iter
            (fun (coords, fragment) ->
              equal int (Tc.threads tc) (Array.length coords);
              Array.iter
                (fun elements ->
                  equal int
                    (1 lsl List.length fragment.Tc.elements)
                    (Array.length elements))
                coords)
            [ (a, tc.frag_a); (b, tc.frag_b); (c, tc.frag_c) ]);
      each_core "relabel pairs each bit of A and of B, lanes then elements"
        (fun tc ->
          let a, b = Tc.relabel tc in
          equal (list bit)
            (tc.frag_a.lanes @ tc.frag_a.elements)
            (List.map fst a);
          equal (list bit)
            (tc.frag_b.lanes @ tc.frag_b.elements)
            (List.map fst b));
      each_core "v rebuilds a core from its fields" (fun tc ->
          equal core tc
            (Tc.v ~dtype_in:tc.dtype_in ~dtype_out:tc.dtype_out
               ~frag_a:tc.frag_a ~frag_b:tc.frag_b ~frag_c:tc.frag_c));
      prop "equal is an equivalence"
        (Gen.pair any_core any_core)
        (Law.equivalence core);
      prop "cores are equal iff they print alike" (Gen.pair any_core any_core)
        (fun (tc0, tc1) ->
          equal bool
            (Format.asprintf "%a" Tc.pp tc0 = Format.asprintf "%a" Tc.pp tc1)
            (Tc.equal tc0 tc1));
    ]

(* Construction *)

let metal = List.hd Tc.metal
let fragment lanes elements = { Tc.lanes; elements }

(* [like ?dtype_out ?frag_a ?frag_b ?frag_c ()] is [metal] with the fields
   given. *)
let like ?(dtype_out = metal.dtype_out) ?(frag_a = metal.frag_a)
    ?(frag_b = metal.frag_b) ?(frag_c = metal.frag_c) () =
  Tc.v ~dtype_in:metal.dtype_in ~dtype_out ~frag_a ~frag_b ~frag_c

let equality =
  group "equal"
    [
      test "cores that differ only in the output type differ" (fun () ->
          not_equal core metal (like ~dtype_out:Dtype.Float16 ()));
      test "cores that differ only in A's fragment differ" (fun () ->
          not_equal core metal
            (like ~frag_a:(fragment Tc.[ K 1; M 1; M 0; K 2; M 2 ] [ K 0 ]) ()));
      test "cores that differ only in B's fragment differ" (fun () ->
          not_equal core metal
            (like ~frag_b:(fragment Tc.[ N 0; K 0; K 1; N 2; K 2 ] [ N 1 ]) ()));
      test "cores that differ only in C's fragment differ" (fun () ->
          not_equal core metal
            (like ~frag_c:(fragment Tc.[ N 0; M 0; M 1; N 2; M 2 ] [ N 1 ]) ()));
    ]

let construction =
  group "v"
    [
      Golden.cases ~key:[ "case" ] "refused.golden" (fun cell ->
          let make () =
            Tc.v ~dtype_in:Dtype.Float16 ~dtype_out:Dtype.Float32
              ~frag_a:(fragment_of_cell (cell "frag_a"))
              ~frag_b:(fragment_of_cell (cell "frag_b"))
              ~frag_c:(fragment_of_cell (cell "frag_c"))
          in
          match cell "accepted" with
          | "True" -> ignore (make ())
          | _ -> rejects make);
    ]

let () =
  exit (run "Tolk.Tc" [ bits; listed; described; laws; equality; construction ])
