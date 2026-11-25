(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx_kernel's dtype rules against their tables, each function of
   Nx_kernel.Shape against the rule its doc states, at its bounds and their
   neighbours, and nx.cpu's name and the devices it computes on. *)

open Windtrap
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module Shape = Nx_kernel.Shape

let pp_ints = Nx_array_gen.pp_ints
let ints = array int
let big = L.max_numel

(* Dtypes *)

let name (D.Any dt) = D.name dt
let boolean (D.Any dt) = D.is D.Boolean dt
let float (D.Any dt) = D.is D.Float dt
let narrow_float (D.Any dt) = D.is D.Float dt && D.bits dt < 32

(* The names of the dtypes where [p] holds, in D.all's order. *)
let where p = List.map name (List.filter p D.all)

(* The doc's table: float32 for a float narrower than it, int8 and uint8 for
   int4 and uint4, bool for bit, else the dtype itself. *)
let carrier_name (D.Any dt as d) =
  match dt with
  | _ when narrow_float d -> "float32"
  | D.Int4 -> "int8"
  | D.Uint4 -> "uint8"
  | D.Bit -> "bool"
  | _ -> D.name dt

let test_carrier d = equal string (carrier_name d) (name (Nx_kernel.carrier d))

let floats =
  List.filter_map
    (fun (D.Any dt) ->
      match D.kind dt with
      | Float -> Some (Nx_kernel.Any_float dt)
      | Complex | Signed | Unsigned | Boolean -> None)
    D.all

let test_float_carrier (Nx_kernel.Any_float dt) =
  let (Nx_kernel.Any_float c) = Nx_kernel.float_carrier dt in
  equal string (name (Nx_kernel.carrier (D.Any dt))) (D.name c)

let reductions : Nx_kernel.reduction list =
  [
    Monoid Sum;
    Monoid Prod;
    Monoid Max;
    Monoid Min;
    Logsumexp;
    Moments;
    Arg Max;
    Arg Min;
  ]

let reduction_name : Nx_kernel.reduction -> string = function
  | Monoid Sum -> "Sum"
  | Monoid Prod -> "Prod"
  | Monoid Max -> "Max"
  | Monoid Min -> "Min"
  | Logsumexp -> "Logsumexp"
  | Moments -> "Moments"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"

let test_accepts (r : Nx_kernel.reduction) =
  let takes =
    match r with
    | Monoid (Sum | Prod) -> fun d -> not (boolean d)
    | Monoid (Max | Min) | Arg _ -> fun _ -> true
    | Logsumexp | Moments -> float
  in
  equal (list string) (where takes)
    (where (fun (D.Any dt) -> Nx_kernel.accepts r dt))

let test_accumulates () =
  equal (list string)
    (where (fun d -> not (boolean d || narrow_float d)))
    (where (fun (D.Any dt) -> Nx_kernel.accumulates dt))

let precision (D.Any dt) =
  match dt with
  | D.Float32 | D.Complex64 -> Some 32
  | D.Float64 | D.Complex128 -> Some 64
  | _ -> None

let test_same_precision () =
  let pairs p =
    List.concat_map
      (fun x ->
        List.filter_map
          (fun y -> if p x y then Some (name x, name y) else None)
          D.all)
      D.all
  in
  equal
    (list (pair string string))
    (pairs (fun x y -> precision x <> None && precision x = precision y))
    (pairs (fun (D.Any x) (D.Any y) -> Nx_kernel.same_precision x y))

let test_full_precision () =
  equal (list string)
    (where (fun d -> precision d <> None))
    (where (fun (D.Any dt) -> Nx_kernel.full_precision dt))

(* Shape's results *)

(* An Error's message is no part of a claim: [reason] equals any, and prints it
   where a shape was expected. *)
let reason = Testable.make ~pp:Format.pp_print_string ~equal:(fun _ _ -> true)
let shape = result ints reason
let misfit = Error "a misfit"

(* Whether a layout admits each of [shapes], as every shape Shape derives must
   be. *)
let admits shapes = List.for_all (fun s -> L.fits (Iarray.of_array s)) shapes

(* Shape.padded *)

let pad ?interior ?(windows = [||]) lo hi : Nx_kernel.pad =
  let interior =
    Option.value interior ~default:(Array.make (Array.length lo) 0)
  in
  {
    lo = Iarray.of_array lo;
    hi = Iarray.of_array hi;
    interior = Iarray.of_array interior;
    windows = Iarray.of_array windows;
  }

let window ?(step = 1) ?(dilation = 1) axis size =
  { M.axis; size; step; dilation }

type padded = { x : int array; pad : Nx_kernel.pad }

let pp_padded ppf (c : padded) =
  Format.fprintf ppf "%a padded lo %a hi %a interior %a, then %a" pp_ints c.x
    pp_ints (Iarray.to_array c.pad.lo) pp_ints (Iarray.to_array c.pad.hi)
    pp_ints
    (Iarray.to_array c.pad.interior)
    Nx_array_gen.pp_move (M.Window c.pad.windows)

(* An operand's padding: crops and pads of a few elements, interiors of up to
   two, and windows on a drawn subset of the axes, of up to three taps, one or
   two apart, steps of one or two. *)
let padded_case =
  Gen.with_pp pp_padded
    (let open Gen in
     let* x = Nx_array_gen.shape in
     let r = Array.length x in
     let small = array ~size:(constant r) (int_range (-2) 3) in
     let* lo = small in
     let* hi = small in
     let* interior = array ~size:(constant r) (int_range 0 2) in
     let* axes = subsequence (List.init r Fun.id) in
     let tap = triple (int_range 1 3) (int_range 1 2) (int_range 1 2) in
     let+ taps = list ~size:(constant (List.length axes)) tap in
     let windows =
       List.map2
         (fun axis (size, step, dilation) -> window ~step ~dilation axis size)
         axes taps
     in
     { x; pad = pad ~interior ~windows:(Array.of_list windows) lo hi })

(* The padded extents by the doc's rule: [lo] elements before the operand's,
   [hi] after and [interior] between neighbours. *)
let padded_extents (p : Nx_kernel.pad) x =
  Array.mapi
    (fun i d ->
      Iarray.get p.lo i + Iarray.get p.hi i + d
      + (Iarray.get p.interior i * max 0 (d - 1)))
    x

(* The padded shape, then the windows Move.Window takes of it, or an Error where
   Move.shape refuses them or the padded shape. *)
let law_padded (c : padded) =
  let padded = padded_extents c.pad c.x in
  let want =
    match M.shape (Window c.pad.windows) (Iarray.of_array padded) with
    | y -> Ok (Iarray.to_array y)
    | exception Invalid_argument _ -> misfit
  in
  let longer (w : M.window) =
    (w.dilation * (w.size - 1)) + 1 > padded.(w.axis)
  in
  cover "refused" (Result.is_error want);
  cover "two windows" (Iarray.length c.pad.windows >= 2);
  cover "a dilated window"
    (Iarray.exists
       (fun (w : M.window) -> w.dilation > 1 && w.size > 1)
       c.pad.windows);
  cover "a window longer than its axis"
    (Result.is_ok want && Iarray.exists longer c.pad.windows);
  equal shape want (Shape.padded c.pad c.x)

let padded_examples =
  [
    { x = [| 2 |]; pad = pad [| -2 |] [| 0 |] };
    { x = [| 2 |]; pad = pad [| -2 |] [| -1 |] };
    { x = [| 2 |]; pad = pad ~windows:[| window 0 2 |] [| 0 |] [| 0 |] };
    { x = [| 2 |]; pad = pad ~windows:[| window 0 3 |] [| 0 |] [| 0 |] };
    { x = [||]; pad = pad [||] [||] };
  ]

let padded_errors =
  let z = pad [| 0 |] [| 0 |] in
  let ws windows = pad ~windows [| 0; 0 |] [| 0; 0 |] in
  let ones = Array.make L.max_rank 1 in
  let zeros = Array.make L.max_rank 0 in
  [
    ("lo of another rank", pad [| 0; 0 |] [| 0 |], [| 2 |]);
    ("hi of another rank", pad [| 0 |] [| 0; 0 |], [| 2 |]);
    ( "interior of another rank",
      pad ~interior:[| 0; 0 |] [| 0 |] [| 0 |],
      [| 2 |] );
    ("a pad of another rank than x", z, [| 2; 2 |]);
    ("a negative interior", pad ~interior:[| -1 |] [| 0 |] [| 0 |], [| 2 |]);
    ("a negative padded extent", pad [| -3 |] [| 0 |], [| 2 |]);
    ("windows out of order", ws [| window 1 1; window 0 1 |], [| 2; 2 |]);
    ("two windows of one axis", ws [| window 0 1; window 0 1 |], [| 2; 2 |]);
    ( "a window past the rank",
      pad ~windows:[| window 1 1 |] [| 0 |] [| 0 |],
      [| 2 |] );
    ("a window of size 0", ws [| window 0 0 |], [| 2; 2 |]);
    ("a window of step 0", ws [| window ~step:0 0 1 |], [| 2; 2 |]);
    ("a window of dilation 0", ws [| window ~dilation:0 0 1 |], [| 2; 2 |]);
    ( "a window past the most axes",
      pad ~windows:[| window 0 1 |] zeros zeros,
      ones );
  ]

let test_padded_error (_, p, x) = is_error ~pp:pp_ints (Shape.padded p x)

let padded_bounds =
  let z = [| 0 |] in
  [
    ("padded to max_numel", pad [| big - 2 |] z, [| 2 |], Ok [| big |]);
    ("padded past max_numel", pad [| big - 1 |] z, [| 2 |], misfit);
    ("lo + hi past every int", pad [| max_int |] [| max_int |], [| 2 |], misfit);
    ("lo + hi below every int", pad [| min_int |] [| min_int |], [| 2 |], misfit);
    ( "interior times extent past every int",
      pad ~interior:[| max_int |] z z,
      [| 3 |],
      misfit );
    ( "interior times extent past max_numel",
      pad ~interior:[| big / 2 |] z z,
      [| 3 |],
      misfit );
    ( "no element, another extent of max_numel",
      pad [| 0; 0 |] [| 0; big - 1 |],
      [| 0; 1 |],
      Ok [| 0; big |] );
    ( "no element, another extent past max_numel",
      pad [| 0; 0 |] [| 0; big |],
      [| 0; 1 |],
      misfit );
  ]

let test_padded_bound (_, p, x, want) = equal shape want (Shape.padded p x)

(* Shape.reduce *)

let extreme : Nx_kernel.reduction -> bool = function
  | Monoid (Max | Min) | Arg _ -> true
  | Monoid (Sum | Prod) | Logsumexp | Moments -> false

type reduce_case = {
  y : int array;
  axes : int array;
  rs : Nx_kernel.reduction array;
}

let pp_reduce_case ppf (c : reduce_case) =
  Format.fprintf ppf "%s along %a of %a"
    (String.concat ", " (Array.to_list (Array.map reduction_name c.rs)))
    pp_ints c.axes pp_ints c.y

(* Reductions along a subset of a shape's axes. One case in three empties the
   first reduced axis, so that an extreme of no term, which a drawn shape alone
   gives in one case in eighteen, comes in one in seven. *)
let reduce_case =
  Gen.with_pp pp_reduce_case
    (let open Gen in
     let* y = Nx_array_gen.shape in
     let rank = Array.length y in
     let* axes = subsequence (List.init rank Fun.id) in
     let axes = Array.of_list axes in
     let* empty = frequency [ (2, constant false); (1, constant true) ] in
     let y =
       if empty && axes <> [||] then
         Array.mapi (fun i n -> if i = axes.(0) then 0 else n) y
       else y
     in
     let+ rs = array ~size:(int_range 1 2) (of_list reductions) in
     { y; axes; rs })

(* The destinations' shape by the rule: the loop's without the axes, and an
   Error for an extreme with a destination element and no term. *)
let law_reduce (c : reduce_case) =
  let kept =
    List.filteri (fun i _ -> not (Array.mem i c.axes)) (Array.to_list c.y)
  in
  let terms = Array.fold_left (fun n a -> n * c.y.(a)) 1 c.axes in
  let no_term = terms = 0 && List.fold_left ( * ) 1 kept > 0 in
  let want =
    if no_term && Array.exists extreme c.rs then misfit
    else Ok (Array.of_list kept)
  in
  cover "fits" (Result.is_ok want);
  cover "no term" no_term;
  cover "an extreme of no term" (Result.is_error want);
  equal shape want (Shape.reduce ~axes:(Iarray.of_array c.axes) c.rs c.y)

let reduce_examples =
  let empty = [| 2; 0 |] and sum = Nx_kernel.Monoid Sum in
  [
    { y = [| 2; 3 |]; axes = [||]; rs = [| sum |] };
    { y = [| 2; 3 |]; axes = [| 0; 1 |]; rs = [| sum |] };
    { y = empty; axes = [| 1 |]; rs = [| sum |] };
    { y = empty; axes = [| 1 |]; rs = [| Moments |] };
    { y = empty; axes = [| 1 |]; rs = [| Monoid Max |] };
    { y = empty; axes = [| 1 |]; rs = [| sum; Monoid Min |] };
    { y = empty; axes = [| 1 |]; rs = [| Arg Min |] };
    { y = [| 0; 0 |]; axes = [| 1 |]; rs = [| Monoid Max |] };
  ]

let reduce_errors =
  [
    ("axes out of order", [| 1; 0 |]);
    ("a repeated axis", [| 0; 0 |]);
    ("a negative axis", [| -1 |]);
    ("an axis past the rank", [| 2 |]);
  ]

let test_reduce_error (_, axes) =
  is_error ~pp:pp_ints
    (Shape.reduce ~axes:(Iarray.of_array axes) [| Monoid Sum |] [| 2; 3 |])

(* Shape.scan and Shape.carry *)

let scans =
  [
    ("[2; 3] along 1", 1, [| 2; 3 |]);
    ("[2; 0] along 1", 1, [| 2; 0 |]);
    ("[5] along 0", 0, [| 5 |]);
  ]

let test_scan (_, axis, y) = equal shape (Ok y) (Shape.scan ~axis y)

let axis_errors =
  [
    ("an axis past the rank", 2, [| 2; 3 |]);
    ("a negative axis", -1, [| 2; 3 |]);
    ("rank 0", 0, [||]);
  ]

let test_scan_error (_, axis, y) = is_error ~pp:pp_ints (Shape.scan ~axis y)

let carries =
  [
    ("[2; 3] along 1", 1, [| 2; 3 |], [| 2; 1 |]);
    ("[0; 3] along 0", 0, [| 0; 3 |], [| 1; 3 |]);
    ("[2; 0] along 1", 1, [| 2; 0 |], [| 2; 1 |]);
    ("[5] along 0", 0, [| 5 |], [| 1 |]);
  ]

let test_carry (_, axis, y, want) = equal shape (Ok want) (Shape.carry ~axis y)
let test_carry_error (_, axis, y) = is_error ~pp:pp_ints (Shape.carry ~axis y)

(* Shape.gather, gather_ranges, scatter and sort *)

let gathers =
  [
    ("positions of x's extents off the axis", 1, [| 2; 5; 4 |], [| 2; 3; 4 |]);
    ("positions from an empty axis", 1, [| 2; 5; 4 |], [| 2; 0; 4 |]);
    ("positions along the last axis", 1, [| 2; 5 |], [| 2; 3 |]);
    ("no position", 0, [| 0 |], [| 3 |]);
  ]

let test_gather (_, axis, idx, x) =
  equal shape (Ok idx) (Shape.gather ~axis idx x)

let gather_errors =
  [
    ("a negative axis", -1, [| 2 |], [| 2 |]);
    ("an axis past the rank", 1, [| 2 |], [| 2 |]);
    ("another extent off the axis", 1, [| 2; 5; 3 |], [| 2; 3; 4 |]);
    ("another rank", 1, [| 2; 5 |], [| 2; 3; 4 |]);
  ]

let test_gather_error (_, axis, idx, x) =
  is_error ~pp:pp_ints (Shape.gather ~axis idx x)

let ranges =
  [
    ("size along the runs' axis", 6, [| 3 |], [| 3 |], [| 4 |], [| 6 |]);
    ( "the cells kept",
      6,
      [| 2; 5 |],
      [| 2; 5 |],
      [| 2; 4; 3; 0 |],
      [| 2; 6; 3; 0 |] );
    ("no run, no element", 0, [| 0 |], [| 0 |], [| 0 |], [| 0 |]);
    ( "max_numel elements",
      big / 4,
      [| 3 |],
      [| 3 |],
      [| 4; 4 |],
      [| big / 4; 4 |] );
  ]

let test_ranges (_, size, s, e, x, want) =
  equal shape (Ok want) (Shape.gather_ranges ~size s e x)

let range_errors =
  [
    ("0-d runs", 6, [||], [||], [| 4 |]);
    ("starts and ends of other shapes", 6, [| 3 |], [| 4 |], [| 4 |]);
    ("another lead extent", 6, [| 2; 3 |], [| 2; 3 |], [| 3; 4 |]);
    ("no runs' axis in x", 6, [| 2; 3 |], [| 2; 3 |], [| 2 |]);
    ("a negative size", -1, [| 3 |], [| 3 |], [| 4 |]);
    ("past max_numel elements", (big / 4) + 1, [| 3 |], [| 3 |], [| 4; 4 |]);
    ("a size past every int's elements", max_int, [| 3 |], [| 3 |], [| 4; 2 |]);
  ]

let test_ranges_error (_, size, s, e, x) =
  is_error ~pp:pp_ints (Shape.gather_ranges ~size s e x)

let scatters =
  [
    ("into's shape", 0, [| 7; 4 |], [| 9; 4 |]);
    ("along the last axis", 0, [| 2 |], [| 3 |]);
    ("no update", 1, [| 2; 3 |], [| 2; 0 |]);
  ]

let test_scatter (_, axis, into, idx) =
  equal shape (Ok into) (Shape.scatter ~axis into idx idx)

let scatter_errors =
  [
    ( "positions and updates of other shapes",
      0,
      [| 7; 4 |],
      [| 9; 4 |],
      [| 8; 4 |] );
    ("another extent off the axis", 0, [| 7; 4 |], [| 9; 3 |], [| 9; 3 |]);
    ("another rank", 0, [| 7; 4 |], [| 9 |], [| 9 |]);
    ("a negative axis", -1, [| 2 |], [| 2 |], [| 2 |]);
    ("an axis past the rank", 1, [| 2 |], [| 2 |], [| 2 |]);
  ]

let test_scatter_error (_, axis, into, idx, u) =
  is_error ~pp:pp_ints (Shape.scatter ~axis into idx u)

let sorts =
  [
    ("x without k", 1, None, [| 2; 5 |], [| 2; 5 |]);
    ("k along the axis", 1, Some 3, [| 2; 5 |], [| 2; 3 |]);
    ("k of the whole axis", 1, Some 5, [| 2; 5 |], [| 2; 5 |]);
    ("a k of 0", 0, Some 0, [| 2 |], [| 0 |]);
  ]

let test_sort (_, axis, k, x, want) =
  equal shape (Ok want) (Shape.sort ~axis ~k x)

let sort_errors =
  [
    ("k past the axis's extent", 1, Some 6, [| 2; 5 |]);
    ("a negative k", 0, Some (-1), [| 2 |]);
    ("a negative axis", -1, None, [| 2 |]);
    ("an axis past the rank", 1, None, [| 2 |]);
  ]

let test_sort_error (_, axis, k, x) =
  is_error ~pp:pp_ints (Shape.sort ~axis ~k x)

(* Shape.unique *)

let three = result (triple ints ints ints) reason

let uniques =
  [
    ("five rows of two words", 3, [| 5; 2 |]);
    ("no row", 0, [| 0; 0 |]);
    ("max_numel entries", big, [| 1; 1 |]);
  ]

let test_unique (_, size, keys) =
  equal three (Ok ([| keys.(0) |], [| size |], [||])) (Shape.unique ~size keys)

let unique_errors =
  [
    ("keys of rank 0", 3, [||]);
    ("keys of rank 1", 3, [| 5 |]);
    ("keys of rank 3", 3, [| 5; 2; 1 |]);
    ("a negative size", -1, [| 5; 2 |]);
    ("a size past max_numel", big + 1, [| 5; 2 |]);
  ]

let test_unique_error (_, size, keys) =
  is_error ~pp:(Testable.pp (triple ints ints ints)) (Shape.unique ~size keys)

(* Shape.assemble *)

type assemble_case = { y : int array; regions : M.range array array }

let pp_regions ppf rs =
  Array.iter
    (fun r ->
      Format.fprintf ppf "[%s]"
        (String.concat "; "
           (Array.to_list
              (Array.map
                 (fun (x : M.range) ->
                   Printf.sprintf "%d+%d*%d" x.start x.count x.step)
                 r))))
    rs

let assemble_case =
  Gen.with_pp
    (fun ppf (c : assemble_case) ->
      Format.fprintf ppf "%a of regions %a" pp_ints c.y pp_regions c.regions)
    (let open Gen in
     let* y = Nx_array_gen.shape in
     let region =
       let rec go i acc =
         if i < 0 then constant (Array.of_list acc)
         else
           let* r = Nx_array_gen.range y.(i) in
           go (i - 1) (r :: acc)
       in
       go (Array.length y - 1) []
     in
     let+ regions = array ~size:(int_range 0 3) region in
     { y; regions })

(* Each piece has its region's shape, as Move.Slice keeps it. *)
let law_assemble (c : assemble_case) =
  let s = Iarray.of_array c.y in
  let piece r = Iarray.to_array (M.shape (Slice (Iarray.of_array r)) s) in
  cover "pieces" (c.regions <> [||]);
  equal
    (result (array ints) reason)
    (Ok (Array.map piece c.regions))
    (Shape.assemble c.y (Array.map Iarray.of_array c.regions))

let range start count step = { M.start; count; step }

let assemble_examples =
  [
    { y = [| 2 |]; regions = [||] };
    { y = [| 2 |]; regions = [| [| range 0 2 1 |] |] };
    { y = [| big |]; regions = [| [| range (big - 1) big (-1) |] |] };
  ]

let assemble_errors =
  [
    ("a region of another rank", [| [| range 0 1 1; range 0 1 1 |] |]);
    ("a region past its axis", [| [| range 1 2 1 |] |]);
    ("a region before its axis", [| [| range (-1) 1 1 |] |]);
    ("a step of 0", [| [| range 0 1 0 |] |]);
    ("a negative count", [| [| range 0 (-1) 1 |] |]);
    ("a region ending past every int", [| [| range 0 max_int max_int |] |]);
    ("a second region past its axis", [| [| range 0 2 1 |]; [| range 0 3 1 |] |]);
  ]

let test_assemble_error (_, regions) =
  is_error
    ~pp:(Testable.pp (array ints))
    (Shape.assemble [| 2 |] (Array.map Iarray.of_array regions))

(* Shape.contract *)

(* A contraction drawn by its groups' extents. [a] is batch, rows, then
   contracted axes, and [b] batch, contracted, then columns, each permuted:
   [pa.(i)] is the group axis of [a]'s axis [i]. *)
type contraction = {
  batch : int array;
  rows : int array;
  contracted : int array;
  columns : int array;
  pa : int array;
  pb : int array;
}

let permuted p s = Array.map (fun m -> s.(m)) p

let a_of (c : contraction) =
  permuted c.pa (Array.concat [ c.batch; c.rows; c.contracted ])

let b_of (c : contraction) =
  permuted c.pb (Array.concat [ c.batch; c.contracted; c.columns ])

let pp_contraction ppf (c : contraction) =
  Format.fprintf ppf "%a by %a, batch %a, contracting %a" pp_ints (a_of c)
    pp_ints (b_of c) pp_ints c.batch pp_ints c.contracted

let inverse p =
  let q = Array.make (Array.length p) 0 in
  Array.iteri (fun i m -> q.(m) <- i) p;
  q

(* The pairs of [c]: each group's axis, as an axis of [a] and of [b]. *)
let pairs_of (c : contraction) =
  let nb = Array.length c.batch and nr = Array.length c.rows in
  let ia = inverse c.pa and ib = inverse c.pb in
  let batch = Array.init nb (fun k -> (ia.(k), ib.(k))) in
  let contracting =
    Array.init (Array.length c.contracted) (fun k ->
        (ia.(nb + nr + k), ib.(nb + k)))
  in
  (batch, contracting)

(* The axes of a rank [r] that no pair names, in axis order. *)
let free r named =
  List.filter (fun ax -> not (List.mem ax named)) (List.init r Fun.id)

let contraction =
  Gen.with_pp pp_contraction
    (let open Gen in
     let extents = array ~size:(int_range 0 2) Nx_array_gen.extent in
     let order n = map Array.of_list (permutation (List.init n Fun.id)) in
     let* batch = extents in
     let* rows = extents in
     let* contracted = extents in
     let* columns = extents in
     let nb = Array.length batch and nk = Array.length contracted in
     let* pa = order (nb + Array.length rows + nk) in
     let+ pb = order (nb + nk + Array.length columns) in
     { batch; rows; contracted; columns; pa; pb })

(* The result by the rule: the batch extents in pair order, then [a]'s free
   extents, then [b]'s, each in axis order. *)
let law_contract (c : contraction) =
  let a = a_of c and b = b_of c in
  let batch, contracting = pairs_of c in
  let named side =
    Array.to_list (Array.map side (Array.append batch contracting))
  in
  let fa = free (Array.length a) (named fst) in
  let fb = free (Array.length b) (named snd) in
  let want =
    Array.concat
      [
        Array.map (fun (i, _) -> a.(i)) batch;
        Array.of_list (List.map (fun ax -> a.(ax)) fa);
        Array.of_list (List.map (fun ax -> b.(ax)) fb);
      ]
  in
  equal shape (Ok want)
    (Shape.contract ~batch:(Iarray.of_array batch)
       ~contracting:(Iarray.of_array contracting)
       ~segments:None (a, None) (b, None))

type call = {
  batch : (int * int) array;
  contracting : (int * int) array;
  segments : int array Nx_kernel.segments option;
  a : int array * int array option;
  b : int array * int array option;
}

let call ?(batch = [| (0, 0) |]) ?(contracting = [| (2, 1) |]) ?segments ?sa ?sb
    a b =
  { batch; contracting; segments; a = (a, sa); b = (b, sb) }

let contract (c : call) =
  Shape.contract ~batch:(Iarray.of_array c.batch)
    ~contracting:(Iarray.of_array c.contracting)
    ~segments:c.segments c.a c.b

let test_contract_fit (_, c, want) = equal shape (Ok want) (contract c)
let test_contract_error (_, c) = is_error ~pp:pp_ints (contract c)

let contract_errors =
  let a = [| 2; 3; 4 |] and sq = [| 4; 4 |] in
  [
    ("pairs of other extents", call a [| 2; 3; 5 |]);
    ("batch pairs of other extents", call a [| 3; 4; 5 |]);
    ("a pair past b's rank", call a [| 2 |]);
    ("a pair past a's rank", call ~contracting:[| (3, 1) |] a [| 2; 4; 5 |]);
    ("a negative axis of a", call ~batch:[| (-1, 0) |] a [| 2; 4; 5 |]);
    ("a negative axis of b", call ~batch:[| (0, -1) |] a [| 2; 4; 5 |]);
    ( "an axis of a in two pairs",
      call ~batch:[| (0, 0) |] ~contracting:[| (0, 1) |] sq sq );
    ( "an axis of b in two pairs",
      call ~batch:[| (0, 1); (1, 1) |] ~contracting:[||] sq sq );
    ( "an axis in two contracting pairs",
      call ~batch:[||] ~contracting:[| (1, 0); (1, 1) |] sq sq );
  ]

(* A matrix product of [a] by [b], scaled by [sa] and [sb]. *)
let mm ?segments ?sa ?sb a b =
  call ~batch:[||] ~contracting:[| (1, 0) |] ?segments ?sa ?sb a b

let scale_fits =
  let a = [| 4; 8 |] and b = [| 8; 6 |] in
  [
    ("blocks of 4 by 2", mm ~sb:[| 2; 3 |] a b, [| 4; 6 |]);
    ("a block of a whole axis", mm ~sb:[| 8; 1 |] a b, [| 4; 6 |]);
    ("a ratio of 1", mm ~sa:a ~sb:b a b, [| 4; 6 |]);
    ("0 along an empty axis", mm ~sa:[| 0; 2 |] [| 0; 8 |] b, [| 0; 6 |]);
    ("1 along an empty axis", mm ~sa:[| 1; 2 |] [| 0; 8 |] b, [| 0; 6 |]);
  ]

let scale_errors =
  let a = [| 4; 8 |] and b = [| 8; 6 |] in
  [
    ("an extent that does not divide", mm ~sb:[| 3; 6 |] a b);
    ("an extent past the operand's", mm ~sa:[| 8; 8 |] a b);
    ("a lower rank", mm ~sa:[| 4 |] a b);
    ("a higher rank", mm ~sa:[| 4; 8; 1 |] a b);
    ("0 along a full axis", mm ~sa:[| 0; 2 |] a b);
    ("a negative extent", mm ~sa:[| -4; 2 |] a b);
  ]

let rows ?(a_axis = 0) ?(b_axis = 0) ids =
  Nx_kernel.Rows { ids; a_axis; b_axis }

let groups ?(groups = 4) ?(pair = 0) ids =
  Nx_kernel.Contracting { ids; groups; pair }

(* Rows of [10; 8] by [3; 8; 6] along their 8s; rows per batch of [2; 10; 8] by
   [2; 3; 8; 6]; groups of [4; 8] by [8; 6]. *)
let by_rows s =
  call ~batch:[||]
    ~contracting:[| (1, 1) |]
    ~segments:s [| 10; 8 |] [| 3; 8; 6 |]

let by_batch_rows ids =
  call
    ~contracting:[| (2, 2) |]
    ~segments:(rows ~a_axis:1 ~b_axis:1 ids)
    [| 2; 10; 8 |] [| 2; 3; 8; 6 |]

let by_groups s = mm ~segments:s [| 4; 8 |] [| 8; 6 |]

let segment_fits =
  [
    ("Rows drops b's b_axis", by_rows (rows [| 10 |]), [| 10; 6 |]);
    ( "Rows of no group",
      call ~batch:[||]
        ~contracting:[| (1, 1) |]
        ~segments:(rows [| 10 |]) [| 10; 8 |] [| 0; 8; 6 |],
      [| 10; 6 |] );
    ("Rows per batch", by_batch_rows [| 2; 10 |], [| 2; 10; 6 |]);
    ( "Contracting puts the groups' axis first",
      by_groups (groups [| 8 |]),
      [| 4; 4; 6 |] );
    ( "Contracting into no group",
      by_groups (groups ~groups:0 [| 8 |]),
      [| 0; 4; 6 |] );
  ]

let segment_errors =
  [
    ("Rows' ids of another extent", by_rows (rows [| 3 |]));
    ("Rows' ids of another rank", by_rows (rows [| 10; 1 |]));
    ("Rows' 0-d ids", by_rows (rows [||]));
    ("an a_axis in a pair", by_rows (rows ~a_axis:1 [| 8 |]));
    ("a b_axis in a pair", by_rows (rows ~b_axis:1 [| 10 |]));
    ("an a_axis past a's rank", by_rows (rows ~a_axis:2 [| 10 |]));
    ("a negative b_axis", by_rows (rows ~b_axis:(-1) [| 10 |]));
    ("Rows' ids without the batch", by_batch_rows [| 10 |]);
    ("Rows' ids of another batch", by_batch_rows [| 3; 10 |]);
    ("Contracting's 0-d ids", by_groups (groups [||]));
    ("Contracting's ids of another extent", by_groups (groups [| 4 |]));
    ("Contracting's ids of another rank", by_groups (groups [| 2; 8 |]));
    ("negative groups", by_groups (groups ~groups:(-1) [| 8 |]));
    ("a pair past the contracting ones", by_groups (groups ~pair:1 [| 8 |]));
    ("a negative pair", by_groups (groups ~pair:(-1) [| 8 |]));
  ]

let contract_bounds =
  let k = 1 lsl 27 in
  let ones n = Array.make n 1 in
  let deep b =
    call ~batch:[||]
      ~contracting:[| (0, 0) |]
      ~segments:(groups ~groups:1 [| 1 |])
      (ones (L.max_rank - 1))
      b
  in
  let none = call ~batch:[||] ~contracting:[||] in
  [
    ( "a result of the most axes",
      none (ones L.max_rank) [||],
      Ok (ones L.max_rank) );
    ("a result past the most axes", none (ones L.max_rank) [| 1 |], misfit);
    ("groups' result of the most axes", deep [| 1; 1 |], Ok (ones L.max_rank));
    ("groups' result past the most axes", deep [| 1; 1; 1 |], misfit);
    ("a result of max_numel", mm [| k / 2; 0 |] [| 0; k |], Ok [| k / 2; k |]);
    ("a result past max_numel", mm [| k; 0 |] [| 0; k |], misfit);
    ( "groups of max_numel",
      mm ~segments:(groups ~groups:big [| 8 |]) [| 1; 8 |] [| 8; 1 |],
      Ok [| big; 1; 1 |] );
    ( "groups past max_numel",
      mm ~segments:(groups ~groups:(big + 1) [| 8 |]) [| 1; 8 |] [| 8; 1 |],
      misfit );
  ]

let test_contract_bound (_, c, want) = equal shape want (contract c)

(* Shape.c2c, r2c and c2r *)

type transform = { x : int array; axes : int array; n : int }

let pp_transform ppf (c : transform) =
  Format.fprintf ppf "%a along %a, n %d" pp_ints c.x pp_ints c.axes c.n

(* A transform of an operand of rank 1 to 4 along a non-empty subset of its
   axes, written three times in four as it is, else emptied, reversed, with its
   last axis repeated, -1 put first or the rank put last; with [n] points, whose
   bins the last of the axes holds half the time. *)
let transform =
  Gen.with_pp pp_transform
    (let open Gen in
     let* rank = int_range 1 4 in
     let* x = array ~size:(constant rank) (int_range 0 9) in
     let* top = int_range 0 (rank - 1) in
     let* kept = subsequence (List.init top Fun.id) in
     let axes = Array.of_list (kept @ [ top ]) in
     let* spoil =
       frequency
         [
           (15, constant Fun.id);
           (1, constant (fun _ -> [||]));
           (1, constant (fun a -> Array.of_list (List.rev (Array.to_list a))));
           (1, constant (fun a -> Array.append a [| top |]));
           (1, constant (fun a -> Array.append [| -1 |] a));
           (1, constant (fun a -> Array.append a [| rank |]));
         ]
     in
     let* n = int_range (-1) 17 in
     let+ binned = bool in
     let x =
       if binned then
         Array.mapi (fun i d -> if i = top then (n / 2) + 1 else d) x
       else x
     in
     { x; axes = spoil axes; n })

(* Whether [axes] are strictly increasing axes of a rank [r], not empty. *)
let takes r axes =
  axes <> [||]
  && Array.for_all (fun a -> a >= 0 && a < r) axes
  && Array.for_all Fun.id
       (Array.init (Array.length axes - 1) (fun i -> axes.(i) < axes.(i + 1)))

let along x i n = Array.mapi (fun j d -> if j = i then n else d) x
let last a = a.(Array.length a - 1)

let law_c2c (c : transform) =
  let fits = takes (Array.length c.x) c.axes in
  cover "fits" fits;
  cover "refused" (not fits);
  equal shape
    (if fits then Ok c.x else misfit)
    (Shape.c2c ~axes:(Iarray.of_array c.axes) c.x)

let law_r2c (c : transform) =
  let fits = takes (Array.length c.x) c.axes in
  cover "fits" fits;
  cover "refused" (not fits);
  let want =
    if not fits then misfit
    else
      let top = last c.axes in
      Ok (along c.x top ((c.x.(top) / 2) + 1))
  in
  equal shape want (Shape.r2c ~axes:(Iarray.of_array c.axes) c.x)

let law_c2r (c : transform) =
  let binned =
    takes (Array.length c.x) c.axes
    && c.n >= 0
    && c.x.(last c.axes) = (c.n / 2) + 1
  in
  let y = if binned then along c.x (last c.axes) c.n else c.x in
  let fits = binned && admits [ y ] in
  cover "fits" fits;
  cover "refused" (not fits);
  cover "a negative n" (c.n < 0);
  let want = if fits then Ok y else misfit in
  equal shape want (Shape.c2r ~axes:(Iarray.of_array c.axes) ~n:c.n c.x)

let r2c_examples = [ { x = [| 2; 0 |]; axes = [| 1 |]; n = 0 } ]

let c2r_examples =
  [
    { x = [| 0 |]; axes = [| 0 |]; n = -1 };
    { x = [| 1 |]; axes = [| 0 |]; n = -1 };
    { x = [| 2; 3 |]; axes = [| 1 |]; n = 5 };
    { x = [| 2; 1 |]; axes = [| 1 |]; n = 0 };
    { x = [| (big / 2) + 1 |]; axes = [| 0 |]; n = big };
    { x = [| (big / 2) + 1 |]; axes = [| 0 |]; n = big + 1 };
  ]

(* Matrices *)

(* A matrix's shape: a batch of rank 0 to 2, then [m] rows and [n] columns,
   square a third of the time; cut to rank 0 or 1 one time in ten. *)
let matrix =
  let open Gen in
  let* batch = array ~size:(int_range 0 2) (int_range 0 3) in
  let* m = int_range 0 5 in
  let* n = int_range 0 5 in
  let* square = frequency [ (2, constant false); (1, constant true) ] in
  let a = Array.append batch [| m; (if square then m else n) |] in
  let+ cut =
    frequency [ (9, constant None); (1, map Option.some (int_range 0 1)) ]
  in
  match cut with None -> a | Some r -> Array.sub a 0 (min r (Array.length a))

let matrices = Gen.with_pp pp_ints matrix

(* [a]'s batch, rows and columns, where it has two axes or more. *)
let split a =
  let r = Array.length a in
  if r < 2 then None else Some (Array.sub a 0 (r - 2), a.(r - 2), a.(r - 1))

let cover_forms a =
  cover "rank below 2" (split a = None);
  let form f = match split a with Some (_, m, n) -> f m n | None -> false in
  cover "wide" (form ( < ));
  cover "tall" (form ( > ));
  cover "square" (form ( = ))

let low_ranks = [ [||]; [| 3 |] ]
let two = result (pair ints ints) reason

let law_cholesky a =
  cover_forms a;
  let want =
    match split a with Some (_, m, n) when m = n -> Ok a | _ -> misfit
  in
  equal shape want (Shape.cholesky a)

let law_lu a =
  cover_forms a;
  let want =
    match split a with
    | None -> misfit
    | Some (batch, m, n) ->
        let y tail = Array.append batch tail in
        Ok (a, y [| min m n |], y [| m |])
  in
  equal three want (Shape.lu a)

let pp_factors ppf = function
  | Nx_kernel.Reduced -> Format.pp_print_string ppf "Reduced"
  | Complete -> Format.pp_print_string ppf "Complete"

let factored =
  Gen.with_pp
    (fun ppf (f, a) -> Format.fprintf ppf "%a %a" pp_factors f pp_ints a)
    (Gen.pair (Gen.of_list [ Nx_kernel.Reduced; Complete ]) matrix)

let law_qr (f, a) =
  cover_forms a;
  let want =
    match (split a, f) with
    | None, _ -> misfit
    | Some (batch, m, n), Nx_kernel.Reduced ->
        let y tail = Array.append batch tail and k = min m n in
        Ok (y [| m; k |], y [| k; n |])
    | Some (batch, m, n), Complete ->
        let y tail = Array.append batch tail in
        Ok (y [| m; m |], y [| m; n |])
  in
  let want =
    match want with Ok (q, r) when not (admits [ q; r ]) -> misfit | w -> w
  in
  equal two want (Shape.qr f a)

let law_svd (f, a) =
  cover_forms a;
  let want =
    match (split a, f) with
    | None, _ -> misfit
    | Some (batch, m, n), Nx_kernel.Reduced ->
        let y tail = Array.append batch tail and k = min m n in
        Ok (y [| k |], y [| m; k |], y [| k; n |])
    | Some (batch, m, n), Complete ->
        let y tail = Array.append batch tail in
        Ok (y [| min m n |], y [| m; m |], y [| n; n |])
  in
  let want =
    match want with
    | Ok (s, u, vh) when not (admits [ s; u; vh ]) -> misfit
    | w -> w
  in
  equal three want (Shape.svd f a)

(* Square factors past max_numel of a matrix of no element. *)
let factored_examples =
  let k = 1 lsl 27 in
  List.concat_map
    (fun f ->
      [
        (f, [||]);
        (f, [| 3 |]);
        (f, [| k / 2; 0 |]);
        (f, [| k; 0 |]);
        (f, [| 0; k |]);
      ])
    [ Nx_kernel.Reduced; Complete ]

let law_eigh a =
  cover_forms a;
  let want =
    match split a with
    | Some (batch, m, n) when m = n -> Ok (Array.append batch [| n |], a)
    | _ -> misfit
  in
  equal two want (Shape.eigh a)

let solve =
  Gen.with_pp
    (fun ppf (a, b) -> Format.fprintf ppf "%a by %a" pp_ints a pp_ints b)
    (let open Gen in
     let* a = matrix in
     let* cols = int_range 0 4 in
     let+ fits = frequency [ (3, constant true); (1, constant false) ] in
     let b =
       match split a with
       | Some (batch, _, n) ->
           Array.append batch [| (if fits then n else n + 1); cols |]
       | None -> [| cols |]
     in
     (a, b))

let law_solve (a, b) =
  let want =
    match split a with
    | Some (batch, m, n) when m = n && b = Array.append batch [| n; last b |] ->
        Ok b
    | _ -> misfit
  in
  cover "fits" (Result.is_ok want);
  cover "refused" (Result.is_error want);
  equal shape want (Shape.solve_triangular a b)

let solve_errors =
  [
    ("b of another batch", [| 2; 3; 3 |], [| 3; 3; 1 |]);
    ("b of another rank", [| 2; 3; 3 |], [| 3; 1 |]);
    ("b of one axis", [| 3; 3 |], [| 3 |]);
  ]

let test_solve_error (_, a, b) =
  is_error ~pp:pp_ints (Shape.solve_triangular a b)

(* Shape.attention *)

type attention = {
  q : int array;
  kv : (int array, int array) Nx_kernel.kv;
  lengths : int array option;
  loads : int array array;
}

let dense ?lengths ?(loads = [||]) q k v =
  { q; kv = Dense { k; v }; lengths; loads }

let paged ?(loads = [||]) q k v table =
  { q; kv = Paged { k; v; table }; lengths = None; loads }

let attend (c : attention) = Shape.attention c.q c.kv ~lengths:c.lengths c.loads

let attentions =
  let q = [| 2; 4; 3; 5 |] in
  let k = [| 2; 2; 7; 5 |] and v = [| 2; 2; 7; 6 |] in
  let pools = ([| 5; 1; 4; 5 |], [| 5; 1; 4; 6 |]) in
  [
    ("dense, grouped heads", dense q k v, ([| 2; 4; 3; 6 |], [| 2; 4; 3 |]));
    ( "lengths and a load",
      dense ~lengths:[| 2 |] ~loads:[| [| 2; 4; 3; 7 |] |] q k v,
      ([| 2; 4; 3; 6 |], [| 2; 4; 3 |]) );
    ( "paged",
      paged ~loads:[| [| 2; 4; 3; 12 |] |] q (fst pools) (snd pools) [| 2; 3 |],
      ([| 2; 4; 3; 6 |], [| 2; 4; 3 |]) );
    ( "no heads",
      dense [| 0; 3; 5 |] [| 0; 7; 5 |] [| 0; 7; 6 |],
      ([| 0; 3; 6 |], [| 0; 3 |]) );
    ( "paged keys of max_numel",
      paged [| 1; 1; 1 |] [| 1; 1; 2; 1 |] [| 1; 1; 2; 1 |] [| big / 2 |],
      ([| 1; 1; 1 |], [| 1; 1 |]) );
  ]

let test_attention (_, c, want) = equal two (Ok want) (attend c)

let attention_errors =
  let q = [| 2; 4; 3; 5 |] in
  let k = [| 2; 2; 7; 5 |] and v = [| 2; 2; 7; 6 |] in
  let kp = [| 5; 1; 4; 5 |] and vp = [| 5; 1; 4; 6 |] in
  let e = 1 lsl 27 in
  [
    ("q of two axes", dense [| 3; 5 |] [| 7; 5 |] [| 7; 6 |]);
    ("heads that do not divide", dense q [| 2; 3; 7; 5 |] [| 2; 3; 7; 6 |]);
    ("no key head for two", dense [| 2; 3; 5 |] [| 0; 7; 5 |] [| 0; 7; 6 |]);
    ("keys of another dimension", dense q [| 2; 2; 7; 4 |] v);
    ("keys of another lead", dense q [| 3; 2; 7; 5 |] [| 3; 2; 7; 6 |]);
    ("values of other keys", dense q k [| 2; 2; 8; 6 |]);
    ("lengths of another shape", dense ~lengths:[| 3 |] q k v);
    ("a load of another shape", dense ~loads:[| [| 2; 4; 3; 6 |] |] q k v);
    ("a table of another rank", paged q kp vp [| 3 |]);
    ("pools of other pages", paged q kp [| 4; 1; 4; 6 |] [| 2; 3 |]);
    ("a key pool of another dimension", paged q [| 5; 1; 4; 4 |] vp [| 2; 3 |]);
    ( "paged keys past max_numel",
      paged [| 1; 1; 1 |] [| 1; 1; 2; 1 |] [| 1; 1; 2; 1 |] [| (big / 2) + 1 |]
    );
    ("o past max_numel", dense [| e; 1; 1 |] [| 1; 1; 1 |] [| 1; 1; e |]);
  ]

let test_attention_error (_, c) =
  is_error ~pp:(Testable.pp (pair ints ints)) (attend c)

(* Libraries *)

(* The host addresses its own memory and a software device's, and not an io
   device's. *)
let test_computes_on () =
  let devices =
    [
      ("the host", Rig.host);
      ("a software device", Result.get_ok (Rig.software "nx-kernel-computes-on"));
      ("an io device", Nx_array_support.io_device ());
    ]
  in
  equal
    (list (pair string bool))
    [ ("the host", true); ("a software device", true); ("an io device", false) ]
    (List.map (fun (n, d) -> (n, Nx_cpu.computes_on d)) devices)

(* The suite *)

let fst3 (n, _, _) = n
let fst4 (n, _, _, _) = n
let fst5 (n, _, _, _, _) = n
let fst6 (n, _, _, _, _, _) = n

let () =
  exit
    (run "nx_kernel.shape"
       [
         group "carrier"
           [
             cases
               "is float32 for a float narrower than it, int8 and uint8 for \
                int4 and uint4, bool for bit, else the dtype"
               ~id:name D.all test_carrier;
           ];
         group "float_carrier"
           [
             cases "is the carrier of the float"
               ~id:(fun (Nx_kernel.Any_float dt) -> D.name dt)
               floats test_float_carrier;
           ];
         group "accepts"
           [
             cases
               "takes for Sum and Prod every dtype but booleans, for Max, Min \
                and Arg every dtype, for Logsumexp and Moments floats"
               ~id:reduction_name reductions test_accepts;
           ];
         group "accumulates"
           [
             test
               "holds for every dtype but booleans and floats narrower than \
                float32"
               test_accumulates;
           ];
         group "same_precision"
           [
             test "pairs float32 and complex64, and float64 and complex128"
               test_same_precision;
           ];
         group "full_precision"
           [
             test "holds for float32, float64, complex64 and complex128"
               test_full_precision;
           ];
         group "Shape"
           [
             group "padded"
               [
                 prop ~examples:padded_examples
                   "is the padded extents, then the windows Move.Window takes"
                   padded_case law_padded;
                 cases
                   "is an Error for lo, hi or interior of another rank, a \
                    negative interior or padded extent, or windows not on \
                    strictly increasing axes or of size, step or dilation 0"
                   ~id:fst3 padded_errors test_padded_error;
                 cases
                   "admits an extent of max_numel and is an Error past it, or \
                    where lo + hi or interior times an extent passes an int"
                   ~id:fst4 padded_bounds test_padded_bound;
               ];
             group "reduce"
               [
                 prop ~examples:reduce_examples
                   "is y without axes, an Error for an extreme with a \
                    destination element and no term"
                   reduce_case law_reduce;
                 cases
                   "is an Error unless axes are strictly increasing axes of y"
                   ~id:fst reduce_errors test_reduce_error;
               ];
             group "scan"
               [
                 cases "is y" ~id:fst3 scans test_scan;
                 cases "is an Error for an axis outside [0, rank)" ~id:fst3
                   axis_errors test_scan_error;
               ];
             group "carry"
               [
                 cases "is y with extent 1 along axis" ~id:fst4 carries
                   test_carry;
                 cases "is an Error for an axis outside [0, rank)" ~id:fst3
                   axis_errors test_carry_error;
               ];
             group "gather"
               [
                 cases "is idx, of x's rank and extents but along axis" ~id:fst4
                   gathers test_gather;
                 cases
                   "is an Error for an axis outside [0, rank), or idx of \
                    another rank or extent off the axis"
                   ~id:fst4 gather_errors test_gather_error;
               ];
             group "gather_ranges"
               [
                 cases "is x with size elements along the runs' axis" ~id:fst6
                   ranges test_ranges;
                 cases
                   "is an Error for 0-d or differing runs, x without the runs' \
                    lead extents or axis, a negative size, or past max_numel"
                   ~id:fst5 range_errors test_ranges_error;
               ];
             group "scatter"
               [
                 cases
                   "is into, where idx and updates have one shape, into's but \
                    along axis"
                   ~id:fst4 scatters test_scatter;
                 cases
                   "is an Error for idx and updates of other shapes, or an \
                    axis outside [0, rank)"
                   ~id:fst5 scatter_errors test_scatter_error;
               ];
             group "sort"
               [
                 cases "is x with k elements along axis, or x" ~id:fst5 sorts
                   test_sort;
                 cases
                   "is an Error for k negative or past the axis's extent, or \
                    an axis outside [0, rank)"
                   ~id:fst4 sort_errors test_sort_error;
               ];
             group "unique"
               [
                 cases "is [n], [size] and [] for keys of [n; w]" ~id:fst3
                   uniques test_unique;
                 cases
                   "is an Error unless keys have rank 2 and size is not \
                    negative, or past max_numel"
                   ~id:fst3 unique_errors test_unique_error;
               ];
             group "assemble"
               [
                 prop ~examples:assemble_examples
                   "is each region's shape, as Move.Slice keeps it"
                   assemble_case law_assemble;
                 cases "is an Error for a region Move.shape refuses as a Slice"
                   ~id:fst assemble_errors test_assemble_error;
               ];
             group "contract"
               [
                 prop
                   "is the batch axes in pair order, then a's free axes, then \
                    b's"
                   contraction law_contract;
                 cases
                   "is an Error for pairs of other extents, axes outside a \
                    rank or in two pairs"
                   ~id:fst contract_errors test_contract_error;
                 cases
                   "takes a scale of its operand's rank, each extent dividing \
                    the operand's, 0 only along an empty axis"
                   ~id:fst3 scale_fits test_contract_fit;
                 cases "is an Error for another scale" ~id:fst scale_errors
                   test_contract_error;
                 cases
                   "drops b's b_axis for Rows; puts the groups' axis first for \
                    Contracting"
                   ~id:fst3 segment_fits test_contract_fit;
                 cases
                   "is an Error for segments' ids of other shapes, axes in \
                    pairs or past a rank, another pair or negative groups"
                   ~id:fst segment_errors test_contract_error;
                 cases
                   "admits max_rank axes and max_numel elements and is an \
                    Error past them"
                   ~id:fst3 contract_bounds test_contract_bound;
               ];
             group "c2c"
               [
                 prop
                   "is x, where axes are strictly increasing axes of x, not \
                    empty"
                   transform law_c2c;
               ];
             group "r2c"
               [
                 prop ~examples:r2c_examples
                   "is x but n/2 + 1 bins along the last of axes" transform
                   law_r2c;
               ];
             group "c2r"
               [
                 prop ~examples:c2r_examples
                   "is x but n points along the last of axes, where x has n/2 \
                    + 1 bins there and n is not negative"
                   transform law_c2r;
               ];
             group "cholesky"
               [ prop ~examples:low_ranks "is a, square" matrices law_cholesky ];
             group "lu"
               [
                 prop ~examples:low_ranks "is a, [...; k] and [...; m]" matrices
                   law_lu;
               ];
             group "qr"
               [
                 prop ~examples:factored_examples
                   "is [...; m; k] and [...; k; n] under Reduced, [...; m; m] \
                    and [...; m; n] under Complete"
                   factored law_qr;
               ];
             group "svd"
               [
                 prop ~examples:factored_examples
                   "is [...; k], then [...; m; k] and [...; k; n] under \
                    Reduced, [...; m; m] and [...; n; n] under Complete"
                   factored law_svd;
               ];
             group "eigh"
               [
                 prop ~examples:low_ranks "is [...; n] and a, square" matrices
                   law_eigh;
               ];
             group "solve_triangular"
               [
                 prop "is b, [...; n; r], for a square of n and b of a's batch"
                   solve law_solve;
                 cases "is an Error for b of another batch or rank" ~id:fst3
                   solve_errors test_solve_error;
               ];
             group "attention"
               [
                 cases "is [lead...; hq; tq; dv] and [lead...; hq; tq]" ~id:fst3
                   attentions test_attention;
                 cases
                   "is an Error for q of fewer than three axes, kv of other \
                    shapes, hk not dividing hq, other lengths or loads, or tk \
                    past max_numel"
                   ~id:fst attention_errors test_attention_error;
               ];
           ];
         group "name"
           [
             test "is \"nx.cpu\"" (fun () -> equal string "nx.cpu" Nx_cpu.name);
           ];
         group "computes_on"
           [
             test
               "is whether the host addresses d's memory: its own and a \
                software device's, not an io device's"
               test_computes_on;
           ];
       ])
