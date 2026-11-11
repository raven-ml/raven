(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions through every kernel library the host runs: within the
   largest bound nx_kernel.mli allows a library of the exact sum, scaled and
   segmented ones too, the same bits into a destination identical to init,
   nothing written where declined, and misfits refused. Each library
   computes every case the laws draw: nx_cpu.mli and nx_cuda.mli state
   it. *)

open Windtrap
open Contract_cases
module A = Nx_array
module D = Nx_array.Dtype
module Support = Nx_kernels_support

(* The reference *)

(* The sum of [o]'s products and init in two parts, with an error under
   2^-100 of the sum of their magnitudes, and that sum. Products and the
   running sum are held in two parts. *)
let exact o =
  (* A loop: refs that a closure takes are boxed, a float allocated a
     term. *)
  let start = Option.value ~default:0. o.init in
  let hi = ref start and lo = ref 0. and mag = ref (Float.abs start) in
  for k = 0 to Array.length o.av - 1 do
    let x = o.av.(k) and y = o.bv.(k) in
    let p = x *. y in
    let s, t = two_sum !hi p in
    hi := s;
    lo := !lo +. t +. Float.fma x y (-.p);
    mag := !mag +. Float.abs p
  done;
  (!hi, !lo, !mag)

let unit_roundoff acc =
  if part_of acc = D.Any D.Float32 then 0x1p-24 else 0x1p-53

let smallest_subnormal acc =
  if part_of acc = D.Any D.Float32 then 0x1p-149 else 0x1p-1074

(* The bound *)

(* [o]'s sum in plain float64 additions, which is not finite where the
   exact one is not. *)
let naive o =
  let s = ref (Option.value ~default:0. o.init) in
  for k = 0 to Array.length o.av - 1 do
    s := !s +. (o.av.(k) *. o.bv.(k))
  done;
  !s

(* Σ|a||b| of an output whose parts are [parts]: a complex one's factors
   are its real part's, [ar] and [ai] at terms [2p] and [2p + 1] against
   [br] and [-bi]. *)
let magnitude parts =
  let o = parts.(0) in
  if Array.length parts = 1 then begin
    let m = ref 0. in
    for k = 0 to Array.length o.av - 1 do
      m := !m +. Float.abs (o.av.(k) *. o.bv.(k))
    done;
    !m
  end
  else begin
    let m = ref 0. in
    for p = 0 to (Array.length o.av / 2) - 1 do
      let a = Float.hypot o.av.(2 * p) o.av.((2 * p) + 1) in
      let b = Float.hypot o.bv.(2 * p) o.bv.((2 * p) + 1) in
      m := !m +. (a *. b)
    done;
    !m
  end

(* nx_kernel.mli's [β(n) = ⌈5 min(n, 256) / 4⌉ + ⌈n / 128⌉]. *)
let beta n = ((5 * min n 256) + 3) / 4 + ((n + 127) / 128)

(* An output [xs] of [c], one value a part, is within the largest bound
   nx_kernel.mli allows a library, [u|r| + (1 + u) e Σ|a||b|] of the exact
   [r], init and its [K] products, a complex one in modulus, [e] being
   [γ(β(K))] for a real [acc] and [√2 γ(β(2K))] for a complex one, and u
   the unit roundoff of [acc]'s parts. Where results underflow it may err by η
   more for each term of each part, [η] [acc]'s smallest subnormal, twice
   the most a rounding into the subnormals errs. A part whose sum is not
   finite is matched to the bits of its naive sum. *)
let within c parts xs =
  let show x = Printf.sprintf "%h" x in
  let sums = Array.map naive parts in
  if not (Array.for_all Float.is_finite sums) then
    let bad = ref None in
    Array.iteri
      (fun p s ->
        if (not (Float.is_finite s)) && bits xs.(p) <> bits s then
          bad := Some (show xs.(p) ^ ", not " ^ show s))
      sums;
    !bad
  else
    let exact = Array.map exact parts in
    (* Each part subtracted apart, so that rounding the exact sum adds no
       error. *)
    let off = Array.mapi (fun p (hi, lo, _) -> xs.(p) -. hi -. lo) exact in
    let err = Array.fold_left Float.hypot 0. off in
    let terms = Array.length parts.(0).av in
    let complex = Array.length parts > 1 in
    let k = terms / Array.length parts in
    let u = unit_roundoff c.acc in
    let gamma n =
      let n = float_of_int n in
      n *. u /. (1. -. (n *. u))
    in
    let e =
      if complex then Float.sqrt 2. *. gamma (beta (2 * k))
      else gamma (beta k)
    in
    let r =
      Array.fold_left (fun s (hi, lo, _) -> Float.hypot s (hi +. lo)) 0. exact
    in
    let has_init = Option.is_some parts.(0).init in
    let parts_n = float_of_int (Array.length parts) in
    let eta = smallest_subnormal c.acc *. float_of_int (terms + 1) *. parts_n in
    let slack =
      Array.fold_left (fun s (_, _, m) -> s +. (0x1p-100 *. m)) 0. exact
    in
    let bound =
      (if has_init then u *. r else 0.)
      +. ((1. +. u) *. e *. magnitude parts)
      +. slack +. eta
    in
    if err <= bound then None
    else
      let all f = String.concat ", " (Array.to_list (Array.map f exact)) in
      Some
        (Printf.sprintf "%s is %h from %s, past %h"
           (String.concat ", " (Array.to_list (Array.map show xs)))
           err (all (fun (hi, lo, _) -> show (hi +. lo))) bound)

(* The first outputs of [y] past their bound. *)
let outside c y =
  let _, each = outputs c in
  let g = planes c.acc y in
  let bad = ref [] in
  each (fun yi k parts ->
      match within c parts (Array.map (fun part -> part.(k)) g) with
      | Some why when List.length !bad < 8 ->
          bad := Format.asprintf "%a: %s" Nx_array_gen.pp_ints yi why :: !bad
      | _ -> ());
  List.rev !bad

let law_bound b c =
  covers c;
  match run b c with
  | None -> fail "declined"
  | Some y -> equal (list string) [] (outside c y)

(* Scales *)

(* A scale of an operand: [Exponent] codes or float32 [Value]s, a ratio
   along each axis, and the seed of its values. *)
type spec = { exponent : bool; ratios : int array; seed : int }

(* The scale [sp] of an operand of shape [s], on the host: codes near 1,
   at times 255; values on the grid. *)
let scale_of s sp : Nx_kernel.scale =
  let r = Random.State.make [| sp.seed |] in
  let shape = Array.mapi (fun i e -> if e = 0 then 0 else e / sp.ratios.(i)) s in
  let n = total shape in
  let dims = Iarray.of_array shape in
  if sp.exponent then
    Exponent
      (A.of_array D.Uint8 dims
         (Array.init n (fun _ ->
              if Random.State.int r 50 = 0 then 255 else 120 + Random.State.int r 15)))
  else
    Value
      (A.of_array D.Float32 dims
         (Array.init n (fun _ -> grid.(Random.State.int r (Array.length grid)))))

(* The scale's value of each element of [x], in C order of its indices: a
   code's power of two or NaN, a value as a cast into [acc] holds it. *)
let scale_values acc (A.Any x) (sc : Nx_kernel.scale) =
  let s = shape_of (A.Any x) in
  let values, ss =
    match sc with
    | Exponent e ->
        ( Array.map
            (fun c -> if c = 255 then Float.nan else Float.ldexp 1. (c - 127))
            (A.to_array e),
          shape_of (A.Any e) )
    | Value v ->
        let v = cast_to acc (A.Any v) in
        (A.to_array (to64 v), shape_of v)
  in
  let st = Array.make (Array.length ss) 1 in
  for i = Array.length ss - 2 downto 0 do
    st.(i) <- st.(i + 1) * ss.(i + 1)
  done;
  Array.init (total s) (fun t ->
      let at = ref 0 and rest = ref t in
      for d = Array.length s - 1 downto 0 do
        let i = !rest mod s.(d) in
        rest := !rest / s.(d);
        at := !at + (i / (s.(d) / ss.(d)) * st.(d))
      done;
      values.(!at))

(* [x] as [acc] holds it times its scale, exactly, as float64: a scaled
   operand's terms are this rounded once into [acc]. *)
let scaled acc x sc =
  let xs = A.to_array (to64 (cast_to acc x)) and ss = scale_values acc x sc in
  A.Any
    (A.of_array D.Float64
       (Iarray.of_array (shape_of x))
       (Array.mapi (fun i v -> v *. ss.(i)) xs))

(* Cases of float32 and float64 accumulators, of operands each holds, with
   a scale on a, on b or on both: ratios 1 or a whole axis. *)
let scaled_case =
  let open Gen in
  let spec x =
    let s = shape_of x in
    let* exponent = bool and+ seed = int in
    let+ whole = array ~size:(constant (Array.length s)) bool in
    { exponent; ratios = Array.mapi (fun i e -> if whole.(i) && e > 0 then e else 1) s; seed }
  in
  Gen.with_pp
    (fun ppf (c, sa, sb) ->
      Format.fprintf ppf "%a; scaled %s%s" pp_case c
        (if Option.is_some sa then "a" else "")
        (if Option.is_some sb then "b" else ""))
    (let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64 ] in
     let* c = case_of ~acc ~out:acc ~dts:(held acc) () in
     let* which = int_range 1 3 in
     let* sa = spec c.a and+ sb = spec c.b in
     constant
       ( c,
         (if which land 1 <> 0 then Some sa else None),
         if which land 2 <> 0 then Some sb else None ))

(* A scaled contraction is within the bound of the exact sum of its terms,
   each element as the accumulator holds it times its scale, rounded once
   into the accumulator. *)
let law_scaled (b : Support.backend) (c, sa, sb) =
  let sa = Option.map (scale_of (shape_of c.a)) sa in
  let sb = Option.map (scale_of (shape_of c.b)) sb in
  match run ?sa ?sb b c with
  | None -> fail "declined"
  | Some y ->
      let a = Option.fold ~none:c.a ~some:(scaled c.acc c.a) sa in
      let b' = Option.fold ~none:c.b ~some:(scaled c.acc c.b) sb in
      equal (list string) [] (outside { c with a; b = b' } y)

(* Segments *)

(* A segmented contraction: its case, unsegmented, of a [z; m; k] and b
   [z; k; n] (Contracting) or [z; k; n; g] (Rows), [z] present where [nb];
   its segments over host ids. *)
type seg = {
  sc : case;
  segments : (int64, D.int64_elt) A.t Nx_kernel.segments;
  yshape : int array;
}

let pp_seg ppf s =
  let what =
    match s.segments with
    | Rows _ -> "rows"
    | Contracting g -> Printf.sprintf "contracting into %d groups" g.groups
  in
  Format.fprintf ppf "%s: %a" what pp_case s.sc

(* An operand of [dt] whose made shape is [s] through the view [v], its
   values finite: on the grid, or small integers. *)
let finite dt s v seed =
  let r = Random.State.make [| seed |] in
  let make made =
    let draw () =
      if is_integer dt then float_of_int (Random.State.int r 8)
      else grid.(Random.State.int r (Array.length grid))
    in
    cast_to dt
      (A.Any (A.of_array D.Float64 (Iarray.of_array made) (Array.init (total made) (fun _ -> draw ()))))
  in
  E.operand_with make (Array.map (fun m -> s.(m)) v.perm) v

(* Segmented cases, Rows ones where [rows] draws true. *)
let seg_cases rows =
  let open Gen in
  let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64; Any Int32; Any Complex64; Any Complex128 ] in
  let dts = if is_integer acc then D.[ Any Int32; Any Int8; Any Uint4 ] else held acc in
  let* da = of_list ~pp:pp_dtype dts and+ db = of_list ~pp:pp_dtype dts in
  let* rows = rows and+ nb = int_range 0 1 in
  let* z = int_range 1 3 and+ m = int_range 1 40 and+ n = int_range 1 20 in
  let* k = frequency [ (3, int_range 0 40); (1, int_range 100 700) ] in
  let* g = int_range 0 5 and+ with_init = bool and+ seed = int in
  let batch = if nb = 1 then [| z |] else [||] in
  let sa = Array.concat [ batch; [| m; k |] ] in
  let sb = Array.concat [ batch; [| k; n |]; (if rows then [| Int.max g 1 |] else [||]) ] in
  let* va = view (Array.length sa) nb and+ vb = view (Array.length sb) nb in
  let a = finite da sa va seed and bx = finite db sb vb (seed + 1) in
  (* Each made axis's place in the arrays the views give. *)
  let ia = E.inverse va.perm and ib = E.inverse vb.perm in
  let pairs = if nb = 1 then [| (ia.(0), ib.(0)) |] else [||] in
  let contracting = [| (ia.(nb + 1), ib.(nb)) |] in
  let r = Random.State.make [| seed + 2 |] in
  let ids extent groups =
    A.of_array D.Int64
      (Iarray.of_array (Array.append batch [| extent |]))
      (Array.init (total batch * extent) (fun _ ->
           Int64.of_int (Random.State.int r (groups + 2) - 1)))
  in
  let segments : _ Nx_kernel.segments =
    if rows then Rows { ids = ids m (Int.max g 1); a_axis = ia.(nb); b_axis = ib.(nb + 2) }
    else Contracting { ids = ids k g; groups = g; pair = 0 }
  in
  let y =
    match
      Nx_kernel.Shape.contract ~batch:(Iarray.of_array pairs)
        ~contracting:(Iarray.of_array contracting)
        ~segments:
          (Some
             (match segments with
             | Rows x -> Nx_kernel.Rows { x with ids = shape_of (A.Any x.ids) }
             | Contracting x -> Contracting { x with ids = shape_of (A.Any x.ids) }))
        (shape_of a, None) (shape_of bx, None)
    with
    | Ok y -> y
    | Error e -> invalid_arg ("seg_case: " ^ e)
  in
  let init =
    if with_init then Some (finite acc y (E.plain (Array.length y)) (seed + 3)) else None
  in
  constant
    {
      sc = { batch = pairs; contracting; acc; a; b = bx; init; out = acc; views = [] };
      segments;
      yshape = y;
    }

let seg_case = Gen.with_pp pp_seg (seg_cases Gen.bool)
let k_seg_case = Gen.with_pp pp_seg (seg_cases (Gen.constant false))

(* [s] on [k], into a fresh destination on [device]; [abs] reads every
   operand's magnitude instead, in float64. *)
let seg_run ?(abs = false) (module K : Nx_kernel.S) device on_ s =
  let c = s.sc in
  let mag x =
    if not abs then x
    else
      let v = to64 (cast_to c.acc x) in
      A.Any (A.of_array D.Float64 (Iarray.of_array (shape_of x)) (Array.map Float.abs (A.to_array v)))
  in
  let acc = if abs then D.Any D.Float64 else c.acc in
  let shape = s.yshape in
  let (D.Any out) = acc in
  let dst = A.create device out (Iarray.of_array shape) in
  let init = Option.map (fun i -> A.expect out (on_ (mag i))) c.init in
  let ids x = A.expect D.Int64 (on_ (A.Any x)) in
  let segments : _ Nx_kernel.segments =
    match s.segments with
    | Rows r -> Rows { r with ids = ids r.ids }
    | Contracting g -> Contracting { g with ids = ids g.ids }
  in
  match
    K.contract ~batch:(Iarray.of_array c.batch)
      ~contracting:(Iarray.of_array c.contracting) ~segments:(Some segments) ~acc
      ~dst ~init
      { array = on_ (mag c.a); scale = None }
      { array = on_ (mag c.b); scale = None }
  with
  | A.Done -> Some (host (A.Any dst))
  | A.Declined -> None
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* A segmented contraction is each group's contraction of its indices
   alone: within twice the largest bound of nx.cpu's result, which keeps
   that bound of the exact sum, measured by nx.cpu's float64 sums of the
   magnitudes; an integer one is nx.cpu's. *)
let segmented (b : Support.backend) s =
  let c = s.sc in
  match seg_run b.kernels b.device (on b) s with
  | None -> fail "declined"
  | Some y ->
      let cpu = Option.get (seg_run (module Nx_cpu) Rig.host Fun.id s) in
      let want = (planes c.acc cpu).(0) and got = (planes c.acc y).(0) in
      if is_integer c.acc then equal (array float_exact) want got
      else begin
        let mag = (planes (D.Any D.Float64) (Option.get (seg_run ~abs:true (module Nx_cpu) Rig.host Fun.id s))).(0) in
        let _, k, _ = sizes c in
        let u = unit_roundoff c.acc and n = float_of_int ((2 * k) + 2) in
        let g = n *. u /. (1. -. (n *. u)) in
        let bad = ref [] in
        Array.iteri
          (fun i w ->
            let bound = (2.0001 *. g *. mag.(i)) +. (smallest_subnormal c.acc *. float_of_int (k + 1)) in
            if Float.abs (got.(i) -. w) > bound && List.length !bad < 8 then
              bad := Printf.sprintf "output %d: %h, nx.cpu %h, past %h" i got.(i) w bound :: !bad)
          want;
        equal (list string) [] (List.rev !bad)
      end

let law_segmented (b : Support.backend) s =
  cover "rows" (match s.segments with Rows _ -> true | _ -> false);
  cover "contracting" (match s.segments with Contracting _ -> true | _ -> false);
  cover "init" (Option.is_some s.sc.init);
  segmented b s

(* The contracted extent past which nx.cuda holds a Contracting call's
   indices in 8 bytes, without Nx_cuda.contract_wide_k: 2^31 - 1, the most
   it takes. *)
let wide_k = (1 lsl 31) - 1

(* nx.cuda's Contracting call past 2^31 - 1 contracted indices holds them
   in 8 bytes: the same law, with 8 bytes from the first index. *)
let law_wide (b : Support.backend) s =
  cover "complex" (is_complex s.sc.acc);
  cover "init" (Option.is_some s.sc.init);
  Fun.protect
    ~finally:(fun () -> Nx_cuda.contract_wide_k wide_k)
    (fun () ->
      Nx_cuda.contract_wide_k 0;
      segmented b s)

(* In place *)

(* A destination identical to [init], a C-contiguous array of the result's
   dtype, gives the bits a fresh one does: a kernel reads each element of
   [init] before it writes the destination there. *)
let law_in_place (b : Support.backend) (c : case) =
  match c.init with
  | None -> cover "no init" true
  | Some i -> (
      let c = { c with init = Some (cast_to c.out i) } in
      let alias = on b (cast_to c.out i) in
      let answer =
        contract b.kernels c ~dst:alias (on b c.a) (on b c.b) (Some alias)
      in
      match (run b c, answer) with
      | Some y, A.Done ->
          cover "computed" true;
          equal (list string) []
            (differ_bits c.acc ~other:"in place" y (host alias))
      | None, A.Declined -> cover "declined" true
      | y, r ->
          failf "a fresh destination %s, in place %a"
            (if Option.is_some y then "computed" else "declined")
            Nx_array_support.pp_answer r)

(* Contractions of 16 × 16 outputs, with init, either side of a block of
   256 products and over several, which a library may sum into scratch
   beside a destination identical to init. *)
let in_place_examples =
  let v = Elements.plain 2 in
  List.concat_map
    (fun acc ->
      List.map
        (fun k ->
          make ~acc ~out:acc ([||], [| 16 |], [| 16 |], [| k |]) (acc, v)
            (acc, v) (Some v) k)
        [ 255; 256; 257; 1025 ])
    D.[ Any Float32; Any Float64; Any Complex64 ]

(* A destination through any view, its axes stored apart, windowed,
   stepped or reversed, made on the backend's device, gets the bits a fresh
   C-contiguous one gets: the association reads no layout. *)
let law_dst_view (b : Support.backend) (c, v) =
  match run b c with
  | None -> fail "declined"
  | Some y -> (
      let ys, _, _ = sizes c in
      let (D.Any out) = c.out in
      let make made = A.Any (A.create b.device out (Iarray.of_array made)) in
      (* The made array's axes in the view's order, as [operand] lays them. *)
      let made = Array.map (fun i -> ys.(i)) (E.inverse v.perm) in
      let dst = E.operand_with make (Array.map (fun m -> made.(m)) v.perm) v in
      let init = Option.map (on b) c.init in
      match contract b.kernels c ~dst (on b c.a) (on b c.b) init with
      | A.Done ->
          equal (list string) []
            (differ_bits c.acc ~other:"through a view" y (host dst))
      | r -> failf "contract answered %a" Nx_array_support.pp_answer r)

let dst_view_case =
  Gen.with_pp
    (fun ppf (c, v) -> Format.fprintf ppf "%a; dst %s" pp_case c (String.concat " " (E.view_names v)))
    (Gen.bind computed (fun c ->
         let ys, _, _ = sizes c in
         (* No windows: a destination's elements are distinct. *)
         Gen.map (fun v -> (c, v)) (E.view_of ~apart:true (Array.length ys))))

(* A kernel that declines writes nothing: [dst] keeps its values. *)
let law_declined (b : Support.backend) c =
  let y, _, _ = sizes c in
  let threes =
    A.Any (A.of_array D.Float64 (Iarray.of_array y) (Array.make (total y) 3.))
  in
  let dst = on b (cast_to c.out threes) in
  let before = planes c.out (host dst) in
  let init = Option.map (on b) c.init in
  match contract b.kernels c ~dst (on b c.a) (on b c.b) init with
  | A.Declined ->
      cover "declined" true;
      equal (array (array float_exact)) before (planes c.out (host dst))
  | A.Done -> cover "computed" true
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* Integers *)

(* A contraction whose operands Shape.contract answers Error for, or whose
   dst's shape is not the result's, does not fit: not Done, and nothing
   written outside dst. Misfits in the pairs' extents, the operands' ranks
   and dst, a result past max_numel, and one behind a group that does not
   merge. *)
let test_misfits (b : Support.backend) () =
  let module K = (val b.kernels) in
  let arr s =
    let ones = Array.make (total s) 1. in
    let x = A.of_array D.Float32 (Iarray.of_array s) ones in
    A.expect D.Float32 (on b (A.Any x))
  in
  let refused ~msg ?(batch : _ iarray = [||]) contracting dst a b' =
    Misfit.check ~msg ~what:"a shape" b [| (D.Any D.Float32, dst) |]
      (fun g ds ->
        K.contract ~batch ~contracting ~segments:None
          ~acc:(D.Any D.Float32) ~dst:(A.expect D.Float32 ds.(0)) ~init:None
          (plain (g.g a)) (plain (g.g b')))
  in
  let mm : _ iarray = [| (1, 0) |] in
  refused ~msg:"contracted extents" mm [| 2; 3 |] (arr [| 2; 4 |])
    (arr [| 5; 3 |]);
  refused ~msg:"dst's shape" mm [| 3; 2 |] (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"dst's rank" mm [| 6 |] (arr [| 2; 4 |]) (arr [| 4; 3 |]);
  refused ~msg:"a's rank" mm [| 2; 3 |] (arr [| 4 |]) (arr [| 4; 3 |]);
  let k = 1 lsl 27 in
  refused ~msg:"a result past max_numel" mm [| 1; 1 |] (arr [| k; 0 |])
    (arr [| 0; k |]);
  (* [a]'s batch axes transposed, its contracted extent 4 against [b]'s 5. *)
  let a =
    let x = arr [| 3; 2; 2; 4 |] in
    Option.get (A.move (Nx_array.Move.Permute [| 1; 0; 2; 3 |]) x)
  in
  refused ~msg:"behind an unmerged group"
    ~batch:[| (0, 0); (1, 1) |]
    [| (3, 2) |] [| 2; 3; 2; 2 |] a
    (arr [| 2; 3; 5; 2 |])

(* An [acc] Nx_kernel.accumulates refuses, a narrow float's or a
   boolean's, does not fit: not Done, and nothing written outside dst. *)
let test_acc_refused (b : Support.backend) () =
  let module K = (val b.kernels) in
  let a = A.create b.device D.Float32 [| 2; 3 |] in
  let x = A.create b.device D.Float32 [| 3; 2 |] in
  List.iter
    (fun acc ->
      Misfit.check
        ~msg:(Format.asprintf "an accumulator of %a" pp_dtype acc)
        ~what:"a dtype" b
        [| (D.Any D.Float32, [| 2; 2 |]) |]
        (fun g ds ->
          K.contract ~batch:[||] ~contracting:[| (1, 0) |] ~segments:None
            ~acc ~dst:(A.expect D.Float32 ds.(0)) ~init:None
            (plain (g.g a)) (plain (g.g x))))
    D.[ Any Float16; Any Bfloat16; Any Bool ]

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop ~examples "a contraction is within its bound of the exact sum"
        computed (run (law_bound b));
      prop "a destination through a view gets a fresh one's bits"
        dst_view_case (run (law_dst_view b));
      prop "a scaled contraction is within its bound of its terms' sum"
        scaled_case (run (law_scaled b));
      prop "a segmented contraction is each group's contraction" seg_case
        (run (law_segmented b));
      prop "a declined contraction writes nothing" any_case
        (run (law_declined b));
      prop ~examples:(examples @ in_place_examples)
        "a destination identical to init gives a fresh one's bits" computed
        (run (law_in_place b));
      test
        "a call of shapes that do not fit is not Done and writes nothing \
         outside dst"
        (fun () -> b.around (test_misfits b));
      test "an acc a contraction does not compute in is not Done" (fun () ->
          b.around (test_acc_refused b));
    ]

(* nx_cuda.mli: an operand whose contracted axes do not form one run of
   strides is copied, here [2k] float32 elements past the 64 MiB the
   device keeps; a sum cut into ranges of k holds each range's sums of a
   tile of outputs, here within 1 MiB. *)
let test_loop_copy (b : Support.backend) () =
  let module K = (val b.kernels) in
  let k = 1 lsl 24 and d = b.device in
  let base = A.create d D.Float32 [| 1; k; 2 |] in
  let a =
    A.v D.Float32 (A.Layout.v ~strides:[| 2 * k; 1; 2 |] [| 1; 2; k |]) (A.buffer base)
  in
  let rhs = A.create d D.Float32 [| 2; k; 1 |] in
  let dst = A.create d D.Float32 [| 1; 1 |] in
  let operand x : Nx_kernel.operand = { array = A.Any x; scale = None } in
  Kernels_cases.loop_held b ~bound:((2 * k * 4) + (1 lsl 20)) (fun () ->
      equal Elements.answer A.Done
        (K.contract ~batch:[||] ~contracting:[| (1, 0); (2, 1) |] ~segments:None
           ~acc:(D.Any D.Float32) ~dst ~init:None (operand a) (operand rhs));
      Rig.Buffer.wait (A.buffer dst) Read_write)

let cuda (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cuda " ^ b.name)
    [
      prop "a Contracting call with 8-byte indices is each group's contraction"
        k_seg_case (run (law_wide b));
      test "contractions in a loop hold one call's copies" (fun () ->
          b.around (test_loop_copy b));
    ]

(* rig.nv *)

let some = function Some y -> [| y |] | None -> [||]

(* Whether [c] sums on the matrix units, as nx_cuda.mli states: operands
   each bfloat16 or float8, or both float16, into float32. *)
let matrix_units (c : case) =
  let narrow (D.Any d) =
    match d with
    | D.Bfloat16 | D.Float8_e4m3fn | D.Float8_e5m2 -> true
    | _ -> false
  in
  let dtype (A.Any x) = D.Any (A.dtype x) in
  let a = dtype c.a and b = dtype c.b in
  c.acc = D.Any D.Float32
  && ((narrow a && narrow b) || (a = D.Any D.Float16 && b = D.Any D.Float16))

(* Contractions of narrow floats into float32, the matrix units' among
   them. *)
let narrow_case =
  Gen.with_pp pp_case
    (case_of ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32)
       ~dts:D.[ Any Bfloat16; Any Float8_e4m3fn; Any Float8_e5m2; Any Float16 ]
       ())

(* rig.nv loads the cubin CUDA loads, so a contraction through it has
   CUDA's bits: what it can get wrong, its launches, memory and order,
   shows in them. Each case runs through CUDA, then through rig.nv. A
   NaN agrees with any NaN: nx_cuda.mli leaves a NaN result's sign and
   payload unspecified. *)
let nv_laws (first : Support.backend) (nv : Support.backend) =
  let agree call = Elements.agree first nv call in
  group nv.name
    [
      group "bits are cuda's, any NaN for a NaN"
        [
          prop ~examples "of float and complex contractions" computed (fun c ->
              cover "complex" (is_complex c.acc);
              cover "init" (Option.is_some c.init);
              agree (fun b -> some (run b c)));
          prop "of narrow floats' contractions into float32" narrow_case
            (fun c ->
              cover "on the matrix units" (matrix_units c);
              agree (fun b -> some (run b c)));
          prop "of contractions of any accumulator and result" any_case
            (fun c ->
              cover "an integer accumulator" (is_integer c.acc);
              agree (fun b -> some (run b c)));
          prop "of scaled contractions" scaled_case (fun (c, sa, sb) ->
              let sa = Option.map (scale_of (shape_of c.a)) sa in
              let sb = Option.map (scale_of (shape_of c.b)) sb in
              agree (fun b -> some (run ?sa ?sb b c)));
          prop "of segmented contractions" seg_case (fun s ->
              cover "rows" (match s.segments with Rows _ -> true | _ -> false);
              cover "contracting"
                (match s.segments with Contracting _ -> true | _ -> false);
              agree (fun b -> some (seg_run b.kernels b.device (on b) s)));
          prop "of Contracting calls with 8-byte indices" k_seg_case (fun s ->
              Fun.protect
                ~finally:(fun () -> Nx_cuda.contract_wide_k wide_k)
                (fun () ->
                  Nx_cuda.contract_wide_k 0;
                  agree (fun b -> some (seg_run b.kernels b.device (on b) s))));
        ];
    ]

(* nx.cpu under each target table, then nx.cuda. *)
let backends = Support.cpus @ Support.gpus ()

let () =
  let nv =
    match (Support.gpus (), Support.nv ()) with
    | first :: _, Some nv -> [ nv_laws first nv ]
    | _ -> []
  in
  exit
    (Windtrap.run "nx_kernel.contract"
       (List.map laws backends @ List.map cuda (Support.gpus ()) @ nv))
