(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Contractions through every kernel library the host runs: within the bound
   Nx_kernel.Spec.contract states of the exact sum, the same bits under every
   layout of the same values, and nothing written where declined. nx.cpu's
   group checks what nx_cpu.mli states: the cases it computes, and its order,
   bit for bit, against a reference built here. *)

open Windtrap
open Nx_array_gen
module A = Nx_array
module D = Nx_array.Dtype
module S = Nx_kernel.Spec
module Support = Nx_kernels_support

let pp_dtype ppf (D.Any dt) = D.pp ppf dt

let pp_any ppf (A.Any x) =
  Format.fprintf ppf "%a %a" D.pp (A.dtype x) L.pp (A.layout x)

let shape_of (A.Any x) = L.shape (A.layout x)
let total s = Array.fold_left ( * ) 1 s

(* [x] cast into [d], by nx.cpu. *)
let cast_into d (A.Any x) =
  match Nx_cpu.apply1 Nx_kernel.Prog.Cast ~dst:d x with
  | A.Done -> ()
  | r -> failf "a cast answered %a" Nx_array_support.pp_answer r

(* [x] cast to [dt], C-contiguous. *)
let cast_to (D.Any dt) x =
  let d = A.create Rig.host dt (shape_of x) in
  cast_into d x;
  A.Any d

(* [x] as float64: every value of a dtype a contraction takes is one of
   float64's. *)
let to64 x =
  let d = A.create Rig.host D.Float64 (shape_of x) in
  cast_into d x;
  d

(* Values *)

(* Values every float format holds, float4 included. *)
let grid = [| 0.; -0.; 0.5; -0.5; 1.; -1.; 1.5; -1.5; 2.; -3.; 4.; -6. |]

(* A value of [dt] from [r]: on the grid, an integer in the range of every
   integer dtype of its kind, or, for float32 and float64, one of 24 or 53
   significant bits in (-4, 4), an infinity or a NaN. *)
let value (D.Any dt) r =
  let pick a = a.(Random.State.int r (Array.length a)) in
  let full p =
    let m = Random.State.int64 r (Int64.shift_left 1L (p + 1)) in
    Float.ldexp (Int64.to_float (Int64.sub m (Int64.shift_left 1L p))) (2 - p)
  in
  let float p =
    match Random.State.int r 40 with
    | 0 -> pick [| infinity; neg_infinity; nan |]
    | n when n < 15 -> pick grid
    | _ -> full p
  in
  match (D.kind dt, dt) with
  | D.Boolean, _ -> float_of_int (Random.State.int r 2)
  | D.Unsigned, _ -> float_of_int (Random.State.int r 16)
  | D.Signed, _ -> float_of_int (Random.State.int r 16 - 8)
  | _, D.Float32 -> float 23
  | _, D.Float64 -> float 52
  | _ -> pick grid

(* Cases *)

(* A contraction: its spec, operands and result dtype, and the views its
   operands were drawn through. *)
type case = {
  spec : S.contract S.t;
  a : A.any;
  b : A.any;
  init : A.any option;
  out : D.any;
  views : string list;
}

let pp_case ppf c =
  let pp_init ppf = function
    | None -> ()
    | Some i -> Format.fprintf ppf "; init %a" pp_any i
  in
  Format.fprintf ppf "batch %a contracting %a acc %a out %a; a %a; b %a%a"
    pp_ints
    (Array.map fst (S.batch c.spec))
    pp_ints
    (Array.map fst (S.contracting c.spec))
    pp_dtype (S.acc c.spec) pp_dtype c.out pp_any c.a pp_any c.b pp_init c.init

let inverse p =
  let q = Array.make (Array.length p) 0 in
  Array.iteri (fun i m -> q.(m) <- i) p;
  q

(* How an operand of rank [r] is viewed: made with extent 1 along its [nb]
   batch axes and broadcast, made with its last axis twice as long and
   stepped by two, then permuted, [perm.(i)] being the made axis of axis
   [i]. *)
type view = { stepped : bool; broadcast : bool; perm : int array }

let view r nb =
  let open Gen in
  let* broadcast = if nb > 0 then bool else constant false in
  let* stepped = bool in
  let stepped = stepped && r > 0 && (r > nb || not broadcast) in
  let+ p = permutation ~pp:Format.pp_print_int (List.init r Fun.id) in
  { stepped; broadcast; perm = Array.of_list p }

let names v =
  (if v.stepped then [ "stepped" ] else [])
  @ (if v.broadcast then [ "broadcast" ] else [])
  @
  if v.perm <> Array.init (Array.length v.perm) Fun.id then [ "permuted" ]
  else []

(* An array of [dt] of shape [s] through the view [v], its values drawn from
   [seed]. *)
let operand dt s nb v seed =
  let r = Random.State.make [| seed |] in
  let last = Array.length s - 1 in
  let made =
    Array.mapi
      (fun i e ->
        if v.broadcast && i < nb then 1
        else if v.stepped && i = last then 2 * e
        else e)
      s
  in
  let xs = Array.init (total made) (fun _ -> value dt r) in
  let (A.Any x) = cast_to dt (A.Any (A.of_array D.Float64 made xs)) in
  let move m x = Option.get (A.move m x) in
  let step i e =
    if i = last then { M.start = 1; count = s.(i); step = 2 }
    else { M.start = 0; count = e; step = 1 }
  in
  let x = if v.stepped then move (M.Slice (Array.mapi step made)) x else x in
  let x = if v.broadcast then move (M.Broadcast s) x else x in
  A.Any (move (M.Permute v.perm) x)

(* Extents of the batch, row, column and contracted axes, in profiles: any
   ranks over a few elements; one output; 64 outputs or more with edge
   tiles; several blocks of the contraction on several threads; dots past a
   block of lanes; several panels of columns; few rows over several blocks
   of the contraction; few columns; several groups of batch elements. *)
let extents =
  let open Gen in
  let one lo hi = map (fun e -> [| e |]) (int_range lo hi) in
  let shaped batch rows cols con =
    let+ batch = batch and+ rows = rows and+ cols = cols and+ con = con in
    (batch, rows, cols, con)
  in
  let axes most lo hi = array ~size:(int_range 0 most) (int_range lo hi) in
  let ranked = shaped (axes 1 1 3) (axes 2 0 5) (axes 2 0 5) (axes 2 0 4) in
  let none = constant [||] in
  frequency
    [
      (6, ranked);
      (1, shaped (axes 1 1 1) (axes 2 1 1) (axes 2 1 1) (axes 2 0 6));
      (3, shaped none (one 4 40) (one 8 40) (one 0 40));
      (1, shaped none (one 4 20) (one 8 24) (one 500 1300));
      (1, shaped (one 1 3) (one 1 2) (one 1 2) (one 1000 5000));
      (1, shaped none (one 1 2) (one 3070 3200) (one 1 6));
      (1, shaped none (one 1 4) (one 16 200) (one 400 1100));
      (1, shaped none (one 16 300) (one 1 4) (one 1 600));
      (1, shaped (one 340 420) (constant [| 8 |]) (constant [| 8 |]) (one 1 4));
    ]

(* A case in [acc] into [out], each operand of [acc] or of [dts]. *)
let case_of ~acc ~out ~dts =
  let open Gen in
  let* batch, rows, cols, con = extents in
  let nb = Array.length batch and fa = Array.length rows in
  let pick = frequency [ (3, constant ~pp:pp_dtype acc); (2, of_list ~pp:pp_dtype dts) ] in
  let sa = Array.concat [ batch; rows; con ] in
  let sb = Array.concat [ batch; con; cols ] in
  let* da = pick in
  let* db = pick in
  let* di = pick in
  let* va = view (Array.length sa) nb in
  let* vb = view (Array.length sb) nb in
  let* with_init = bool in
  let* seed = int in
  let ia = inverse va.perm and ib = inverse vb.perm in
  let spec init =
    S.contract
      ~batch:(Array.init nb (fun k -> (ia.(k), ib.(k))))
      ~contracting:
        (Array.init (Array.length con) (fun k ->
             (ia.(nb + fa + k), ib.(nb + k))))
      ~acc ~out ~init
  in
  let a = operand da sa nb va seed and b = operand db sb nb vb (seed + 1) in
  let y =
    match S.shapes (spec false) [| shape_of a; shape_of b |] with
    | Ok [| y |] -> y
    | Ok _ -> invalid_arg "case_of: results"
    | Error e -> invalid_arg ("case_of: " ^ e)
  in
  let+ vi = view (Array.length y) 0 in
  let init =
    if with_init then
      Some (operand di (Array.map (fun i -> y.(i)) (inverse vi.perm)) 0 vi (seed + 2))
    else None
  in
  let views = names va @ names vb @ if with_init then names vi else [] in
  { spec = spec with_init; a; b; init; out; views }

(* The dtypes each accumulator holds, as nx_cpu.mli lists them. *)
let held = function
  | D.Any D.Float32 ->
      D.
        [
          Any Float32;
          Any Float16;
          Any Bfloat16;
          Any Float8_e4m3fn;
          Any Float8_e5m2;
          Any Float4_e2m1fn;
          Any Int8;
          Any Uint8;
          Any Int16;
          Any Uint16;
          Any Int4;
          Any Uint4;
          Any Bool;
          Any Bit;
        ]
  | _ ->
      D.
        [
          Any Float64;
          Any Float32;
          Any Float16;
          Any Bfloat16;
          Any Int32;
          Any Uint32;
          Any Int8;
          Any Bool;
        ]

let accs = D.[ Any Float32; Any Float64 ]

let computed =
  Gen.with_pp pp_case
    (Gen.bind (Gen.of_list ~pp:pp_dtype accs) (fun acc ->
         case_of ~acc ~out:acc ~dts:(held acc)))

(* Cases nx.cpu may decline: accumulators it lacks, other results, operands an
   accumulator does not hold. *)
let any_case =
  let open Gen in
  let dts = D.[ Any Float64; Any Float32; Any Int32; Any Int64; Any Bfloat16; Any Complex64 ] in
  Gen.with_pp pp_case
    (let* acc = of_list ~pp:pp_dtype D.[ Any Float32; Any Float64; Any Int32; Any Complex64 ] in
     let* out = frequency [ (2, constant ~pp:pp_dtype acc); (1, of_list ~pp:pp_dtype D.all) ] in
     case_of ~acc ~out ~dts)

(* The reference *)

(* An output's init and the factors of its products, in increasing order of
   the contracted index: the contracting pairs' indices in C order. *)
type output = { init : float option; av : float array; bv : float array }

(* The result's shape and its outputs in C order. *)
let outputs c =
  let a = to64 c.a and b = to64 c.b in
  let init = Option.map to64 c.init in
  let la = A.layout a and lb = A.layout b in
  let bp = S.batch c.spec and cp = S.contracting c.spec in
  let free side r =
    List.filter
      (fun ax ->
        not
          (Array.exists (fun p -> side p = ax) bp
          || Array.exists (fun p -> side p = ax) cp))
      (List.init r Fun.id)
    |> Array.of_list
  in
  let fa = free fst (L.rank la) and fb = free snd (L.rank lb) in
  let nb = Array.length bp and nfa = Array.length fa in
  let y =
    Array.concat
      [
        Array.map (fun (i, _) -> L.dim la i) bp;
        Array.map (L.dim la) fa;
        Array.map (L.dim lb) fb;
      ]
  in
  let ks = indices (Array.map (fun (i, _) -> L.dim la i) cp) in
  let each yi =
    let ai = Array.make (L.rank la) 0 and bi = Array.make (L.rank lb) 0 in
    Array.iteri
      (fun p (i, j) ->
        ai.(i) <- yi.(p);
        bi.(j) <- yi.(p))
      bp;
    Array.iteri (fun q ax -> ai.(ax) <- yi.(nb + q)) fa;
    Array.iteri (fun q ax -> bi.(ax) <- yi.(nb + nfa + q)) fb;
    let terms =
      List.map
        (fun ki ->
          Array.iteri
            (fun q (i, j) ->
              ai.(i) <- ki.(q);
              bi.(j) <- ki.(q))
            cp;
          (A.get a ai, A.get b bi))
        ks
    in
    {
      init = Option.map (fun i -> A.get i yi) init;
      av = Array.of_list (List.map fst terms);
      bv = Array.of_list (List.map snd terms);
    }
  in
  (y, List.map each (indices y))

(* [x + y] as a double and its error, exactly (Knuth's TwoSum). *)
let two_sum x y =
  let s = x +. y in
  let z = s -. x in
  (s, x -. (s -. z) +. (y -. z))

(* The sum of [o]'s products and init, with an error under 2^-100 of the sum
   of their magnitudes, and that sum. Products and the running sum are held
   in two parts. *)
let exact o =
  let start = Option.value ~default:0. o.init in
  let hi = ref start and lo = ref 0. and mag = ref (Float.abs start) in
  Array.iteri
    (fun k x ->
      let y = o.bv.(k) in
      let p = x *. y in
      let s, t = two_sum !hi p in
      hi := s;
      lo := !lo +. t +. Float.fma x y (-.p);
      mag := !mag +. Float.abs p)
    o.av;
  (!hi +. !lo, !mag)

let unit_roundoff (D.Any dt) = if D.bits dt = 32 then 0x1p-24 else 0x1p-53

(* The extents of [c]'s result, its contracted index and its batch. *)
let sizes c =
  let sa = shape_of c.a in
  let y =
    let ops = [ sa; shape_of c.b ] @ Option.to_list (Option.map shape_of c.init) in
    match S.shapes c.spec (Array.of_list ops) with
    | Ok [| y |] -> y
    | Ok _ -> failf "the case's spec has several results"
    | Error e -> failf "the case's shapes do not fit its spec: %s" e
  in
  let along pairs = total (Array.map (fun (i, _) -> sa.(i)) pairs) in
  (y, along (S.contracting c.spec), along (S.batch c.spec))

let operands c = Array.of_list ([ c.a; c.b ] @ Option.to_list c.init)

(* Whether nx_cpu.mli says nx.cpu computes [c]: its dtypes, and a layout
   Contract_view groups. *)
let cpu_computes c =
  let acc = S.acc c.spec in
  let y, _, _ = sizes c in
  let (D.Any out) = c.out in
  let dst = A.Any (A.create Rig.host out y) in
  List.mem acc accs && c.out = acc
  && Array.for_all
       (fun (A.Any x) -> List.mem (D.Any (A.dtype x)) (held acc))
       (operands c)
  && S.Contract_view.fill (S.Contract_view.make ()) c.spec ~dst (operands c)

(* Outputs per batch element: what picks nx.cpu's order. *)
let per_batch c =
  let y, _, batch = sizes c in
  if batch = 0 then 0 else total y / batch

let covers c =
  let y, k, batch = sizes c in
  let per = per_batch c in
  cover "no output" (total y = 0);
  cover "one output" (total y = 1);
  cover "no product" (k = 0 && total y > 0);
  cover "fewer than 64 outputs" (per > 0 && per < 64);
  cover "64 outputs or more" (per >= 64);
  cover "init" (c.init <> None);
  cover "several blocks of the contraction" (per >= 64 && k > 512);
  cover "several blocks of lanes" (per > 0 && per < 64 && k > 1024);
  cover "several panels" (per >= 64 && Array.exists (fun e -> e > 3072) y);
  cover "several groups" (per >= 64 && batch > 341);
  let nb = Array.length (S.batch c.spec) in
  let fa =
    Array.length (shape_of c.a) - nb - Array.length (S.contracting c.spec)
  in
  let m = total (Array.sub y nb fa) in
  let n = total (Array.sub y (nb + fa) (Array.length y - nb - fa)) in
  cover "4 rows or fewer" (per >= 64 && m <= 4);
  cover "4 columns or fewer" (per >= 64 && n <= 4);
  List.iter
    (fun v -> cover v (List.mem v c.views))
    [ "stepped"; "broadcast"; "permuted" ]

(* The kernels' result; [None] if they declined. *)
let run (b : Support.backend) c =
  let module K = (val b.kernels) in
  let y, _, _ = sizes c in
  let (D.Any out) = c.out in
  let dst = A.create Rig.host out y in
  let ops = Array.of_list ([ c.a; c.b ] @ Option.to_list c.init) in
  match K.contract c.spec ~dst:(A.Any dst) ops with
  | A.Done -> Some (A.Any dst)
  | A.Declined -> None
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* A float's bits, every NaN's as one: a NaN result is some NaN. *)
let bits x =
  if Float.is_nan x then Int64.bits_of_float nan else Int64.bits_of_float x

(* The first outputs of [y] at which [check] answers why it is wrong. *)
let wrong c y check =
  let shape, outs = outputs c in
  let g = to64 y in
  let bad = ref [] in
  List.iter2
    (fun yi o ->
      match check o (A.get g yi) with
      | Some why when List.length !bad < 8 ->
          bad := Format.asprintf "%a: %s" pp_ints yi why :: !bad
      | _ -> ())
    (indices shape) outs;
  List.rev !bad

(* The laws of S *)

(* An output [x] of [c] is within γ(K + 1, 2u) (|init| + Σ|a·b|) of the exact
   sum of [o], u being [acc]'s unit roundoff; a NaN or infinite sum is
   matched. *)
let within c o x =
  let naive =
    Array.fold_left ( +. )
      (Option.value ~default:0. o.init)
      (Array.mapi (fun k a -> a *. o.bv.(k)) o.av)
  in
  let show = Printf.sprintf "%h" in
  if not (Float.is_finite naive) then
    if bits x = bits naive then None else Some (show x ^ ", not " ^ show naive)
  else
    let want, mag = exact o in
    let u = unit_roundoff (S.acc c.spec) in
    let n = float_of_int (Array.length o.av + 1) in
    let g = n *. 2. *. u /. (1. -. (n *. 2. *. u)) in
    let bound = (g *. mag) +. (0x1p-100 *. mag) in
    if Float.abs (x -. want) <= bound then None
    else
      Some
        (Printf.sprintf "%s is %h from %s, past %h" (show x) (x -. want)
           (show want) bound)

let law_bound b c =
  covers c;
  match run b c with
  | None -> cover "declined" true
  | Some y ->
      cover "computed" true;
      equal (list string) [] (wrong c y (within c))

(* Each operand as a C-contiguous copy of its view gives the same bits,
   where both layouts are computed. *)
let law_layouts b c =
  let plain (A.Any x) = A.Any (A.copy x) in
  let c' =
    { c with a = plain c.a; b = plain c.b; init = Option.map plain c.init }
  in
  match (run b c, run b c') with
  | Some y, Some y' ->
      cover "computed" true;
      cover "computed from views" (c.views <> []);
      let g = to64 y and g' = to64 y' in
      let differs i = bits (A.get g i) <> bits (A.get g' i) in
      let bad = List.filter differs (indices (shape_of y)) in
      let show i =
        Format.asprintf "%a: %h, contiguous %h" pp_ints i (A.get g i)
          (A.get g' i)
      in
      equal (list string) [] (List.filteri (fun n _ -> n < 8) (List.map show bad))
  | _ -> cover "declined" true

(* A kernel that declines writes nothing: [dst] keeps its values. *)
let law_declined (b : Support.backend) c =
  let module K = (val b.kernels) in
  let y, _, _ = sizes c in
  let threes = A.Any (A.of_array D.Float64 y (Array.make (total y) 3.)) in
  let dst = cast_to c.out threes in
  let before = A.to_array (to64 dst) in
  let ops = Array.of_list ([ c.a; c.b ] @ Option.to_list c.init) in
  match K.contract c.spec ~dst ops with
  | A.Declined ->
      cover "declined" true;
      equal (array float_exact) before (A.to_array (to64 dst))
  | A.Done -> cover "computed" true
  | r -> failf "contract answered %a" Nx_array_support.pp_answer r

(* nx.cpu's order *)

(* [x] rounded to float32. *)
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* float32's fused multiply-add: a · b, exact in a double for float32 [a] and
   [b], plus [c], rounded to odd in a double and then to float32, which
   rounds once since 53 >= 2·24 + 2. *)
let fma32 a b c =
  let p = a *. b in
  let s, e = two_sum c p in
  if e = 0. || not (Float.is_finite s) then round32 s
  else if Int64.logand (Int64.bits_of_float s) 1L = 1L then round32 s
  else round32 (if e > 0. then Float.succ s else Float.pred s)

(* float32's addition, rounded once through a double for the same reason. *)
let add32 x y = round32 (x +. y)

(* [o]'s result in nx.cpu's order, with [fma] and [add] of the accumulator:
   one fused chain from init or +0, or, with fewer than 64 outputs per batch
   element, blocks of 1024 terms in 16 lanes, summed by trees, then init. *)
let ordered ~chain ~fma ~add o =
  let k = Array.length o.av in
  if chain then begin
    let acc = ref (Option.value ~default:0. o.init) in
    for t = 0 to k - 1 do
      acc := fma o.av.(t) o.bv.(t) !acc
    done;
    !acc
  end
  else begin
    let block x =
      let l = Array.make 16 0. in
      for t = x * 1024 to min k ((x + 1) * 1024) - 1 do
        l.(t mod 16) <- fma o.av.(t) o.bv.(t) l.(t mod 16)
      done;
      List.iter
        (fun w ->
          for i = 0 to w - 1 do
            l.(i) <- add l.(i) l.(i + w)
          done)
        [ 8; 4; 2; 1 ];
      l.(0)
    in
    let rec tree s n =
      if n = 1 then s.(0)
      else
        let h = ref 1 in
        while 2 * !h < n do
          h := 2 * !h
        done;
        add (tree s !h) (tree (Array.sub s !h (n - !h)) (n - !h))
    in
    let n = (k + 1023) / 1024 in
    let sum = if n = 0 then 0. else tree (Array.init n block) n in
    match o.init with None -> sum | Some i -> add i sum
  end

let law_order b c =
  covers c;
  match run b c with
  | None when cpu_computes c ->
      failf "nx.cpu declined a case nx_cpu.mli says it computes"
  | None -> cover "a layout the view does not group" true
  | Some y ->
      let chain = per_batch c >= 64 in
      let fma, add =
        if S.acc c.spec = D.Any D.Float32 then (fma32, add32)
        else (Float.fma, ( +. ))
      in
      let want = ordered ~chain ~fma ~add in
      equal (list string) []
        (wrong c y (fun o x ->
             let w = want o in
             if bits x = bits w then None
             else Some (Printf.sprintf "%h, expected %h" x w)))

let law_computes b c =
  let computes = cpu_computes c in
  cover "computed" computes;
  cover "declined" (not computes);
  equal ~msg:"computed" bool computes (Option.is_some (run b c))

(* The suite *)

let laws (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group b.name
    [
      prop "a contraction is within its bound of the exact sum"
        computed (run (law_bound b));
      prop "a contraction's bits do not depend on layouts" computed
        (run (law_layouts b));
      prop "a declined contraction writes nothing" any_case
        (run (law_declined b));
    ]

(* nx.cpu under each table the host runs. *)
let cpu (b : Support.backend) =
  let run f x = b.around (fun () -> f x) in
  group ("nx.cpu " ^ b.name)
    [
      prop "each output adds its products in nx.cpu's order"
        computed (run (law_order b));
      prop "computes the cases nx_cpu.mli lists, declines others"
        any_case (run (law_computes b));
    ]

let () =
  exit
    (Windtrap.run "nx_kernel.contract"
       (List.map laws Support.backends @ List.map cpu Support.backends))
