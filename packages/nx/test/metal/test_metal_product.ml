(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The matrix routines' products (contract.mli), on the machine's Metal GPU 0,
   skipping without one: their outputs are their sums, of the bits of the
   association contract.mli states, negated on every tile, whole steps and edge
   tiles included, with subnormal operands read as zeros of their sign, and at a
   budget that chunks k. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module S = Nx_metal_support
module Support = Nx_kernels_support
module C = Nx_metal.Contract

let answer =
  Testable.make ~pp:Nx_array_support.pp_answer ~equal:(fun a b -> a = b)

let strf = Printf.sprintf

(* Values *)

(* Values whose products, and sums of them over the extents drawn below, are
   exact in float32, so that every association and every count of ranges gives
   the one exact sum. *)
let grid = [| 0.; 0.5; -0.5; 1.; 1.5; 2.; -3.; 4.; -6. |]

type acc = F32 | C64

let accs = [ F32; C64 ]
let acc_name = function F32 -> "float32" | C64 -> "complex64"
let acc_dtype = function F32 -> D.Any D.Float32 | C64 -> D.Any D.Complex64
let is_complex = function C64 -> true | F32 -> false
let width = function F32 -> 4 | C64 -> 8

let encode acc (z : Complex.t) =
  let b = Bytes.create (width acc) in
  let f32 at v = Bytes.set_int32_le b at (Int32.bits_of_float v) in
  f32 0 z.re;
  if is_complex acc then f32 4 z.im;
  Bytes.to_string b

let decode acc s : Complex.t =
  let f32 at = Int32.float_of_bits (String.get_int32_le s at) in
  match acc with
  | F32 -> { re = f32 0; im = 0. }
  | C64 -> { re = f32 0; im = f32 4 }

(* [x] rounded into float32. *)
let round x = Int32.float_of_bits (Int32.bits_of_float x)

(* [x] as the GPU reads it: a subnormal float32 is a zero of its sign. *)
let flush x =
  if Float.abs x < Float.ldexp 1. (-126) then Float.copy_sign 0. x else x

let flush_c (z : Complex.t) = { Complex.re = flush z.re; im = flush z.im }

(* How a matrix [batch; rows; cols] lies: its rows contiguous, its columns
   contiguous, its rows reversed and every other column of a store twice as
   wide, or one matrix broadcast along the batch. *)
type lay = Rows | Columns | Strided | Broadcast_batch

let lays = [ Rows; Columns; Strided; Broadcast_batch ]

let lay_name = function
  | Rows -> "rows"
  | Columns -> "columns"
  | Strided -> "strided"
  | Broadcast_batch -> "broadcast batch"

(* A matrix of [acc] and shape [batch; r; c] lying as [lay] on the GPU, its
   elements' parts drawn by [draw], from [grid] by default, real where [acc]
   is. *)
let matrix ?draw st acc lay (batch, r, c) =
  let (D.Any dt) = acc_dtype acc in
  let store =
    match lay with
    | Rows -> [| batch; r; c |]
    | Columns -> [| batch; c; r |]
    | Strided -> [| batch; r; 2 * c |]
    | Broadcast_batch -> [| 1; r; c |]
  in
  let x = A.create (S.device ()) dt (Iarray.of_array store) in
  let draw =
    match draw with
    | Some d -> d
    | None -> fun () -> grid.(Random.State.int st (Array.length grid))
  in
  let n = Array.fold_left ( * ) 1 store in
  let bytes =
    String.concat ""
      (List.init n (fun _ ->
           let im = if is_complex acc then draw () else 0. in
           encode acc { re = draw (); im }))
  in
  Rig.Buffer.blit_from_string bytes 0 (A.buffer x) 0 (String.length bytes);
  let moved m = A.Any (Option.get (A.move m x)) in
  match lay with
  | Rows -> A.Any x
  | Columns -> moved (M.Permute (Iarray.of_array [| 0; 2; 1 |]))
  | Strided ->
      moved
        (M.Slice
           (Iarray.of_array
              [|
                { M.start = 0; count = batch; step = 1 };
                { M.start = Int.max 0 (r - 1); count = r; step = -1 };
                { M.start = 0; count = c; step = 2 };
              |]))
  | Broadcast_batch -> moved (M.Broadcast (Iarray.of_array [| batch; r; c |]))

let values acc x = Array.map (decode acc) (Support.elements x)

let tri_name = function
  | None -> "none"
  | Some `Lower -> "lower"
  | Some `Upper -> "upper"

(* Whether output [i, j] lies in [tri]'s triangle, the diagonal included. *)
let in_triangle tri i j =
  match tri with None -> true | Some `Lower -> j <= i | Some `Upper -> j >= i

let same (u : Complex.t) (v : Complex.t) =
  Int64.bits_of_float u.re = Int64.bits_of_float v.re
  && Int64.bits_of_float u.im = Int64.bits_of_float v.im

(* Products *)

type product = {
  acc : acc;
  batch : int;
  rows : int;
  cols : int;
  k : int;
  lay_a : lay;
  lay_m : lay;
  lay_y : lay;
  init : bool;
  init_first : bool;
  neg : bool;
  conj_a : bool;
  conj_b : bool;
  tri : [ `Lower | `Upper ] option;
  ranges : int option;
  workspace : bool;
  pseed : int;
}

let pp_product ppf c =
  let flag name b = if b then " " ^ name else "" in
  Format.fprintf ppf
    "%s [%d; %d; %d] k %d, a %s, m %s, y %s,%s%s%s%s%s tri %s, ranges %s%s, \
     seed %d"
    (acc_name c.acc) c.batch c.rows c.cols c.k (lay_name c.lay_a)
    (lay_name c.lay_m) (lay_name c.lay_y) (flag "init" c.init)
    (flag "init_first" c.init_first)
    (flag "neg" c.neg) (flag "conj_a" c.conj_a) (flag "conj_b" c.conj_b)
    (tri_name c.tri)
    (match c.ranges with None -> "model's" | Some s -> string_of_int s)
    (flag ", workspace" c.workspace)
    c.pseed

(* Extents of no element, of one, of a few, within a tile of 64 and past it; k
   of no term, of a few spans of 32 and of several. Half the cases ask no
   option, a contraction's product. *)
let product =
  let open Gen in
  let extent =
    frequency
      [
        (1, int_range 0 1);
        (1, int_range 2 8);
        (3, int_range 9 40);
        (2, int_range 65 160);
      ]
  in
  let k =
    frequency
      [
        (1, constant 0);
        (1, constant 1);
        (3, int_range 2 64);
        (2, int_range 65 400);
      ]
  in
  with_pp pp_product
    (let+ acc = of_list accs
     and+ batch = frequency [ (1, constant 0); (6, int_range 1 3) ]
     and+ rows = extent
     and+ cols = frequency [ (1, constant 1); (4, extent) ]
     and+ k
     and+ lay_a = of_list lays
     and+ lay_m = of_list lays
     and+ lay_y = of_list [ Rows; Columns; Strided ]
     and+ init = bool
     and+ init_first = bool
     and+ neg = bool
     and+ conj_a = bool
     and+ conj_b = bool
     and+ tri = of_list [ None; Some `Lower; Some `Upper ]
     and+ ranges =
       frequency [ (1, constant None); (2, map Option.some (int_range 1 8)) ]
     and+ workspace = bool
     and+ routine = bool
     and+ pseed = nat in
     let asks b = routine && b in
     let complex = is_complex acc in
     {
       acc;
       batch;
       rows;
       cols;
       k;
       lay_a;
       lay_m;
       lay_y;
       init;
       init_first = asks (init && init_first);
       neg = asks neg;
       conj_a = asks (complex && conj_a);
       conj_b = asks (complex && conj_b);
       tri = (if routine then tri else None);
       ranges = (if routine then ranges else None);
       workspace;
       pseed;
     })

(* A product of no term into y, and one that sums y first, so that each run
   reaches them: drawn, a case has no term one time in nine and sums y first one
   time in eight. *)
let product_examples =
  let plain =
    {
      acc = F32;
      batch = 2;
      rows = 20;
      cols = 30;
      k = 0;
      lay_a = Rows;
      lay_m = Rows;
      lay_y = Rows;
      init = true;
      init_first = false;
      neg = false;
      conj_a = false;
      conj_b = false;
      tri = None;
      ranges = None;
      workspace = false;
      pseed = 1;
    }
  in
  [
    plain; { plain with k = 100; init_first = true; ranges = Some 3; pseed = 2 };
  ]

(* [y <- [y +] op(a) op(m)] computed on the host, exact on [grid]'s values. *)
let reference c ~y ~a ~m =
  let op ~conj z = if conj then Complex.conj z else z in
  let at b i j ~rows ~cols = (((b * rows) + i) * cols) + j in
  Array.init (Array.length y) (fun e ->
      let j = e mod c.cols and i = e / c.cols mod c.rows in
      let b = e / (c.cols * c.rows) in
      let s = ref (if c.init then y.(e) else Complex.zero) in
      for u = 0 to c.k - 1 do
        let x = op ~conj:c.conj_a a.(at b i u ~rows:c.rows ~cols:c.k) in
        let x = if c.neg then Complex.neg x else x in
        let w = op ~conj:c.conj_b m.(at b u j ~rows:c.k ~cols:c.cols) in
        s := Complex.add !s (Complex.mul x w)
      done;
      !s)

let run ?(init_first = false) ?(neg = false) ?(conj_a = false) ?(conj_b = false)
    ?tri ?ranges ?(workspace = false) ~acc ~init ~y a m =
  Nx_metal.product ~init_first ~neg ~conj_a ~conj_b ?tri ?ranges ~workspace
    ~acc:(acc_dtype acc) ~init ~y a m

let law_product c =
  let st = Random.State.make [| c.pseed |] in
  let a = matrix st c.acc c.lay_a (c.batch, c.rows, c.k)
  and m = matrix st c.acc c.lay_m (c.batch, c.k, c.cols)
  and y = matrix st c.acc c.lay_y (c.batch, c.rows, c.cols) in
  let was = values c.acc y in
  let want = reference c ~y:was ~a:(values c.acc a) ~m:(values c.acc m) in
  cover "complex" (is_complex c.acc);
  cover "no output" (c.batch * c.rows * c.cols = 0);
  cover "no term" (c.k = 0 && c.batch * c.rows * c.cols > 0);
  cover "one column" (c.cols = 1);
  cover "past one tile of 64" (c.rows > 64 || c.cols > 64);
  cover "several ranges" (c.k > 32 && Option.value c.ranges ~default:1 > 1);
  cover "init_first" c.init_first;
  cover "neg" c.neg;
  cover "conjugated" (c.conj_a || c.conj_b);
  cover "a triangle past one tile of 64"
    (c.tri <> None && (c.rows > 64 || c.cols > 64));
  cover "workspace" c.workspace;
  List.iter (fun l -> cover ("a by " ^ lay_name l) (c.lay_a = l)) lays;
  List.iter (fun l -> cover ("m by " ^ lay_name l) (c.lay_m = l)) lays;
  equal answer A.Done
    (run ~init_first:c.init_first ~neg:c.neg ~conj_a:c.conj_a ~conj_b:c.conj_b
       ?tri:c.tri ?ranges:c.ranges ~workspace:c.workspace ~acc:c.acc
       ~init:c.init ~y a m);
  let now = values c.acc y in
  (* Outside the triangle an output is as it was. Values compare as numbers: a
     zero's sign follows the association, which law_bits states. *)
  let wrong = ref [] in
  Array.iteri
    (fun e got ->
      let i = e / c.cols mod c.rows and j = e mod c.cols in
      let ok =
        if in_triangle c.tri i j then got = want.(e) else got = was.(e)
      in
      if (not ok) && List.length !wrong < 8 then
        wrong :=
          strf "output %d (%d, %d): %g%+gi, want %g%+gi" e i j got.re got.im
            want.(e).re want.(e).im
          :: !wrong)
    now;
  equal (list string) [] (List.rev !wrong)

(* Bits *)

(* A part of a value whose products float64 holds exactly: 6 significant bits
   within 2^-7 and 2^6, whose sums of a few hundred terms float64 holds too, so
   that float32's roundings alone decide the bits. A zero, of either sign, with
   probability [zeros]; a subnormal, of either sign, with probability
   [subnormals]. *)
let part st ~zeros ~subnormals () =
  let u = Random.State.float st 1. in
  let sign x = if Random.State.bool st then -.x else x in
  if u < zeros then sign 0.
  else if u < zeros +. subnormals then
    sign (Float.ldexp (Float.of_int (1 + Random.State.int st 0x7fffff)) (-149))
  else
    let m = 32 + Random.State.int st 32 in
    sign (Float.ldexp (float_of_int m) (Random.State.int st 13 - 6 - 6))

(* An output part's sum as contract.mli states it, of the terms [ts] in order,
   each a pair of factors read as the GPU reads them: in [S = min s ⌈k' / 32⌉]
   ranges of spans of 32 terms, each summed by fused multiply-adds from +0, the
   first from [init] where [init_first]; the ranges' sums added in increasing k,
   then [init] where not [init_first]. Without terms, [init] as it is, or +0. A
   float32 fused multiply-add is exact in float64 on {!part}'s values, then
   rounded once. *)
let sum_model ~ranges ~init_first ~init ts =
  let k' = Array.length ts in
  if k' = 0 then Option.value init ~default:0.
  else
    let init = Option.map flush init in
    let fma x w c = round ((flush x *. flush w) +. c) in
    let r = (k' + 31) / 32 in
    let n = Int.max 1 (Int.min ranges r) in
    let range i =
      let lo = 32 * (i * r / n) and hi = Int.min k' (32 * ((i + 1) * r / n)) in
      let from =
        match init with Some y when i = 0 && init_first -> y | _ -> 0.
      in
      let acc = ref from in
      for t = lo to hi - 1 do
        let x, w = ts.(t) in
        acc := fma x w !acc
      done;
      !acc
    in
    let total = ref (range 0) in
    for i = 1 to n - 1 do
      total := round (!total +. range i)
    done;
    match init with
    | Some y when not init_first -> round (!total +. y)
    | _ -> !total

type bits = {
  bacc : acc;
  bbatch : int;
  brows : int;
  bcols : int;
  bk : int;
  blay_a : lay;
  blay_m : lay;
  blay_y : lay;
  binit : bool;
  binit_first : bool;
  bneg : bool;
  bconj_a : bool;
  bconj_b : bool;
  btri : [ `Lower | `Upper ] option;
  branges : int;
  bzeros : float;
  bsubnormals : float;
  bworkspace : bool;
  bseed : int;
}

let pp_bits ppf c =
  let flag name b = if b then " " ^ name else "" in
  Format.fprintf ppf
    "%s [%d; %d; %d] k %d, a %s, m %s, y %s,%s%s%s%s%s tri %s, ranges %d, \
     zeros %g, subnormals %g%s, seed %d"
    (acc_name c.bacc) c.bbatch c.brows c.bcols c.bk (lay_name c.blay_a)
    (lay_name c.blay_m) (lay_name c.blay_y) (flag "init" c.binit)
    (flag "init_first" c.binit_first)
    (flag "neg" c.bneg) (flag "conj_a" c.bconj_a) (flag "conj_b" c.bconj_b)
    (tri_name c.btri) c.branges c.bzeros c.bsubnormals
    (flag ", workspace" c.bworkspace)
    c.bseed

(* Products in ranges a routine asks, init first or last, many zeros of either
   sign at times, subnormals at times; few rows of columns past two tiles of
   64. *)
let bits =
  let open Gen in
  let extent =
    frequency
      [
        (1, int_range 0 1);
        (2, int_range 2 8);
        (3, int_range 9 40);
        (1, int_range 65 160);
      ]
  in
  with_pp pp_bits
    (let+ bacc = of_list accs
     and+ bbatch = int_range 1 2
     and+ shape =
       frequency
         [
           (4, pair extent extent);
           (1, pair (int_range 1 4) (int_range 129 200));
           (1, pair (int_range 129 200) (int_range 1 4));
         ]
     and+ bk =
       frequency
         [
           (2, constant 0);
           (1, constant 1);
           (4, int_range 2 64);
           (3, int_range 65 300);
         ]
     and+ blay_a = of_list lays
     and+ blay_m = of_list lays
     and+ blay_y = of_list [ Rows; Columns; Strided ]
     and+ binit = bool
     and+ binit_first = bool
     and+ bneg = bool
     and+ bconj_a = bool
     and+ bconj_b = bool
     and+ btri =
       frequency
         [
           (2, constant None);
           (1, constant (Some `Lower));
           (1, constant (Some `Upper));
         ]
     and+ branges = int_range 1 8
     and+ bzeros = frequency [ (3, constant 0.02); (1, constant 0.6) ]
     and+ bsubnormals = frequency [ (1, constant 0.); (1, constant 0.2) ]
     and+ bworkspace = bool
     and+ bseed = nat in
     let complex = is_complex bacc in
     {
       bacc;
       bbatch;
       brows = fst shape;
       bcols = snd shape;
       bk;
       blay_a;
       blay_m;
       blay_y;
       binit;
       binit_first = binit && binit_first;
       bneg;
       bconj_a = complex && bconj_a;
       bconj_b = complex && bconj_b;
       btri;
       branges;
       bzeros;
       bsubnormals;
       bworkspace;
       bseed;
     })

(* Output [e]'s value as contract.mli states it, of [c]'s matrices' values [va],
   [vm] and [was]. *)
let model c ~va ~vm ~was e =
  let at b i j ~cols ~rows = (((b * rows) + i) * cols) + j in
  let j = e mod c.bcols and i = e / c.bcols mod c.brows in
  let b = e / (c.bcols * c.brows) in
  let factors u =
    let x = va.(at b i u ~rows:c.brows ~cols:c.bk) in
    let w = vm.(at b u j ~rows:c.bk ~cols:c.bcols) in
    let x = if c.bconj_a then Complex.conj x else x in
    let x = if c.bneg then Complex.neg x else x in
    let w = if c.bconj_b then Complex.conj w else w in
    (x, w)
  in
  let sum init ts =
    sum_model ~ranges:c.branges ~init_first:c.binit_first
      ~init:(if c.binit then Some init else None)
      (Array.concat (List.init c.bk ts))
  in
  if not (is_complex c.bacc) then
    {
      Complex.re =
        sum was.(e).Complex.re (fun u ->
            let x, w = factors u in
            [| (x.re, w.re) |]);
      im = 0.;
    }
  else
    {
      re =
        sum was.(e).re (fun u ->
            let x, w = factors u in
            [| (x.re, w.re); (x.im, -.w.im) |]);
      im =
        sum was.(e).im (fun u ->
            let x, w = factors u in
            [| (x.re, w.im); (x.im, w.re) |]);
    }

(* [c]'s outputs against the model: an output in the triangle its sum's, one
   outside it y's value, flushed where the workspace's copies carried it. *)
let check_bits c ~was ~want now =
  let kept e = if c.bworkspace then flush_c was.(e) else was.(e) in
  let wrong = ref [] in
  Array.iteri
    (fun e got ->
      let i = e / c.bcols mod c.brows and j = e mod c.bcols in
      let want = if in_triangle c.btri i j then want e else kept e in
      if (not (same got want)) && List.length !wrong < 8 then
        wrong :=
          strf "output %d (%d, %d): %h%+hi, want %h%+hi, was %h%+hi" e i j
            got.re got.im want.re want.im was.(e).re was.(e).im
          :: !wrong)
    now;
  equal (list string) [] (List.rev !wrong)

let bits_matrices c =
  let st = Random.State.make [| c.bseed |] in
  let draw = part st ~zeros:c.bzeros ~subnormals:c.bsubnormals in
  let a = matrix ~draw st c.bacc c.blay_a (c.bbatch, c.brows, c.bk)
  and m = matrix ~draw st c.bacc c.blay_m (c.bbatch, c.bk, c.bcols)
  and y = matrix ~draw st c.bacc c.blay_y (c.bbatch, c.brows, c.bcols) in
  (a, m, y)

let run_bits c ~y a m =
  run ~init_first:c.binit_first ~neg:c.bneg ~conj_a:c.bconj_a ~conj_b:c.bconj_b
    ?tri:c.btri ~ranges:c.branges ~workspace:c.bworkspace ~acc:c.bacc
    ~init:c.binit ~y a m

(* A product's outputs have the bits of the association contract.mli states, a
   complex one's parts summing the terms Nx_kernel numbers, its operands'
   subnormal elements read as zeros of their sign. *)
let law_bits c =
  let a, m, y = bits_matrices c in
  let was = values c.bacc y in
  (* The workspace's copies carry y flushed. *)
  let was = if c.bworkspace then Array.map flush_c was else was in
  let va = values c.bacc a and vm = values c.bacc m in
  let subnormal = Array.exists (fun (z : Complex.t) -> flush_c z <> z) in
  cover "complex" (is_complex c.bacc);
  cover "several ranges" (c.bk > 32 && c.branges > 1);
  cover "init first" c.binit_first;
  cover "init last" (c.binit && not c.binit_first);
  cover "no term, with init" (c.bk = 0 && c.binit);
  cover "many zeros" (c.bzeros > 0.5);
  cover "outputs outside a triangle"
    (c.btri <> None && c.brows > 1 && c.bcols > 1);
  cover "workspace" c.bworkspace;
  cover "a subnormal in a" (c.brows * c.bk > 0 && subnormal va);
  cover "a subnormal in m" (c.bk * c.bcols > 0 && subnormal vm);
  cover "a subnormal in an init_first y" (c.binit_first && subnormal was);
  (* The complex tiles take each flip's signs from one table, indexed by a
     block's row and column, the other way round where m is n x k (Columns) than
     k x n (Rows). *)
  List.iter
    (fun lay ->
      List.iter
        (fun (name, flip) ->
          cover
            (Printf.sprintf "complex, m by %s, %s" (lay_name lay) name)
            (is_complex c.bacc && c.blay_m = lay && c.bk > 1 && flip c))
        [
          ("neg", fun c -> c.bneg);
          ("conj_a", fun c -> c.bconj_a);
          ("conj_b", fun c -> c.bconj_b);
        ])
    [ Rows; Columns ];
  equal answer A.Done (run_bits c ~y a m);
  check_bits c ~was ~want:(model c ~va ~vm ~was) (values c.bacc y)

module Plan = Nx_metal.Plan
module K = Nx_metal.Kernels

(* Runs [f] with launches of at most [t] seconds. *)
let bounded t f =
  Fun.protect
    ~finally:(fun () -> Plan.bound_budget Plan.budget)
    (fun () ->
      Plan.bound_budget t;
      f ())

(* [c]'s product of [y], [a] and [m], each its own buffer, planned without
   running it: [None] where it raises. *)
let planned c ~y a m =
  let b = Plan.make () in
  let o =
    {
      C.init_first = c.binit_first;
      neg = c.bneg;
      conj_a = c.bconj_a;
      conj_b = c.bconj_b;
      tri = c.btri;
      ranges = Some c.branges;
    }
  in
  match
    C.product b o ~acc:(acc_dtype c.bacc) ~init:c.binit ~y:(C.array 0 y)
      (C.array 1 a) (C.array 2 m)
  with
  | () -> Plan.finish_some b
  | exception C.Decline -> None

(* Complex products of m k x n and n x k under each flip alone, over two tiles
   of rows and columns and several steps; and a product of no term. *)
let bits_examples =
  List.concat_map
    (fun blay_m ->
      List.map
        (fun (bneg, bconj_a, bconj_b) ->
          {
            bacc = C64;
            bbatch = 1;
            brows = 40;
            bcols = 70;
            bk = 50;
            blay_a = Rows;
            blay_m;
            blay_y = Rows;
            binit = false;
            binit_first = false;
            bneg;
            bconj_a;
            bconj_b;
            btri = None;
            branges = 1;
            bzeros = 0.02;
            bsubnormals = 0.;
            bworkspace = false;
            bseed = 7;
          })
        [ (true, false, false); (false, true, false); (false, false, true) ])
    [ Rows; Columns ]
  @ [
      (* No term from an init summed first, of subnormal values: drawn, no term
         with an init comes one time in ten, a subnormal in an init_first y
         about one in twenty. *)
      {
        bacc = F32;
        bbatch = 1;
        brows = 8;
        bcols = 8;
        bk = 0;
        blay_a = Rows;
        blay_m = Rows;
        blay_y = Rows;
        binit = true;
        binit_first = true;
        bneg = false;
        bconj_a = false;
        bconj_b = false;
        btri = None;
        branges = 1;
        bzeros = 0.02;
        bsubnormals = 0.2;
        bworkspace = false;
        bseed = 3;
      };
    ]

(* A product of 300 terms, whose tiles take chunks of k at the least budget it
   plans at, so that each run reaches law_chunked's cover. *)
let chunked_examples =
  [
    {
      bacc = F32;
      bbatch = 1;
      brows = 40;
      bcols = 70;
      bk = 300;
      blay_a = Rows;
      blay_m = Rows;
      blay_y = Rows;
      binit = true;
      binit_first = false;
      bneg = false;
      bconj_a = false;
      bconj_b = false;
      btri = None;
      branges = 1;
      bzeros = 0.02;
      bsubnormals = 0.;
      bworkspace = false;
      bseed = 5;
    };
  ]

(* Whether a plan's tiles take a chunk of their ranges. *)
let chunks (p : Plan.t) =
  Array.exists
    (fun (l : Plan.launch) ->
      K.words l.kernel = K.words K.Contract_f32_64_kn_a
      && Bytes.get_int64_le l.words (8 * K.contract_length) > 0L)
    p.launches

(* The least budget, within 0.1%, at which [c]'s product plans. *)
let least_budget c ~y a m =
  let fits t = bounded t (fun () -> Option.is_some (planned c ~y a m)) in
  let rec go lo hi =
    if hi /. lo < 1.001 then hi
    else
      let mid = sqrt (lo *. hi) in
      if fits mid then go lo mid else go mid hi
  in
  go 1e-7 Plan.budget

(* At the least budget the product plans at, its tiles' launches cut k into
   chunks, each resuming from the sum the last stored: the bits are the
   unchunked product's. *)
let law_chunked c =
  let c = { c with bworkspace = false } in
  let a, m, y = bits_matrices c in
  let was = values c.bacc y in
  let va = values c.bacc a and vm = values c.bacc m in
  let t = least_budget c ~y a m in
  bounded t (fun () ->
      cover "chunked"
        (match planned c ~y a m with Some p -> chunks p | None -> false);
      equal answer A.Done (run_bits c ~y a m));
  check_bits c ~was ~want:(model c ~va ~vm ~was) (values c.bacc y)

(* Copies *)

(* A product of 32 x 24 outputs, which reads a's rows of terms in place, whose a
   lies by columns reads each of a's rows of 20000 terms 32 elements apart, so
   the copy of one such row takes about 50 us. At 30 us, at which the tiles
   plan, the copy cuts the row's columns into launches, and the outputs are
   their sums. *)
let test_copy_cut acc =
  let c =
    {
      acc;
      batch = 1;
      rows = 32;
      cols = 24;
      k = 20000;
      lay_a = Columns;
      lay_m = Rows;
      lay_y = Rows;
      init = false;
      init_first = false;
      neg = false;
      conj_a = false;
      conj_b = false;
      tri = None;
      ranges = None;
      workspace = false;
      pseed = 7;
    }
  in
  let st = Random.State.make [| c.pseed |] in
  let a = matrix st acc c.lay_a (c.batch, c.rows, c.k)
  and m = matrix st acc c.lay_m (c.batch, c.k, c.cols)
  and y = matrix st acc c.lay_y (c.batch, c.rows, c.cols) in
  let want =
    reference c ~y:(values acc y) ~a:(values acc a) ~m:(values acc m)
  in
  let got, ts =
    bounded 30e-6 (fun () ->
        Plan.traced (fun () -> run ~acc ~init:false ~y a m))
  in
  equal answer A.Done got;
  let starts =
    List.filter_map
      (fun (t : Plan.traced) ->
        if t.kernel <> K.Contract_copy then None
        else
          Some (Int64.to_int (Bytes.get_int64_le t.words (8 * K.ccopy_start))))
      ts
  in
  at_least ~msg:"a copy's launches from a row's later columns" int ~than:1
    (List.length (List.filter (fun s -> s > 0) starts));
  let now = values acc y in
  let wrong =
    List.filter
      (fun e -> now.(e) <> want.(e))
      (List.init (Array.length now) Fun.id)
  in
  equal ~msg:"outputs other than their sums" (list int) [] wrong

(* Negated tiles *)

(* Extents of each float32 tile, as contract.mli's plan takes them: 16 x 64 for
   few rows, 64 x 64 for a call of many tiles, 32 x 32 otherwise; each of more
   terms than a step, so that whole steps run negated, and a tail. *)
let tiles =
  [
    (C.T16, (1, 8, 40, 75));
    (C.T32, (1, 40, 40, 37));
    (C.T64, (24, 128, 128, 37));
  ]

let tile_name = function
  | C.T16 -> "16 x 64"
  | C.T32 -> "32 x 32"
  | C.T64 -> "64 x 64"
  | C.G32 | C.G16 -> "grouped"

(* A negated product on [tile], [m] k x n or n x k, has the bits of its negated
   chain from y. *)
let test_negated (tile, (batch, rows, cols, k), lay_m) =
  let e =
    {
      C.batch;
      m = rows;
      n = cols;
      k;
      stage = C.F32;
      nk = lay_m = Columns;
      complex = false;
      source = 0;
    }
  in
  equal ~msg:"the tile" string (tile_name tile)
    (tile_name (Option.get (C.plan e)).tile);
  let c =
    {
      bacc = F32;
      bbatch = batch;
      brows = rows;
      bcols = cols;
      bk = k;
      blay_a = Rows;
      blay_m = lay_m;
      blay_y = Rows;
      binit = true;
      binit_first = true;
      bneg = true;
      bconj_a = false;
      bconj_b = false;
      btri = None;
      branges = 1;
      bzeros = 0.1;
      bsubnormals = 0.;
      bworkspace = false;
      bseed = rows + k;
    }
  in
  let a, m, y = bits_matrices c in
  let was = values c.bacc y in
  let va = values c.bacc a and vm = values c.bacc m in
  equal answer A.Done (run_bits c ~y a m);
  check_bits c ~was ~want:(model c ~va ~vm ~was) (values c.bacc y)

(* A negated chain standing at -0 stays there: y -0, a's and m's elements +0, k
   a whole step and k with a tail, whose padding a negated tile must not flip.
   And a sum that cancels to zero is +0, where y - a m is computed as y + (-a)
   m. *)
let test_zeros () =
  let dev = S.device () in
  let filled shape v =
    let x = A.create dev D.Float32 (Iarray.of_array shape) in
    let n = Array.fold_left ( * ) 1 shape in
    let s =
      String.concat "" (List.init n (fun _ -> encode F32 { re = v; im = 0. }))
    in
    Rig.Buffer.blit_from_string s 0 (A.buffer x) 0 (String.length s);
    A.Any x
  in
  let bits_of y =
    List.map
      (fun (z : Complex.t) -> Int64.bits_of_float z.re)
      (Array.to_list (values F32 y))
  in
  List.iter
    (fun k ->
      let y = filled [| 1; 3; 5 |] (-0.) in
      let a = filled [| 1; 3; k |] 0. and m = filled [| 1; k; 5 |] 0. in
      equal answer A.Done
        (run ~init_first:true ~neg:true ~ranges:1 ~acc:F32 ~init:true ~y a m);
      equal ~msg:(strf "k %d" k) (list int64)
        (List.init 15 (fun _ -> Int64.bits_of_float (-0.)))
        (bits_of y))
    [ 5; 16; 32; 37 ];
  let y = filled [| 1; 3; 5 |] 1. in
  let a = filled [| 1; 3; 1 |] 1. and m = filled [| 1; 1; 5 |] 1. in
  equal answer A.Done
    (run ~init_first:true ~neg:true ~ranges:1 ~acc:F32 ~init:true ~y a m);
  equal ~msg:"cancelled" (list int64)
    (List.init 15 (fun _ -> Int64.bits_of_float 0.))
    (bits_of y)

(* Triangles *)

(* Whether [s] occurs in [n]. *)
let contains n s =
  let k = String.length s in
  let rec at i =
    i + k <= String.length n && (String.sub n i k = s || at (i + 1))
  in
  at 0

(* The kernel of [c]'s plan's tiles, whose name gives their size. *)
let tile_of c ~y a m =
  let name (l : Plan.launch) =
    let n = K.name l.kernel in
    let tile prefix = String.starts_with ~prefix n in
    if tile "contract_f32" || tile "contract_c64" then Some n else None
  in
  match planned c ~y a m with
  | None -> None
  | Some p -> Array.find_map name p.launches

(* A product under [tri] on the tile of [extents]: an output inside the triangle
   has its chain's bits, one outside it y's bits as they were, in the tiles on
   the diagonal too, at one range of k and at several. *)
let test_tri (tile, (batch, rows, cols, k), acc, tri, branges) =
  let c =
    {
      bacc = acc;
      bbatch = batch;
      brows = rows;
      bcols = cols;
      bk = k;
      blay_a = Rows;
      blay_m = Rows;
      blay_y = Rows;
      binit = true;
      binit_first = true;
      bneg = true;
      bconj_a = false;
      bconj_b = false;
      btri = Some tri;
      branges;
      bzeros = 0.1;
      bsubnormals = 0.;
      bworkspace = false;
      bseed = rows + k;
    }
  in
  let a, m, y = bits_matrices c in
  let size =
    match tile with C.T16 -> "_16_" | C.T32 -> "_32_" | _ -> "_64_"
  in
  equal ~msg:"the tile" bool true
    (match tile_of c ~y a m with Some n -> contains n size | None -> false);
  let was = values c.bacc y in
  let va = values c.bacc a and vm = values c.bacc m in
  equal answer A.Done (run_bits c ~y a m);
  check_bits c ~was ~want:(model c ~va ~vm ~was) (values c.bacc y)

(* Each tile at each dtype, and the 16 x 64 tile of an exchanged float32 call,
   [C.exchanges ~m:40 ~n:8], whose tiles' rows are y's columns, so that its
   triangle is the other one of the tiles'. *)
let tri_cases =
  List.concat_map
    (fun (t, e, acc) ->
      List.concat_map
        (fun tri -> [ (t, e, acc, tri, 1); (t, e, acc, tri, 4) ])
        [ `Lower; `Upper ])
    (List.concat_map
       (fun (t, e) -> List.map (fun acc -> (t, e, acc)) accs)
       tiles
    @ [ (C.T16, (2, 40, 8, 75), F32) ])

(* Ranges *)

(* contract.mli's count: one where a matrix's tiles of 64 fill half of 192, else
   as many as fill it, each of at least 8 spans of 32, at most 64. *)
let law_ranges (rows, cols, k) =
  let cd a b = (a + b - 1) / b in
  let t = cd rows 64 * cd cols 64 and r = cd k 32 in
  let want =
    if t = 0 || 2 * t >= 192 then 1
    else Int.max 1 (Int.min (Int.min 64 (r / 8)) (cd 192 t))
  in
  equal int want (C.ranges ~rows ~cols ~k)

let extents =
  let open Gen in
  let x = frequency [ (1, int_range 0 4); (3, int_range 1 2000) ] in
  triple x x (frequency [ (1, int_range 0 300); (2, int_range 0 100_000) ])

let () =
  exit
    (Windtrap.run "nx.metal products"
       [
         group "products"
           [
             prop ~examples:product_examples
               "a product's outputs are their sums" product law_product;
             prop ~examples:bits_examples
               "a product's outputs have the stated association's bits" bits
               law_bits;
             prop ~examples:chunked_examples
               "a chunked product's bits are the unchunked one's" bits
               law_chunked;
             cases "a negated product on each tile"
               ~id:(fun (t, _, l) -> tile_name t ^ ", m by " ^ lay_name l)
               (List.concat_map
                  (fun (t, e) -> [ (t, e, Rows); (t, e, Columns) ])
                  tiles)
               test_negated;
             test "a negated -0 stays -0, a cancelled sum is +0" test_zeros;
             cases "a copy's row past a launch is cut by its columns"
               ~id:acc_name accs test_copy_cut;
             cases "outputs outside a triangle are left as they are"
               ~id:(fun (t, (_, rows, cols, _), acc, tri, s) ->
                 strf "%s, %d x %d, %s, %s, ranges %d" (tile_name t) rows cols
                   (acc_name acc) (tri_name (Some tri)) s)
               tri_cases test_tri;
             prop "a routine's ranges follow one matrix's extents" extents
               law_ranges;
           ];
       ])
