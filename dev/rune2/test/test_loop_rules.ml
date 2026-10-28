(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.scan under forward mode, reverse mode and maps: the scan of a step that
   captures an array, with a carry of two arrays whose second starts constant
   and gains a tangent or lanes from the first, against the OCaml fold of the
   same step, eagerly and under jit: bit for bit, but for the zeros a loop holds
   where the fold has none (Compose.lenient). Each leaf is drawn whole on both
   devices of one set or on one of them. *)

open Windtrap
open Compose
module Rune = Rune2
module T = Nx.Ptree
module D = Nx.Dtype
module Two = (val Nx.devices [ Nx_support.memory 0; Nx_support.memory 1 ])

type where = Whole | Dev of int

let pp_where ppf = function
  | Whole -> Format.pp_print_string ppf "both devices"
  | Dev k -> Format.fprintf ppf "device %d" k

(* A leaf of shape [s]: a host operand's values placed on a device, or whole on
   both a formula, [a + i b] at the element [i] in C order, [a] and [b] the
   operand's first values. *)
let put (type s) w (dt : (float, s) Nx.dtype) shape o : (float, s, Two.d) Nx.t =
  match w with
  | Dev k ->
      Nx.place
        (Nx.Placement.device Two.v (Nx_support.memory k))
        (make dt shape o)
  | Whole ->
      let at j = if j < Array.length o.values then o.values.(j) else 1. in
      let here = Nx.Placement.on Two.v in
      let n = Float.of_int (numel shape) in
      let i = Nx.reshape shape (Nx.arange_f_at here dt 0. n 1.) in
      Nx.add (Nx.full_at here dt shape (at 0)) (Nx.mul_s i (at 1))

let fold step ~init xs =
  let n = (Nx.shape xs).(0) in
  let rec go i c ys =
    if i = n then (c, List.rev ys)
    else
      let c, y = step c (Nx.get [ i ] xs) in
      go (i + 1) c (y :: ys)
  in
  go 0 init []

(* A law's draw: dtype, row shape, rows, the step's three expressions, the
   operands of [c0], [c1], [xs], [w] and their tangents, and where each leaf
   lies. *)
type case =
  | Case : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      n : int;
      g : expr;
      k : expr;
      h : expr;
      ops : operand array;
      tans : operand array;
      at : where array;  (** Of [c0], [c1], [xs], [w], then their tangents. *)
      free : bool;  (** Whether [c1'] reads [c0] only where [k] reads [Y]. *)
    }
      -> case

let pp_case ppf (Case c) =
  Format.fprintf ppf "%s; tangents %a, %a, %a; "
    (if c.free then "c1' as k" else "c1' + c0 w")
    pp_where c.at.(4) pp_where c.at.(5) pp_where c.at.(6);
  Format.fprintf ppf
    "%a %a, %d rows: c0' %a, c1' %a, y %a; c0 %a, c1 %a, xs %a, w %a"
    Nx.pp_dtype c.dt pp_shape c.s c.n pp c.g pp c.k pp c.h pp_where c.at.(0)
    pp_where c.at.(1) pp_where c.at.(2) pp_where c.at.(3)

let cases =
  let open Gen in
  (let* s = shape in
   let* n = of_list [ 0; 1; 2; 5 ] in
   let rank = Array.length s in
   let* g = expr ~rank and+ k = expr ~rank and+ h = expr ~rank in
   let rows = Array.append [| n |] s in
   let* c0 = operand s and+ c1 = operand s and+ xs = operand rows in
   let* w = operand s in
   let* t0 = operand s and+ tx = operand rows and+ tw = operand s in
   (* One device of the set's two per case: values on two devices meet only
      where the program places one beside the other. *)
   let* d = int_range 0 1 in
   let where = frequency [ (1, constant Whole); (1, constant (Dev d)) ] in
   let* at = array ~size:(constant 7) where in
   let* free = bool in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any Nx.float64 ] in
   match D.kind dt with
   | Float ->
       Case
         {
           dt;
           s;
           n;
           g;
           k;
           h;
           free;
           ops = [| c0; c1; xs; w |];
           tans = [| t0; tx; tw |];
           at;
         }
   | _ -> assert false)
  |> with_pp pp_case

(* The law's functions of [(c0, xs, w)]: the scan, and the fold of the same step
   from the carry [(c0, c1)], each returning the final carry and the outputs.
   [c1'] reads [c0], so that [c1] gains what [c0] has after a step. [nested]
   scans the step over its row twice inside the step, summing the outputs. *)
let functions (type s) ?(nested = false) ~free (dt : (float, s) Nx.dtype) s n g
    k h (c1 : (float, s, Two.d) Nx.t) =
  let b = Nx.broadcast_to s in
  let step w (c0, c1) x =
    let c0' = b (eval c0 x g) in
    let c0w = Nx.mul c0 w in
    let c1' =
      if free then b (eval c1 c0w k) else b (Nx.add (eval c1 c0w k) c0w)
    in
    ((c0', c1'), b (eval c1 x h))
  in
  let cs = T.(pair tensor tensor) in
  let inner ~scan w c x =
    if not nested then step w c x
    else
      let two = Nx.stack [ x; x ] in
      let c, ys =
        if scan then Rune.scan cs T.tensor T.tensor ~f:(step w) ~init:c two
        else
          let c, ys = fold (step w) ~init:c two in
          (c, Nx.stack ys)
      in
      (c, Nx.sum ~axes:[ 0 ] ys)
  in
  let scanned (c0, (xs, w)) =
    let (c0, c1), ys =
      Rune.scan cs T.tensor T.tensor ~f:(inner ~scan:true w) ~init:(c0, c1) xs
    in
    (c0, (c1, ys))
  in
  let step w = inner ~scan:false w in
  let folded c1 (c0, (xs, w)) =
    (* The step is traced at the call, so with no row its errors raise. *)
    let here = Nx.Placement.without_leading_axis (Nx.placement xs) in
    ignore (step w (c0, c1) (Nx.zeros_at here dt s));
    let (c0, c1), ys = fold (step w) ~init:(c0, c1) xs in
    let rows = Array.append [| n |] s in
    let ys =
      if n = 0 then Nx.zeros_at (Nx.placement xs) dt rows
      else Nx.stack ~axis:0 ys
    in
    (c0, (c1, ys))
  in
  (scanned, folded c1, folded)

(* Outcomes, where two failures of operands on two devices are alike whichever
   operation meets them first: a backward pass orders its operations otherwise
   than the fold's. *)
let mixed m =
  let key = "place one beside the other" in
  let n = String.length key and l = String.length m in
  let rec go i = i + n <= l && (String.sub m i n = key || go (i + 1)) in
  go 0

let outcome_t' t =
  Testable.make
    ~pp:(Testable.pp (outcome_t t))
    ~equal:(fun a b ->
      match (a, b) with
      | Raised a, Raised b -> String.equal a b || (mixed a && mixed b)
      | _ -> Testable.equal (outcome_t t) a b)

let triple () = T.(pair tensor (pair tensor tensor))
let t3 = Testable.triple bits_t bits_t bits_t
let bits3 (a, (b, c)) = (bits a, bits b, bits c)

(* The leaves [c0], [c1], [xs], [w] of a case, where it puts them. *)
let operands (type s) (dt : (float, s) Nx.dtype) s n ops at :
    (float, s, Two.d) Nx.t array =
  let rows = Array.append [| n |] s in
  [|
    put at.(0) dt s ops.(0);
    put at.(1) dt s ops.(1);
    put at.(2) dt rows ops.(2);
    put at.(3) dt s ops.(3);
  |]

(* Whether [c1] gains what [c0] has after a step. *)
let gains (Case c) = (not c.free) || reads_y c.k

(* Whether a step leaves [c0] or [c1] unread: the reversed loop's carry then
   holds a zero cotangent where the fold has none. *)
let unread (Case c) =
  (not (reads_x c.k || reads_x c.h))
  || (c.free && not (reads_x c.g || reads_y c.k))

let covers (Case c) =
  cover "no row" (c.n = 0);
  cover "several rows" (c.n > 1);
  cover "a placed leaf" (Array.exists (fun w -> w <> Whole) c.at);
  cover "the second device" (Array.exists (fun w -> w = Dev 1) c.at);
  cover "a leaf on both devices" (Array.exists (fun w -> w = Whole) c.at)

(* A compiled loop gives the fold's bits where its step's functions are exact
   (Rune.jit's classes): the laws compare compiled calls on the case's
   functions made exact. *)
let exactly (Case c) =
  let x e = elementwise (exact e) in
  Case { c with g = x c.g; k = x c.k; h = x c.h; at = Array.make 7 (Dev 0) }

(* [f ()], covering that its compiled calls compiled: their leaves are placed
   on one device, so that only a program with a result of every set, a
   constant, is evaluated. *)
let compiled f =
  let before = Rune.evaluations () in
  let r = f () in
  cover "the compiled calls compiled"
    ((match r with Raised _ -> false | _ -> true)
    && Rune.evaluations () = before);
  r

let law_jvp' ~jit (Case c as case) =
  covers case;
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let rows = Array.append [| c.n |] c.s in
  let tan k sh = put c.at.(4 + k) c.dt sh c.tans.(k) in
  let t = (tan 0 c.s, (tan 1 rows, tan 2 c.s)) in
  cover "a placed tangent" (Array.exists (( <> ) Whole) (Array.sub c.at 4 3));
  let scanned, folded, from =
    functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1
  in
  let jvp f = Rune.jvp (triple ()) (triple ()) f (c0, (xs, w)) t in
  let both (y, t) = (bits3 y, bits3 t) in
  (* Where [c1] gains a tangent at the first step, the scan's starts from a
     zero: the fold's is a zero from the start. *)
  cover "a carry with no tangent" (not (gains case));
  let expected =
    let p = T.(pair tensor (pair tensor (pair tensor tensor))) in
    outcome (fun () ->
        if not (gains case) then both (jvp folded)
        else
          let y, (_, t) =
            Rune.jvp p p
              (fun (c1, x) -> (c1, from c1 x))
              (c1, (c0, (xs, w)))
              (Nx.full_at (Nx.placement c1) c.dt c.s (-0.), t)
          in
          both (snd y, t))
  in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  (* A carry whose step reads nothing keeps a zero tangent where the fold's has
     none. *)
  let l = lenient ~nan:(constant c.g) in
  let l = if constant c.g then l else bits_t in
  let tt = outcome_t' (Testable.pair t3 (Testable.triple l l l)) in
  equal ~msg:"eager" tt expected (outcome (fun () -> both (jvp scanned)));
  if jit then begin
    let a3 =
      T.(
        tensor @-> tensor @-> tensor @-> tensor @-> tensor @-> tensor
        @-> returns
              (pair
                 (pair tensor (pair tensor tensor))
                 (pair tensor (pair tensor tensor))))
    in
    let staged =
      Rune.jit a3 (fun c0 xs w t0 tx tw ->
          Rune.jvp (triple ()) (triple ()) scanned (c0, (xs, w)) (t0, (tx, tw)))
    in
    let t0, (tx, tw) = t in
    equal ~msg:"jit of jvp" tt expected
      (compiled (fun () -> outcome (fun () -> both (staged c0 xs w t0 tx tw))));
    let jitted =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor
          @-> returns (pair tensor (pair tensor tensor)))
        (fun c0 xs w -> scanned (c0, (xs, w)))
    in
    equal ~msg:"jvp of jit" tt expected
      (compiled (fun () ->
           outcome (fun () ->
               both
                 (Rune.jvp (triple ()) (triple ())
                    (fun (a, (b, c)) -> jitted a b c)
                    (c0, (xs, w))
                    t))))
  end

let law_jvp case =
  law_jvp' ~jit:false case;
  law_jvp' ~jit:true (exactly case)

let law_grad' ~jit (Case c as case) =
  covers case;
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let scanned, folded, _ = functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1 in
  let objective f p =
    let a, (b, y) = f p in
    Nx.add (Nx.add (Nx.sum a) (Nx.sum b)) (Nx.sum y)
  in
  let grad f =
    outcome (fun () ->
        bits3 (Rune.grad (triple ()) (objective f) (c0, (xs, w))))
  in
  let expected = grad folded in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  (* [c1]'s cotangent is a zero where the step reads no [c1]. *)
  let unread = unread case in
  let l = lenient ~nan:unread in
  let tt = outcome_t' (Testable.triple l l l) in
  equal ~msg:"eager" tt expected (grad scanned);
  if jit then begin
    let staged =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor
          @-> returns (pair tensor (pair tensor tensor)))
        (fun c0 xs w -> Rune.grad (triple ()) (objective scanned) (c0, (xs, w)))
    in
    equal ~msg:"jit of grad" tt expected
      (compiled (fun () -> outcome (fun () -> bits3 (staged c0 xs w))));
    let jitted =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor
          @-> returns (pair tensor (pair tensor tensor)))
        (fun c0 xs w -> scanned (c0, (xs, w)))
    in
    equal ~msg:"grad of jit" tt expected
      (compiled (fun () -> grad (fun (a, (b, c)) -> jitted a b c)))
  end

let law_grad case =
  law_grad' ~jit:false case;
  law_grad' ~jit:true (exactly case)

(* A case whose compiled transpose tolk declines to link, so that the call
   evaluates over its arguments, a cotangent of every set among them. *)
let grad_examples =
  let plain values = { view = Plain; values } in
  [
    Case
      {
        dt = Nx.float32;
        s = [| 2 |];
        n = 1;
        g = Y;
        k = Un (Neg, Y);
        h =
          Un
            ( Cumsum 0,
              Bin
                ( Add,
                  Bin
                    (Add, Y, Bin (Add, Bin (Mul, X, K 0.), Bin (Add, Y, K 0.))),
                  Un (Cumsum 0, Un (Neg, Bin (Add, Y, X))) ) );
        ops =
          [|
            plain [| 0.5; -1. |];
            plain [| 2.; 1. |];
            plain [| 0.25; 3. |];
            plain [| 1.5; -2. |];
          |];
        tans = [| plain [| 1.; 1. |]; plain [| 1.; 1. |]; plain [| 1.; 1. |] |];
        at = Array.make 7 (Dev 0);
        free = false;
      };
  ]

(* Maps: one of [c0], [xs] and [w] has lanes, the others are captured. *)
type mapped = Carry | Rows | Capture

let map_cases =
  Gen.(pair cases (pair (of_list [ Carry; Rows; Capture ]) (of_list [ 1; 3 ])))
  |> Gen.with_pp (fun ppf (c, (m, l)) ->
      Format.fprintf ppf "%a, %d lanes of the %s" pp_case c l
        (match m with
        | Carry -> "carry"
        | Rows -> "rows"
        | Capture -> "capture"))

let law_vmap' ~jit ((Case c as case), (m, lanes)) =
  covers case;
  cover "lanes in the carry" (m = Carry);
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let scanned, folded, _ = functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1 in
  let flat f p =
    let a, (b, y) = f p in
    Nx.concatenate ~axis:0
      [ Nx.reshape [| -1 |] a; Nx.reshape [| -1 |] b; Nx.reshape [| -1 |] y ]
  in
  (* Lane [l] scales the mapped operand by [l + 1]. *)
  let pick = function Carry -> c0 | Rows -> xs | Capture -> w in
  let x = pick m in
  let batch =
    Nx.stack ~axis:0
      (List.init lanes (fun l -> Nx.mul_s x (Float.of_int (l + 1))))
  in
  let at f v =
    match m with
    | Carry -> flat f (v, (xs, w))
    | Rows -> flat f (c0, (v, w))
    | Capture -> flat f (c0, (xs, v))
  in
  let one_by_one f =
    Nx.stack ~axis:0 (List.init lanes (fun l -> at f (Nx.get [ l ] batch)))
  in
  let expected = outcome (fun () -> bits (one_by_one folded)) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  let tt = outcome_t' bits_t in
  equal ~msg:"eager" tt expected
    (outcome (fun () -> bits (Rune.vmap' (at scanned) batch)));
  if jit then begin
    equal ~msg:"jit of vmap" tt expected
      (compiled (fun () ->
           outcome (fun () ->
               bits (Rune.jit' (Rune.vmap' (at scanned)) batch))));
    equal ~msg:"vmap of jit" tt expected
      (compiled (fun () ->
           outcome (fun () ->
               bits (Rune.vmap' (Rune.jit' (at scanned)) batch))))
  end

let law_vmap (case, m) =
  law_vmap' ~jit:false (case, m);
  law_vmap' ~jit:true (exactly case, m)

(* Compositions: a scan whose step scans, under jvp, grad and a map, and second
   orders through a scan. Their zeros are a loop's wherever a step reads no
   carry, so values compare leniently. *)
(* [e] with every partial derivative finite at finite operands: exp, sqrt,
   log, division and power replaced by smooth functions, a constant within
   [-2, 2]. *)
let rec smoothed = function
  | (X | Y) as e -> e
  | K v -> K (if Float.is_nan v then 0.5 else Float.min 2. (Float.max (-2.) v))
  | Un (Exp, e) -> Un (Sin, smoothed e)
  | Un (Sqrt, e) -> Un (Tanh, smoothed e)
  | Un (Log, e) -> Un (Neg, smoothed e)
  | Un (u, e) -> Un (u, smoothed e)
  | Bin ((Div | Pow), e, e') -> Bin (Mul, smoothed e, smoothed e')
  | Bin (b, e, e') -> Bin (b, smoothed e, smoothed e')

(* An operand's values within [-2, 2], a NaN as [0.5]. *)
let moderated o =
  {
    o with
    values =
      Array.map
        (fun v ->
          if Float.is_nan v then 0.5 else Float.min 2. (Float.max (-2.) v))
        o.values;
  }

let law_compose (Case c as case) =
  covers case;
  let o = operands c.dt c.s c.n c.ops c.at in
  let c0 = o.(0) and c1 = o.(1) and xs = o.(2) and w = o.(3) in
  let x = (c0, (xs, w)) in
  let rows = Array.append [| c.n |] c.s in
  let t =
    ( put Whole c.dt c.s c.tans.(0),
      (put Whole c.dt rows c.tans.(1), put Whole c.dt c.s c.tans.(2)) )
  in
  let p = triple () in
  let objective f p =
    let a, (b, y) = f p in
    Nx.add (Nx.add (Nx.sum a) (Nx.sum b)) (Nx.sum y)
  in
  (* A second order through a loop meets more zeros a loop holds than a step's
     reads tell, so it is compared where every partial derivative is finite: the
     case's functions smoothed, over operands within [-2, 2]. It sums a step's
     contributions before adding them to the loop's, where the fold adds each in
     turn: within [tol] of the larger, or of 1. *)
  let tol = if D.equal c.dt Nx.float32 then 1e-4 else 1e-10 in
  let within ~nan =
    let l = lenient ~nan in
    Testable.make ~pp:(Testable.pp bits_t)
      ~equal:(fun ((s, a) as u) ((s', b) as v) ->
        Testable.equal l u v
        || s = s'
           && Array.for_all2
                (fun x y ->
                  let x = Int64.float_of_bits x and y = Int64.float_of_bits y in
                  x = y
                  || (Float.is_nan x && Float.is_nan y)
                  || (nan && Float.is_nan y)
                  || Float.abs (x -. y)
                     <= tol
                        *. Float.max 1. (Float.max (Float.abs x) (Float.abs y)))
                a b)
  in
  let check msg cmp (scanned, folded) g =
    equal ~msg
      (outcome_t' (Testable.triple cmp cmp cmp))
      (outcome (fun () -> bits3 (g folded)))
      (outcome (fun () -> bits3 (g scanned)))
  in
  let nested_s, nested_f, nested_from =
    functions ~nested:true ~free:c.free c.dt c.s c.n c.g c.k c.h c1
  in
  let nested = (nested_s, nested_f) in
  let ps, pf, _ = functions ~free:c.free c.dt c.s c.n c.g c.k c.h c1 in
  let plain = (ps, pf) in
  (* jvp of a scan whose step scans, against its fold's from a zero tangent of
     [c1] where it gains one, as [law_jvp]. *)
  let jl = if constant c.g then lenient ~nan:true else bits_t in
  let reference =
    outcome (fun () ->
        if not (gains case) then bits3 (snd (Rune.jvp p p nested_f x t))
        else
          let q = T.(pair tensor (pair tensor (pair tensor tensor))) in
          let _, (_, t) =
            Rune.jvp q q
              (fun (c1, x) -> (c1, nested_from c1 x))
              (c1, x)
              (Nx.full_at (Nx.placement c1) c.dt c.s (-0.), t)
          in
          bits3 t)
  in
  equal ~msg:"jvp of a nested scan"
    (outcome_t' (Testable.triple jl jl jl))
    reference
    (outcome (fun () -> bits3 (snd (Rune.jvp p p nested_s x t))));
  (* An inner reversed loop sums its steps' contributions to a capture before
     the outer step adds them. *)
  check "grad of a nested scan"
    (within ~nan:(unread case))
    nested
    (fun f -> Rune.grad p (objective f) x);
  let lanes f =
    let batch = Nx.stack [ c0; Nx.neg c0 ] in
    let y =
      Rune.vmap'
        (fun c0 ->
          let a, (b, y) = f (c0, (xs, w)) in
          Nx.concatenate ~axis:0
            [
              Nx.reshape [| -1 |] a;
              Nx.reshape [| -1 |] b;
              Nx.reshape [| -1 |] y;
            ])
        batch
    in
    (y, (y, y))
  in
  check "vmap of a nested scan" bits_t nested lanes;
  let sq f x =
    let a, (b, y) = Rune.grad p (objective f) x in
    Nx.add
      (Nx.add (Nx.sum (Nx.mul a a)) (Nx.sum (Nx.mul b b)))
      (Nx.sum (Nx.mul y y))
  in
  let o2 = operands c.dt c.s c.n (Array.map moderated c.ops) c.at in
  let x2 = (o2.(0), (o2.(2), o2.(3))) in
  let t2 =
    let m k sh = put Whole c.dt sh (moderated c.tans.(k)) in
    (m 0 c.s, (m 1 rows, m 2 c.s))
  in
  let s2, f2, _ =
    functions ~free:c.free c.dt c.s c.n (smoothed c.g) (smoothed c.k)
      (smoothed c.h) o2.(1)
  in
  (* Within [tol] of the derivative's largest magnitude, or of 1: a sum that
     cancels leaves a residue of that scale's rounding. *)
  let scaled =
    let values (_, a) = Array.map Int64.float_of_bits a in
    let scale (a, b, c) =
      Array.fold_left
        (fun m v -> if Float.is_finite v then Float.max m (Float.abs v) else m)
        1.
        (Array.concat [ values a; values b; values c ])
    in
    Testable.make ~pp:(Testable.pp t3)
      ~equal:(fun ((a, b, c) as u) (a', b', c') ->
        let m = scale u in
        let near (s, x) (s', y) =
          s = s'
          && Array.for_all2
               (fun x y ->
                 let x = Int64.float_of_bits x and y = Int64.float_of_bits y in
                 x = y || Float.abs (x -. y) <= tol *. m)
               x y
        in
        near a a' && near b b' && near c c')
  in
  (* Where the fold's second derivative is finite: an infinity there meets a
     zero the loop holds, as at first order. *)
  let finite (_, a) =
    Array.for_all (fun x -> Float.is_finite (Int64.float_of_bits x)) a
  in
  let second msg g =
    let expected = outcome (fun () -> bits3 (g f2)) in
    match expected with
    | Value ((a, b, c) as e) when finite a && finite b && finite c ->
        cover ("a finite " ^ msg) true;
        equal ~msg (outcome_t' scaled) (Value e)
          (outcome (fun () -> bits3 (g s2)))
    | _ -> cover ("a finite " ^ msg) false
  in
  second "grad of grad" (fun f -> Rune.grad p (sq f) x2);
  second "jvp of grad" (fun f ->
      snd (Rune.jvp p p (Rune.grad p (objective f)) x2 t2));
  let per_lane f =
    let batch = Nx.stack [ c0; Nx.mul_s c0 2. ] in
    let g =
      Rune.vmap' (fun c0 -> fst (Rune.grad p (objective f) (c0, (xs, w)))) batch
    in
    (g, (g, g))
  in
  check "vmap of grad" (lenient ~nan:(unread case)) plain per_lane

(* Interface *)

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs
let floats = Testable.array Testable.float_exact
let pair = T.(pair tensor tensor)

(* [a' = 2a], [b' = a] from [b = 0], a constant: [b] gains a tangent, lanes and
   a cotangent after the first step, and after three [b = 4 a0]. *)
let doubling x =
  let (_, b), () =
    Rune.scan pair T.tensor T.unit
      ~f:(fun (a, _) _ -> ((Nx.mul_s a 2., a), ()))
      ~init:(x, Nx.zeros f64 [| 2 |])
      (Nx.zeros f64 [| 3; 1 |])
  in
  b

let test_gains () =
  let x = vec [| 1.; -2. |] and t = vec [| 0.5; 3. |] in
  let y, dy = Rune.jvp' doubling x t in
  equal ~msg:"primal" floats [| 4.; -8. |] (Nx.to_array y);
  equal ~msg:"tangent" floats [| 2.; 12. |] (Nx.to_array dy);
  equal ~msg:"gradient" floats [| 4.; 4. |]
    (Nx.to_array (Rune.grad' (fun x -> Nx.sum (doubling x)) x));
  let xs = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  equal ~msg:"lanes" floats [| 4.; 8.; 12.; 16. |]
    (Nx.to_array (Rune.vmap' doubling xs))

module Total = Rune.Total

(* A loop's additions under a transformation are the fold's, from the collect's
   running total; inside a map, each lane's. *)
let test_totals () =
  let t = Total.make () in
  let xs = Nx.create f64 [| 3; 2 |] [| 1e16; 3.; 1.; 5.; 1.; 7. |] in
  let step c x =
    Total.add t (Nx.sum x);
    (Nx.mul c x, ())
  in
  let scanned c =
    Total.add t (Nx.scalar f64 1.);
    Nx.sum (fst (Rune.scan T.tensor T.tensor T.unit ~f:step ~init:c xs))
  in
  let folded c =
    Total.add t (Nx.scalar f64 1.);
    Nx.sum (fst (fold step ~init:c xs))
  in
  let total f = snd (Total.collect t ~zero:(Nx.scalar f64 0.) f) in
  let c = vec [| 0.5; 2. |] in
  let check msg g =
    equal ~msg bits_t
      (bits (total (fun () -> ignore (g folded))))
      (bits (total (fun () -> ignore (g scanned))))
  in
  check "eager" (fun f -> f c);
  check "jvp" (fun f -> Rune.jvp' f c c);
  check "grad" (fun f -> Rune.grad' f c);
  (* A collect inside a map: each lane's total, chained from its own running
     value. *)
  let lanes f =
    Rune.vmap'
      (fun c ->
        snd
          (Total.collect t ~zero:(Nx.scalar f64 0.) (fun () ->
               Total.add t (Nx.sum c);
               ignore (f c))))
      (Nx.create f64 [| 2; 2 |] [| 1e16; 1.; -1e16; 2. |])
  in
  equal ~msg:"inside a map" bits_t (bits (lanes folded)) (bits (lanes scanned))

(* A total inside a differentiated function has the derivative of the additions
   a loop makes, as of those written out; one outside it receives their primals,
   from an iterate too. *)
let test_totals_inside () =
  let t = Total.make () in
  let xs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let step x c r =
    Total.add t (Nx.sum (Nx.mul x r));
    (Nx.add c r, ())
  in
  let inside loop x =
    snd
      (Total.collect t ~zero:(Nx.scalar f64 0.) (fun () ->
           Total.add t (Nx.sum (Nx.sin x));
           ignore (loop x)))
  in
  let scanned x = Rune.scan T.tensor T.tensor T.unit ~f:(step x) ~init:x xs in
  let folded x = fold (step x) ~init:x xs in
  let x = vec [| 0.5; -1. |] and v = vec [| 1.; 2. |] in
  equal ~msg:"grad" bits_t
    (bits (Rune.grad' (inside folded) x))
    (bits (Rune.grad' (inside scanned) x));
  equal ~msg:"jvp" bits_t
    (bits (snd (Rune.jvp' (inside folded) x v)))
    (bits (snd (Rune.jvp' (inside scanned) x v)));
  let remat x =
    Rune.remat
      T.(tensor @-> returns tensor)
      (fun x ->
        Total.add t (Nx.sum (Nx.mul x x));
        x)
      x
  in
  let written x =
    Total.add t (Nx.sum (Nx.mul x x));
    x
  in
  equal ~msg:"grad of a remat" bits_t
    (bits (Rune.grad' (inside written) x))
    (bits (Rune.grad' (inside remat) x));
  (* An iterate's chain reaches a collect outside the differentiation as its
     primal. *)
  let iterated x =
    Rune.iterate' ~max:5
      ~until:(fun c -> Nx.greater (Nx.max c) (Nx.scalar f64 4.))
      ~f:(fun c ->
        Total.add t (Nx.sum c);
        Nx.mul_s c 2.)
      x
  in
  let (), total =
    Total.collect t ~zero:(Nx.scalar f64 0.) (fun () ->
        ignore (Rune.jvp' iterated (vec [| 1. |]) (vec [| 1. |])))
  in
  equal ~msg:"an iterate under jvp" bits_t
    (bits (Nx.scalar f64 7.))
    (bits total)

(* A scan whose step runs an iterate over a tracked value is differentiated and
   mapped as its fold. *)
let test_iterate_in_step () =
  let xs = Nx.create f64 [| 2; 1 |] [| 3.; 0.25 |] in
  let step c r =
    let c =
      Rune.iterate' ~max:10
        ~until:(fun c -> Nx.less (Nx.max c) (Nx.scalar f64 1.))
        ~f:(fun c -> Nx.mul_s c 0.5)
        (Nx.add c r)
    in
    (c, ())
  in
  let scanned x = fst (Rune.scan T.tensor T.tensor T.unit ~f:step ~init:x xs) in
  let folded x = fst (fold step ~init:x xs) in
  let x = vec [| 0.5 |] in
  equal ~msg:"jvp" bits_t
    (bits (snd (Rune.jvp' folded x x)))
    (bits (snd (Rune.jvp' scanned x x)));
  equal ~msg:"grad" bits_t
    (bits (Rune.grad' (fun x -> Nx.sum (folded x)) x))
    (bits (Rune.grad' (fun x -> Nx.sum (scanned x)) x));
  (* Under a map, an iterate whose stop the lanes share: three halvings. *)
  let step c r =
    let c, _ =
      Rune.iterate
        T.(pair tensor tensor)
        ~max:10
        ~until:(fun (_, k) -> Nx.greater_equal k (Nx.scalar f64 3.))
        ~f:(fun (c, k) -> (Nx.mul_s c 0.5, Nx.add_s k 1.))
        (Nx.add c r, Nx.scalar f64 0.)
    in
    (c, ())
  in
  let scanned x = fst (Rune.scan T.tensor T.tensor T.unit ~f:step ~init:x xs) in
  let folded x = fst (fold step ~init:x xs) in
  let lanes = Nx.create f64 [| 2; 1 |] [| 0.5; 2. |] in
  equal ~msg:"vmap" bits_t
    (bits
       (Nx.stack [ folded (Nx.get [ 0 ] lanes); folded (Nx.get [ 1 ] lanes) ]))
    (bits (Rune.vmap' scanned lanes))

(* Memory *)

let device = Nx_support.memory 1

module Mem = (val Nx.devices [ device ])

let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create device 0);
    Rig.free_cache device
  done;
  Rig.allocated device

(* The bytes [keep] holds after a run beyond those held before it, after one
   uncounted run; what it kept is dropped before the next measurement. *)
let held_after kept keep =
  keep ();
  let before = settled () in
  keep ();
  let held = settled () - before in
  ignore (Sys.opaque_identity !kept);
  kept := [];
  held

(* Reverse mode keeps per step the loads of the tangent's maps, stacked, but
   reads a row or a capture where it lies. *)
let test_residuals () =
  let n = 1024 and rows = 8 in
  let f32 = Nx.float32 in
  let x = Nx.place Mem.on (Nx.ones f32 [| n |]) in
  let w = Nx.place Mem.on (Nx.full f32 [| n |] 0.5) in
  let xs = Nx.place Mem.on (Nx.full f32 [| rows; n |] 0.25) in
  let kept = ref [] in
  let held step =
    held_after kept (fun () ->
        kept :=
          Rune.vjp'
            (fun c ->
              fst (Rune.scan T.tensor T.tensor T.unit ~f:step ~init:c xs))
            x
          :: !kept)
  in
  let bytes k = 4 * n * k in
  (* The result, and each step's carry, which the sine's tangent reads. *)
  equal ~msg:"a step's carry" int
    (bytes (1 + rows))
    (held (fun c _ -> (Nx.sin c, ())));
  equal ~msg:"a capture" int (bytes 1) (held (fun c _ -> (Nx.mul c w, ())));
  equal ~msg:"a row" int (bytes 1) (held (fun c r -> (Nx.mul c r, ())));
  (* Held to the end, so that no measurement sees them collected. *)
  ignore (Sys.opaque_identity (x, w, xs))

(* A single contribution keeps its sign through a capture and a carry. *)
let test_signs () =
  let w = vec [| -0.; 1. |] in
  let g =
    Rune.grad'
      (fun w ->
        Nx.sum
          (fst
             (Rune.scan T.tensor T.tensor T.unit
                ~f:(fun c r -> (Nx.mul c (Nx.mul w r), ()))
                ~init:w
                (Nx.create f64 [| 1; 2 |] [| 1.; -0. |]))))
      w
  in
  (* [w r w] at row [1; -0]: [2 w r]: [-0.] and [-0.]. *)
  equal bits_t (bits (vec [| -0.; -0. |])) (bits g)

let () =
  exit
    (run "rune2 loop rules"
       [
         group "scan"
           [
             prop "jvp of scan is jvp of its fold" cases law_jvp;
             prop ~examples:grad_examples "grad of scan is grad of its fold"
               cases law_grad;
             prop "vmap of scan is the fold per lane" map_cases law_vmap;
             prop "compositions are the fold's" cases law_compose;
           ];
         group "interface"
           [
             test "a carry that gains a tangent and lanes" test_gains;
             test "additions under transformations" test_totals;
             test "a total inside a differentiation" test_totals_inside;
             test "a step that runs an iterate" test_iterate_in_step;
             test "a single contribution keeps its sign" test_signs;
           ];
         group "memory" [ test "what reverse mode keeps" test_residuals ];
       ])
