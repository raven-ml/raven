(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Vectorizing maps, one row of nx's operations at a time: a map of the row is
   its loop over the rows of its batched operands, whichever operands are
   batched, at any batch length, through a moved axis and inside another map;
   and each row's Jacobians, which batch its tangent and its pullback. *)

open Windtrap
module Rune = Rune_next.Rune

(* How a batch is drawn: which operands carry it, how many rows, whether the
   batch axis is a moved one, whether the map sits inside another. *)
type batching = {
  which : bool list;
  length : int;
  moved : bool;
  nested : bool;
  seed : int;
}

let pp_batching ppf b =
  Format.fprintf ppf "batched %s, %d rows%s%s, seed %d"
    (String.concat "" (List.map (fun t -> if t then "x" else "-") b.which))
    b.length
    (if b.moved then ", through a moved axis" else "")
    (if b.nested then ", inside another map" else "")
    b.seed

let draw ~length ~moved ~nested (c : Case.t) =
  let open Gen in
  let* (Case.Instance i as inst) = c.smooth Case.float64 in
  let n = List.length i.x in
  let* which =
    map
      (fun bits ->
        if List.exists Fun.id bits then bits else true :: List.tl bits)
      (list ~size:(constant n) bool)
  in
  let+ length = length and+ seed = int_range 0 1_000_000 in
  (inst, { which; length; moved; nested; seed })

let draw ?(length = Gen.int_range 1 3) ?(moved = false) ?(nested = false) c =
  Gen.with_pp
    (fun ppf (i, b) ->
      Format.fprintf ppf "%a@ %a" Case.pp_instance i pp_batching b)
    (draw ~length ~moved ~nested c)

let rec merge which all xs =
  match (which, all, xs) with
  | true :: which, _ :: all, x :: xs -> x :: merge which all xs
  | false :: which, a :: all, xs -> a :: merge which all xs
  | [], [], [] -> []
  | _ -> invalid_arg "merge"

let take which xs = List.filteri (fun i _ -> List.nth which i) xs

(* [rows seed k x] is the [k]th row of a batch around [x]: [x] moved a little
   along a direction, so that each row stays in the row's domain. *)
let row seed k x =
  let v =
    Reference.direction (Random.State.make [| seed |]) Nx.Ptree.tensor x
  in
  Nx.add x
    (Nx.mul v
       (Nx.full (Nx.dtype v) [||]
          (Nx_dtype.of_float (Nx.dtype v) (1e-3 *. float_of_int k))))

(* The rows whose elements each come from their own inputs compare bit for bit;
   those that accumulate compare to rounding. *)
let accumulates : Row.t -> bool = function
  | Reduce _ | Scan _ | Scatter | Fold | Matmul | Fft | Rfft | Irfft | Cholesky
  | Qr | Lu | Svd | Eig | Eigh | Solve_triangular ->
      true
  | Unary _ | Binary _ | Compare _ | Where | Arg_reduce _ | Sort | Argsort | Pad
  | Cat | Cast | Bitcast | Threefry | Gather | Update | Unfold | Contiguous
  | Move _ | Place | Read ->
      false

let compare_rows (c : Case.t) expected actual =
  if accumulates c.row then
    let rel = match c.row with Eig -> 1e-5 | _ -> 1e-10 in
    List.iter2
      (fun e a ->
        equal ~msg:"dtype and shape"
          (pair string (array int))
          (Nx_dtype.to_string (Nx.dtype e), Nx.shape e)
          (Nx_dtype.to_string (Nx.dtype a), Nx.shape a);
        equal
          (Reference.close ~rel ~floor:rel ())
          [ Reference.complexes e ]
          [ Reference.complexes a ])
      expected actual
  else List.iter2 (equal (Reference.exact ())) expected actual

(* [stack ~moved rows] is the batch of [rows] on a new axis 0, or, for rows of
   at least one axis, on axis 1 moved to 0: a view whose batch axis is not its
   leading one in memory. *)
let stack ~moved rows =
  match rows with
  | r :: _ when moved && Nx.ndim r >= 1 ->
      Nx.moveaxis 1 0 (Nx.stack ~axis:1 rows)
  | _ -> Nx.stack rows

let is_the_loop (c : Case.t) (Case.Instance i, b) =
  let mapped bs = i.f (merge b.which i.x bs) in
  let sig_ = Nx.Ptree.(list tensor @-> returns (list tensor)) in
  let leaves = take b.which i.x in
  let row_operands k = List.mapi (fun j x -> row (b.seed + j) k x) leaves in
  if b.length = 0 then begin
    let empty =
      List.map
        (fun x -> Nx.zeros (Nx.dtype x) (Array.append [| 0 |] (Nx.shape x)))
        leaves
    in
    let ys = Rune.vmap sig_ mapped empty in
    List.iter2
      (fun y e ->
        equal ~msg:"an empty batch" (array int)
          (Array.append [| 0 |] (Nx.shape e))
          (Nx.shape y))
      ys (i.f i.x)
  end
  else if b.nested then begin
    let inner = 2 in
    let at k l =
      List.mapi (fun j x -> row (b.seed + j) ((k * inner) + l) x) leaves
    in
    let batch =
      List.mapi
        (fun j _ ->
          stack ~moved:b.moved
            (List.init b.length (fun k ->
                 Nx.stack (List.init inner (fun l -> List.nth (at k l) j)))))
        leaves
    in
    let ys = Rune.vmap sig_ (Rune.vmap sig_ mapped) batch in
    let expected =
      List.mapi
        (fun o _ ->
          Nx.stack
            (List.init b.length (fun k ->
                 Nx.stack
                   (List.init inner (fun l -> List.nth (mapped (at k l)) o)))))
        ys
    in
    compare_rows c expected ys
  end
  else begin
    let batch =
      List.mapi
        (fun j _ ->
          stack ~moved:b.moved
            (List.init b.length (fun k -> List.nth (row_operands k) j)))
        leaves
    in
    let ys = Rune.vmap sig_ mapped batch in
    let expected =
      List.mapi
        (fun o _ ->
          Nx.stack
            (List.init b.length (fun k -> List.nth (mapped (row_operands k)) o)))
        ys
    in
    compare_rows c expected ys
  end

(* Jacobians: [jacfwd'] is the loop of [jvp] over the input's basis, and
   [jacrev'] the loop of the pullback over the output's basis, conjugated, of
   the row in its first operand. *)

let basis x j =
  let n = Nx.numel x in
  Nx.cast (Nx.dtype x)
    (Nx.create Nx.float64 (Nx.shape x)
       (Array.init n (fun k -> if k = j then 1. else 0.)))

let jacobians (c : Case.t) (Case.Instance i, _) =
  let rel = match c.row with Eig -> 1e-4 | _ -> 1e-10 in
  let close e a =
    equal ~msg:"dtype and shape"
      (pair string (array int))
      (Nx_dtype.to_string (Nx.dtype e), Nx.shape e)
      (Nx_dtype.to_string (Nx.dtype a), Nx.shape a);
    equal
      (Reference.close ~rel ~floor:rel ())
      [ Reference.complexes e ]
      [ Reference.complexes a ]
  in
  let x = List.hd i.x and rest = List.tl i.x in
  let f x = List.hd (i.f (x :: rest)) in
  let y = f x in
  let shape = Array.append (Nx.shape y) (Nx.shape x) in
  if Nx.numel x = 0 || Nx.numel y = 0 then begin
    equal ~msg:"jacfwd's shape" (array int) shape (Nx.shape (Rune.jacfwd' f x));
    equal ~msg:"jacrev's shape" (array int) shape (Nx.shape (Rune.jacrev' f x))
  end
  else begin
    let columns =
      List.init (Nx.numel x) (fun j -> snd (Rune.jvp' f x (basis x j)))
    in
    close (Nx.reshape shape (Nx.stack ~axis:(-1) columns)) (Rune.jacfwd' f x);
    let _, pullback = Rune.vjp' f x in
    let rows =
      List.init (Nx.numel y) (fun k -> Nx.conjugate (pullback (basis y k)))
    in
    close (Nx.reshape shape (Nx.stack rows)) (Rune.jacrev' f x)
  end

let rows ~count =
  List.map
    (fun (c : Case.t) ->
      group (Row.name c.row)
        ([
           prop ~count "a map is its loop" (draw c) (is_the_loop c);
           prop ~count "a map through a moved axis is its loop"
             (draw ~moved:true c) (is_the_loop c);
           prop ~count "a map inside another map is its loop"
             (draw ~nested:true c) (is_the_loop c);
           prop ~count "a map over no row gives no row"
             (draw ~length:(Gen.constant 0) c)
             (is_the_loop c);
         ]
        @
        match c.kind with
        | Case.Tangent ->
            [
              prop ~count
                "its Jacobians are the loops of its tangent and pullback"
                (Gen.with_pp
                   (fun ppf (i, seed) ->
                     Format.fprintf ppf "%a@ seed %d" Case.pp_instance i seed)
                   (Gen.pair (c.smooth Case.float64)
                      (Gen.int_range 0 1_000_000)))
                (jacobians c);
            ]
        | Case.Plain | Case.Integer -> []))
    Case.all

let () =
  exit
    (run "rune.next batching"
       (rows ~count:10
       @ [
           group "edges" Batching_edges.tests;
           group ~tags:[ "slow" ] "swept" (rows ~count:300);
         ]))
