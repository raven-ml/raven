(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.associative_scan: over int32, where wrapping arithmetic associates
   exactly, it is the sequential fold, along any axis of every array of a
   structure; over floats a prefix's scan is the scan's prefix, bit for bit; it
   calls its function at most 2⌈log₂ n⌉ times. *)

open Windtrap
open Nx_support
module P = Nx.Ptree

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f
let numel s = Array.fold_left ( * ) 1 s

let ceil_log2 n =
  let rec go k p = if p >= n then k else go (k + 1) (2 * p) in
  go 0 1

(* Affine maps [y -> a y + b] of int32, composed first then second. *)
let compose (a1, b1) (a2, b2) = Nx.(mul a1 a2, add (mul a2 b1) b2)
let affine = P.(pair tensor tensor)

let cases =
  let open Gen in
  (let* s =
     array ~size:(int_range 1 3)
       (frequency [ (1, constant 0); (1, constant 1); (4, int_range 2 9) ])
   in
   let r = Array.length s in
   let* axis = int_range (-r) (r - 1) in
   let word = map Int32.of_int (int_range (-3) 3) in
   let+ a = array ~size:(constant (numel s)) word
   and+ b = array ~size:(constant (numel s)) word in
   (s, axis, a, b))
  |> with_pp (fun ppf (s, axis, _, _) ->
      Format.fprintf ppf "%a along %d" pp_ints s axis)

(* The sequential scan of the maps along [axis], at each index. *)
let reference s axis a b =
  let r = Array.length s in
  let ax = if axis < 0 then axis + r else axis in
  let strides = Array.make r 1 in
  for i = r - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * s.(i + 1)
  done;
  let ya = Array.copy a and yb = Array.copy b in
  Array.iteri
    (fun j _ ->
      let along = j / strides.(ax) mod s.(ax) in
      if along > 0 then begin
        let p = j - strides.(ax) in
        ya.(j) <- Int32.mul ya.(p) a.(j);
        yb.(j) <- Int32.add (Int32.mul a.(j) yb.(p)) b.(j)
      end)
    a;
  (ya, yb)

let law (s, axis, a, b) =
  let r = Array.length s in
  let n = s.(if axis < 0 then axis + r else axis) in
  cover "no element along the axis" (n = 0);
  cover "one element" (n = 1);
  cover "an odd extent past 2" (n > 2 && n mod 2 = 1);
  cover "an even extent" (n >= 2 && n mod 2 = 0);
  cover "a negative axis" (axis < 0);
  let calls = ref 0 in
  let f x y =
    incr calls;
    compose x y
  in
  let x = (Nx.create Nx.int32 s a, Nx.create Nx.int32 s b) in
  let ya, yb = Nx.associative_scan ~axis affine f x in
  let ea, eb = reference s axis a b in
  equal ~msg:"a" (array int32) ea (Nx.to_array ya);
  equal ~msg:"b" (array int32) eb (Nx.to_array yb);
  at_most ~msg:"calls of f" int ~than:(2 * ceil_log2 n) !calls

let prefixes =
  let open Gen in
  (let* n = int_range 0 40 in
   let* k = int_range 0 n in
   let+ xs = array ~size:(constant n) (float_range (-1e3) 1e3) in
   (k, xs))
  |> with_pp (fun ppf (k, xs) ->
      Format.fprintf ppf "%d of %d" k (Array.length xs))

let law_prefix (k, xs) =
  let n = Array.length xs in
  cover "a proper prefix" (k > 0 && k < n);
  let scan xs =
    Nx.to_array
      (Nx.associative_scan P.tensor Nx.add
         (Nx.create Nx.float32 [| Array.length xs |] xs))
  in
  let whole = scan xs and part = scan (Array.sub xs 0 k) in
  equal (array float_exact) (Array.sub whole 0 k) part

(* A structure of two dtypes whose float32 sums are a fixed array. *)
type ('a, 'd) mixed = { counts : 'a; sums : 'd Nx.float32_t }

module Mixed = struct
  type ('a, 'd) t = ('a, 'd) mixed

  let walk c x =
    let open P.Walk in
    let counts = field c "counts" leaf x.counts in
    let sums = field c "sums" tensor x.sums in
    { counts; sums }
end

let m = Nx_support.memory

module S = (val Nx.devices [ m 0 ])

let mixed_cases =
  let open Gen in
  (let* n = int_range 0 17 in
   let+ counts = array ~size:(constant (2 * n)) (int_range (-9) 9)
   and+ sums = array ~size:(constant n) (int_range (-9) 9) in
   (n, counts, sums))
  |> with_pp (fun ppf (n, _, _) -> Format.fprintf ppf "%d elements" n)

let running n width xs =
  let ys = Array.copy xs in
  for i = 1 to n - 1 do
    for j = 0 to width - 1 do
      ys.((i * width) + j) <- ys.((i * width) + j) + ys.(((i - 1) * width) + j)
    done
  done;
  ys

(* Each array summed along axis 0, on the host and on a memory set: an int32
   parameter of shape [n; 2] and a float32 fixed array of shape [n]. *)
let law_mixed (n, counts, sums) =
  cover "an odd extent past 2" (n > 2 && n mod 2 = 1);
  cover "an even extent" (n >= 2 && n mod 2 = 0);
  let x =
    {
      counts = Nx.create Nx.int32 [| n; 2 |] (Array.map Int32.of_int counts);
      sums = Nx.create Nx.float32 [| n |] (Array.map float_of_int sums);
    }
  in
  let f a b =
    { counts = Nx.add a.counts b.counts; sums = Nx.add a.sums b.sums }
  in
  let check msg y =
    equal ~msg:(msg ^ ": counts") (array int32)
      (Array.map Int32.of_int (running n 2 counts))
      (Nx.to_array y.counts);
    equal ~msg:(msg ^ ": sums") (array float_exact)
      (Array.map float_of_int (running n 1 sums))
      (Nx.to_array y.sums)
  in
  check "host" (Nx.associative_scan (P.instantiate (module Mixed)) f x);
  let on_s = P.place (module Mixed) S.on x in
  check "memory set" (Nx.associative_scan (P.instantiate (module Mixed)) f on_s)

let scans =
  group "associative_scan"
    [
      prop "over int32 maps it is the sequential scan along the axis" cases law;
      prop "a prefix's scan is the scan's prefix, bit for bit" prefixes
        law_prefix;
      test "each array counts its axis from its own last" (fun () ->
          let a = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |]
          and b = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 1l; 1l; 2l; 2l; 2l |] in
          let ys =
            Nx.associative_scan ~axis:(-1)
              P.(list tensor)
              (List.map2 Nx.add) [ a; b ]
          in
          equal
            (list (array int32))
            [ [| 1l; 3l; 6l |]; [| 1l; 2l; 3l; 2l; 4l; 6l |] ]
            (List.map Nx.to_array ys));
      prop
        "a structure of two dtypes and a fixed array scans every array, on the \
         host and on another set"
        mixed_cases law_mixed;
      test "a structure without arrays is itself" (fun () ->
          Nx.associative_scan P.unit (fun () () -> ()) ());
      test
        "arrays of two extents, a missing axis, and an f that reshapes or \
         casts raise" (fun () ->
          let a = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |]
          and b = Nx.create Nx.int32 [| 4 |] [| 1l; 2l; 3l; 4l |] in
          invalid ~by:"Nx.associative_scan" (fun () ->
              Nx.associative_scan affine compose (a, b));
          invalid ~by:"Nx.associative_scan" (fun () ->
              Nx.associative_scan ~axis:1 affine compose (a, a));
          invalid ~by:"Nx.associative_scan" (fun () ->
              Nx.associative_scan P.tensor
                (fun x _ -> Nx.reshape [| 1; -1 |] x)
                b);
          let packed = P.v (fun c (Nx.P a) -> Nx.P (P.Walk.tensor c a)) in
          invalid ~by:"Nx.associative_scan" (fun () ->
              Nx.associative_scan packed
                (fun (Nx.P a) _ -> Nx.P (Nx.cast Nx.float64 a))
                (Nx.P b)));
    ]

let () = exit (run "nx scan" [ scans ])
