(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* SPK files: states of bodies as CSPICE reads them from the same files, the
   errors of an instant, a body or a segment the file does not serve, and the
   laws of the states. *)

open Windtrap
open Ymir

let km = Unit.(kilo metre)
let km_s = Unit.(kilo metre / second)
let seconds x = Quantity.v Unit.second x
let scalar x = Nx.scalar Nx.float64 x

let ok = function Ok x -> x | Error e -> failwith e
let excerpt = lazy (ok (Ymir_spk.read ~at:Nx.Host.on "golden/de440s-2024.bsp"))
let types = lazy (ok (Ymir_spk.read ~at:Nx.Host.on "golden/types.bsp"))
let types_big = lazy (ok (Ymir_spk.read ~at:Nx.Host.on "golden/types-big.bsp"))
let many = lazy (ok (Ymir_spk.read ~at:Nx.Host.on "golden/many.bsp"))

let tdb day second =
  Time.v Time.tdb
    ~day:(Nx.scalar Nx.int64 (Int64.of_int day))
    ~second:(seconds (scalar second))

let state file ~target ~center t =
  let p, v = Ymir_spk.state (Lazy.force file) ~target ~center t in
  (Nx.to_array (Quantity.value km p), Nx.to_array (Quantity.value km_s v))

(* [close tol expected actual] holds when each component is within [tol]
   times the vector's norm. *)
let close tol expected actual =
  let norm = sqrt (Array.fold_left (fun a x -> a +. (x *. x)) 0. expected) in
  let gap =
    Array.fold_left max 0.
      (Array.map2 (fun a b -> Float.abs (a -. b)) expected actual)
  in
  less float_exact ~than:(tol *. norm) gap

(* CSPICE's states *)

(* CSPICE evaluates a record by Clenshaw's recurrence and ymir by the
   Chebyshev polynomials themselves; the two round apart by a few ulps of
   the largest term. *)
let rel = 1e-15

let cspice name file rows =
  let id (d, s, _) = Printf.sprintf "TDB MJD %d + %.6f s" d s in
  cases name ~id rows (fun (d, s, states) ->
      List.iter
        (fun (target, center, st) ->
          let p, v = state file ~target ~center (tdb d s) in
          close rel (Array.sub st 0 3) p;
          close rel (Array.sub st 3 3) v)
        states)

let spice =
  group "CSPICE's states"
    [
      cspice "DE440s: type 2 segments, chained through the bodies' centres"
        excerpt Spk_reference.excerpt;
      cspice "type 3 segments, the later one where it covers"
        types Spk_reference.types;
      cspice
        "two summary records; a later segment of another centre serves the \
         Moon where it covers"
        many Spk_reference.many;
      cases "a BIG-IEEE file reads as its LTL-IEEE twin, bit for bit"
        ~id:(fun (d, s, _) -> Printf.sprintf "TDB MJD %d + %.6f s" d s)
        Spk_reference.types (fun (d, s, _) ->
          equal
            (pair (array float_exact) (array float_exact))
            (state types ~target:301 ~center:3 (tdb d s))
            (state types_big ~target:301 ~center:3 (tdb d s)));
    ]

(* Errors *)

let failures =
  group "Errors"
    [
      test "an instant outside a body's segments raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"outside") (fun () ->
              state excerpt ~target:301 ~center:399 (tdb 62502 0.)));
      test "a body no segment names raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"no segment of body 499")
            (fun () -> state excerpt ~target:499 ~center:0 (tdb 60500 0.)));
      test "a segment of a type ymir does not read raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"SPK type 9") (fun () ->
              state types ~target:1000 ~center:399 (tdb 60500 0.)));
      test "bytes that are no SPK file are an error" (fun () ->
          equal (result unit string)
            (Error "x: 4 bytes, shorter than a DAF file record")
            (Result.map ignore
               (Ymir_spk.of_string ~at:Nx.Host.on ~name:"x" "abcd")));
      test "a file cut short is an error" (fun () ->
          let s =
            In_channel.with_open_bin "golden/types.bsp" In_channel.input_all
          in
          match
            Ymir_spk.of_string ~at:Nx.Host.on ~name:"cut" (String.sub s 0 3000)
          with
          | Ok _ -> failwith "a cut file read"
          | Error e -> equal bool true (String.starts_with ~prefix:"cut: " e));
      test "segments lists the file's, in order, with their instants"
        (fun () ->
          let segs = Ymir_spk.segments (Lazy.force types) in
          equal (list (pair int int))
            [ (301, 3); (301, 3); (1000, 399) ]
            (List.map
               (fun (s : Ymir_spk.segment) -> (s.target, s.center))
               segs);
          (* The first covers 760622400 s past J2000: MJD 60348, 0h. *)
          let s = List.hd segs in
          equal (pair int float_exact) (60348, 0.)
            ( Int64.to_int (Nx.item [] (Time.day s.start)),
              Nx.item [] (Quantity.value Unit.second (Time.second s.start)) ));
      test "a NaN instant gives NaN" (fun () ->
          let p, v =
            state excerpt ~target:301 ~center:399 (tdb 60500 Float.nan)
          in
          equal (array bool) [| true; true; true; true; true; true |]
            (Array.map Float.is_nan (Array.append p v)));
      test "a segment on a frame other than J2000 raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"is on frame 17") (fun () ->
              state many ~target:1001 ~center:399 (tdb 60310 0.)));
    ]

(* Corrupt files *)

let little = In_channel.with_open_bin "golden/types.bsp" In_channel.input_all

(* [corrupt f] is [little] with [f] applied to its bytes, read. *)
let corrupt f =
  let b = Bytes.of_string little in
  f b;
  Ymir_spk.of_string ~at:Nx.Host.on ~name:"bad" (Bytes.to_string b)

let fward = Int32.to_int (String.get_int32_le little 76)
let summary = (fward - 1) * 1024

(* The first segment's last address: its directory ends there. *)
let last_word =
  Int32.to_int (String.get_int32_le little (summary + 24 + 16 + 20))

(* [contains sub e] is whether [sub] occurs in [e]. *)
let contains sub e =
  let n = String.length sub and m = String.length e in
  let rec at i = i + n <= m && (String.sub e i n = sub || at (i + 1)) in
  at 0

let error_with sub = function
  | Ok _ -> failwith "a corrupt file read"
  | Error e ->
      equal bool true (String.starts_with ~prefix:"bad: " e && contains sub e)

let set_double b o x = Bytes.set_int64_le b o (Int64.bits_of_float x)

let corrupted =
  group "Corrupt files"
    [
      test "ND and NI other than 2 and 6" (fun () ->
          error_with "ND = 3"
            (corrupt (fun b -> Bytes.set_int32_le b 8 3l)));
      test "a FWARD outside the file" (fun () ->
          error_with "FWARD, 9999"
            (corrupt (fun b -> Bytes.set_int32_le b 76 9999l)));
      test "a NEXT that names its own record" (fun () ->
          error_with "comes twice"
            (corrupt (fun b -> set_double b summary (float_of_int fward))));
      test "a NEXT outside the file" (fun () ->
          error_with "NEXT, 1e+06"
            (corrupt (fun b -> set_double b summary 1e6)));
      test "an NSUM above 25" (fun () ->
          error_with "NSUM, 26"
            (corrupt (fun b -> set_double b (summary + 16) 26.)));
      test "an INTLEN of 0" (fun () ->
          error_with "INTLEN 0"
            (corrupt (fun b -> set_double b ((last_word - 3) * 8) 0.)));
      test "an epoch beyond 2^62 s" (fun () ->
          error_with "within 2^62 s"
            (corrupt (fun b -> set_double b (summary + 24) 1e300)));
      test "a file read inside a traced function" (fun () ->
          let f x =
            match Ymir_spk.of_string ~at:Nx.Host.on ~name:"t" little with
            | Ok _ -> x
            | Error e -> failwith e
          in
          equal float_exact 1. (Nx.item [] (Rune.jit' f (scalar 1.))));
      test "an INTLEN that is NaN" (fun () ->
          error_with "INTLEN nan"
            (corrupt (fun b -> set_double b ((last_word - 3) * 8) Float.nan)));
    ]

(* Laws *)

(* TDB instants within the excerpt's segments: 2024-01-01 to 2025-12-31. *)
let instants =
  let open Gen in
  pair (int_range 60310 61040) (float_range 0. 86399.)

let pairs =
  Gen.of_list
    ~pp:(fun ppf (t, c) -> Format.fprintf ppf "(%d, %d)" t c)
    [ (301, 399); (399, 0); (10, 0); (5, 3); (3, 301); (199, 10) ]

let laws =
  group "Laws"
    [
      prop "the velocity is the position's derivative within 1e-12 relative"
        Gen.(pair instants pairs)
        (fun ((d, s), (target, center)) ->
          let f s =
            Quantity.value km
              (Ymir_spk.position (Lazy.force excerpt) ~target ~center
                 (Time.v Time.tdb
                    ~day:(Nx.scalar Nx.int64 (Int64.of_int d))
                    ~second:(seconds s)))
          in
          let _, v = state excerpt ~target ~center (tdb d s) in
          let jv = Nx.to_array (snd (Rune.jvp' f (scalar s) (scalar 1.))) in
          close 1e-12 v jv);
      prop "the state of b from a is minus a's from b"
        Gen.(pair instants pairs)
        (fun ((d, s), (target, center)) ->
          let p, v = state excerpt ~target ~center (tdb d s) in
          let p', v' = state excerpt ~target:center ~center:target (tdb d s) in
          equal (array float_exact) p (Array.map Float.neg p');
          equal (array float_exact) v (Array.map Float.neg v'));
      prop "the Moon from the Earth is its chain through their barycentre"
        instants (fun (d, s) ->
          let p, _ = state excerpt ~target:301 ~center:399 (tdb d s) in
          let pm, _ = state excerpt ~target:301 ~center:3 (tdb d s) in
          let pe, _ = state excerpt ~target:399 ~center:3 (tdb d s) in
          equal (array float_exact) (Array.map2 ( -. ) pm pe) p);
      prop "a batch of instants is each instant's, bit for bit"
        Gen.(list ~size:(int_range 1 8) instants)
        (fun xs ->
          let n = List.length xs in
          let t =
            Time.v Time.tdb
              ~day:
                (Nx.create Nx.int64 [| n |]
                   (Array.of_list (List.map (fun (d, _) -> Int64.of_int d) xs)))
              ~second:
                (seconds
                   (Nx.create Nx.float64 [| n |]
                      (Array.of_list (List.map snd xs))))
          in
          let batch, _ = state excerpt ~target:301 ~center:399 t in
          let each =
            List.map
              (fun (d, s) ->
                fst (state excerpt ~target:301 ~center:399 (tdb d s)))
              xs
          in
          equal (array float_exact) (Array.concat each) batch);
      test "the Moon's state compiles to the eager one within 1e-15 relative"
        (fun () ->
          let second = Nx.linspace Nx.float64 0. 86399. 8 in
          let f s =
            Quantity.value km
              (Ymir_spk.position (Lazy.force excerpt) ~target:301 ~center:10
                 (Time.v Time.tdb ~day:(Nx.scalar Nx.int64 60700L)
                    ~second:(seconds s)))
          in
          let eager = Nx.to_array (f second)
          and compiled = Nx.to_array (Rune.jit' f second) in
          close 1e-15 eager compiled);
    ]

let () = exit (run "Spk" [ spice; failures; corrupted; laws ])
