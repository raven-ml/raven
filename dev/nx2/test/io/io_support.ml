(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Files, values compared bit for bit, and values drawn to break a format: every
   dtype it holds, no element or one, strided and broadcast views, NaN with a
   payload and signalling, -0., infinities and the integers' bounds. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

(* Files *)

let fixture name = Filename.concat "support" name
let read path = In_channel.with_open_bin path In_channel.input_all

let write path s =
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc s)

let file ?(suffix = "") s =
  let path = temp_file ~suffix () in
  write path s;
  path

(* [fails ~path f] asserts that [f ()] raises [Failure] whose message starts
   with [path]. *)
let fails ?(naming = "") ~path f =
  raises_match
    (function
      | Failure m ->
          String.starts_with ~prefix:(path ^ ": ") m
          && (naming = "" || Exn.failure ~substring:naming (Failure m))
      | _ -> false)
    (fun () -> ignore (f ()))

let invalid ~by f =
  raises_match
    (Exn.invalid_arg ~substring:(by ^ ": "))
    (fun () -> ignore (f ()))

(* [loads_or_fails ~path f] asserts that [f ()] returns or raises [Failure]
   naming [path]: a corrupted file never raises anything else. *)
let loads_or_fails ~path f =
  match f () with
  | _ -> ()
  | exception Failure m when String.starts_with ~prefix:(path ^ ": ") m -> ()
  | exception e -> failf "%s raised %s" path (Printexc.to_string e)

let flip_byte s i =
  String.mapi
    (fun j c -> if i = j then Char.chr (Char.code c lxor 0xff) else c)
    s

(* Every proper prefix of [whole] fails to load, naming its file. *)
let prefixes_fail load whole =
  for n = 0 to String.length whole - 1 do
    let path = file (String.sub whole 0 n) in
    fails ~path (fun () -> load path)
  done

(* Every byte of [whole] below [upto], flipped, loads or fails. *)
let flips_load_or_fail load ?(upto = max_int) whole =
  for i = 0 to Int.min upto (String.length whole) - 1 do
    let path = file (flip_byte whole i) in
    loads_or_fails ~path (fun () -> load path)
  done

let saved suffix save x =
  let path = temp_file ~suffix () in
  save path x;
  path

(* Memory *)

(* [grown d make f] is [f (make ())] and the bytes [d] holds more after it, read
   once collected memory has returned. One uncounted run comes first. *)
let grown d make f =
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      ignore (Rig.Buffer.create d 0);
      Rig.free_cache d
    done;
    Rig.allocated d
  in
  ignore (f (make ()));
  let x = make () in
  let before = settled () in
  let y = f x in
  (y, Rig.allocated d - before)

(* Where a loaded value's elements lie: over the file's pages, which the host
   borrows, or in memory of its own. *)
type memory = Over_file | Own_memory

let memory_of (Nx.P x) =
  match Nx.Repr.array x with
  | Some a when Rig.Buffer.is_borrowed (Nx_array.buffer a) -> Over_file
  | Some _ | None -> Own_memory

let memory =
  Testable.make ~equal:( = ) ~pp:(fun ppf m ->
      Format.pp_print_string ppf
        (match m with
        | Over_file -> "over the file's pages"
        | Own_memory -> "in memory of its own"))

(* Values *)

let pp_shape = Nx.pp_shape

(* A byte-wide value's bytes, in C order. *)
let bytes (type v s) (x : (v, s, _) Nx.t) = Nx.to_array (Nx.bitcast Nx.uint8 x)

let same_value (Nx.P a) (Nx.P b) =
  D.equal (Nx.dtype a) (Nx.dtype b)
  && Nx.shape a = Nx.shape b
  && bytes a = bytes b

let pp_packed ppf (Nx.P x) = Nx.pp ppf x

(* Host values compare by dtype, shape and bytes: every NaN payload counts. *)
let packed : Nx.host Nx.packed testable =
  Testable.make ~pp:pp_packed ~equal:same_value

(* Drawn values *)

(* The values a format must keep: zeros of both signs, NaN, the infinities, the
   extremes and the least normal and subnormal magnitudes of a float format, and
   an integer format's bounds. *)
let specials (type v s) (dt : (v, s) D.t) : v list =
  match D.kind dt with
  | D.Float ->
      let f = D.float_format dt in
      [
        0.;
        -0.;
        1.;
        -1.5;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
        f.max_finite;
        -.f.max_finite;
        f.min_normal;
        f.min_normal /. 2.;
      ]
  | D.Complex ->
      [
        Complex.zero;
        { re = -0.; im = Float.nan };
        { re = Float.infinity; im = -0. };
      ]
  | D.Boolean -> [ false; true ]
  | D.Signed | D.Unsigned ->
      [ D.zero dt; D.one dt; D.min_value dt; D.max_value dt ]

(* One element of [dt] from drawn bits: any value of the format. *)
let drawn (type v s) (dt : (v, s) D.t) : v Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt 1)) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  (A.to_array (A.v dt (L.contiguous [| 1 |]) (Rig.Buffer.of_string data))).(0)

(* A special drawn by its index: [Gen.of_list] would merge [0.] and [-0.], which
   compare equal. *)
let element dt =
  let specials = Array.of_list (specials dt) in
  let special =
    Gen.map (Array.get specials) (Gen.int_range 0 (Array.length specials - 1))
  in
  Gen.frequency [ (1, special); (2, drawn dt) ]

(* How a saved value lies in memory. *)
type view = Plain | Transposed | Flipped | Broadcast

let pp_view ppf v =
  Format.pp_print_string ppf
    (match v with
    | Plain -> "plain"
    | Transposed -> "transposed"
    | Flipped -> "flipped"
    | Broadcast -> "broadcast")

type case = Case : ('v, 's) D.t * int array * 'v array * view -> case

(* What a case holds. Each draws about as often as the others, so that every
   seed reaches the cases a round trip must cover: a value of no element, of
   one, and values of two or more with a NaN, a [-0.] or both of an integer
   type's bounds planted among drawn elements. *)
type kind = Empty | Single | Nan | Negative_zero | Bounds | Many

let supports kind (D.Any dt) =
  match kind with
  | Empty | Single | Many -> true
  | Nan -> (
      match D.kind dt with
      | D.Float -> (D.float_format dt).nans
      | D.Complex -> true
      | D.Signed | D.Unsigned | D.Boolean -> false)
  | Negative_zero -> D.is D.Float dt
  | Bounds -> D.is D.Signed dt || D.is D.Unsigned dt

(* [plant kind dt vs i] puts [kind]'s special value into [vs] at [i]. *)
let plant (type v s) kind (dt : (v, s) D.t) (vs : v array) i =
  let n = Array.length vs in
  match (kind, D.kind dt) with
  | Nan, D.Float -> vs.(i) <- Float.nan
  | Nan, D.Complex -> vs.(i) <- { Complex.re = 0.; im = Float.nan }
  | Negative_zero, D.Float -> vs.(i) <- -0.
  | Bounds, (D.Signed | D.Unsigned) ->
      vs.(0) <- D.min_value dt;
      vs.(n - 1) <- D.max_value dt
  | _ -> ()

let extent = Gen.int_range 1 4

(* A shape of [kind], of at most [rank] axes. *)
let shape ~rank = function
  | Empty ->
      let open Gen in
      let* s = array ~size:(int_range 1 rank) extent in
      let+ i = int_range 0 (Array.length s - 1) in
      s.(i) <- 0;
      s
  | Single -> Gen.array ~size:(Gen.int_range 0 rank) (Gen.constant 1)
  | Nan | Negative_zero | Bounds | Many ->
      let open Gen in
      let+ s = array ~size:(int_range 1 rank) extent in
      if Array.fold_left ( * ) 1 s < 2 then s.(0) <- 2;
      s

let kinds = [ Empty; Single; Nan; Negative_zero; Bounds; Many ]

(* Arrays of [dts], of at most [rank] axes, of the [kinds] drawn, laid out by
   [views]. *)
let drawn_cases ?(rank = 3) ?(kinds = kinds)
    ?(views = [ Plain; Transposed; Flipped; Broadcast ]) dts =
  let open Gen in
  let* kind = of_list kinds in
  let kind, dts =
    match List.filter (supports kind) dts with
    | [] -> (Many, dts)
    | dts -> (kind, dts)
  in
  let* (D.Any dt) = of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) dts in
  let* s = shape ~rank kind in
  let n = Array.fold_left ( * ) 1 s in
  let* vs = array ~size:(constant n) (element dt) in
  let* i = int_range 0 (Int.max 0 (n - 1)) in
  let+ view = of_list views in
  plant kind dt vs i;
  Case (dt, s, vs, view)

let case ?rank ?kinds ?views dts =
  Gen.with_pp
    (fun ppf (Case (dt, s, _, view)) ->
      Format.fprintf ppf "%a %a %a" D.pp dt pp_shape s pp_view view)
    (drawn_cases ?rank ?kinds ?views dts)

(* The value of a case, laid out by its view. An odd number of elements holds
   its NaNs signalling, which only bits keep. *)
let value (Case (dt, s, vs, view)) =
  let x = Nx.create dt s vs in
  let x =
    if Array.length vs mod 2 = 0 then x
    else
      Nx.Repr.of_array Nx.Host.v
        (Nx_support.signalled (Option.get (Nx.Repr.array x)))
  in
  match view with
  | Plain -> Nx.P x
  | Transposed -> Nx.P (Nx.transpose x)
  | Flipped -> Nx.P (Nx.flip x)
  | Broadcast -> Nx.P (Nx.broadcast_to (Array.append [| 2 |] s) x)

let has_nan (type v s) (dt : (v, s) D.t) (vs : v array) =
  match D.kind dt with
  | D.Float -> Array.exists Float.is_nan vs
  | D.Complex ->
      Array.exists
        (fun (c : Complex.t) -> Float.is_nan c.re || Float.is_nan c.im)
        vs
  | D.Signed | D.Unsigned | D.Boolean -> false

let has_negative_zero (type v s) (dt : (v, s) D.t) (vs : v array) =
  match D.kind dt with
  | D.Float -> Array.exists (fun v -> v = 0. && Float.sign_bit v) vs
  | D.Complex | D.Signed | D.Unsigned | D.Boolean -> false

let has_bounds (type v s) (dt : (v, s) D.t) (vs : v array) =
  match D.kind dt with
  | D.Signed | D.Unsigned ->
      Array.mem (D.min_value dt) vs && Array.mem (D.max_value dt) vs
  | D.Float | D.Complex | D.Boolean -> false

(* [covers c] states the cases a round trip must reach; [~empty:false] and
   [~broadcast:false] leave out the value of no element and the broadcast
   view. *)
let covers ?(empty = true) ?(broadcast = true) (Case (dt, s, vs, view)) =
  let n = Array.fold_left ( * ) 1 s in
  if empty then cover "no element" (n = 0);
  cover "one element" (n = 1);
  cover "strided"
    ((view = Transposed && Array.length s >= 2) || (view = Flipped && n >= 2));
  if broadcast then cover "broadcast" (view = Broadcast && n >= 1);
  cover "NaN" (has_nan dt vs);
  cover "-0." (has_negative_zero dt vs);
  cover "an integer type's bounds" (has_bounds dt vs)

(* Dtypes *)

let npy_dtypes =
  D.
    [
      Any Float16;
      Any Float32;
      Any Float64;
      Any Int8;
      Any Int16;
      Any Int32;
      Any Int64;
      Any Uint8;
      Any Uint16;
      Any Uint32;
      Any Uint64;
      Any Complex64;
      Any Complex128;
      Any Bool;
    ]

let safetensors_dtypes =
  D.
    [
      Any Bool;
      Any Uint8;
      Any Int8;
      Any Float8_e5m2;
      Any Float8_e4m3fn;
      Any Int16;
      Any Uint16;
      Any Float16;
      Any Bfloat16;
      Any Int32;
      Any Uint32;
      Any Float32;
      Any Float64;
      Any Int64;
      Any Uint64;
    ]

(* Archives *)

module Archive = Nx_io.Archive

(* An archive compares as its named values, in name order. *)
let listed a =
  List.map (fun n -> (n, Option.get (Archive.find n a))) (Archive.names a)

let archive = Testable.contramap listed (list (pair string packed))
let find a name = Option.get (Archive.find name a)

let entry_names = [ "w"; "layer.0/weight"; "a b"; "é"; "日本"; "deep/er/name" ]

(* Archives of one to three values of [dts], under distinct names. *)
let archives dts =
  let open Gen in
  let* names = subsequence entry_names in
  let names =
    match names with
    | [] -> [ "w" ]
    | n :: ns -> n :: List.filteri (fun i _ -> i < 2) ns
  in
  let+ cs = list ~size:(constant (List.length names)) (case dts) in
  List.combine names cs

let archive_of entries =
  Archive.of_list (List.map (fun (n, c) -> (n, value c)) entries)

let pp_entries ppf l =
  Format.pp_print_list
    (fun ppf (n, c) ->
      Format.fprintf ppf "%s: %a" n
        (fun ppf (Case (dt, s, _, v)) ->
          Format.fprintf ppf "%a %a %a" D.pp dt pp_shape s pp_view v)
        c)
    ppf l

(* Bytes as stored *)

let pattern n seed =
  String.init n (fun i -> Char.chr (((i * 37) + seed) land 255))

(* A value's dtype, shape and bytes as stored. *)
let stored (Nx.P x) =
  let a = Option.get (Nx.Repr.array (Nx.copy x)) in
  let b = Nx_array.buffer a in
  ( D.name (Nx.dtype x),
    Nx.shape x,
    String.init (Rig.Buffer.length b) (fun i ->
        let s = Bytes.create 1 in
        Rig.Buffer.blit_to_bytes b i s 0 1;
        Bytes.get s 0) )
