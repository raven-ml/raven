(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The ring writer's templates, with no GPU: a packet loaded into a device
   state's template, as the driver loads its own at open, then filled as the
   writer fills it, is the packet's encoding. A template holds at most 16 words
   and 6 holes, each a term of at most 3 operations on argument 0, 1 or 2. *)

open Windtrap
open Rig_nv_abi

external create : int -> int -> int -> int -> int = "caml_rig_nv_create"

external set_template : int -> int -> string -> string -> unit
  = "caml_rig_nv_template"

external fill : int -> int -> int64 -> int64 -> int64 -> string
  = "rig_nv_test_fill"

let self = create 0 0 0 0
let max_words = 16
let max_holes = 6

let load p =
  let words, holes = Template.flatten (fun _ -> None) p in
  set_template self 0 words holes

(* Printing *)

let rec pp_term ppf : int Packet.term -> unit = function
  | Value i -> Format.fprintf ppf "Value %d" i
  | Add (t, n) -> Format.fprintf ppf "Add (%a, 0x%Lx)" pp_term t n
  | Shift (t, n) -> Format.fprintf ppf "Shift (%a, %d)" pp_term t n

let pp_word ppf : int Packet.word -> unit = function
  | Dword n -> Format.fprintf ppf "Dword 0x%x" n
  | W32 t -> Format.fprintf ppf "W32 (%a)" pp_term t
  | W64 t -> Format.fprintf ppf "W64 (%a)" pp_term t

let pp_case ppf (p, (a, b, c)) =
  Format.fprintf ppf "[%a]@ with 0x%Lx, 0x%Lx, 0x%Lx"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
       pp_word)
    p a b c

(* Drawing *)

(* 64-bit integers at the edges of their halves and of the type. *)
let u64 =
  Gen.frequency
    [
      (3, Gen.int64);
      ( 1,
        Gen.of_list
          [
            0L;
            1L;
            -1L;
            0xffff_ffffL;
            0x1_0000_0000L;
            0x4000_0000_0000_0000L;
            Int64.min_int;
            Int64.max_int;
          ] );
    ]

let shift = Gen.(frequency [ (3, int_range 0 63); (1, of_list [ 0; 63 ]) ])
let map2 f a b = Gen.map (fun (a, b) -> f a b) (Gen.pair a b)

(* Terms on an argument with at most [depth] operations. *)
let rec term depth : int Packet.term Gen.t =
  let open Gen in
  let leaf = map (fun i -> Packet.Value i) (int_range 0 2) in
  if depth = 0 then leaf
  else
    let sub = term (depth - 1) in
    frequency
      [
        (2, leaf);
        (1, map2 (fun t n -> Packet.Add (t, n)) sub u64);
        (1, map2 (fun t n -> Packet.Shift (t, n)) sub shift);
      ]

let word : int Packet.word Gen.t =
  let open Gen in
  frequency
    [
      (4, map (fun n -> Packet.Dword n) int);
      (1, map (fun t -> Packet.W32 t) (term 3));
      (1, map (fun t -> Packet.W64 t) (term 3));
    ]

let is_hole : int Packet.word -> bool = function
  | Dword _ -> false
  | W32 _ | W64 _ -> true

(* The words of [ws] kept in turn while the packet stays within [words] words
   and [holes] holes. *)
let within ~words ~holes ws =
  let fits (p, n, h) w =
    let n' = n + Packet.size [ w ] and h' = if is_hole w then h + 1 else h in
    if n' > words || h' > holes then (p, n, h) else (w :: p, n', h')
  in
  let p, _, _ = List.fold_left fits ([], 0, 0) ws in
  List.rev p

(* Packets within a template's bounds, many reaching them, some ending with a
   hole at the last word. *)
let bounded =
  let words = Gen.list ~size:(Gen.int_range 0 24) word in
  let ending ws h =
    let n = max_words - Packet.size [ h ] in
    let p = within ~words:n ~holes:(max_holes - 1) ws in
    p @ List.init (n - Packet.size p) (fun i -> Packet.Dword i) @ [ h ]
  in
  let hole =
    Gen.frequency
      [
        (1, Gen.map (fun t -> Packet.W32 t) (term 3));
        (1, Gen.map (fun t -> Packet.W64 t) (term 3));
      ]
  in
  Gen.frequency
    [
      (3, Gen.map (within ~words:max_words ~holes:max_holes) words);
      (1, map2 ending words hole);
    ]

let case = Gen.with_pp pp_case (Gen.pair bounded (Gen.triple u64 u64 u64))

(* Coverage *)

let rec eval value : int Packet.term -> int64 = function
  | Value i -> value i
  | Add (t, n) -> Int64.add (eval value t) n
  | Shift (t, n) -> Int64.shift_right_logical (eval value t) n

(* Whether an addition in [t] wraps past 2^64 with the arguments [value]. *)
let rec wraps value : int Packet.term -> bool = function
  | Value _ -> false
  | Add (t, n) ->
      let x = eval value t in
      wraps value t || Int64.unsigned_compare (Int64.add x n) x < 0
  | Shift (t, _) -> wraps value t

let rec has f : int Packet.term -> bool = function
  | Value _ -> false
  | (Add (t, _) | Shift (t, _)) as n -> f n || has f t

let any f p =
  List.exists (function Packet.Dword _ -> false | W32 t | W64 t -> f t) p

(* The index after the last word that holds a term. *)
let last_hole p =
  let _, last =
    List.fold_left
      (fun (i, last) w ->
        let i' = i + Packet.size [ w ] in
        (i', if is_hole w then i' else last))
      (0, 0) p
  in
  last

(* Tests *)

let law =
  prop "a loaded template, filled, is the packet's encoding" case
    (fun (p, (a, b, c)) ->
      let value i = [| a; b; c |].(i) in
      cover "16 words" (Packet.size p = max_words);
      cover "6 holes" (List.length (List.filter is_hole p) = max_holes);
      cover "a hole ending at word 16" (last_hole p = max_words);
      cover "a 64-bit hole"
        (List.exists (function Packet.W64 _ -> true | _ -> false) p);
      cover "a shift by 0"
        (any (has (function Shift (_, 0) -> true | _ -> false)) p);
      cover "a shift by 63"
        (any (has (function Shift (_, 63) -> true | _ -> false)) p);
      cover "an addition with bit 63"
        (any
           (has (function
             | Add (_, n) -> Int64.logand n Int64.min_int <> 0L
             | _ -> false))
           p);
      cover "an addition that wraps" (any (wraps value) p);
      load p;
      equal string (Packet.encode value p) (fill self 0 a b c))

let refused name p =
  test name (fun () ->
      raises_match (Exn.invalid_arg ~substring:"Rig_nv.make") (fun () -> load p))

let templates =
  group "templates"
    [
      law;
      test "an addition of bit 63 keeps it" (fun () ->
          let p = [ Packet.W64 (Add (Value 0, Int64.min_int)) ] in
          load p;
          equal string
            (Packet.encode (fun _ -> 0x100L) p)
            (fill self 0 0x100L 0L 0L));
      test "a load replaces the template" (fun () ->
          load [ Dword 1; W32 (Value 0); W64 (Value 1) ];
          load [ W32 (Value 2) ];
          equal string
            (Packet.encode (fun _ -> 7L) [ W32 (Value 2) ])
            (fill self 0 5L 6L 7L));
      test "a refused template leaves the one before" (fun () ->
          let p = [ Packet.Dword 1; W32 (Value 0) ] in
          load p;
          (try load (List.init 17 (fun i -> Packet.Dword i))
           with Invalid_argument _ -> ());
          equal string (Packet.encode (fun _ -> 5L) p) (fill self 0 5L 6L 7L));
      refused "17 words are refused" (List.init 17 (fun i -> Packet.Dword i));
      refused "7 holes are refused"
        (List.init 7 (fun _ -> Packet.W32 (Value 0)));
      refused "a term of 4 operations is refused"
        [ W32 (Add (Add (Add (Add (Value 0, 1L), 1L), 1L), 1L)) ];
      cases ~name:string_of_int "an argument outside [0;2] is refused"
        [ -1; 3; max_int ] (fun i ->
          raises_match (Exn.invalid_arg ~substring:"Rig_nv.make") (fun () ->
              load [ W32 (Value i) ]));
      cases ~name:string_of_int "a shift outside [0;63] is refused"
        [ -1; 64; 256 ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Rig_nv.make") (fun () ->
              load [ W32 (Shift (Value 0, n)) ]));
    ]

let () = exit (run "rig_nv.template" [ templates ])
