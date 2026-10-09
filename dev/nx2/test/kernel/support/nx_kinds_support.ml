(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external f32 : string -> int array -> int = "nx_kinds_support_f32"
external f64 : string -> float array -> float = "nx_kinds_support_f64"
external int : string -> string -> int64 array -> int64 = "nx_kinds_support_int"
external threefry : int64 -> int64 -> int64 = "nx_kinds_support_threefry"

external run :
  string ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  (float, 's, Bigarray.c_layout) Bigarray.Array1.t ->
  unit = "nx_kinds_support_run"
[@@noalloc]

type worst = { errors : (int * int) array; ambiguous : int }

external sweep_raw : string -> int -> int -> int -> (int * int) array * int
  = "nx_kinds_support_sweep"

external points_raw : string -> int array -> (int * int) array * int
  = "nx_kinds_support_points"

external strata_raw : string -> (string * int * ((int * int) array * int)) array
  = "nx_kinds_support_strata"

external narrow : string -> int -> int * int = "nx_kinds_support_narrow"

external binary_raw : string -> int -> int -> int * int * int
  = "nx_kinds_support_binary"

external targets : unit -> string list = "nx_kinds_support_targets"

external digest_raw : string -> string -> string -> string
  = "nx_kinds_support_digest"

let worst (errors, ambiguous) = { errors; ambiguous }
let sweep_range kind lo hi step = worst (sweep_raw kind lo hi step)
let points kind ps = worst (points_raw kind ps)
let strata kind = Array.map (fun (r, n, w) -> (r, n, worst w)) (strata_raw kind)
let binary kind ~seed n = binary_raw kind seed n
let digest ~target kind ty = digest_raw target kind ty

let f32_bounds =
  [
    ("exp", 2);
    ("exp2", 2);
    ("expm1", 1);
    ("log", 1);
    ("log2", 1);
    ("log1p", 1);
    ("sin", 2);
    ("cos", 2);
    ("tan", 4);
    ("asin", 2);
    ("acos", 2);
    ("atan", 2);
    ("sinh", 3);
    ("cosh", 2);
    ("tanh", 2);
    ("erf", 2);
  ]

let kinds =
  let real = [ "f32"; "f64" ] and both = [ "f32"; "f64"; "int" ] in
  List.map
    (fun k -> (k, real))
    [
      "exp";
      "exp2";
      "expm1";
      "log";
      "log2";
      "log1p";
      "sin";
      "cos";
      "tan";
      "asin";
      "acos";
      "atan";
      "sinh";
      "cosh";
      "tanh";
      "erf";
      "sqrt";
      "floor";
      "ceil";
      "round";
      "trunc";
      "fdiv";
      "atan2";
    ]
  @ List.map
      (fun k -> (k, both))
      [
        "neg";
        "abs";
        "sign";
        "recip";
        "add";
        "sub";
        "mul";
        "mod";
        "pow";
        "maximum";
        "minimum";
        "equal";
        "not_equal";
        "less";
        "less_equal";
        "fma";
        "where";
      ]
  @ List.map
      (fun k -> (k, [ "int" ]))
      [ "idiv"; "and"; "or"; "xor"; "threefry" ]

(* Worst points of two checks together, ordered as the C stubs order them: by
   error, then by a scramble of the pattern. A binade lies within one chunk, so
   each keeps at most two ties. *)
let scramble p = p * 0x9E3779B1 land 0xFFFFFFFF

let merge a b =
  let all = Array.append a.errors b.errors in
  Array.sort
    (fun (e, p) (e', p') ->
      if e <> e' then compare e' e else compare (scramble p) (scramble p'))
    all;
  {
    errors = Array.sub all 0 (min 64 (Array.length all));
    ambiguous = a.ambiguous + b.ambiguous;
  }

(* Chunks of 2^24 patterns go to the domains in turn. *)
let sweep ?(step = 1) ?(offset = 0) kind =
  let chunk = 1 lsl 24 and total = 1 lsl 32 in
  if step <= 0 || chunk mod step <> 0 || offset < 0 || offset >= step then
    invalid_arg "Nx_kinds_support.sweep: step must divide 2^24, offset below it";
  let next = Atomic.make 0 in
  let none = { errors = [||]; ambiguous = 0 } in
  let rec work acc =
    let lo = Atomic.fetch_and_add next chunk in
    if lo >= total then acc
    else work (merge acc (sweep_range kind (lo + offset) (lo + chunk) step))
  in
  let n = max 0 (Domain.recommended_domain_count () - 1) in
  let others = List.init n (fun _ -> Domain.spawn (fun () -> work none)) in
  let mine = work none in
  List.fold_left (fun acc d -> merge acc (Domain.join d)) mine others

(* The full sweep's record *)

let headers =
  [ "../../lib/kernel/nx_kinds.h"; "../../lib/kernel/nx_kinds_real.h" ]

let record = "golden/kinds/f32-sweep.txt"
let stamp = String.concat "+" (List.map Filename.basename headers)

let header_digest () =
  Digest.to_hex
    (Digest.string (String.concat "" (List.map Digest.file headers)))

let write_record sweeps =
  let oc = open_out record in
  Printf.fprintf oc
    "# written by gen/sweep_kinds.exe, run from dev/nx2/test/kernel; do not edit\n";
  Printf.fprintf oc "# kind, largest error in ulps, the worst patterns\n";
  Printf.fprintf oc "%s %s\n" stamp (header_digest ());
  List.iter
    (fun (kind, w) ->
      let e = if Array.length w.errors = 0 then 0 else fst w.errors.(0) in
      Printf.fprintf oc "%s %d" kind e;
      Array.iter (fun (_, p) -> Printf.fprintf oc " 0x%08x" p) w.errors;
      output_char oc '\n')
    sweeps;
  close_out oc

let read_record () =
  let ic = open_in record in
  let lines = ref [] in
  (try
     while true do
       lines := input_line ic :: !lines
     done
   with End_of_file -> close_in ic);
  let fields l = String.split_on_char ' ' l |> List.filter (( <> ) "") in
  let bad l = failwith ("Nx_kinds_support.read_record: " ^ l) in
  match List.filter (fun l -> l <> "" && l.[0] <> '#') (List.rev !lines) with
  | first :: rows -> (
      match fields first with
      | [ s; digest ] when s = stamp ->
          ( digest,
            List.map
              (fun l ->
                match fields l with
                | kind :: e :: ps -> (
                    try (kind, int_of_string e, List.map int_of_string ps)
                    with Failure _ -> bad l)
                | _ -> bad l)
              rows )
      | _ -> bad first)
  | [] -> failwith "Nx_kinds_support.read_record: empty record"
