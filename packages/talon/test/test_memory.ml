(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The columns that talon makes are host memory: the host counts their
   arrays' bytes while they are live, and has them back once they are
   collected ({!Talon_held.check}). Drawn columns of up to forty rows, some
   of them null and none in at least one case of five. *)

open Talon
open Windtrap
module G = Talon_gen

(* [computed c e] is the column [r] that [e] computes over the table whose
   column [s] is [c]. *)
let computed c e =
  let t = Talon.v [ ("s", c) ] in
  let q = Query.select Expr.[ "r" := e ] (Query.of_table t) in
  Talon.column (Result.get_ok (Query.run q)) "r"

(* The text kernels, each an expression over the text column [s]. *)
let kernels =
  let s = Col.string "s" in
  Expr.
    [
      ("slice", fun c -> computed c (Str.slice ~offset:1 ~length:2 s));
      ( "slice from the end",
        fun c -> computed c (Str.slice ~offset:(-2) ~length:5 s) );
      ("trim", fun c -> computed c (Str.trim s));
      ("length", fun c -> computed c (Str.length s));
      ( "matches a literal",
        fun c -> computed c (Str.matches (Str.literal "a") s) );
      ( "matches a prefix",
        fun c -> computed c (Str.matches (Str.prefix "a") s) );
      ( "matches a suffix",
        fun c -> computed c (Str.matches (Str.suffix "a") s) );
      ( "matches pieces",
        fun c -> computed c (Str.matches (Str.pieces [ "a"; "b" ]) s) );
      ("split", fun c -> computed c (Str.split "a" s));
      ("replace", fun c -> computed c (Str.replace "a" ~by:"xyz" s));
      ("concat", fun c -> computed c (Str.concat "," [ s; string "é"; s ]));
      ("compare", fun c -> computed c (s < string "m"));
      ("equal", fun c -> computed c (s = string "a"));
    ]

let texts = G.options Type.string
let oneofl xs = Gen.one_of (List.map (fun x -> Gen.constant x) xs)

let kernel =
  Gen.pair (oneofl (List.map fst kernels)) texts
  |> Gen.with_pp (fun ppf (k, rows) ->
         Format.fprintf ppf "%s computed %d rows" k (Array.length rows))

let law_kernel (k, rows) =
  List.iter (fun (k', _) -> cover k' (String.equal k k')) kernels;
  cover "no row" (Array.length rows = 0);
  cover "a null" (Array.exists Option.is_none rows);
  let c = Column.of_options Type.string rows in
  Talon_held.check ~inputs:[ c ] (fun () -> (List.assoc k kernels) c)

(* The types that [Column.parse] reads and [Column.print] writes. *)
let parsed =
  Type.
    [
      Any bool; Any int8; Any int16; Any int32; Any int64; Any uint8;
      Any uint16; Any uint32; Any uint64; Any float16; Any float32;
      Any float64; Any string; Any binary; Any date; Any (wall S);
      Any (wall Ns); Any (instant Us);
      Any (categorical [| "a"; "b"; "é" |]);
    ]

type sample = Sample : 'a Type.t * 'a option array -> sample

let sample =
  Gen.bind (oneofl parsed) (fun (Type.Any ty) ->
      Gen.map (fun rows -> Sample (ty, rows)) (G.options ty))
  |> Gen.with_pp (fun ppf (Sample (ty, rows)) ->
         Format.fprintf ppf "%d rows of %a" (Array.length rows) Type.pp ty)

let name ty = Format.asprintf "%a" Type.pp ty

(* [Column.print] of a column and [Column.parse] of its texts. *)
let law_parse (Sample (ty, rows)) =
  List.iter
    (fun (Type.Any t) -> cover (name t) (String.equal (name ty) (name t)))
    parsed;
  cover "no row" (Array.length rows = 0);
  let c = Column.of_options ty rows in
  Talon_held.check ~inputs:[ c ] (fun () -> Column.print c);
  let texts = Column.print c in
  Talon_held.check ~inputs:[ texts ] (fun () ->
      Result.get_ok (Column.parse (Any ty) texts))

(* [Temporal.parse] of dates printed in its format. *)
let law_parse_with rows =
  cover "no row" (Array.length rows = 0);
  let texts = Column.print (Column.of_options Type.date rows) in
  let parse () =
    computed texts (Expr.Temporal.parse "%Y-%m-%d" Type.date (Col.string "s"))
  in
  Talon_held.check ~inputs:[ texts ] parse

let () =
  exit
    (run "talon memory"
       [
         prop ~count:300 "a text kernel's column is the host's memory" kernel
           law_kernel;
         prop ~count:300 "a printed or parsed column is the host's memory"
           sample law_parse;
         prop ~count:50 "a column Temporal.parse reads is the host's memory"
           (G.options Type.date) law_parse_with;
       ])
