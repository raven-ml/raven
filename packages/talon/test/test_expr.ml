(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let x = Col.float "x"
let y = Col.float "y"
let n = Col.int "n"
let m = Col.int "m"
let b = Col.bool "b"
let s = Col.string "s"
let ts = Col.instant "ts"
let p e = Format.asprintf "%a" Expr.pp e

(* [print64 v] is the text Column.print writes for the float64 [v]. *)
let print64 v =
  let c = Column.print (Column.v Type.float64 [| v |]) in
  (Column.values Kind.string c).(0)

(* [unquote t] is the name that the printed handle [t] reads back as: an OCaml
   string literal, or a bare identifier. *)
let unquote t = if t.[0] = '"' then Scanf.sscanf t "%S%!" Fun.id else t

(* Raises at the call *)

let construction =
  let open Expr in
  [
    group "Expr.string"
      [
        test "raises on text that is not UTF-8" (fun () ->
            raises_match Exn.invalid_arg (fun () -> string "\xff"));
      ];
    group "Expr.cut"
      [
        cases "raises on fewer than two edges"
          ~id:(fun e -> Printf.sprintf "%d edges" (Array.length e))
          [ [||]; [| 1 |] ]
          (fun edges -> raises_match Exn.invalid_arg (fun () -> cut edges n));
      ];
    group "Expr.quantile"
      [
        cases "accepts p in [0, 1]" ~id:(Printf.sprintf "%h")
          [ 0.; -0.; 0.5; 1. ] (fun q -> ignore (quantile q x));
        cases "raises unless p lies in [0, 1]" ~id:(Printf.sprintf "%h")
          [ -0.1; 1.1; Float.nan; Float.infinity; Float.neg_infinity ] (fun q ->
            raises_match Exn.invalid_arg (fun () -> quantile q x));
      ];
    group "Expr.Str.slice"
      [
        test "raises on a negative length and accepts zero" (fun () ->
            ignore (Str.slice ~offset:0 ~length:0 s);
            raises_match Exn.invalid_arg (fun () ->
                Str.slice ~offset:0 ~length:(-1) s));
      ];
    group "Expr.Str.concat"
      [
        test "raises on a separator that is not UTF-8" (fun () ->
            raises_match Exn.invalid_arg (fun () -> Str.concat "\xff" [ s ]));
      ];
    group "Expr.Temporal.floor"
      [
        cases "raises on a step that is not positive"
          ~id:(Format.asprintf "%a" Time.pp_step)
          [ Time.Days 0; Time.Exact (Time.Span.s (-1)) ]
          (fun step ->
            raises_match Exn.invalid_arg (fun () -> Temporal.floor step ts));
      ];
    group "Expr.Temporal.parse"
      [
        test "raises on a directive outside its list" (fun () ->
            raises_match Exn.invalid_arg (fun () ->
                Temporal.parse "%Y-%q" Type.date s));
      ];
    group "Expr.Temporal.format"
      [
        cases "raises as parse does" ~id:Fun.id [ "%q"; "%H%" ] (fun fmt ->
            raises_match Exn.invalid_arg (fun () -> Temporal.format fmt ts));
      ];
  ]

(* Formatting *)

let catalogue () =
  let open Expr in
  let per_user e = over ~by:[ "user" ] ~order:[ Order.asc "ts" ] e in
  let lines =
    [
      p ((x -. over (mean x)) /. over (std x));
      p (per_user (shift 1 x));
      p (over ~order:[ Order.nulls_first (Order.desc "x") ] (rank x));
      p (cast Type.float32 n);
      p (store Type.int8 (const (fun a -> a) $ n));
      p (nx2 { f2 = Nx.atan2 } y x);
      p (is_in [ 1; 2 ] (cut [| 1; 2 |] n));
      p (Str.matches (Str.pieces [ "special"; "requests" ]) s);
      p (Str.matches (Str.prefix "wk") s || Str.matches (Str.suffix "!") s);
      p (Str.extract ~group:1 (Str.regex "id=(\\d+)") s);
      p (Str.split (Str.literal ", ") s);
      p (Str.parse Type.int8 s);
      p (Temporal.floor (Time.Days 1) ts);
      p (Temporal.offset (Time.Months (-1)) ts);
      p (Temporal.add ts (span (Time.Span.s 90)));
      p (Temporal.parse "%Y-%m-%d" Type.date s);
      p (Temporal.wall Tz.utc ts);
    ]
  in
  expect (String.concat "\n" lines)
  @@ __POS_OF__
       {|
    (x -. over (mean x)) /. over (std x)
    over ~by:["user"] ~order:[asc "ts"] (shift 1 x)
    over ~order:[nulls_first (desc "x")] (rank x)
    cast float32 n
    store int8 (<const> $ n)
    nx2 <fn> y x
    is_in […] (cut […] n)
    Str.matches (pieces ["special"; "requests"]) s
    Str.matches (prefix "wk") s || Str.matches (suffix "!") s
    Str.extract ~group:1 (regex "id=(\\d+)") s
    Str.split (literal ", ") s
    Str.parse int8 s
    Temporal.floor 1d ts
    Temporal.offset (-1mo) ts
    Temporal.add ts 1m30s
    Temporal.parse "%Y-%m-%d" date s
    Temporal.wall "UTC" ts
    |}

let float_literal v =
  let text = print64 v in
  let has_point = String.exists (fun c -> c = '.' || c = 'e') text in
  cover "a text with a point or an exponent" has_point;
  cover "a text without" (not has_point);
  equal string (if has_point then text else text ^ ".") (p (Expr.float v))

let column_name name =
  let printed = p (Col.float name) in
  cover "a bare name" (printed.[0] <> '"');
  cover "a quoted name" (printed.[0] = '"');
  equal string name (unquote printed)

let printing =
  let open Expr in
  group "Expr.pp"
    [
      cases "parenthesizes only where precedence needs it" ~id:snd
        [
          (p (n + (m * int 2)), "n + m * 2");
          (p ((n + m) * int 2), "(n + m) * 2");
          (p (n - m - int 1), "n - m - 1");
          (p (n - (m - int 1)), "n - (m - 1)");
          (p (n / m mod int 3), "n / m mod 3");
          (p (x ** y ** float 2.), "x ** y ** 2.");
          (p ((x ** y) ** float 2.), "(x ** y) ** 2.");
          (p (x *. (y +. float 1.)), "x *. (y +. 1.)");
          (p ((x > float 15. && b) || not b), "x > 15. && b || not b");
          (p (x > float 15. && (b || not b)), "x > 15. && (b || not b)");
          (p (n + m = int 3), "n + m = 3");
          (p (if_ b n (n + int 1)), "if_ b n (n + 1)");
          (p (shift (-1) n + shift 1 m), "shift (-1) n + shift 1 m");
          (p (n + int (-3)), "n + -3");
          (p (if_ (n < int 0) (int 0 - n) n), "if_ (n < 0) (0 - n) n");
        ]
        (fun (printed, expected) -> equal Windtrap.string expected printed);
      cases "formats a float literal as its value" ~id:snd
        [
          (15., "15.");
          (0.1, "0.1");
          (-0., "-0.");
          (Float.nan, "nan");
          (Float.infinity, "infinity");
          (Float.neg_infinity, "neg_infinity");
          (1e300, "1e+300");
          (1e-8, "1e-08");
          (1e20, "100000000000000000000.");
          (0.30000000000000004, "0.30000000000000004");
        ]
        (fun (v, expected) -> equal Windtrap.string expected (p (float v)));
      prop
        "formats a finite float literal as Column.print writes a float64, with \
         a point after a text that has none"
        ~examples:[ 15.; -0.; 1e21 ] Gen.float float_literal;
      cases "formats each other literal as its value" ~id:snd
        [
          (p (int 15), "15");
          (p (string "a \"b\""), {|"a \"b\""|});
          (p (bool true), "true");
          (p (span (Time.Span.minutes 15)), "15m");
          ( p (date (Option.get (Time.Date.of_civil (2024, 3, 15)))),
            "2024-03-15" );
          ( p (instant (Time.of_ns 1710495000_000_000_000L)),
            "2024-03-15T09:30:00" );
          (p null, "null");
        ]
        (fun (printed, expected) -> equal Windtrap.string expected printed);
      cases "leaves a lowercase identifier bare and quotes any other name"
        ~id:fst
        [
          ("dep_delay", "dep_delay");
          ("x1", "x1");
          ("Delay", {|"Delay"|});
          ("dep delay", {|"dep delay"|});
          ("", {|""|});
          ("let", {|"let"|});
          ("rows", {|"rows"|});
          ("sum", {|"sum"|});
          ("over", {|"over"|});
          ("cut", {|"cut"|});
          ("délai", {|"délai"|});
        ]
        (fun (name, expected) ->
          equal Windtrap.string expected (p (Col.float name)));
      prop "quotes a column name as an OCaml string that reads back as it"
        ~examples:[ "dep_delay"; {|a"b\c|}; "a\nb"; "a\xffb" ]
        Gen.string column_name;
      test "formats types, orders, steps, patterns and OCaml values as stated"
        catalogue;
    ]

let ext =
  group "Ext.v"
    [
      test "raises as Type.ext does" (fun () ->
          let declare ~name storage =
            Ext.v ~name ~ordered:true storage ~dec:Fun.id ~enc:Fun.id
          in
          raises_match Exn.invalid_arg (fun () -> declare ~name:"" Type.int8);
          raises_match Exn.invalid_arg (fun () ->
              declare ~name:"\xff" Type.int8);
          raises_match Exn.invalid_arg (fun () ->
              declare ~name:"a" (Type.ext ~name:"b" Type.int8)));
    ]

let () = exit (run "talon Expr" (construction @ [ printing; ext ]))
