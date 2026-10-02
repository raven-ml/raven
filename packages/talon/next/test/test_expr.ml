(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next
open Windtrap

let str pp v = Format.asprintf "%a" pp v
let rejects f = raises_match (fun e -> Exn.invalid_arg e) f

(* [messages fs] is the message of the [Invalid_argument] that each of [fs]
   raises, one per line. *)
let messages fs =
  let message f =
    match f () with () -> "no exception" | exception Invalid_argument m -> m
  in
  String.concat "\n" (List.map message fs)

let x = Col.float "x"
let y = Col.float "y"
let n = Col.int "n"
let m = Col.int "m"
let b = Col.bool "b"
let s = Col.string "s"
let ts = Col.instant "ts"

(* Printing *)

let p e = str Expr.pp e
let abs_like n = Expr.(if_ (n < int 0) (int 0 - n) n)

let precedence =
  let open Expr in
  group "Expr.pp precedence"
    [
      cases
        ~name:(fun (_, printed) -> Printf.sprintf "prints %s" printed)
        "fewest parentheses"
        [
          ((fun () -> p (n + (m * int 2))), "n + m * 2");
          ((fun () -> p ((n + m) * int 2)), "(n + m) * 2");
          ((fun () -> p (n - m - int 1)), "n - m - 1");
          ((fun () -> p (n - (m - int 1))), "n - (m - 1)");
          ((fun () -> p (n / m mod int 3)), "n / m mod 3");
          ((fun () -> p (x ** y ** float 2.)), "x ** y ** 2.");
          ((fun () -> p ((x ** y) ** float 2.)), "(x ** y) ** 2.");
          ((fun () -> p (x *. (y +. float 1.))), "x *. (y +. 1.)");
          ( (fun () -> p ((x > float 15. && b) || not b)),
            "x > 15. && b || not b" );
          ( (fun () -> p (x > float 15. && (b || not b))),
            "x > 15. && (b || not b)" );
          ((fun () -> p (n + m = int 3)), "n + m = 3");
          ((fun () -> p (if_ b n (n + int 1))), "if_ b n (n + 1)");
          ((fun () -> p (shift (-1) n + shift 1 m)), "shift (-1) n + shift 1 m");
          ((fun () -> p (n + int (-3))), "n + -3");
          ((fun () -> p (abs_like n)), "if_ (n < 0) (0 - n) n");
        ]
        (fun (print, expected) -> equal Windtrap.string expected (print ()));
    ]

let literals =
  let open Expr in
  group "Expr.pp literals"
    [
      cases
        ~name:(fun (_, printed) -> Printf.sprintf "prints %s" printed)
        "values"
        [
          ((fun () -> p (float 15.)), "15.");
          ((fun () -> p (float 0.1)), "0.1");
          ((fun () -> p (float (-0.))), "-0.");
          ((fun () -> p (float Float.nan)), "nan");
          ((fun () -> p (float Float.infinity)), "infinity");
          ((fun () -> p (float 1e300)), "1e+300");
        ]
        (fun (print, expected) -> equal Windtrap.string expected (print ()));
      cases
        ~name:(fun (_, printed) -> Printf.sprintf "prints %s" printed)
        "other kinds"
        [
          ((fun () -> p (is_null (string "a \"b\""))), {|is_null "a \"b\""|});
          ((fun () -> p (if_ (bool true) n (int 0))), "if_ true n 0");
          ((fun () -> p (is_null (span (Time.Span.minutes 15)))), "is_null 15m");
          ( (fun () ->
              p (is_null (date (Option.get (Time.Date.of_civil (2024, 3, 15)))))),
            "is_null 2024-03-15" );
          ( (fun () ->
              p (is_null (instant (Time.of_ns 1710495000_000_000_000L)))),
            "is_null 2024-03-15T09:30:00" );
          ((fun () -> p (is_null null)), "is_null null");
          ((fun () -> p (is_null (const 3))), "is_null <const>");
        ]
        (fun (print, expected) -> equal Windtrap.string expected (print ()));
    ]

let names =
  group "Expr.pp column names"
    [
      cases
        ~name:(fun (name, _) -> Printf.sprintf "prints the column %S" name)
        "quoting"
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
          ("délai", {|"délai"|});
          ({|a"b\c|}, {|"a\"b\\c"|});
          ("a\nb", {|"a\x0ab"|});
          ("a\xffb", {|"a\xffb"|});
        ]
        (fun (name, expected) ->
          equal Windtrap.string expected (str Expr.pp (Col.float name)));
    ]

let print_operations () =
  let open Expr in
  let per_user e = over ~by:[ "user" ] ~order:[ Order.asc "ts" ] e in
  let exprs =
    [
      str pp ((x -. over (mean x)) /. over (std x));
      str pp (per_user (shift 1 x));
      str pp (over ~order:[ Order.nulls_first (Order.desc "x") ] (rank x));
      str pp (cast Type.float32 n);
      str pp (store Type.int8 (const (fun a -> a) $ n));
      str pp
        (const (fun a b -> Stdlib.(a + Option.value ~default:0 b))
        $ n $ option m);
      str pp (of_option (const (fun a -> Some a) $ n));
      str pp (is_in [ 1; 2 ] n);
      str pp (coalesce [ x; y; float 0. ]);
      str pp (nx { f = Nx.exp } x);
      str pp (nx2 { f2 = Nx.atan2 } y x);
      str pp (quantile 0.9 x +. median x);
      str pp (Str.length s + int 1);
      str pp (Str.slice ~offset:(-3) ~length:2 s);
      str pp (Str.matches (Str.pieces [ "special"; "requests" ]) s);
      str pp (Str.matches (Str.literal ",") s || Str.matches (Str.suffix "!") s);
      str pp (Str.matches (Str.prefix "wk") s);
      str pp (Str.parse Type.int8 s);
      str pp (Temporal.field `Hour ts);
      str pp (Temporal.field `Year (Col.date "d"));
      str pp (Temporal.floor (Time.Days 1) ts);
      str pp (Temporal.offset (Time.Months (-1)) ts);
      str pp (Temporal.diff ts (Temporal.add ts (span (Time.Span.s 90))));
      str pp (Temporal.parse "%Y-%m-%d" Type.date s);
      str pp (Temporal.format "%H:%M" ts);
      str pp (rows - count x);
      str pp (first x +. last x -. only x);
      str pp (var x /. min x);
      str pp (n_unique s + arg_min x - arg_max x);
      str pp ((n <> m || n < m) && (n <= m || n >= m));
      str pp (n + null);
    ]
  in
  expect (String.concat "\n" exprs)
  @@ __POS_OF__
       {|
            (x -. over (mean x)) /. over (std x)
            over ~by:["user"] ~order:[asc "ts"] (shift 1 x)
            over ~order:[nulls_first (desc "x")] (rank x)
            cast float32 n
            store int8 (<const> $ n)
            <const> $ n $ option m
            of_option (<const> $ n)
            is_in […] n
            coalesce [x; y; 0.]
            nx <fn> x
            nx2 <fn> y x
            quantile 0.9 x +. median x
            Str.length s + 1
            Str.slice ~offset:(-3) ~length:2 s
            Str.matches (pieces ["special"; "requests"]) s
            Str.matches (literal ",") s || Str.matches (suffix "!") s
            Str.matches (prefix "wk") s
            Str.parse int8 s
            Temporal.field `Hour ts
            Temporal.field `Year d
            Temporal.floor 1d ts
            Temporal.offset (-1mo) ts
            Temporal.diff ts (Temporal.add ts 1m30s)
            Temporal.parse "%Y-%m-%d" date s
            Temporal.format "%H:%M" ts
            rows - count x
            first x +. last x -. only x
            var x /. min x
            n_unique s + arg_min x - arg_max x
            (n <> m || n < m) && (n <= m || n >= m)
            n + null
            |}

let catalogue =
  group "Expr.pp catalogue"
    [ test "prints every operation as written" print_operations ]

(* Construction *)

let construction =
  let open Expr in
  group "construction"
    [
      cases
        ~name:(fun p -> Printf.sprintf "quantile accepts %g" p)
        "quantile bounds" [ 0.; 0.5; 1. ]
        (fun p -> ignore (quantile p x));
      cases
        ~name:(fun p -> Printf.sprintf "quantile rejects %g" p)
        "quantile outside"
        [ -0.1; 1.1; Float.nan; Float.infinity ]
        (fun p -> rejects (fun () -> quantile p x));
      test "Str.slice accepts a zero length and rejects a negative one"
        (fun () ->
          ignore (Str.slice ~offset:0 ~length:0 s);
          rejects (fun () -> Str.slice ~offset:0 ~length:(-1) s));
      test "reports each invalid argument" (fun () ->
          expect
            (messages
               [
                 (fun () -> ignore (string "\xff"));
                 (fun () -> ignore (Str.literal ""));
                 (fun () -> ignore (Str.prefix "\xc3"));
                 (fun () -> ignore (Str.suffix ""));
                 (fun () -> ignore (Str.pieces []));
                 (fun () -> ignore (Str.pieces [ "a"; "" ]));
                 (fun () -> ignore (Temporal.parse "%Y-%q" Type.date s));
                 (fun () -> ignore (Temporal.format "%H%" ts));
                 (fun () -> ignore (Temporal.floor (Time.Days 0) ts));
                 (fun () ->
                   ignore (Temporal.floor (Time.Exact (Time.Span.s (-1))) ts));
                 (fun () ->
                   ignore
                     (Ext.v ~name:"" ~ordered:true Type.int8 ~dec:Fun.id
                        ~enc:Fun.id));
                 (fun () ->
                   ignore
                     (Ext.v ~name:"a" ~ordered:true
                        (Type.ext ~name:"b" Type.int8)
                        ~dec:Fun.id ~enc:Fun.id));
               ])
          @@ __POS_OF__
               {|
            Expr.string: "\255" is not valid UTF-8
            Expr.Str.literal: empty pattern
            Expr.Str.prefix: "\195" is not valid UTF-8
            Expr.Str.suffix: empty pattern
            Expr.Str.pieces: no pieces
            Expr.Str.pieces: empty pattern
            Expr.Temporal.parse: "%Y-%q" holds the unknown directive %q
            Expr.Temporal.format: "%H%" ends with %
            Expr.Temporal.floor: 0d is not positive
            Expr.Temporal.floor: -1s is not positive
            Type.ext: empty name
            Type.ext: "a" is stored as an extension type
            |});
    ]

let () =
  exit
    (run "talon.next Expr"
       [ precedence; literals; names; catalogue; construction ])
