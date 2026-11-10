(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

(* Files built byte by byte from the ESRI Shapefile Technical Description's
   tables. *)

let be32 b n = Buffer.add_int32_be b (Int32.of_int n)
let le32 b n = Buffer.add_int32_le b (Int32.of_int n)
let f64 b x = Buffer.add_int64_le b (Int64.bits_of_float x)

type shape =
  | Null
  | Poly of (float * float) list list
  | Kind of int * (float * float) list list

(* A record's content: shape type, box, parts, points, and for z and m types the
   ranges and arrays the reader drops. *)
let content kind rings =
  let b = Buffer.create 64 in
  le32 b kind;
  if kind <> 0 then begin
    let pts = List.concat rings in
    List.iter (f64 b) [ 0.; 0.; 1.; 1. ];
    le32 b (List.length rings);
    le32 b (List.length pts);
    ignore
      (List.fold_left
         (fun start r ->
           le32 b start;
           start + List.length r)
         0 rings);
    List.iter
      (fun (x, y) ->
        f64 b x;
        f64 b y)
      pts;
    let extra = if kind = 15 then 2 else if kind = 25 then 1 else 0 in
    for _ = 1 to extra do
      f64 b 0.;
      f64 b 9.;
      List.iter (fun _ -> f64 b 7.) pts
    done
  end;
  Buffer.contents b

let shp ?(code = 9994) ?(version = 1000) ?(kind = 5) ?(extra_length = 0) shapes
    =
  let records = Buffer.create 256 in
  List.iteri
    (fun k s ->
      let c =
        match s with
        | Null -> content 0 []
        | Poly r -> content kind r
        | Kind (t, r) -> content t r
      in
      be32 records (k + 1);
      be32 records (String.length c / 2);
      Buffer.add_string records c)
    shapes;
  let b = Buffer.create 512 in
  be32 b code;
  for _ = 1 to 5 do
    be32 b 0
  done;
  be32 b (((100 + Buffer.length records) / 2) + extra_length);
  le32 b version;
  le32 b kind;
  for _ = 1 to 8 do
    f64 b 0.
  done;
  Buffer.add_buffer b records;
  Buffer.contents b

(* A dBASE III table of character fields; a record starting with '*' is
   deleted. *)
let dbf ?(width_error = 0) fields records =
  let b = Buffer.create 256 in
  let rlen = 1 + List.fold_left (fun n (_, w) -> n + w) 0 fields in
  Buffer.add_char b '\003';
  Buffer.add_string b "\126\011\010";
  le32 b (List.length records);
  Buffer.add_uint16_le b (32 + (32 * List.length fields) + 1);
  Buffer.add_uint16_le b (rlen + width_error);
  Buffer.add_string b (String.make 20 '\000');
  List.iter
    (fun (name, w) ->
      Buffer.add_string b (name ^ String.make (11 - String.length name) '\000');
      Buffer.add_char b 'C';
      Buffer.add_string b (String.make 4 '\000');
      Buffer.add_char b (Char.chr w);
      Buffer.add_string b (String.make 15 '\000'))
    fields;
  Buffer.add_char b '\r';
  List.iter
    (fun (flag, values) ->
      Buffer.add_char b flag;
      List.iter2
        (fun v (_, w) ->
          Buffer.add_string b (v ^ String.make (w - String.length v) ' '))
        values fields)
    records;
  Buffer.add_char b '\026';
  Buffer.contents b

let square x = [ (x, 0.); (x, 1.); (x +. 1., 1.); (x +. 1., 0.); (x, 0.) ]
let hole = [ (0.25, 0.25); (0.75, 0.25); (0.75, 0.75); (0.25, 0.25) ]
let fields = [ ("crop", 8); ("id", 4) ]
let utm = Crs.utm ~zone:33 ~north:true
let columns = list (pair string (array string))

let read_polygons =
  test "polygons, holes and null shapes, one feature each" (fun () ->
      let p, cols =
        Result.get_ok
          (Shapefile.read utm
             ~shp:
               (shp
                  [
                    Poly [ square 0.; hole ];
                    Null;
                    Poly [ square 5.; square 7. ];
                  ])
             ~dbf:
               (dbf fields
                  [
                    (' ', [ "maize"; "1" ]);
                    (' ', [ ""; "2" ]);
                    (' ', [ "wheat"; "3" ]);
                  ]))
      in
      equal bool true (Crs.equal utm p.crs);
      equal (array int) [| 0; 2; 2; 4 |] p.features;
      equal (array int) [| 0; 5; 9; 14; 19 |] p.rings;
      equal (array float_exact)
        [| 0.; 0.; 0.; 1.; 1.; 1.; 1.; 0.; 0.; 0. |]
        (Array.sub (Nx.to_array p.points) 0 10);
      equal columns
        [ ("crop", [| "maize"; ""; "wheat" |]); ("id", [| "1"; "2"; "3" |]) ]
        cols)

let z_and_m =
  cases "PolygonZ and PolygonM drop z and m" ~id:string_of_int [ 15; 25 ]
    (fun kind ->
      let p, _ =
        Result.get_ok
          (Shapefile.read utm
             ~shp:(shp ~kind [ Poly [ square 2. ] ])
             ~dbf:(dbf fields [ (' ', [ "a"; "1" ]) ]))
      in
      equal (array float_exact)
        [| 2.; 0.; 2.; 1.; 3.; 1.; 3.; 0.; 2.; 0. |]
        (Nx.to_array p.points))

let deleted =
  test "a deleted record is dropped with its shape" (fun () ->
      let p, cols =
        Result.get_ok
          (Shapefile.read utm
             ~shp:(shp [ Poly [ square 0. ]; Poly [ square 5. ] ])
             ~dbf:
               (dbf fields [ ('*', [ "gone"; "1" ]); (' ', [ "kept"; "2" ]) ]))
      in
      equal int 1 (Polygons.count p);
      equal float_exact 5. (Nx.item [ 0; 0 ] p.points);
      equal columns [ ("crop", [| "kept" |]); ("id", [| "2" |]) ] cols)

let refusals =
  let table = dbf fields [ (' ', [ "a"; "1" ]) ] in
  cases "refusals name what breaks the description" ~id:fst
    [
      ( "a file code other than 9994",
        (shp ~code:9995 [ Poly [ square 0. ] ], table, "file code") );
      ( "another version",
        (shp ~version:999 [ Poly [ square 0. ] ], table, "version") );
      ( "a PolyLine file",
        (shp ~kind:3 [ Poly [ square 0. ] ], table, "shape type 3") );
      ( "a shape of another type",
        (shp [ Kind (15, [ square 0. ]) ], table, "shape 0: shape type 15") );
      ( "a length past the file",
        (shp ~extra_length:4 [ Poly [ square 0. ] ], table, "length") );
      ( "a truncated record",
        (String.sub (shp [ Poly [ square 0. ] ]) 0 150, table, "length") );
      ( "more shapes than records",
        (shp [ Poly [ square 0. ]; Null ], table, "2 shapes and the table 1") );
      ( "a table whose widths disagree",
        ( shp [ Poly [ square 0. ] ],
          dbf ~width_error:1 fields [ (' ', [ "a"; "1" ]) ],
          "widths" ) );
      ( "a table cut short",
        (shp [ Poly [ square 0. ] ], String.sub table 0 70, "the table") );
    ]
    (fun (_, (shp, dbf, sub)) ->
      match Shapefile.read utm ~shp ~dbf with
      | Ok _ -> fail "read"
      | Error e -> contains ~sub e)

let parts =
  test "a polygon's first part must start at its first point" (fun () ->
      let s = Bytes.of_string (shp [ Poly [ square 0. ] ]) in
      (* The record's first part index, after its 8-byte header, type, box, and
         part and point counts. *)
      Bytes.set_int32_le s (100 + 8 + 44) 1l;
      match
        Shapefile.read utm ~shp:(Bytes.to_string s)
          ~dbf:(dbf fields [ (' ', [ "a"; "1" ]) ])
      with
      | Ok _ -> fail "read"
      | Error e -> contains ~sub:"part 0" e)

let () =
  exit
    (run "jord.shapefile"
       [ group "reading" [ read_polygons; z_and_m; deleted; refusals; parts ] ])
