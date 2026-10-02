(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

let pdf ?(w = 100.) ?(h = 100.) p =
  Hugin_next_vg_pdf.render (Renderable.v w h p)

(* Reading documents *)

let find_from s i sub =
  let n = String.length sub in
  let rec loop i =
    if i + n > String.length s then None
    else if String.sub s i n = sub then Some i
    else loop (i + 1)
  in
  loop i

let contains s sub = find_from s 0 sub <> None

(* [count s sub] is the number of occurrences of [sub] in [s]. *)
let count s sub =
  let rec loop i k =
    match find_from s i sub with None -> k | Some j -> loop (j + 1) (k + 1)
  in
  loop 0 0

(* [objects doc] is the objects that the cross-reference table of [doc] lists,
   in order: each object's number, its text up to its stream if it has one, and
   its stream inflated. It fails on a table that does not point at the object it
   numbers. *)
let objects doc =
  let find sub i =
    match find_from doc i sub with Some j -> j | None -> failf "no %S" sub
  in
  let table = find "\nxref\n0 " 0 + 8 in
  let eol = String.index_from doc table '\n' in
  let size = int_of_string (String.sub doc table (eol - table)) in
  List.init (size - 1) (fun k ->
      let n = k + 1 in
      let offset = int_of_string (String.sub doc (eol + 1 + (20 * n)) 10) in
      let head = Printf.sprintf "%d 0 obj\n" n in
      if String.sub doc offset (String.length head) <> head then
        failf "object %d is not at %d" n offset;
      let start = offset + String.length head in
      let body = String.sub doc start (find "\nendobj\n" start - start) in
      match find_from body 0 "\nstream\n" with
      | None -> (n, body, None)
      | Some i -> (
          let dict = String.sub body 0 i in
          let length =
            let j = Option.get (find_from dict 0 "/Length ") + 8 in
            let k = ref j in
            while dict.[!k] >= '0' && dict.[!k] <= '9' do
              incr k
            done;
            int_of_string (String.sub dict j (!k - j))
          in
          let data = String.sub body (i + 8) length in
          match Compress_deflate.Zlib.decompress data with
          | Ok data -> (n, dict, Some data)
          | Error e -> failf "object %d: %s" n e))

(* [content doc] is the page's content stream, object 4. *)
let content doc =
  match List.find (fun (n, _, _) -> n = 4) (objects doc) with
  | _, _, Some data -> data
  | _ -> failf "no content stream"

(* [dicts doc] is the text of every object of [doc] but its streams. *)
let dicts doc = String.concat "\n" (List.map (fun (_, d, _) -> d) (objects doc))

(* [stream_of doc sub] is the stream of the first object whose dictionary holds
   [sub]. *)
let stream_of doc sub =
  match
    List.find_opt (fun (_, d, s) -> s <> None && contains d sub) (objects doc)
  with
  | Some (_, _, Some s) -> s
  | _ -> failf "no stream with %s" sub

(* Tokens of a content stream: numbers, operators and names, hex strings, and
   the brackets of arrays and dictionaries. *)
type token = Num of float | Hex of string | Word of string

let tokens s =
  let acc = ref [] and i = ref 0 and n = String.length s in
  let is_space c = c = ' ' || c = '\n' || c = '\r' in
  while !i < n do
    let c = s.[!i] in
    let pair = if !i + 1 < n then String.sub s !i 2 else "" in
    if is_space c then incr i
    else if pair = "<<" || pair = ">>" then begin
      acc := Word pair :: !acc;
      i := !i + 2
    end
    else if c = '[' || c = ']' then begin
      acc := Word (String.make 1 c) :: !acc;
      incr i
    end
    else if c = '<' then begin
      let j = String.index_from s !i '>' in
      acc := Hex (String.sub s (!i + 1) (j - !i - 1)) :: !acc;
      i := j + 1
    end
    else begin
      let j = ref !i in
      while
        !j < n
        && (not (is_space s.[!j]))
        && s.[!j] <> '['
        && s.[!j] <> ']'
        && s.[!j] <> '<'
        && s.[!j] <> '>'
      do
        incr j
      done;
      let w = String.sub s !i (!j - !i) in
      acc :=
        (match float_of_string_opt w with Some v -> Num v | None -> Word w)
        :: !acc;
      i := !j
    end
  done;
  List.rev !acc

(* [operands op s] is the numbers before each operator [op] of [s]. *)
let operands op s =
  let rec loop nums acc = function
    | [] -> List.rev acc
    | Num v :: rest -> loop (v :: nums) acc rest
    | Word w :: rest when w = op -> loop [] (List.rev nums :: acc) rest
    | _ :: rest -> loop [] acc rest
  in
  loop [] [] (tokens s)

(* [points s] is the points that the moves and lines of [s] go to, in order. *)
let points s =
  let rec loop nums acc = function
    | [] -> List.rev acc
    | Num v :: rest -> loop (v :: nums) acc rest
    | Word ("m" | "l") :: rest -> (
        match nums with
        | y :: x :: _ -> loop [] ((x, y) :: acc) rest
        | _ -> loop [] acc rest)
    | _ :: rest -> loop [] acc rest
  in
  loop [] [] (tokens s)

(* [shown s] is the glyph ids that the text of [s] shows, in order. *)
let shown s =
  List.filter_map (function Hex h -> Some h | _ -> None) (tokens s)

(* [widths doc] is the glyph widths of the first font of [doc]. *)
let widths doc =
  let d = dicts doc in
  let start = Option.get (find_from d 0 "/W [") + 4 in
  let stop = Option.get (find_from d start "] /CIDToGIDMap") in
  let rec loop acc = function
    | Num g :: Word "[" :: Num w :: Word "]" :: rest ->
        loop ((int_of_float g, w) :: acc) rest
    | [] -> acc
    | _ -> failf "widths"
  in
  loop [] (tokens (String.sub d start (stop - start)))

(* [glyph_xs doc] is where the content of [doc] shows each glyph along x, for
   text set with a matrix that turns nothing. *)
let glyph_xs doc =
  let w = widths doc in
  let rec loop size x pen acc = function
    | [] -> List.rev acc
    | Num s :: Word "Tf" :: rest -> loop s x pen acc rest
    | Num _ :: Num _ :: Num _ :: Num _ :: Num e :: Num _ :: Word "Tm" :: rest ->
        loop size e 0. acc rest
    | Hex g :: rest ->
        let g = int_of_string ("0x" ^ g) in
        loop size x
          (pen +. (size *. List.assoc g w /. 1000.))
          ((x +. pen) :: acc) rest
    | Num n :: rest -> loop size x (pen -. (n *. size /. 1000.)) acc rest
    | _ :: rest -> loop size x pen acc rest
  in
  loop 0. 0. 0. [] (tokens (content doc))

(* [forms doc] is the number of forms of [doc]. *)
let forms doc = count (dicts doc) "/Subtype /Form"

(* [bbox doc] is the box of the first form of [doc]. *)
let bbox doc =
  let d = dicts doc in
  let start = Option.get (find_from d 0 "/BBox [") + 7 in
  let stop = String.index_from d start ']' in
  match tokens (String.sub d start (stop - start)) with
  | [ Num a; Num b; Num c; Num d ] -> (a, b, c, d)
  | _ -> failf "bbox"

(* Pictures *)

let red = Color.red
let rect x y w h = Path.rect (Box2.v x y w h)
let square = rect 10. 20. 30. 40.
let disc = Path.circle (P2.v 0. 0.) 2.
let glyph c = Font.glyph Font.regular (Uchar.of_char c)

(* [one text g] is a run of the one glyph [g] rendering [text]. *)
let one text g =
  Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text ~glyphs:[| g |]
    ~xs:[| 0. |] ()

let marker =
  Picture.group
    [
      Picture.fill Color.black disc;
      Picture.stroke (Stroke.v 0.5) Color.white disc;
    ]

let document =
  group "document"
    [
      test "is a PDF 1.7 file whose table finds every object" (fun () ->
          let doc = pdf (Picture.fill red square) in
          equal string "%PDF-1.7\n" (String.sub doc 0 9);
          equal string "%%EOF\n" (String.sub doc (String.length doc - 6) 6);
          equal int 5 (List.length (objects doc));
          is_true ~msg:"trailer"
            (contains doc "trailer\n<< /Size 6 /Root 1 0 R >>"));
      test "has one page of the renderable's size" (fun () ->
          let d = dicts (pdf ~w:360.5 ~h:240. Picture.empty) in
          is_true ~msg:"media box" (contains d "/MediaBox [0 0 360.5 240]");
          is_true ~msg:"one page" (contains d "/Count 1"));
      test "has no date and no identifier" (fun () ->
          let doc = pdf Vg_corpus.(Renderable.picture marks) in
          equal int 0 (count doc "/CreationDate");
          equal int 0 (count doc "/ID"));
      test "deflates every stream" (fun () ->
          let doc = pdf Vg_corpus.(Renderable.picture marks) in
          List.iter
            (fun (n, d, s) ->
              if s <> None then
                is_true ~msg:(string_of_int n)
                  (contains d "/Filter /FlateDecode"))
            (objects doc));
      test "draws in the page's y-down coordinates" (fun () ->
          expect (content (pdf ~h:240. (Picture.fill red square)))
          @@ __POS_OF__
               {|
            1 0 0 -1 0 240 cm
            q
            1 0 0 rg
            10 20 m
            40 20 l
            40 60 l
            10 60 l
            h
            f
            Q
          |});
    ]

(* Leaves *)

let stroked ?(cap = `Butt) ?(join = `Miter) ?(miter_limit = 10.) ?dash
    ?dash_offset w =
  content
    (pdf
       (Picture.stroke
          (Stroke.v ~cap ~join ~miter_limit ?dash ?dash_offset w)
          red
          (Path.polyline [| 10.; 50.; 50. |] [| 10.; 10.; 50. |])))

let typeset_at y ?font text size =
  Picture.glyphs red (P2.v 5. y) (Vg_corpus.typeset ?font text size)

let leaves =
  group "leaves"
    [
      test "an even-odd fill says so" (fun () ->
          is_true
            (contains
               (content (pdf (Picture.fill ~rule:`Even_odd red square)))
               "f*"));
      test "an alpha below 1 is a graphics state's ca for fills" (fun () ->
          let doc = pdf (Picture.fill (Color.with_alpha 0.25 red) square) in
          is_true ~msg:"state"
            (contains (dicts doc) "/G1 << /Type /ExtGState /ca 0.25 >>");
          is_true ~msg:"set" (contains (content doc) "/G1 gs"));
      test "an alpha below 1 is a graphics state's CA for strokes" (fun () ->
          let doc =
            pdf (Picture.stroke (Stroke.v 1.) (Color.with_alpha 0.5 red) square)
          in
          is_true (contains (dicts doc) "/CA 0.5"));
      test "a stroke writes its pen, PDF's defaults left out" (fun () ->
          let s = stroked 2. in
          equal (list (list float_exact)) [ [ 1.; 0.; 0. ] ] (operands "RG" s);
          equal (list (list float_exact)) [ [ 2. ] ] (operands "w" s);
          equal int 0 (count s " J\n");
          equal int 0 (count s " j\n");
          equal int 0 (count s " M\n");
          equal int 0 (count s " d\n"));
      test "a stroke writes caps, joins and limits that differ" (fun () ->
          let s = stroked ~cap:`Square ~join:`Round 2. in
          is_true ~msg:"cap" (contains s "2 J\n");
          is_true ~msg:"join" (contains s "1 j\n");
          is_true ~msg:"limit" (contains (stroked ~miter_limit:4. 2.) "4 M\n"));
      test "a dashed stroke writes its pattern and phase" (fun () ->
          equal int 1
            (count (stroked ~dash:[ 4.; 2. ] ~dash_offset:1. 2.) "[4 2] 1 d\n"));
      prop "dashes keep within the accuracy over a thousand lengths"
        (Gen.pair (Gen.float_range 0.3 3.)
           (Gen.pair (Gen.float_range 0.1 10.) (Gen.float_range 0.1 10.)))
        (fun (k, (a, b)) ->
          let s =
            content
              (pdf
                 (Picture.transform (Affine.scale k k)
                    (Picture.stroke
                       (Stroke.v ~dash:[ a; b ] 1.)
                       red
                       (Path.polyline [| 10.; 20. |] [| 10.; 10. |]))))
          in
          let rec pattern = function
            | Word "[" :: rest ->
                let rec nums acc = function
                  | Num v :: rest -> nums (v :: acc) rest
                  | _ -> List.rev acc
                in
                nums [] rest
            | _ :: rest -> pattern rest
            | [] -> failf "no dash pattern"
          in
          match pattern (tokens s) with
          | [ a'; b' ] ->
              at_most float_exact ~than:0.0005
                (500. *. Float.abs (a' +. b' -. (k *. (a +. b))))
          | l -> failf "%d lengths" (List.length l));
      test "a dash pattern written as zeros is solid" (fun () ->
          let s = stroked ~dash:[ 1e-7; 2e-7 ] 2. in
          equal int 0 (count s " d\n");
          equal int 1 (count s "S\n"));
      test "a stroke thinner than the accuracy is left out" (fun () ->
          equal int 0 (count (stroked 0.0004) "S\n");
          equal
            (list (list float_exact))
            [ [ 0.001 ] ]
            (operands "w" (stroked 0.001)));
      test "a stroke whose matrix rounds to a flat one is left out" (fun () ->
          let flat =
            Picture.transform (Affine.scale 1. 1e-20)
              (Picture.stroke (Stroke.v 1.) red square)
          in
          equal int 0 (count (content (pdf flat)) "S\n"));
      test "a subpath of zero length with square caps is left out" (fun () ->
          let point =
            Path.empty |> Path.move_to (P2.v 5. 5.) |> Path.line_to (P2.v 5. 5.)
          in
          equal int 0
            (count
               (content
                  (pdf (Picture.stroke (Stroke.v ~cap:`Square 2.) red point)))
               " m\n"));
      test "a run is text in an embedded CID font" (fun () ->
          let doc = pdf (typeset_at 50. "AV" 10.) in
          let d = dicts doc and c = content doc in
          is_true ~msg:"Type0"
            (contains d
               "/Subtype /Type0 /BaseFont /Inter-Regular /Encoding /Identity-H");
          is_true ~msg:"CIDFontType2" (contains d "/Subtype /CIDFontType2");
          is_true ~msg:"size" (contains c "/F1 10 Tf");
          is_true ~msg:"upright at the origin" (contains c "1 0 0 -1 5 50 Tm");
          equal string (Font.bytes Font.regular) (stream_of doc "/Length1"));
      test "a run off one baseline places each line of glyphs on its own"
        (fun () ->
          let r =
            Run.v ~ys:[| 0.; 2.5 |] ~font:Font.regular ~size:10. ~text:"ab"
              ~glyphs:[| glyph 'a'; glyph 'b' |]
              ~xs:[| 0.; 6. |] ()
          in
          let c = content (pdf (Picture.glyphs red (P2.v 5. 50.) r)) in
          equal int 2 (count c " Tm");
          is_true ~msg:"a at its y" (contains c "1 0 0 -1 5 50 Tm");
          is_true ~msg:"b at its y" (contains c "1 0 0 -1 11 52.5 Tm"));
      prop ~tags:[ "slow" ]
        ~examples:
          [ (1.0000185299292945, [ 18.918466897345255; 26.100956639689798 ]) ]
        "glyphs are shown within 0.001 of where the run puts them"
        (Gen.pair (Gen.float_range 1. 40.)
           (Gen.list ~size:(Gen.int_range 1 40) (Gen.float_range (-3.) 30.)))
        (fun (size, steps) ->
          let n = List.length steps in
          let xs = Array.of_list steps in
          for i = 1 to n - 1 do
            xs.(i) <- xs.(i - 1) +. xs.(i)
          done;
          let r =
            Run.v ~font:Font.regular ~size ~text:(String.make n 'a')
              ~glyphs:(Array.make n (glyph 'a'))
              ~xs ()
          in
          let got =
            glyph_xs (pdf ~w:500. (Picture.glyphs red (P2.v 7. 50.) r))
          in
          equal int n (List.length got);
          List.iteri
            (fun i x ->
              at_most ~msg:(string_of_int i) float_exact ~than:0.001
                (Float.abs (x -. (7. +. xs.(i)))))
            got);
      test "a font is embedded once for runs of equal fonts" (fun () ->
          let copy = Result.get_ok (Font.of_string (Font.bytes Font.regular)) in
          let doc =
            pdf
              (Picture.group
                 [ typeset_at 20. "a" 10.; typeset_at 40. ~font:copy "a" 10. ])
          in
          equal int 1 (count (dicts doc) "/FontFile2");
          equal int 2 (count (content doc) "/F1 10 Tf"));
      test "a map gives each glyph the character it renders" (fun () ->
          let doc = pdf (typeset_at 50. "AV" 10.) in
          let map =
            Option.get
              (List.find_map
                 (fun (_, _, s) ->
                   match s with
                   | Some s when contains s "beginbfchar" -> Some s
                   | _ -> None)
                 (objects doc))
          in
          is_true ~msg:"A"
            (contains map (Printf.sprintf "<%04X> <0041>" (glyph 'A')));
          is_true ~msg:"V"
            (contains map (Printf.sprintf "<%04X> <0056>" (glyph 'V')));
          equal int 0 (count (content doc) "ActualText"));
      test "a map of more than 100 glyphs is written in blocks of 100"
        (fun () ->
          let chars =
            List.map Uchar.of_int
              (List.init 94 (fun i -> 33 + i) @ List.init 95 (fun i -> 0xA1 + i))
          in
          let text =
            let b = Buffer.create 256 in
            List.iter (Buffer.add_utf_8_uchar b) chars;
            Buffer.contents b
          in
          let glyphs =
            Array.of_list (List.map (Font.glyph Font.regular) chars)
          in
          let r =
            Run.v ~font:Font.regular ~size:10. ~text ~glyphs
              ~xs:(Array.init (Array.length glyphs) (fun i -> Float.of_int i))
              ()
          in
          (* Each glyph but .notdef renders the first character set with it. *)
          let distinct =
            List.length
              (List.sort_uniq Int.compare
                 (List.filter (fun g -> g <> 0) (Array.to_list glyphs)))
          in
          let map =
            List.find_map
              (fun (_, _, s) ->
                match s with
                | Some s when contains s "beginbfchar" -> Some s
                | _ -> None)
              (objects (pdf ~w:200. (Picture.glyphs red (P2.v 0. 50.) r)))
          in
          equal
            (list (list float_exact))
            [ [ 100. ]; [ Float.of_int (distinct - 100) ] ]
            (operands "beginbfchar" (Option.get map)));
      cases ~name:fst "a run carries its text when"
        [
          ("a glyph renders two characters", `Ligature);
          ("a glyph is .notdef", `Notdef);
          ("two characters share a glyph", `Shared);
        ]
        (fun (_, case) ->
          let at y r = Picture.glyphs red (P2.v 5. y) r in
          let p =
            match case with
            | `Ligature -> at 50. (one "fi" (glyph 'f'))
            | `Notdef -> at 50. (one "\u{4E2D}" 0)
            | `Shared ->
                Picture.group
                  [ at 20. (one "a" (glyph 'a')); at 50. (one "b" (glyph 'a')) ]
          in
          equal int 1 (count (content (pdf p)) "/ActualText <FEFF"));
      test "a run's text beyond the Basic Multilingual Plane is UTF-16"
        (fun () ->
          let p = Picture.glyphs red (P2.v 5. 50.) (one "\u{10000}" 0) in
          is_true (contains (content (pdf p)) "/ActualText <FEFFD800DC00>"));
      test "a run of glyphs without text carries its empty text" (fun () ->
          let r =
            Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text:""
              ~glyphs:[| glyph 'a' |]
              ~xs:[| 0. |] ()
          in
          equal (list string)
            [ "FEFF"; Printf.sprintf "%04X" (glyph 'a') ]
            (shown (content (pdf (Picture.glyphs red (P2.v 5. 50.) r)))));
      test "a run whose size is written as zero is left out" (fun () ->
          equal int 0 (count (content (pdf (typeset_at 50. "a" 1e-9))) "Tf");
          equal int 1
            (count (content (pdf (typeset_at 50. "a" 0.001))) "/F1 0.001 Tf"));
      test "an image is painted over its box, unsmoothed, its alpha a mask"
        (fun () ->
          let px =
            Nx.init Nx.uint8 [| 2; 3; 4 |] (fun i ->
                (i.(0) * 100) + (i.(1) * 50) + (i.(2) * 9))
          in
          let doc = pdf (Picture.image (Box2.v 10. 20. 30. 40.) px) in
          is_true ~msg:"unsmoothed"
            (contains (dicts doc)
               "/Width 3 /Height 2 /ColorSpace /DeviceRGB /BitsPerComponent 8 \
                /Interpolate false /SMask");
          is_true ~msg:"placed" (contains (content doc) "30 0 0 -40 10 60 cm");
          let a = Nx.to_array px in
          equal string ~msg:"colour"
            (String.init 18 (fun k -> Char.chr a.((k / 3 * 4) + (k mod 3))))
            (stream_of doc "/SMask");
          equal string ~msg:"alpha"
            (String.init 6 (fun k -> Char.chr a.((k * 4) + 3)))
            (stream_of doc "/DeviceGray"));
      test "a grey image is DeviceGray, without a mask" (fun () ->
          let px = Nx.init Nx.uint8 [| 1; 2; 1 |] (fun i -> 7 + i.(1)) in
          let doc = pdf (Picture.image (Box2.v 0. 0. 2. 1.) px) in
          equal int 0 (count (dicts doc) "/SMask");
          equal string "\007\008" (stream_of doc "/DeviceGray"));
      test "an image thinner than the accuracy is left out" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let doc = pdf (Picture.image (Box2.v 10. 10. 0.00001 10.) px) in
          equal int 0 (count (content doc) "Do"));
      test "a clip is a clipping path" (fun () ->
          let s =
            content
              (pdf
                 (Picture.clip ~rule:`Even_odd square (Picture.fill red square)))
          in
          is_true (contains s "W* n\n"));
      test "a clip of no area within the margin hides its picture" (fun () ->
          let far = rect 1e6 1e6 10. 10. in
          let s = content (pdf (Picture.clip far (Picture.fill red square))) in
          equal int 0 (count s " n\n");
          equal int 0 (count s "f\n"));
      test "a transform writes no operator of its own" (fun () ->
          let s =
            content
              (pdf
                 (Picture.transform (Affine.translate 5. 7.)
                    (Picture.fill red square)))
          in
          equal int 1 (count s " cm\n");
          equal
            (list (pair float_exact float_exact))
            [ (15., 27.); (45., 27.); (45., 67.); (15., 67.) ]
            (points s));
      test "an opacity is a transparency group painted with its alpha"
        (fun () ->
          let doc = pdf (Picture.opacity 0.5 (Picture.fill red square)) in
          let d = dicts doc in
          is_true ~msg:"group"
            (contains d "/Group << /S /Transparency /I true >>");
          is_true ~msg:"alpha"
            (contains d "<< /Type /ExtGState /ca 0.5 /CA 0.5 >>");
          is_true ~msg:"painted" (contains (content doc) "/G1 gs\n/X1 Do"));
      test "an opacity of 0 paints nothing" (fun () ->
          let doc = pdf (Picture.opacity 0. (Picture.fill red square)) in
          equal int 0 (count (dicts doc) "/Subtype /Form");
          equal int 0 (count (content doc) "Do"));
      test "tags are ignored" (fun () ->
          let p = Picture.fill red square in
          let t =
            { Picture.id = Nx.Ptree.Path.v [ Field "a" ]; rows = Rows [| 1 |] }
          in
          equal string (pdf p) (pdf (Picture.tag t p)));
    ]

(* Accuracy *)

let gen_far_map =
  Gen.map
    (fun ((ox, oy), (sx, sy)) -> Affine.(scale sx sy * translate (-.ox) (-.oy)))
    (Gen.pair
       (Gen.pair (Gen.float_range (-2e9) 2e9) (Gen.float_range (-2e9) 2e9))
       (Gen.pair (Gen.float_range 0.01 400.) (Gen.float_range 0.01 400.)))

let gen_page_points =
  Gen.list ~size:(Gen.int_range 3 6)
    (Gen.pair (Gen.float_range 0. 100.) (Gen.float_range 0. 100.))

(* [written ~w m pts] checks that the points [pts] of the page, drawn as a
   polygon in the coordinates that [m] maps to the page, are written within
   0.001 of where they are. *)
let written ?(w = 100.) m pts =
  let inv = Option.get (Affine.invert m) in
  let user = List.map (fun (x, y) -> P2.transform inv (P2.v x y)) pts in
  let q =
    Path.polyline
      (Array.of_list (List.map P2.x user))
      (Array.of_list (List.map P2.y user))
  in
  let got =
    points (content (pdf ~w (Picture.transform m (Picture.fill red q))))
  in
  equal int (List.length pts) (List.length got);
  List.iter2
    (fun (x, y) (x', y') ->
      at_most ~msg:"x" float_exact ~than:0.001 (Float.abs (x -. x'));
      at_most ~msg:"y" float_exact ~than:0.001 (Float.abs (y -. y')))
    pts got

(* [far_line] runs from (-123456789, -123456700) to (123456789, 123456900): it
   crosses x = 0 at y = 100. *)
let far_line =
  Path.polyline [| -123456789.; 123456789. |] [| -123456700.; 123456900. |]

(* [gen_marker_points] is the corners of a quadrilateral around the origin, one
   in each quadrant, so that it has an area to fill. *)
let gen_marker_points =
  let coord = Gen.float_range 0.1 1. in
  Gen.map
    (fun ((a, b), (c, d), (e, f), (g, h)) ->
      [ (-.a, -.b); (c, -.d); (e, f); (-.g, h) ])
    (Gen.quad (Gen.pair coord coord) (Gen.pair coord coord)
       (Gen.pair coord coord) (Gen.pair coord coord))

(* [placed scale x y pts] checks that the points [pts] of a polygon stamped at
   [(x, y)] and scaled by [scale] are placed within 0.001 of where they are. *)
let placed scale x y pts =
  let q =
    Path.polyline
      (Array.of_list (List.map fst pts))
      (Array.of_list (List.map snd pts))
  in
  let doc =
    pdf (Picture.stamp ~scales:[| scale |] [| x |] [| y |] (Picture.fill red q))
  in
  match operands "cm" (content doc) with
  | [ _; [ sx; 0.; 0.; sy; tx; ty ] ] ->
      let got = points (stream_of doc "/Subtype /Form") in
      equal int (List.length pts) (List.length got);
      List.iter2
        (fun (px, py) (fx, fy) ->
          at_most ~msg:"x" float_exact ~than:0.001
            (Float.abs (tx +. (sx *. fx) -. (x +. (scale *. px))));
          at_most ~msg:"y" float_exact ~than:0.001
            (Float.abs (ty +. (sy *. fy) -. (y +. (scale *. py)))))
        pts got
  | _ -> failf "one instance matrix"

let accuracy =
  group "accuracy"
    [
      prop "points are written within 0.001 of where the page puts them"
        (Gen.pair gen_far_map gen_page_points) (fun (m, pts) -> written m pts);
      test "an offset of 1.6e9 under a scale keeps a 400-point line level"
        (fun () ->
          written ~w:400.
            Affine.(scale 4. 1. * translate (-1.6e9) 0.)
            [ (0., 50.); (400., 50.); (400., 60.) ]);
      test "a line crossing the page far beyond it crosses where it should"
        (fun () ->
          let s =
            content
              (pdf ~w:200. ~h:200. (Picture.stroke (Stroke.v 1.) red far_line))
          in
          match points s with
          | [ (x0, y0); (x1, y1) ] ->
              (* Cut at the margin, with numbers of the order of the page. *)
              List.iter
                (fun v ->
                  at_most ~msg:"magnitude" float_exact ~than:1000. (Float.abs v))
                [ x0; y0; x1; y1 ];
              let y = y0 +. ((y1 -. y0) *. (0. -. x0) /. (x1 -. x0)) in
              at_most float_exact ~than:0.001 (Float.abs (y -. 100.))
          | pts -> failf "%d points" (List.length pts));
      test "a dashed line cut at the margin keeps the phase of its dashes"
        (fun () ->
          let s =
            content
              (pdf
                 (Picture.stroke
                    (Stroke.v ~cap:`Butt ~dash:[ 3.; 1. ] 1.)
                    red
                    (Path.polyline [| -999999.5; 1e6 |] [| 50.; 50. |])))
          in
          (* The page, its margin and the pen's reach of 0.5 start 999899 along
             the line, 3 into the pattern of period 4. *)
          is_true (contains s "[3 1] 3 d\n-100.5 50 m\n200.5 50 l\nS\n"));
      test "an odd dash pattern cut at the margin keeps its phase" (fun () ->
          let s =
            content
              (pdf
                 (Picture.stroke
                    (Stroke.v ~cap:`Butt ~dash:[ 3. ] 1.)
                    red
                    (Path.polyline [| -999999.5; 1e6 |] [| 50.; 50. |])))
          in
          (* The pattern repeats as [3 3]: the cut, 999899 along the line, is 5
             into its period of 6. *)
          is_true (contains s "[3] 5 d\n"));
      test "a closed subpath cut at the margin keeps its join at its start"
        (fun () ->
          let q =
            Path.polygon [| 50.; 1e5; 1e5; 50. |] [| 50.; 50.; 60.; 60. |]
          in
          let s =
            content
              (pdf
                 (Picture.stroke
                    (Stroke.v ~cap:`Butt ~join:`Miter ~miter_limit:4. 1.)
                    red q))
          in
          (* Cut where the margin and the miter's reach of 2 end. *)
          is_true (contains s "202 60 m\n50 60 l\n50 50 l\n202 50 l\nS\n"));
      test "a curve crossing the margin is flattened, one within it kept"
        (fun () ->
          let big = Path.circle (P2.v 50. 1e6) (1e6 -. 50.) in
          let s = content (pdf (Picture.fill red big)) in
          equal int 0 (count s " c\n");
          List.iter
            (fun (x, y) ->
              at_most ~msg:"magnitude" float_exact ~than:1000.
                (Float.max (Float.abs x) (Float.abs y)))
            (points s);
          let small = pdf (Picture.fill red (Path.circle (P2.v 50. 50.) 10.)) in
          equal int 4 (count (content small) " c\n"));
      test "an image is cropped to the pixels that meet the margin" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1000; 3 |] in
          let doc = pdf (Picture.image (Box2.v (-1e6) 0. 2e6 10.) px) in
          (* Each pixel is 2000 points wide: the margin of 100 points around the
             page meets two, from -2000 to 2000. *)
          is_true ~msg:"two pixels" (contains (dicts doc) "/Width 2 /Height 1");
          is_true ~msg:"placed"
            (contains (content doc) "4000 0 0 -10 -2000 10 cm"));
      test "a cropped image keeps the pixels it shows" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 3; 1000; 1 |] (fun i ->
                ((i.(0) * 100) + i.(1)) mod 256)
          in
          let doc = pdf (Picture.image (Box2.v (-1e6) 0. 2e6 30.) px) in
          (* Columns 499 and 500 meet the margin. *)
          equal string
            (String.init 6 (fun k ->
                 Char.chr (((k / 2 * 100) + 499 + (k mod 2)) mod 256)))
            (stream_of doc "/DeviceGray"));
      test "an image is cropped to the rows that meet the margin" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1000; 1; 3 |] in
          let doc = pdf (Picture.image (Box2.v 0. (-1e6) 10. 2e6) px) in
          (* Each row is 2000 points high: the margin meets two, from -2000 to
             2000. *)
          is_true ~msg:"two rows" (contains (dicts doc) "/Width 1 /Height 2");
          is_true ~msg:"placed"
            (contains (content doc) "10 0 0 -4000 0 2000 cm"));
      cases ~name:fst "an image beyond the margin on one side is cropped there"
        [
          ("left", (1, 1000, Box2.v (-999900.) 0. 1e6 10.));
          ("right", (1, 1000, Box2.v (-100.) 0. 1e6 10.));
          ("top", (1000, 1, Box2.v 0. (-999900.) 10. 1e6));
          ("bottom", (1000, 1, Box2.v 0. (-100.) 10. 1e6));
        ]
        (fun (_, (h, w, box)) ->
          let px = Nx.zeros Nx.uint8 [| h; w; 3 |] in
          is_true
            (contains
               (dicts (pdf (Picture.image box px)))
               "/Width 1 /Height 1 /ColorSpace"));
      test "an image is placed within the accuracy" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let doc =
            pdf (Picture.image (Box2.v 10.0004 20.0004 30.0037 40.0042) px)
          in
          match operands "cm" (content doc) with
          | [ _; [ xx; yx; xy; yy; x0; y0 ] ] ->
              List.iter2
                (fun (name, expected) v ->
                  at_most ~msg:name float_exact ~than:0.001
                    (Float.abs (v -. expected)))
                [
                  ("xx", 30.0037);
                  ("yx", 0.);
                  ("xy", 0.);
                  ("yy", -40.0042);
                  ("x0", 10.0004);
                  ("y0", 60.0046);
                ]
                [ xx; yx; xy; yy; x0; y0 ]
          | _ -> failf "one image matrix");
      test "an image turned by 45 degrees is painted" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let p =
            Picture.transform
              Affine.(translate 50. 50. * rotate (Float.pi /. 4.))
              (Picture.image (Box2.v 0. 0. 20. 20.) px)
          in
          equal int 1 (count (content (pdf p)) "Do"));
      test "an image is written once per pixels and crop" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1000; 3 |]
          and other = Nx.ones Nx.uint8 [| 1; 1000; 3 |] in
          let whole = Box2.v 0. 0. 100. 10.
          and far = Box2.v (-1e6) 0. 2e6 10. in
          let images p = count (dicts (pdf p)) "/Subtype /Image" in
          equal int ~msg:"one tensor, one crop" 1
            (images
               (Picture.group
                  [ Picture.image whole px; Picture.image whole px ]));
          equal int ~msg:"one tensor, two crops" 2
            (images
               (Picture.group [ Picture.image whole px; Picture.image far px ]));
          equal int ~msg:"equal tensors, one crop" 1
            (images
               (Picture.group
                  [ Picture.image whole px; Picture.image whole (Nx.copy px) ]));
          equal int ~msg:"other pixels, one crop" 2
            (images
               (Picture.group
                  [ Picture.image whole px; Picture.image whole other ])));
      test "an opacity beyond the margin writes nothing" (fun () ->
          let p =
            Picture.opacity 0.5 (Picture.fill red (rect 10. 1e6 10. 10.))
          in
          equal int 0 (forms (pdf p)));
      test "a run or an instance beyond the margin is left out" (fun () ->
          let r =
            Picture.glyphs red (P2.v 1e5 50.) (Vg_corpus.typeset "far" 10.)
          in
          equal int 0 (count (content (pdf r)) "TJ");
          let st =
            Picture.stamp [| 50.; 1e5 |] [| 50.; 50. |]
              (Picture.fill red square)
          in
          equal int 1 (count (content (pdf st)) "Do"));
      test "glyphs beyond the margin are left out, the run keeping its text"
        (fun () ->
          (* The margin spans -100 to 200: the ink of the first four glyphs,
             just within each of its edges, meets it, and the last two are far
             beyond. *)
          let r =
            Run.v ~font:Font.regular ~size:10. ~text:"abcdef"
              ~glyphs:(Array.map glyph [| 'a'; 'b'; 'c'; 'd'; 'e'; 'f' |])
              ~xs:[| 199.; -99.; 50.; 50.; 1e300; 50. |]
              ~ys:[| 50.; 50.; 199.; -99.; 50.; 1e300 |]
              ()
          in
          let s = content (pdf (Picture.glyphs red (P2.v 0. 0.) r)) in
          equal (list string)
            ("FEFF006100620063006400650066"
            :: List.map
                 (fun c -> Printf.sprintf "%04X" (glyph c))
                 [ 'a'; 'b'; 'c'; 'd' ])
            (shown s));
      test "a pen stretched unevenly is written under a matrix" (fun () ->
          let p =
            Picture.transform (Affine.scale 1. 4.)
              (Picture.stroke (Stroke.v 1.) red
                 (Path.polyline [| 0.; 10. |] [| 0.; 10. |]))
          in
          let s = content (pdf p) in
          is_true ~msg:"matrix" (contains s "0.25 0 0 1 0 0 cm\n");
          equal (list (list float_exact)) [ [ 4. ] ] (operands "w" s);
          (* Under the matrix, the end (10, 40) on the page is (40, 40). *)
          is_true ~msg:"path" (contains s "0 0 m\n40 40 l\n"));
      test "a pen turned and scaled evenly is written on the page" (fun () ->
          let p =
            Picture.transform
              Affine.(rotate (Float.pi /. 3.) * scale 2. 2.)
              (Picture.stroke (Stroke.v 1.) red
                 (Path.polyline [| 0.; 10. |] [| 0.; 0. |]))
          in
          let s = content (pdf p) in
          equal int 1 (count s " cm\n");
          equal (list (list float_exact)) [ [ 2. ] ] (operands "w" s));
      test "a turned run is written under a matrix at its origin" (fun () ->
          let p =
            Picture.transform
              Affine.(translate 50. 50. * rotate (Float.pi /. 2.) * scale 2. 2.)
              (Picture.glyphs red (P2.v 0. 0.) (Vg_corpus.typeset "a" 10.))
          in
          let s = content (pdf p) in
          is_true ~msg:"matrix" (contains s "0 1 1 0 50 50 Tm");
          is_true ~msg:"size" (contains s "/F1 20 Tf"));
      test "an image under a turn is written under its matrix" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let p =
            Picture.transform
              (Affine.rotate (Float.pi /. 2.))
              (Picture.image (Box2.v 10. 0. 20. 10.) px)
          in
          is_true (contains (content (pdf p)) "0 20 10 0 -10 10 cm"));
      prop
        "a stamp's points are placed within 0.001 of where the page puts them"
        (Gen.triple
           (Gen.float_range 0.01 1000.)
           (Gen.pair (Gen.float_range 0. 100.) (Gen.float_range 0. 100.))
           gen_marker_points)
        (fun (scale, (x, y), pts) -> placed scale x y pts);
    ]

(* Stamps *)

let stamps =
  group "stamps"
    [
      test "a stamp writes its picture once and paints it per instance"
        (fun () ->
          let doc =
            pdf (Picture.stamp [| 10.; 20.; 30. |] [| 5.; 6.; 7. |] marker)
          in
          equal int 1 (forms doc);
          equal
            (list (list float_exact))
            [
              [ 1.; 0.; 0.; 1.; 10.; 5. ];
              [ 1.; 0.; 0.; 1.; 20.; 6. ];
              [ 1.; 0.; 0.; 1.; 30.; 7. ];
            ]
            (List.tl (operands "cm" (content doc))));
      test "fills and strokes are set before each instance" (fun () ->
          let doc =
            pdf
              (Picture.stamp
                 ~fills:[| red; Color.with_alpha 0.5 Color.blue |]
                 ~strokes:[| Color.blue; red |] [| 10.; 20. |] [| 5.; 5. |]
                 marker)
          in
          let c = content doc in
          equal
            (list (list float_exact))
            [ [ 1.; 0.; 0. ]; [ 0.; 0.; 1. ] ]
            (operands "rg" c);
          equal
            (list (list float_exact))
            [ [ 0.; 0.; 1. ]; [ 1.; 0.; 0. ] ]
            (operands "RG" c);
          is_true ~msg:"alpha" (contains (dicts doc) "/ca 0.5 >>");
          (* The form leaves the colours to its instances. *)
          let form = stream_of doc "/Subtype /Form" in
          equal int 0 (count form "rg");
          equal int 0 (count form "RG"));
      test "an image in a form that fills inherit resets ca" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let p = Picture.image (Box2.v 0. 0. 2. 2.) px in
          let doc =
            pdf
              (Picture.stamp
                 ~fills:[| Color.with_alpha 0.5 red |]
                 [| 10. |] [| 5. |] p)
          in
          is_true ~msg:"state" (contains (dicts doc) "/ca 1 >>");
          is_true ~msg:"set" (contains (stream_of doc "/Subtype /Form") " gs\n"));
      test "a scaling stamp scales its instances and keeps their pens"
        (fun () ->
          let ring = Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc in
          let doc =
            pdf
              (Picture.stamp ~scales:[| 2.; 4. |] [| 10.; 20. |] [| 5.; 5. |]
                 ring)
          in
          let c = content doc in
          equal (list (list float_exact)) [ [ 0.5 ]; [ 0.25 ] ] (operands "w" c);
          is_true ~msg:"dashes" (contains c "[1 0.5] 0 d\n");
          let form = stream_of doc "/Subtype /Form" in
          equal int ~msg:"the form's widths" 0 (count form " w\n");
          equal int ~msg:"the form's dashes" 0 (count form " d\n");
          equal
            (list (list float_exact))
            [ [ 2.; 0.; 0.; 2.; 10.; 5. ]; [ 4.; 0.; 0.; 4.; 20.; 5. ] ]
            (List.tl (operands "cm" c)));
      test "a form's box holds the pens its shrunk instances keep" (fun () ->
          let ring =
            Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 1.)
          in
          let doc =
            pdf (Picture.stamp ~scales:[| 0.1 |] [| 50. |] [| 50. |] ring)
          in
          (* In the form, the pen of width 10 reaches 5 beyond the circle. *)
          let x0, y0, x1, y1 = bbox doc in
          at_most ~msg:"left" float_exact ~than:(-6.) x0;
          at_most ~msg:"top" float_exact ~than:(-6.) y0;
          at_least ~msg:"right" float_exact ~than:6. x1;
          at_least ~msg:"bottom" float_exact ~than:6. y1);
      test "a scaling stamp of a stroke within an opacity is written in full"
        (fun () ->
          let ring =
            Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 1.)
          in
          let doc =
            pdf
              (Picture.stamp ~scales:[| 0.1 |] [| 50. |] [| 50. |]
                 (Picture.opacity 0.5 ring))
          in
          (* The only form is the group, which strokes with the pen kept. *)
          equal int 1 (forms doc);
          equal
            (list (list float_exact))
            [ [ 1. ] ]
            (operands "w" (stream_of doc "/Group")));
      test "a scaling stamp of strokes of differing pens is written in full"
        (fun () ->
          let two =
            Picture.group
              [
                Picture.stroke (Stroke.v 1.) red disc;
                Picture.stroke (Stroke.v 2.) red disc;
              ]
          in
          let doc =
            pdf (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] two)
          in
          equal int 0 (forms doc);
          equal
            (list (list float_exact))
            [ [ 1. ]; [ 2. ] ]
            (operands "w" (content doc)));
      test "a scaling stamp holding a scaling stamp is written in full"
        (fun () ->
          let inner =
            Picture.stamp ~scales:[| 2. |] [| 0. |] [| 0. |]
              (Picture.fill red disc)
          in
          let doc =
            pdf (Picture.stamp ~scales:[| 3. |] [| 10. |] [| 5. |] inner)
          in
          equal int 1 (forms doc);
          (* The inner form is drawn at the outer instance's scale. *)
          equal
            (list (list float_exact))
            [ [ 2.; 0.; 0.; 2.; 10.; 5. ] ]
            (List.tl (operands "cm" (content doc))));
      test "an instance within a form that sets strokes sets its alpha"
        (fun () ->
          let inner =
            Picture.stamp ~strokes:[| red |] [| 0. |] [| 0. |]
              (Picture.stroke (Stroke.v 1.) red disc)
          in
          let doc =
            pdf
              (Picture.stamp
                 ~strokes:[| Color.with_alpha 0.5 red |]
                 [| 10. |] [| 5. |] inner)
          in
          is_true ~msg:"the inner alpha" (contains (dicts doc) "/CA 1 >>");
          is_true ~msg:"the outer alpha" (contains (dicts doc) "/CA 0.5 >>"));
      test "an instance whose pen alone reaches within the margin is drawn"
        (fun () ->
          (* The margin spans -100 to 200, and each ring reaches 3 from its
             centre, 0.1 within it. *)
          let ring =
            Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 2.5)
          in
          let doc =
            pdf
              (Picture.stamp
                 [| 202.9; -102.9; 50.; 50.; 300. |]
                 [| 50.; 50.; 202.9; -102.9; 50. |]
                 ring)
          in
          equal int 4 (count (content doc) "Do"));
      test "a translucent replaced colour over an opacity is written in full"
        (fun () ->
          let faded = Picture.opacity 0.8 (Picture.fill red disc) in
          let doc =
            pdf
              (Picture.stamp
                 ~fills:[| Color.with_alpha 0.5 red |]
                 [| 10. |] [| 5. |] faded)
          in
          (* The only form is the opacity's group, its fill's alpha its own. *)
          equal int 1 (forms doc);
          is_true (contains (stream_of doc "/Group") " gs\n"));
      test "an instance whose scale the form cannot carry is written in full"
        (fun () ->
          let doc =
            pdf
              (Picture.stamp ~scales:[| 1.; 1e-20 |] [| 10.; 20. |] [| 5.; 5. |]
                 marker)
          in
          let c = content doc in
          equal int ~msg:"instances through the form" 1 (count c "Do");
          (* The second keeps its pen, with its round caps, around the point it
             shrinks to. *)
          is_true ~msg:"in full" (contains c "0.5 w\n1 J\n1 j\n20 5 m\n");
          List.iter
            (function
              | [ a; _; _; d; _; _ ] ->
                  is_false ~msg:"singular" (a = 0. || d = 0.)
              | _ -> failf "cm")
            (operands "cm" c));
      test
        "instances at non-finite positions or scales or of scale 0 are left out"
        (fun () ->
          let doc =
            pdf
              (Picture.stamp
                 ~scales:[| 1.; 0.; 1.; 1.; Float.nan; Float.infinity |]
                 [| 10.; 20.; Float.nan; 25.; 30.; 40. |]
                 [| 5.; 5.; 5.; Float.nan; 5.; 5. |]
                 marker)
          in
          equal int 1 (count (content doc) "Do"));
      test "instances at non-finite positions or scales in a form are left out"
        (fun () ->
          let inner =
            Picture.stamp ~scales:[| 1.; Float.nan; 1. |] [| 0.; 1.; 2. |]
              [| 0.; 0.; Float.nan |] marker
          in
          let doc = pdf (Picture.stamp [| 50. |] [| 50. |] inner) in
          let forms =
            List.filter_map
              (fun (_, d, s) -> if contains d "/Subtype /Form" then s else None)
              (objects doc)
          in
          let all = String.concat "" forms in
          equal int ~msg:"instances" 1 (count all " Do\n");
          equal int ~msg:"numbers" 0 (count all "nan"));
      cases ~name:fst "an instance changes nothing"
        [
          ("at x = NaN", (Float.nan, 5., 1000.));
          ("at y = NaN", (20., Float.nan, 1000.));
          ("at y = infinity", (20., Float.infinity, 1000.));
          ("of scale NaN", (20., 5., Float.nan));
          ("of scale infinity", (20., 5., Float.infinity));
        ]
        (fun (_, (x, y, k)) ->
          equal string
            (pdf (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] marker))
            (pdf
               (Picture.stamp ~scales:[| 2.; k |] [| 10.; x |] [| 5.; y |]
                  marker)));
      test "equal stamps share their form" (fun () ->
          let st = Picture.stamp [| 10. |] [| 5. |] marker in
          let doc =
            pdf
              (Picture.group
                 [ st; Picture.transform (Affine.translate 0. 20.) st ])
          in
          equal int 1 (forms doc);
          equal int 2 (count (content doc) "Do"));
    ]

(* Limits *)

(* [numbers_finite doc] checks that the objects of [doc], their streams but font
   files and image samples inflated, hold no number that is not finite. *)
let numbers_finite doc =
  List.iter
    (fun (n, dict, stream) ->
      let text =
        match stream with
        | Some data
          when not (contains dict "/Length1" || contains dict "/Subtype /Image")
          ->
            dict ^ "\n" ^ data
        | _ -> dict
      in
      List.iter
        (function
          | Word ("inf" | "-inf" | "nan" | "-nan") as w ->
              failf "object %d holds %s" n
                (match w with Word w -> w | _ -> assert false)
          | _ -> ())
        (tokens text))
    (objects doc)

let limits =
  group "limits"
    [
      cases ~name:fst
        "transforms whose composition overflows or underflows paint nothing"
        [ ("overflows", 1e200); ("underflows", 1e-200) ]
        (fun (_, k) ->
          let twice p =
            Picture.transform (Affine.scale k k)
              (Picture.transform (Affine.scale k k) p)
          in
          equal
            (list (pair float_exact float_exact))
            []
            (points (content (pdf (twice (Picture.fill red square))))));
      prop ~examples:Vg_corpus.extremes "numbers are finite whatever the scales"
        Vg_corpus.gen_extreme (fun p -> numbers_finite (pdf p));
      cases ~name:(Printf.sprintf "under a scale of %g")
        "a pen keeps its width" [ 1e-170; 1e170 ] (fun k ->
          let p =
            Picture.transform (Affine.scale k k)
              (Picture.stroke
                 (Stroke.v (10. /. k))
                 red
                 (Path.polyline [| 0.; 50. /. k |] [| 50. /. k; 50. /. k |]))
          in
          let c = content (pdf p) in
          equal (list (list float_exact)) [ [ 10. ] ] (operands "w" c);
          equal
            (list (pair float_exact float_exact))
            [ (0., 50.); (50., 50.) ]
            (points c));
    ]

(* Determinism *)

let determinism =
  group "determinism"
    [
      prop "equal renderables give equal documents" Vg_corpus.gen_picture
        (fun p -> equal string (pdf p) (pdf (Vg_corpus.respell p)));
      test "a number that rounds to zero is written 0" (fun () ->
          let s =
            content
              (pdf
                 (Picture.fill red
                    (Path.polygon [| -0.; 10.; -0.0001 |] [| -0.; 0.; 10. |])))
          in
          is_true (contains s "0 0 m\n10 0 l\n0 10 l\n"));
      test "objects are numbered in the order of first use" (fun () ->
          let d =
            dicts
              (pdf
                 (Picture.group
                    [
                      typeset_at 20. ~font:Font.bold "a" 10.;
                      typeset_at 40. "b" 10.;
                    ]))
          in
          let at s = Option.get (find_from d 0 s) in
          is_true (at "/BaseFont /Inter-Bold" < at "/BaseFont /Inter-Regular"));
    ]

(* Goldens *)

(* [dump doc] is [doc] as text: its objects in order, their streams inflated,
   font files by their length and image samples in hexadecimal. *)
let dump doc =
  let b = Buffer.create 4096 in
  List.iter
    (fun (n, dict, stream) ->
      Printf.bprintf b "%d 0 obj\n%s\n" n dict;
      match stream with
      | None -> ()
      | Some data when contains dict "/Length1" ->
          Printf.bprintf b "stream <%d bytes>\n" (String.length data)
      | Some data when contains dict "/Subtype /Image" ->
          Buffer.add_string b "stream\n";
          String.iteri
            (fun i c ->
              Printf.bprintf b "%02x%s" (Char.code c)
                (if i mod 32 = 31 then "\n" else ""))
            data;
          Buffer.add_string b "\nendstream\n"
      | Some data -> Printf.bprintf b "stream\n%s\nendstream\n" data)
    (objects doc);
  Buffer.contents b

let goldens =
  group "goldens"
    (List.map
       (fun (name, r) ->
         test (name ^ " writes its golden document") (fun () ->
             expect_file
               (dump (Hugin_next_vg_pdf.render r))
               ("packages/hugin/next/test/vg/golden/" ^ name ^ ".pdf.txt")))
       Vg_corpus.pages)

let () =
  exit
    (run "hugin.next.vg pdf"
       [ document; leaves; accuracy; stamps; limits; determinism; goldens ])
