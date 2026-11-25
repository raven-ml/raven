(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_vg

let pdf ?(w = 100.) ?(h = 100.) p = Hugin_vg_pdf.render (Renderable.v w h p)
let picture = Testable.make ~pp:Picture.pp ~equal:Picture.equal

(* Reading documents *)

let find_from s i sub =
  let n = String.length sub in
  let rec loop i =
    if i + n > String.length s then None
    else if String.sub s i n = sub then Some i
    else loop (i + 1)
  in
  loop i

let occurs s sub = find_from s 0 sub <> None

(* [count s sub] is the number of occurrences of [sub] in [s]. *)
let count s sub =
  let rec loop i k =
    match find_from s i sub with None -> k | Some j -> loop (j + 1) (k + 1)
  in
  loop 0 0

(* [inflate n s] is the data of the zlib stream [s], object [n]'s, which ends at
   [s]'s end. *)
let inflate n s =
  let open Bigarray in
  let src = Array1.create int8_unsigned c_layout (String.length s) in
  String.iteri (fun i c -> Array1.unsafe_set src i (Char.code c)) s;
  (* A destination that is too small is doubled. *)
  let rec go size =
    let dst = Array1.create int8_unsigned c_layout size in
    match Lz.Zlib.decode src dst with
    | Ok { consumed; produced } ->
        if consumed <> String.length s then
          failf "object %d: %d bytes after its stream" n
            (String.length s - consumed);
        String.init produced (fun i ->
            Char.unsafe_chr (Array1.unsafe_get dst i))
    | Error { why = Too_long; _ } -> go (2 * size)
    | Error e -> failf "object %d: %a" n Lz.pp_error e
  in
  go (4 * (String.length s + 64))

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
      | Some i ->
          let dict = String.sub body 0 i in
          let length =
            let j = Option.get (find_from dict 0 "/Length ") + 8 in
            let k = ref j in
            while dict.[!k] >= '0' && dict.[!k] <= '9' do
              incr k
            done;
            int_of_string (String.sub dict j (!k - j))
          in
          (n, dict, Some (inflate n (String.sub body (i + 8) length))))

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
    List.find_opt (fun (_, d, s) -> s <> None && occurs d sub) (objects doc)
  with
  | Some (_, _, Some s) -> s
  | _ -> failf "no stream with %s" sub

(* [to_unicode doc] is the first stream of [doc] that maps glyphs to
   characters. *)
let to_unicode doc =
  match
    List.find_map
      (fun (_, _, s) ->
        match s with Some s when occurs s "beginbfchar" -> Some s | _ -> None)
      (objects doc)
  with
  | Some map -> map
  | None -> fail "no ToUnicode map"

(* [subset_tag doc] is the tag that names the subset of the first font of
   [doc]. *)
let subset_tag doc =
  let d = dicts doc and base = "/Subtype /Type0 /BaseFont /" in
  match find_from d 0 base with
  | Some i -> String.sub d (i + String.length base) 6
  | None -> fail "no Type0 font"

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

let numbers = list (list float_exact)

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

let points_w = list (pair float_exact float_exact)

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

(* Factors of scales each with an inverse, two of which compose to a map without
   one, under which a picture paints nothing. *)
let lost_maps = [ ("an underflow to zero", 1e-200); ("an overflow", 1e200) ]
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

let typeset_at y ?font text size =
  Picture.glyphs red (P2.v 5. y) (Vg_corpus.typeset ?font text size)

let load_font file =
  match Font.of_string (In_channel.with_open_bin file In_channel.input_all) with
  | Ok font -> font
  | Error e -> failf "%s: %a" file Font.pp_error e

let stroked ?(cap = `Butt) ?(join = `Miter) ?(miter_limit = 10.) ?dash
    ?dash_offset w =
  content
    (pdf
       (Picture.stroke
          (Stroke.v ~cap ~join ~miter_limit ?dash ?dash_offset w)
          red
          (Path.polyline [| 10.; 50.; 50. |] [| 10.; 10.; 50. |])))

(* The document *)

let document =
  group "the document"
    [
      test "is a PDF 1.7 file" (fun () ->
          let doc = pdf (Picture.fill red square) in
          starts_with ~prefix:"%PDF-1.7\n" doc;
          ends_with ~suffix:"%%EOF\n" doc);
      test "has a cross-reference table that finds each of its objects"
        (fun () ->
          (* [objects] fails where the table does not point at the object it
             numbers. *)
          let doc = pdf (Picture.fill red square) in
          contains ~sub:"/Root 1 0 R >>" doc;
          at_least int ~than:1 (List.length (objects doc)));
      test "deflates every stream" (fun () ->
          let unfiltered =
            List.filter_map
              (fun (n, d, s) ->
                if s <> None && not (occurs d "/Filter /FlateDecode") then
                  Some n
                else None)
              (objects (pdf Vg_corpus.(Renderable.picture marks)))
          in
          equal (list int) [] unfiltered);
      test "has one page" (fun () ->
          contains ~sub:"/Count 1" (dicts (pdf Picture.empty)));
      test "has a media box of the renderable's size" (fun () ->
          contains ~sub:"/MediaBox [0 0 360.5 240]"
            (dicts (pdf ~w:360.5 ~h:240. Picture.empty)));
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

(* Fills and strokes *)

let fills_and_strokes =
  group "fills and strokes"
    [
      test "an even-odd fill says so" (fun () ->
          contains ~sub:"h\nf*\n"
            (content (pdf (Picture.fill ~rule:`Even_odd red square))));
      test "an alpha below 1 is a graphics state's ca for fills" (fun () ->
          let doc = pdf (Picture.fill (Color.with_alpha 0.25 red) square) in
          contains ~sub:"/G1 << /Type /ExtGState /ca 0.25 >>" (dicts doc);
          contains ~sub:"/G1 gs" (content doc));
      test "an alpha below 1 is a graphics state's ca for glyphs" (fun () ->
          let doc =
            pdf
              (Picture.glyphs
                 (Color.with_alpha 0.25 red)
                 (P2.v 5. 50.)
                 (Vg_corpus.typeset "a" 10.))
          in
          contains ~sub:"/G1 << /Type /ExtGState /ca 0.25 >>" (dicts doc);
          contains ~sub:"/G1 gs" (content doc));
      test "an alpha below 1 is a graphics state's CA for strokes" (fun () ->
          let doc =
            pdf (Picture.stroke (Stroke.v 1.) (Color.with_alpha 0.5 red) square)
          in
          contains ~sub:"/G1 << /Type /ExtGState /CA 0.5 >>" (dicts doc);
          contains ~sub:"/G1 gs" (content doc));
      test "a stroke writes its colour and width" (fun () ->
          let s = stroked 2. in
          equal numbers [ [ 1.; 0.; 0. ] ] (operands "RG" s);
          equal numbers [ [ 2. ] ] (operands "w" s));
      (* PDF 1.7, Table 52: a graphics state starts with butt caps (0 J), miter
         joins (0 j), a miter limit of 10 (10 M) and a solid line ([] 0 d). *)
      test "a stroke leaves out the parts of its pen that are PDF's defaults"
        (fun () ->
          let s = stroked 2. in
          List.iter
            (fun op -> not_contains ~sub:op s)
            [ " J\n"; " j\n"; " M\n"; " d\n" ]);
      test "a stroke writes caps, joins and limits that differ from PDF's"
        (fun () ->
          let s = stroked ~cap:`Square ~join:`Round ~miter_limit:4. 2. in
          equal numbers [ [ 2. ] ] (operands "J" s);
          equal numbers [ [ 1. ] ] (operands "j" s);
          equal numbers [ [ 4. ] ] (operands "M" (stroked ~miter_limit:4. 2.)));
      test "a dashed stroke writes its pattern and phase" (fun () ->
          contains ~sub:"[4 2] 1 d\n"
            (stroked ~dash:[ 4.; 2. ] ~dash_offset:1. 2.));
      test "a subpath of zero length with square caps is left out" (fun () ->
          let point =
            Path.empty |> Path.move_to (P2.v 5. 5.) |> Path.line_to (P2.v 5. 5.)
          in
          not_contains ~sub:" m\n"
            (content
               (pdf (Picture.stroke (Stroke.v ~cap:`Square 2.) red point))));
    ]

(* Glyph runs *)

let glyph_runs =
  group "glyph runs"
    [
      test "a run is shown in its font at its size from its origin" (fun () ->
          let c = content (pdf (typeset_at 50. "AV" 10.)) in
          equal numbers [ [ 10. ] ] (operands "Tf" c);
          equal numbers [ [ 1.; 0.; 0.; -1.; 5.; 50. ] ] (operands "Tm" c));
      test "a run off one baseline places each line of glyphs on its own"
        (fun () ->
          let r =
            Run.v ~ys:[| 0.; 2.5 |] ~font:Font.regular ~size:10. ~text:"ab"
              ~glyphs:[| glyph 'a'; glyph 'b' |]
              ~xs:[| 0.; 6. |] ()
          in
          equal numbers
            [ [ 1.; 0.; 0.; -1.; 5.; 50. ]; [ 1.; 0.; 0.; -1.; 11.; 52.5 ] ]
            (operands "Tm" (content (pdf (Picture.glyphs red (P2.v 5. 50.) r)))));
      test "a TrueType font is a CIDFontType2 font of its subset in Identity-H"
        (fun () ->
          let doc = pdf (typeset_at 50. "AV" 10.) in
          let d = dicts doc in
          contains ~sub:"/Subtype /CIDFontType2" d;
          contains
            ~sub:(subset_tag doc ^ "+Inter-Regular /Encoding /Identity-H")
            d;
          equal string
            (Font.subset Font.regular [ glyph 'A'; glyph 'V' ])
            (stream_of doc "/Length1"));
      cases
        "a font of CFF outlines is a CIDFontType0 font of an OpenType program"
        ~id:Fun.id
        [
          "../../font/golden/cff-small.otf"; "../../font/golden/cff2-small.otf";
        ] (fun file ->
          let font = load_font file in
          let h = Font.glyph font (Uchar.of_char 'H') in
          let doc = pdf (typeset_at 50. ~font "H" 10.) in
          let d = dicts doc in
          contains ~sub:"/Subtype /CIDFontType0" d;
          contains ~sub:"/FontFile3" d;
          (* CIDs are glyph ids, PDF's default for a CIDFontType0 font. *)
          not_contains ~sub:"/CIDToGIDMap" d;
          equal string (Font.subset font [ h ])
            (stream_of doc "/Subtype /OpenType"));
      test "a font is embedded once for runs of equal fonts" (fun () ->
          let copy = Result.get_ok (Font.of_string (Font.bytes Font.regular)) in
          let doc =
            pdf
              (Picture.group
                 [ typeset_at 20. "a" 10.; typeset_at 40. ~font:copy "a" 10. ])
          in
          equal int 1 (count (dicts doc) "/FontFile2");
          equal int 2 (count (content doc) "/F1 10 Tf"));
      test "a subset is named by a tag of six capital letters" (fun () ->
          satisfies ~claim:"six capital letters" string
            (String.for_all (function 'A' .. 'Z' -> true | _ -> false))
            (subset_tag (pdf (typeset_at 50. "AV" 10.))));
      test "a subset's tag depends on the subset" (fun () ->
          let tag text size = subset_tag (pdf (typeset_at 50. text size)) in
          not_equal string (tag "AV" 10.) (tag "AW" 10.);
          equal string (tag "AV" 10.) (tag "VA" 12.));
      test "a map gives each glyph the character it renders" (fun () ->
          let map = to_unicode (pdf (typeset_at 50. "AV" 10.)) in
          contains ~sub:(Printf.sprintf "<%04X> <0041>" (glyph 'A')) map;
          contains ~sub:(Printf.sprintf "<%04X> <0056>" (glyph 'V')) map);
      (* Adobe Technical Note #5411: a beginbfchar block holds at most 100
         mappings. *)
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
            to_unicode (pdf ~w:200. (Picture.glyphs red (P2.v 0. 50.) r))
          in
          equal numbers
            [ [ 100. ]; [ Float.of_int (distinct - 100) ] ]
            (operands "beginbfchar" map));
      test "a run whose text the map reproduces carries no ActualText span"
        (fun () ->
          not_contains ~sub:"ActualText"
            (content (pdf (typeset_at 50. "AV" 10.))));
      cases "a run carries its text in an ActualText span when" ~id:fst
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
          contains ~sub:"/ActualText <FEFFD800DC00>" (content (pdf p)));
      test "a run of glyphs without text carries its empty text" (fun () ->
          let r =
            Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text:""
              ~glyphs:[| glyph 'a' |]
              ~xs:[| 0. |] ()
          in
          equal (list string)
            [ "FEFF"; Printf.sprintf "%04X" (glyph 'a') ]
            (shown (content (pdf (Picture.glyphs red (P2.v 5. 50.) r)))));
    ]

(* Images *)

(* Pixels of two rows of three, RGBA, each value its own. *)
let pixels () =
  Nx.init Nx.uint8 [| 2; 3; 4 |] (fun i ->
      (i.(0) * 100) + (i.(1) * 50) + (i.(2) * 9))

let images =
  group "images"
    [
      test "an image is painted over its box" (fun () ->
          let doc = pdf (Picture.image (Box2.v 10. 20. 30. 40.) (pixels ())) in
          equal numbers
            [ [ 30.; 0.; 0.; -40.; 10.; 60. ] ]
            (List.tl (operands "cm" (content doc))));
      test "an image's samples are its pixels, shown without interpolation"
        (fun () ->
          let px = pixels () in
          let doc = pdf (Picture.image (Box2.v 10. 20. 30. 40.) px) in
          contains
            ~sub:
              "/Width 3 /Height 2 /ColorSpace /DeviceRGB /BitsPerComponent 8 \
               /Interpolate false"
            (dicts doc);
          let a = Nx.to_array px in
          equal string
            (String.init 18 (fun k -> Char.chr a.((k / 3 * 4) + (k mod 3))))
            (stream_of doc "/SMask"));
      test "an RGBA image has its alpha as a soft mask" (fun () ->
          let px = pixels () in
          let doc = pdf (Picture.image (Box2.v 10. 20. 30. 40.) px) in
          let a = Nx.to_array px in
          equal string
            (String.init 6 (fun k -> Char.chr a.((k * 4) + 3)))
            (stream_of doc "/DeviceGray"));
      test "each image shows its own pixels" (fun () ->
          (* One tensor shown whole and cropped to the 2 columns that meet the
             margin, 499 and 500, and another tensor shown whole: three images,
             numbered in the order of first use. *)
          let px = Nx.init Nx.uint8 [| 1; 1000; 1 |] (fun i -> i.(1) mod 256)
          and other =
            Nx.init Nx.uint8 [| 1; 1000; 1 |] (fun i -> 255 - (i.(1) mod 256))
          in
          let whole = Box2.v 0. 0. 100. 10.
          and far = Box2.v (-1e6) 0. 2e6 10. in
          let doc =
            pdf
              (Picture.group
                 [
                   Picture.image whole px;
                   Picture.image far px;
                   Picture.image whole other;
                 ])
          in
          let samples t =
            String.init 1000 (fun i -> Char.chr (Nx.item [ 0; i; 0 ] t))
          in
          equal (list string)
            [ samples px; "\243\244"; samples other ]
            (List.filter_map
               (fun (_, d, s) -> if occurs d "/Subtype /Image" then s else None)
               (objects doc));
          equal int 3 (count (content doc) " Do\n"));
      test "a grey image is DeviceGray, without a mask" (fun () ->
          let px = Nx.init Nx.uint8 [| 1; 2; 1 |] (fun i -> 7 + i.(1)) in
          let doc = pdf (Picture.image (Box2.v 0. 0. 2. 1.) px) in
          not_contains ~sub:"/SMask" (dicts doc);
          equal string "\007\008" (stream_of doc "/DeviceGray"));
    ]

(* Clips, transforms and opacities *)

let clips =
  group "clips"
    [
      test "a clip is a clipping path" (fun () ->
          contains ~sub:"W* n\n"
            (content
               (pdf
                  (Picture.clip ~rule:`Even_odd square (Picture.fill red square)))));
    ]

let transforms =
  group "transforms"
    [
      test "a transform writes no operator of its own" (fun () ->
          let s =
            content
              (pdf
                 (Picture.transform (Affine.translate 5. 7.)
                    (Picture.fill red square)))
          in
          equal int 1 (count s " cm\n");
          equal points_w
            [ (15., 27.); (45., 27.); (45., 67.); (15., 67.) ]
            (points s));
    ]

let opacities =
  group "opacities"
    [
      test "an opacity is a transparency group" (fun () ->
          contains ~sub:"/Group << /S /Transparency /I true >>"
            (dicts (pdf (Picture.opacity 0.5 (Picture.fill red square)))));
      test "an opacity's group is painted with the opacity as its alpha"
        (fun () ->
          let doc = pdf (Picture.opacity 0.5 (Picture.fill red square)) in
          contains ~sub:"/G1 << /Type /ExtGState /ca 0.5 /CA 0.5 >>" (dicts doc);
          contains ~sub:"/G1 gs\n/X1 Do" (content doc));
      test "an opacity of 0 paints nothing" (fun () ->
          let s =
            content (pdf (Picture.opacity 0. (Picture.fill red square)))
          in
          not_contains ~sub:"Do" s;
          not_contains ~sub:"f\n" s);
    ]

(* Stamps *)

(* Stamps of a stroked marker at positions around the margin, 100 points beyond
   the page, some within it and some just beyond, at scales that both renderers
   carry in one definition. *)
let gen_margin_stamp =
  let open Gen in
  let coord =
    frequency
      [
        (1, float_range (-130.) 230.);
        (2, map (fun d -> -100. -. d) (float_range (-6.) 6.));
        (2, map (fun d -> 200. +. d) (float_range (-6.) 6.));
      ]
  in
  let+ pts = list ~size:(int_range 1 12) (pair coord coord)
  and+ scaled = bool
  and+ width = float_range 0.5 6. in
  let xs = Array.of_list (List.map fst pts)
  and ys = Array.of_list (List.map snd pts) in
  let scales =
    if scaled then
      Some (Array.mapi (fun i _ -> 0.5 +. (Float.of_int (i mod 4) /. 2.)) xs)
    else None
  in
  Picture.stamp ?scales xs ys
    (Picture.group
       [
         Picture.fill red disc;
         Picture.stroke (Stroke.v ~join:`Miter width) Color.black disc;
       ])

let gen_margin_stamp = Gen.with_pp Picture.pp gen_margin_stamp

(* Both renderers cull instances by the one rule of their shared walk: they draw
   the same instances. *)
let same_instances p =
  let n = match p with Picture.Stamp { xs; _ } -> Array.length xs | _ -> 0 in
  let svg = Hugin_vg_svg.render (Renderable.v 100. 100. p) in
  let drawn = count (content (pdf p)) " Do\n" in
  cover "an instance left out" (drawn < n);
  cover "an instance drawn" (drawn > 0);
  equal int (count svg "<use ") drawn

(* The margin, 100 points around the page. *)
let margin = Box2.v (-100.) (-100.) 300. 300.

(* An instance whose own bounds meet the margin is drawn: the rule may draw
   more, never fewer. That it draws no instance beyond the margin follows from
   [same_instances] and the SVG suite's law of that side. *)
let bounded_instances_drawn (p : Picture.t) =
  match p with
  | Stamp { xs; ys; scales; picture; _ } ->
      let meets i =
        let scales = Option.map (fun a -> [| a.(i) |]) scales in
        match
          Picture.bounds
            (Picture.stamp ?scales [| xs.(i) |] [| ys.(i) |] picture)
        with
        | Some b -> Option.is_some (Box2.inter b margin)
        | None -> false
      in
      let expected =
        List.length (List.filter meets (List.init (Array.length xs) Fun.id))
      in
      cover "an instance beyond the margin" (expected < Array.length xs);
      at_least int ~than:expected (count (content (pdf p)) " Do\n")
  | _ -> fail "not a stamp"

let stamps =
  group "stamps"
    [
      test "a stamp writes its picture once and paints it per instance"
        (fun () ->
          let doc =
            pdf (Picture.stamp [| 10.; 20.; 30. |] [| 5.; 6.; 7. |] marker)
          in
          equal int 1 (forms doc);
          equal numbers
            [
              [ 1.; 0.; 0.; 1.; 10.; 5. ];
              [ 1.; 0.; 0.; 1.; 20.; 6. ];
              [ 1.; 0.; 0.; 1.; 30.; 7. ];
            ]
            (List.tl (operands "cm" (content doc))));
      test "equal stamps share their form" (fun () ->
          let st = Picture.stamp [| 10. |] [| 5. |] marker in
          let doc =
            pdf
              (Picture.group
                 [ st; Picture.transform (Affine.translate 0. 20.) st ])
          in
          equal int 1 (forms doc);
          equal int 2 (count (content doc) "Do"));
      test "fills and strokes are set before each instance" (fun () ->
          let c =
            content
              (pdf
                 (Picture.stamp
                    ~fills:[| red; Color.with_alpha 0.5 Color.blue |]
                    ~strokes:[| Color.blue; red |] [| 10.; 20. |] [| 5.; 5. |]
                    marker))
          in
          equal numbers [ [ 1.; 0.; 0. ]; [ 0.; 0.; 1. ] ] (operands "rg" c);
          equal numbers [ [ 0.; 0.; 1. ]; [ 1.; 0.; 0. ] ] (operands "RG" c));
      test "a replaced alpha is set before its instance" (fun () ->
          let doc =
            pdf
              (Picture.stamp
                 ~fills:[| red; Color.with_alpha 0.5 Color.blue |]
                 [| 10.; 20. |] [| 5.; 5. |] marker)
          in
          contains ~sub:"/ca 0.5 >>" (dicts doc));
      test "a stamp's form leaves the colours it replaces to its instances"
        (fun () ->
          let form =
            stream_of
              (pdf
                 (Picture.stamp ~fills:[| red |] ~strokes:[| Color.blue |]
                    [| 10. |] [| 5. |] marker))
              "/Subtype /Form"
          in
          not_contains ~sub:"rg" form;
          not_contains ~sub:"RG" form);
      test "an image in a form that fills inherit resets ca" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let p = Picture.image (Box2.v 0. 0. 2. 2.) px in
          let doc =
            pdf
              (Picture.stamp
                 ~fills:[| Color.with_alpha 0.5 red |]
                 [| 10. |] [| 5. |] p)
          in
          contains ~sub:"/ca 1 >>" (dicts doc);
          contains ~sub:" gs\n" (stream_of doc "/Subtype /Form"));
      test "an instance within a form that sets strokes sets its alpha"
        (fun () ->
          let inner =
            Picture.stamp ~strokes:[| red |] [| 0. |] [| 0. |]
              (Picture.stroke (Stroke.v 1.) red disc)
          in
          let d =
            dicts
              (pdf
                 (Picture.stamp
                    ~strokes:[| Color.with_alpha 0.5 red |]
                    [| 10. |] [| 5. |] inner))
          in
          contains ~msg:"the inner alpha" ~sub:"/CA 1 >>" d;
          contains ~msg:"the outer alpha" ~sub:"/CA 0.5 >>" d);
      test "a scaling stamp scales its instances" (fun () ->
          let ring = Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc in
          let c =
            content
              (pdf
                 (Picture.stamp ~scales:[| 2.; 4. |] [| 10.; 20. |] [| 5.; 5. |]
                    ring))
          in
          equal numbers
            [ [ 2.; 0.; 0.; 2.; 10.; 5. ]; [ 4.; 0.; 0.; 4.; 20.; 5. ] ]
            (List.tl (operands "cm" c)));
      test "a scaling stamp sets its picture's pen before each instance"
        (fun () ->
          let ring = Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc in
          let doc =
            pdf
              (Picture.stamp ~scales:[| 2.; 4. |] [| 10.; 20. |] [| 5.; 5. |]
                 ring)
          in
          let c = content doc and form = stream_of doc "/Subtype /Form" in
          equal numbers [ [ 0.5 ]; [ 0.25 ] ] (operands "w" c);
          contains ~sub:"[1 0.5] 0 d\n" c;
          not_contains ~msg:"the form's widths" ~sub:" w\n" form;
          not_contains ~msg:"the form's dashes" ~sub:" d\n" form);
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
      test "a scaling stamp of strokes of differing widths is written in full"
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
          equal numbers [ [ 1. ]; [ 2. ] ] (operands "w" (content doc)));
      test "a scaling stamp of strokes of differing dashes is written in full"
        (fun () ->
          let two =
            Picture.group
              [
                Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc;
                Picture.stroke (Stroke.v ~dash:[ 3.; 1. ] 1.) red disc;
              ]
          in
          let doc =
            pdf (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] two)
          in
          let c = content doc in
          equal int 0 (forms doc);
          contains ~sub:"[2 1] 0 d\n" c;
          contains ~sub:"[3 1] 0 d\n" c);
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
          equal numbers
            [ [ 2.; 0.; 0.; 2.; 10.; 5. ] ]
            (List.tl (operands "cm" (content doc))));
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
          equal numbers [ [ 1. ] ] (operands "w" (stream_of doc "/Group")));
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
          contains ~sub:" gs\n" (stream_of doc "/Group"));
      test "an instance whose scale the form cannot carry is written in full"
        (fun () ->
          let c =
            content
              (pdf
                 (Picture.stamp ~scales:[| 1.; 1e-20 |] [| 10.; 20. |]
                    [| 5.; 5. |] marker))
          in
          equal int ~msg:"instances through the form" 1 (count c "Do");
          (* The second keeps its pen, with its round caps, around the point it
             shrinks to. *)
          contains ~msg:"in full" ~sub:"0.5 w\n1 J\n1 j\n20 5 m\n" c;
          satisfies ~claim:"no matrix of zero scale" numbers
            (List.for_all (function
              | [ a; _; _; d; _; _ ] -> a <> 0. && d <> 0.
              | _ -> false))
            (operands "cm" c));
      cases "an instance holding a stroke that paints nothing is painted"
        ~id:fst lost_maps (fun (_, k) ->
          let lost = Picture.transform (Affine.scale k k) in
          let p =
            Picture.group
              [
                Picture.fill red disc;
                lost (lost (Picture.stroke (Stroke.v 1.) red disc));
              ]
          in
          let doc = pdf (Picture.stamp [| 10.; 20. |] [| 5.; 5. |] p) in
          equal int 2 (count (content doc) " Do\n"));
      cases "an instance changes nothing" ~id:fst
        [
          ("at x = NaN", (Float.nan, 5., 1000.));
          ("at y = NaN", (20., Float.nan, 1000.));
          ("at y = infinity", (20., Float.infinity, 1000.));
          ("of scale NaN", (20., 5., Float.nan));
          ("of scale infinity", (20., 5., Float.infinity));
          ("of scale 0", (20., 5., 0.));
        ]
        (fun (_, (x, y, k)) ->
          equal string
            (pdf (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] marker))
            (pdf
               (Picture.stamp ~scales:[| 2.; k |] [| 10.; x |] [| 5.; y |]
                  marker)));
      test "instances at non-finite positions or scales in a form are left out"
        (fun () ->
          let inner =
            Picture.stamp ~scales:[| 1.; Float.nan; 1. |] [| 0.; 1.; 2. |]
              [| 0.; 0.; Float.nan |] marker
          in
          let doc = pdf (Picture.stamp [| 50. |] [| 50. |] inner) in
          let forms =
            String.concat ""
              (List.filter_map
                 (fun (_, d, s) ->
                   if occurs d "/Subtype /Form" then s else None)
                 (objects doc))
          in
          equal int ~msg:"instances" 1 (count forms " Do\n");
          not_contains ~msg:"numbers" ~sub:"nan" forms);
      test "a scaled instance whose pen alone meets a clip is painted"
        (fun () ->
          (* The stroke misses the clip at scale 1, and meets it at 0.5 with the
             pen the instance keeps. *)
          let p =
            Picture.clip (rect 0. 0. 1. 1.)
              (Picture.stroke (Stroke.v ~cap:`Butt 2.) red
                 (Path.close (Path.polyline [| 0.; 0. |] [| 2.5; 3.5 |])))
          in
          let s = Picture.stamp ~scales:[| 0.5 |] [| 0. |] [| 0. |] p in
          equal int 1 (count (content (pdf s)) " Do\n"));
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
      prop "the SVG and PDF renderers draw the same instances" gen_margin_stamp
        same_instances;
      prop "an instance whose bounds meet the margin is drawn" gen_margin_stamp
        bounded_instances_drawn;
    ]

(* Tags *)

let tags =
  group "tags"
    [
      prop "are ignored" Vg_corpus.gen_picture (fun p ->
          let cells =
            Picture.Cells { box = Box2.v 0. 0. 1. 1.; width = 1; height = 1 }
          in
          let t =
            { Picture.id = Nx.Ptree.Path.v [ Field "a" ]; rows = cells }
          in
          Law.ignores picture string (fun p -> pdf p) (Picture.tag t) p);
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

(* [glyphs_placed (size, steps)] checks that a run of [size] whose glyphs are
   [steps] apart is shown within 0.001 of where the run puts each glyph. *)
let glyphs_placed (size, steps) =
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
  let got = glyph_xs (pdf ~w:500. (Picture.glyphs red (P2.v 7. 50.) r)) in
  equal int n (List.length got);
  List.iteri
    (fun i x ->
      at_most ~msg:(string_of_int i) float_exact ~than:0.001
        (Float.abs (x -. (7. +. xs.(i)))))
    got

(* [dash_lengths (k, (a, b))] checks that the dash pattern [a b] under a scale
   of [k] is written so that a thousand of its lengths add up to within 0.0005
   of theirs: the ends of the dashes they place are written with three decimals,
   which takes the other 0.0005 of the accuracy. *)
let dash_lengths (k, (a, b)) =
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
  | l -> failf "%d lengths" (List.length l)

(* [far_line] runs from (-123456789, -123456700) to (123456789, 123456900): it
   crosses x = 0 at y = 100. *)
let far_line =
  Path.polyline [| -123456789.; 123456789. |] [| -123456700.; 123456900. |]

(* [numbers_finite doc] checks that the objects of [doc], their streams but font
   files and image samples inflated, hold no number that is not finite. *)
let numbers_finite doc =
  List.iter
    (fun (n, dict, stream) ->
      let text =
        match stream with
        | Some data
          when not (occurs dict "/Length1" || occurs dict "/Subtype /Image") ->
            dict ^ "\n" ^ data
        | _ -> dict
      in
      List.iter
        (function
          | Word (("inf" | "-inf" | "nan" | "-nan") as w) ->
              failf "object %d holds %s" n w
          | _ -> ())
        (tokens text))
    (objects doc)

(* A fill and a stroke of a path with a gap, a point that is not finite. *)
let gapped =
  let q =
    Path.polyline [| 10.; Float.nan; 30.; 40. |] [| 10.; 20.; 30.; 10. |]
  in
  Picture.group [ Picture.fill red q; Picture.stroke (Stroke.v 1.) red q ]

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
      prop
        "a stamp's points are placed within 0.001 of where the page puts them"
        (Gen.triple
           (Gen.float_range 0.01 1000.)
           (Gen.pair (Gen.float_range 0. 100.) (Gen.float_range 0. 100.))
           gen_marker_points)
        (fun (scale, (x, y), pts) -> placed scale x y pts);
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
          contains ~sub:"[3 1] 3 d\n-100.5 50 m\n200.5 50 l\nS\n" s);
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
          contains ~sub:"[3] 5 d\n" s);
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
          (* Cut where the margin and the miter's reach of 2 end, the part
             beyond running along that edge, whose ink lies off the page. *)
          contains ~sub:"50 50 m\n202 50 l\n202 60 l\n50 60 l\nh\nS\n" s);
      test "an area's curve crossing the margin is cut at it" (fun () ->
          let big = Path.circle (P2.v 50. 1e6) (1e6 -. 50.) in
          let s = content (pdf (Picture.fill red big)) in
          satisfies ~claim:"every number within 1000 of the page" numbers
            (List.for_all (List.for_all (fun v -> Float.abs v <= 1000.)))
            (operands "m" s @ operands "l" s @ operands "c" s));
      test "an area's curve within the margin keeps its curves" (fun () ->
          let s =
            content (pdf (Picture.fill red (Path.circle (P2.v 50. 50.) 10.)))
          in
          equal int 4 (count s " c\n"));
      test "a clip of no area within the margin hides its picture" (fun () ->
          let far = rect 1e6 1e6 10. 10. in
          let s = content (pdf (Picture.clip far (Picture.fill red square))) in
          not_contains ~sub:" n\n" s;
          not_contains ~sub:"f\n" s);
      test "an image is cropped to the pixels that meet the margin" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1000; 3 |] in
          let doc = pdf (Picture.image (Box2.v (-1e6) 0. 2e6 10.) px) in
          (* Each pixel is 2000 points wide: the margin of 100 points around the
             page meets two, from -2000 to 2000. *)
          contains ~sub:"/Width 2 /Height 1" (dicts doc);
          equal numbers
            [ [ 4000.; 0.; 0.; -10.; -2000.; 10. ] ]
            (List.tl (operands "cm" (content doc))));
      test "an image is cropped to the rows that meet the margin" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1000; 1; 3 |] in
          let doc = pdf (Picture.image (Box2.v 0. (-1e6) 10. 2e6) px) in
          (* Each row is 2000 points high: the margin meets two, from -2000 to
             2000. *)
          contains ~sub:"/Width 1 /Height 2" (dicts doc);
          equal numbers
            [ [ 10.; 0.; 0.; -4000.; 0.; 2000. ] ]
            (List.tl (operands "cm" (content doc))));
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
      cases "an image beyond the margin on one side is cropped there" ~id:fst
        [
          ("left", (1, 1000, Box2.v (-999900.) 0. 1e6 10.));
          ("right", (1, 1000, Box2.v (-100.) 0. 1e6 10.));
          ("top", (1000, 1, Box2.v 0. (-999900.) 10. 1e6));
          ("bottom", (1000, 1, Box2.v 0. (-100.) 10. 1e6));
        ]
        (fun (_, (h, w, box)) ->
          let px = Nx.zeros Nx.uint8 [| h; w; 3 |] in
          contains ~sub:"/Width 1 /Height 1 /ColorSpace"
            (dicts (pdf (Picture.image box px))));
      test "a run beyond the margin is left out" (fun () ->
          let r =
            Picture.glyphs red (P2.v 1e5 50.) (Vg_corpus.typeset "far" 10.)
          in
          not_contains ~sub:"TJ" (content (pdf r)));
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
          equal numbers
            [ [ 0.25; 0.; 0.; 1.; 0.; 0. ] ]
            (List.tl (operands "cm" s));
          equal numbers [ [ 4. ] ] (operands "w" s);
          (* Under the matrix, the end (10, 40) on the page is (40, 40). *)
          equal points_w [ (0., 0.); (40., 40.) ] (points s));
      test "a pen turned and scaled evenly is written on the page" (fun () ->
          let p =
            Picture.transform
              Affine.(rotate (Float.pi /. 3.) * scale 2. 2.)
              (Picture.stroke (Stroke.v 1.) red
                 (Path.polyline [| 0.; 10. |] [| 0.; 0. |]))
          in
          let s = content (pdf p) in
          equal int 1 (count s " cm\n");
          equal numbers [ [ 2. ] ] (operands "w" s));
      test "a turned run is written under a matrix at its origin" (fun () ->
          let p =
            Picture.transform
              Affine.(translate 50. 50. * rotate (Float.pi /. 2.) * scale 2. 2.)
              (Picture.glyphs red (P2.v 0. 0.) (Vg_corpus.typeset "a" 10.))
          in
          let s = content (pdf p) in
          equal numbers [ [ 0.; 1.; 1.; 0.; 50.; 50. ] ] (operands "Tm" s);
          equal numbers [ [ 20. ] ] (operands "Tf" s));
      test "an image under a turn is written under its matrix" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let p =
            Picture.transform
              (Affine.rotate (Float.pi /. 2.))
              (Picture.image (Box2.v 10. 0. 20. 10.) px)
          in
          equal numbers
            [ [ 0.; 20.; 10.; 0.; -10.; 10. ] ]
            (List.tl (operands "cm" (content (pdf p)))));
      test "an image turned by 45 degrees is painted" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let p =
            Picture.transform
              Affine.(translate 50. 50. * rotate (Float.pi /. 4.))
              (Picture.image (Box2.v 0. 0. 20. 20.) px)
          in
          equal int 1 (count (content (pdf p)) "Do"));
      test "an image is placed within the accuracy" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let doc =
            pdf (Picture.image (Box2.v 10.0004 20.0004 30.0037 40.0042) px)
          in
          (* The corners that the image's matrix maps its unit square to,
             against those of its box, y down. *)
          let corners = function
            | [ _; [ a; b; c; d; e; f ] ] ->
                List.map
                  (fun (u, v) ->
                    (e +. (a *. u) +. (c *. v), f +. (b *. u) +. (d *. v)))
                  [ (0., 0.); (1., 0.); (0., 1.); (1., 1.) ]
            | _ -> fail "one image matrix"
          in
          equal
            (list (pair (float 0.001) (float 0.001)))
            [
              (10.0004, 60.0046);
              (40.0041, 60.0046);
              (10.0004, 20.0004);
              (40.0041, 20.0004);
            ]
            (corners (operands "cm" (content doc))));
      prop
        ~examples:
          [ (1.0000185299292945, [ 18.918466897345255; 26.100956639689798 ]) ]
        "glyphs are shown within 0.001 of where the run puts them"
        (Gen.pair (Gen.float_range 1. 40.)
           (Gen.list ~size:(Gen.int_range 1 40) (Gen.float_range (-3.) 30.)))
        glyphs_placed;
      prop "dashes keep within the accuracy over a thousand lengths"
        (Gen.pair (Gen.float_range 0.3 3.)
           (Gen.pair (Gen.float_range 0.1 10.) (Gen.float_range 0.1 10.)))
        dash_lengths;
      test "colour components and alphas are written to three decimals"
        (fun () ->
          let c = Color.v ~alpha:0.12345 0.0004 0.5006 0.99951 in
          let doc = pdf (Picture.fill c square) in
          equal numbers [ [ 0.; 0.501; 1. ] ] (operands "rg" (content doc));
          contains ~sub:"/ca 0.123 >>" (dicts doc));
      test "a stroke whose width rounds to zero is left out" (fun () ->
          not_contains ~sub:"S\n" (stroked 0.0004);
          equal numbers [ [ 0.001 ] ] (operands "w" (stroked 0.001)));
      test "a run whose size rounds to zero is left out" (fun () ->
          not_contains ~sub:"Tf" (content (pdf (typeset_at 50. "a" 1e-9)));
          equal numbers [ [ 0.001 ] ]
            (operands "Tf" (content (pdf (typeset_at 50. "a" 0.001)))));
      test "a stroke whose matrix rounds to a flat one is left out" (fun () ->
          let flat =
            Picture.transform (Affine.scale 1. 1e-20)
              (Picture.stroke (Stroke.v 1.) red square)
          in
          not_contains ~sub:"S\n" (content (pdf flat)));
      test "an image whose matrix rounds to a flat one is left out" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let doc = pdf (Picture.image (Box2.v 10. 10. 0.00001 10.) px) in
          not_contains ~sub:"Do" (content doc));
      test "a dash pattern whose lengths round to zero is written solid"
        (fun () ->
          let s = stroked ~dash:[ 1e-7; 2e-7 ] 2. in
          not_contains ~sub:" d\n" s;
          equal int 1 (count s "S\n"));
      cases "transforms whose composition overflows or underflows paint nothing"
        ~id:fst
        [ ("overflows", 1e200); ("underflows", 1e-200) ]
        (fun (_, k) ->
          let twice p =
            Picture.transform (Affine.scale k k)
              (Picture.transform (Affine.scale k k) p)
          in
          equal points_w []
            (points (content (pdf (twice (Picture.fill red square))))));
      test "a width beyond 1e15 is written as 1e15" (fun () ->
          let p =
            Picture.stroke (Stroke.v 1e20) red
              (Path.polyline [| 40.; 60. |] [| 50.; 50. |])
          in
          equal numbers [ [ 1e15 ] ] (operands "w" (content (pdf p))));
      test "a coordinate below -1e15 is written as -1e15" (fun () ->
          (* A stamp's picture is written once, uncut, relative to its
             instances. *)
          let far = Path.polygon [| 0.; -1e20; 0. |] [| 0.; 0.; 10. |] in
          let doc =
            pdf (Picture.stamp [| 50. |] [| 50. |] (Picture.fill red far))
          in
          equal points_w
            [ (0., 0.); (-1e15, 0.); (0., 10.) ]
            (points (stream_of doc "/Subtype /Form")));
      prop ~examples:(gapped :: Vg_corpus.extremes)
        "numbers are finite whatever the scales" Vg_corpus.gen_extreme (fun p ->
          numbers_finite (pdf p));
      cases "a pen keeps its width" ~id:(Printf.sprintf "under a scale of %g")
        [ 1e-170; 1e170 ] (fun k ->
          let p =
            Picture.transform (Affine.scale k k)
              (Picture.stroke
                 (Stroke.v (10. /. k))
                 red
                 (Path.polyline [| 0.; 50. /. k |] [| 50. /. k; 50. /. k |]))
          in
          let c = content (pdf p) in
          equal numbers [ [ 10. ] ] (operands "w" c);
          equal points_w [ (0., 50.); (50., 50.) ] (points c));
    ]

(* Determinism *)

(* [image_leaves p] is the number of images of [p]. *)
let rec image_leaves (p : Picture.t) =
  match p with
  | Image _ -> 1
  | Group ps -> List.fold_left (fun n p -> n + image_leaves p) 0 ps
  | Clip { picture; _ }
  | Transform { picture; _ }
  | Opacity { picture; _ }
  | Stamp { picture; _ }
  | Tag { picture; _ } ->
      image_leaves picture
  | Empty | Fill _ | Stroke _ | Glyphs _ -> 0

(* Two images of the corpus's one tensor, each whole on the page: written once
   as long as equal tensors are one image, which [respell]'s copies are not. *)
let one_tensor_twice =
  Picture.group
    [
      Picture.image (Box2.v 10. 10. 20. 20.) Vg_corpus.tensor;
      Picture.image (Box2.v 50. 50. 20. 20.) Vg_corpus.tensor;
    ]

let determinism =
  group "determinism"
    [
      prop "equal renderables give equal documents"
        ~examples:[ one_tensor_twice ] Vg_corpus.gen_picture (fun p ->
          cover "two images of one tensor" (image_leaves p >= 2);
          Law.ignores picture string (fun p -> pdf p) Vg_corpus.respell p);
      test "a document holds no creation date and no identifier" (fun () ->
          let doc = pdf Vg_corpus.(Renderable.picture marks) in
          not_contains ~sub:"/CreationDate" doc;
          not_contains ~sub:"/ID" doc);
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
          in_order ~subs:[ "+Inter-Bold"; "+Inter-Regular" ] d);
      test "a number that rounds to zero is written 0" (fun () ->
          let s =
            content
              (pdf
                 (Picture.fill red
                    (Path.polygon [| -0.; 10.; -0.0001 |] [| -0.; 0.; 10. |])))
          in
          contains ~sub:"0 0 m\n10 0 l\n0 10 l\n" s);
    ]

(* Render *)

(* [dump doc] is [doc] as text: its objects in order, their streams inflated,
   font files by their length and image samples in hexadecimal. *)
let dump doc =
  let b = Buffer.create 4096 in
  List.iter
    (fun (n, dict, stream) ->
      Printf.bprintf b "%d 0 obj\n%s\n" n dict;
      match stream with
      | None -> ()
      | Some data when occurs dict "/Length1" ->
          Printf.bprintf b "stream <%d bytes>\n" (String.length data)
      | Some data when occurs dict "/Subtype /Image" ->
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

let render =
  group "render"
    [
      cases "writes each page of the corpus as its baseline" ~id:fst
        Vg_corpus.pages (fun (name, r) ->
          expect_file
            ~file:("packages/hugin/test/vg/pdf/golden/" ^ name ^ ".pdf.txt")
            (dump (Hugin_vg_pdf.render r)));
    ]

let () =
  exit
    (run "hugin.vg pdf"
       [
         document;
         fills_and_strokes;
         glyph_runs;
         images;
         clips;
         transforms;
         opacities;
         stamps;
         tags;
         accuracy;
         determinism;
         render;
       ])
