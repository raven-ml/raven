(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg
open Hugin_next_text
open Text_support

let inter = [ Font.regular; Font.bold ]
let cjk = face [ 0x4E2D ]

(* [draw l] paints the runs of [l] as {!Text.Layout.fold} says, in black where
   they have no colour. *)
let draw l =
  Picture.group
    (List.rev
       (Text.Layout.fold
          (fun acc c at r ->
            Picture.glyphs (Option.value c ~default:Color.black) at r :: acc)
          [] l))

let box_w =
  let near a b = Float.abs (a -. b) <= 1e-9 in
  Testable.make ~pp:Box2.pp ~equal:(fun a b ->
      near (Box2.minx a) (Box2.minx b)
      && near (Box2.miny a) (Box2.miny b)
      && near (Box2.maxx a) (Box2.maxx b)
      && near (Box2.maxy a) (Box2.maxy b))

let drawing =
  group "drawing"
    [
      prop "a drawn layout paints where its ink is" (gen_tree ()) (fun t ->
          let l = Text.Layout.v ~width:30. ~fonts:inter ~size:10. (text t) in
          equal (option box_w) (Text.Layout.ink l) (Picture.bounds (draw l)));
    ]

(* Goldens *)

let find_from s i sub =
  let n = String.length sub in
  let rec loop i =
    if i + n > String.length s then None
    else if String.sub s i n = sub then Some i
    else loop (i + 1)
  in
  loop i

(* [masked s] is [s] with the data of its fonts replaced by their length, which
   leaves the goldens readable. *)
let masked s =
  let key = "base64," and b = Buffer.create (String.length s) in
  let rec loop i =
    match find_from s i "@font-face{" with
    | None -> Buffer.add_string b (String.sub s i (String.length s - i))
    | Some j ->
        let start = Option.get (find_from s j key) + String.length key in
        let stop = String.index_from s start ')' in
        Buffer.add_string b (String.sub s i (start - i));
        Printf.bprintf b "<%d bytes>" (stop - start);
        loop stop
  in
  loop 0;
  Buffer.contents b

(* [page l] is [l] drawn on a page that holds its box with a margin of 2
   points. *)
let page l =
  let b = Text.Layout.box l in
  let m = Affine.translate (2. -. Box2.minx b) (2. -. Box2.miny b) in
  Renderable.v (Box2.w b +. 4.) (Box2.h b +. 4.) (Picture.transform m (draw l))

let layouts =
  [
    ( "axis_title",
      Text.Layout.v ~halign:`Center ~valign:`Cap ~fonts:inter ~size:10.
        Text.(concat [ v "step size \u{03B7}"; sub (v "0") ]) );
    ( "fallback",
      Text.Layout.v
        ~fonts:[ Font.regular; Font.bold; cjk ]
        ~size:10.
        Text.(
          concat
            [ v "loss "; bold (v "\u{4E2D}x"); color Color.red (v " diverged") ])
    );
    ("notdef", Text.Layout.v ~fonts:inter ~size:10. (Text.v "x\u{6587}y"));
  ]

let goldens =
  group "goldens"
    (List.map
       (fun (name, l) ->
         test (name ^ " writes its golden document") (fun () ->
             expect_file
               (masked (Hugin_next_vg_svg.render (page l)))
               ("packages/hugin/next/test/text/golden/" ^ name ^ ".svg")))
       layouts)

let () = exit (run "hugin.next.text: drawing" [ drawing; goldens ])
