open Vdom
open Js_of_ocaml
open Js_browser

let app = simple_app ~init:Model.init ~update:Update.update ~view:View.view ()

let setup_editor editor =
  editor##setAttribute (Js.string "contentEditable") (Js.string "true");
  editor##setAttribute (Js.string "tabindex") (Js.string "0")

let () =
  match Document.get_element_by_id Js_browser.document "editor" with
  | None ->
      Js_of_ocaml.Console.console##error (Js.string "No #editor element found")
  | Some container ->
      setup_editor (Obj.magic container);
      let mounted_app = Vdom_blit.run ~container app in
      let dom_el = Vdom_blit.dom mounted_app in
      Events.setup_event_listeners (Obj.magic dom_el) mounted_app;
      Element.append_child (Document.body document) dom_el
