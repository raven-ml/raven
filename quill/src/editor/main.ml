open Vdom
open Js_of_ocaml
open Js_browser

let app = simple_app ~init:Model.init ~update:Update.update ~view:View.view ()

let () =
  match Document.get_element_by_id Js_browser.document "editor-app" with
  | None ->
      Js_of_ocaml.Console.console##error (Js.string "No #editor element found")
  | Some container ->
      let mounted_app = Vdom_blit.run ~container app in
      let dom_el = Vdom_blit.dom mounted_app in
      Events.setup_event_listeners (Obj.magic dom_el) mounted_app;
      Element.append_child (Document.body document) dom_el;
      let path = Js.to_string Dom_html.window##.location##.pathname in
      let api_url =
        if path = "/" then "/api/doc"
        else "/api/doc/" ^ String.sub path 1 (String.length path - 1)
      in
      let xhr = XmlHttpRequest.create () in
      xhr##.onreadystatechange :=
        Js.wrap_callback (fun () ->
            if xhr##.readyState = XmlHttpRequest.DONE then
              if xhr##.status = 200 then
                match Js.Opt.to_option xhr##.responseText with
                | Some response_text ->
                    let response = Js.to_string response_text in
                    let document = Model.document_of_md response in
                    Vdom_blit.process mounted_app (Set_document document)
                | None ->
                    Js_of_ocaml.Console.console##error
                      (Js.string "Response text is None")
              else
                Js_of_ocaml.Console.console##error
                  (Js.string "Failed to fetch document"));
      xhr##_open (Js.string "GET") (Js.string api_url) Js._true;
      xhr##send Js.null
