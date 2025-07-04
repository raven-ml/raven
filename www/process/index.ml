open Soup

(* Extract the main content from an odoc-generated HTML file *)
let extract_odoc_content soup =
  (* For odoc 3.x, the main content is in the body directly *)
  match soup $? "body" with
  | Some body ->
      (* Extract header and content sections *)
      let header = body $? "header.odoc-preamble" in
      let content_div = body $? "div.odoc-content" in
      let create_wrapper () = create_element "div" ~class_:"odoc-extracted" in

      let wrapper = create_wrapper () in

      (* Add header if it exists *)
      (match header with Some h -> append_child wrapper h | None -> ());

      (* Add content if it exists *)
      (match content_div with Some c -> append_child wrapper c | None -> ());

      wrapper
  | None -> failwith "Could not find body in odoc HTML"

(* Remove odoc-specific navigation and header elements *)
let remove_odoc_navigation content =
  (* Remove the odoc nav elements *)
  content $$ "nav.odoc-nav" |> iter delete;
  content $$ "nav.odoc-toc" |> iter delete;

  (* Remove the odoc search bar *)
  content $$ "div.odoc-search" |> iter delete;

  (* Remove any script tags *)
  content $$ "script" |> iter delete;

  (* Remove any link tags for stylesheets *)
  content $$ "link[rel=stylesheet]" |> iter delete;

  content

(* Convert odoc CSS classes to Raven website classes *)
let adapt_css_classes content =
  (* Map odoc classes to Raven classes *)
  let class_mappings =
    [
      ("odoc-doc", "doc-content");
      ("odoc-spec", "api-spec");
      ("odoc-val", "api-value");
      ("odoc-type", "api-type");
      ("odoc-module", "api-module");
      ("odoc-include", "api-include");
      ("odoc-comment", "api-comment");
    ]
  in

  (* Apply class mappings *)
  List.iter
    (fun (old_class, new_class) ->
      content $$ "." ^ old_class
      |> iter (fun elem ->
             remove_class old_class elem;
             add_class new_class elem))
    class_mappings;

  content

(* Update internal links to match site structure *)
let fix_internal_links ~library content =
  (* Find all links *)
  content $$ "a"
  |> iter (fun link ->
         match attribute "href" link with
         | Some href when not (String.contains href ':') ->
             (* This is a relative link - update it *)
             let new_href =
               if String.starts_with ~prefix:"../" href then
                 (* Link to parent module *)
                 "/docs/" ^ library ^ "/api/"
                 ^ String.sub href 3 (String.length href - 3)
               else if String.contains href '#' then
                 (* Link with anchor *)
                 href
               else
                 (* Link to another module *)
                 "/docs/" ^ library ^ "/api/" ^ href
             in
             set_attribute "href" new_href link
         | _ -> ())

(* Apply syntax highlighting to code blocks *)
let enhance_code_blocks content =
  (* Find all code blocks *)
  content $$ "pre code"
  |> iter (fun code ->
         (* Add syntax highlighting class *)
         add_class "language-ocaml" code;

         (* Ensure the pre element has proper styling *)
         match parent code with
         | Some pre -> add_class "code-block" pre
         | None -> ());

  (* Also handle inline code *)
  content $$ "code"
  |> iter (fun code ->
         match parent code with
         | Some p when name p <> "pre" -> add_class "inline-code" code
         | _ -> ())

(* Add backing lines to headers for visual style *)
let add_backing_lines content =
  (* Add visual backing to h1, h2, h3 elements *)
  content $$ "h1" |> iter (fun header -> add_class "with-backing" header);
  content $$ "h2" |> iter (fun header -> add_class "with-backing" header);
  content $$ "h3" |> iter (fun header -> add_class "with-backing" header);

  (* Add backing to specification blocks *)
  content $$ ".api-spec" |> iter (fun spec -> add_class "spec-backing" spec)

(* Clean up Stdlib references *)
let remove_stdlib_prefix content =
  (* Pattern to match Stdlib. prefixes *)
  let stdlib_regex = Str.regexp "\\bStdlib\\." in

  (* Process all text nodes *)
  content |> descendants |> elements
  |> iter (fun elem ->
         let text_content = texts elem |> String.concat "" in
         if
           String.length text_content > 0
           && Str.string_match stdlib_regex text_content 0
         then (
           let new_text = Str.global_replace stdlib_regex "" text_content in
           (* Don't parse as HTML, just set as text *)
           clear elem;
           append_child elem (Soup.create_text new_text)))

(* Extract module name from the page title or content *)
let extract_module_name content =
  (* Try to find module name from h1 or title *)
  match content $? "h1" with
  | Some h1 ->
      (* Get only the direct text content, not from child elements *)
      let get_direct_text node =
        List.fold_left
          (fun acc child ->
            match element child with
            | None -> acc ^ to_string child
            | Some elem ->
                if name elem = "a" then acc (* Skip anchor links *)
                else acc ^ (texts elem |> String.concat ""))
          ""
          (children node |> to_list)
      in
      let text = get_direct_text h1 |> String.trim in
      (* Extract module name from "Module Nx.Tensor" -> "Nx.Tensor" *)
      if String.starts_with ~prefix:"Module " text then
        String.sub text 7 (String.length text - 7)
      else text
  | None -> "Unknown"

(* Main processing function *)
let process_odoc_html ~source ~library ~destination =
  (* Read HTML using permissive parsing *)
  let soup =
    let stream, close = Markup.file source in
    let signals =
      stream |> Markup.parse_html ~context:`Document |> Markup.signals
    in
    let result = Soup.from_signals signals in
    close ();
    result
  in

  (* Extract the main content *)
  let content = extract_odoc_content soup in

  (* Apply transformations *)
  let content = remove_odoc_navigation content in
  let content = adapt_css_classes content in
  fix_internal_links ~library content;
  enhance_code_blocks content;
  add_backing_lines content;
  remove_stdlib_prefix content;

  (* Extract module name for metadata *)
  let module_name = extract_module_name content in

  (* Generate the final HTML with proper structure *)
  let final_html =
    Printf.sprintf "---\nlayout: layout_docs_%s\ntitle: %s\n---\n\n%s" library
      module_name (Soup.to_string content)
  in

  (* Write to destination *)
  let oc = open_out destination in
  output_string oc final_html;
  close_out oc

(* Entry point *)
let () =
  match Sys.argv with
  | [| _; source; library; destination |] ->
      process_odoc_html ~source ~library ~destination
  | _ ->
      Printf.eprintf "Usage: %s <source.html> <library> <destination.html>\n"
        Sys.argv.(0);
      Printf.eprintf
        "Example: %s _html/nx/Nx.html nx www/site/docs/nx/api/Nx.html\n"
        Sys.argv.(0);
      exit 1
