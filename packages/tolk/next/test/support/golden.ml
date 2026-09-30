(* dune copies a suite's goldens beside its executable, which is where the suite
   runs under dune test but not under dune exec. *)
let path file =
  if Filename.is_relative file then
    Filename.concat (Filename.dirname Sys.executable_name) file
  else file

let body file =
  let contents = In_channel.with_open_bin (path file) In_channel.input_all in
  let last = String.length contents - 1 in
  match String.index_opt contents '\n' with
  | Some eol
    when String.starts_with ~prefix:"# tinygrad " contents
         && last > eol
         && contents.[last] = '\n' ->
      String.sub contents (eol + 1) (last - eol - 1)
  | _ -> failwith (file ^ ": not a golden")

let text file actual =
  Windtrap.test file (fun () ->
      Windtrap.equal Windtrap.text (body file) (actual ()))

(* Graphs *)

let graph file sink =
  Windtrap.test file (fun () ->
      Windtrap.equal Windtrap.text (body file) (Graph.to_string (sink ())))

let sink file =
  try Graph.of_string (body file) with Failure e -> failwith (file ^ ": " ^ e)

(* Tables *)

let table file =
  match String.split_on_char '\n' (body file) with
  | [] | [ _ ] -> failwith (file ^ ": the table has no row")
  | header :: lines ->
      let columns = String.split_on_char '\t' header in
      let row i line =
        let cells = String.split_on_char '\t' line in
        if List.compare_lengths cells columns <> 0 then
          failwith
            (Printf.sprintf "%s, line %d: %d cells for %d columns" file (i + 3)
               (List.length cells) (List.length columns));
        List.combine columns cells
      in
      (columns, List.mapi row lines)

let cell file columns row column =
  match List.assoc_opt column row with
  | Some cell -> cell
  | None ->
      invalid_arg
        (Printf.sprintf "%s: no column %S; the columns are %s" file column
           (String.concat ", " columns))

let columns file = fst (table file)

let rows file =
  let columns, rows = table file in
  List.map (cell file columns) rows

let cases ?key file check =
  let columns, rows = table file in
  let key = Option.value key ~default:[ List.hd columns ] in
  let name row =
    String.concat " "
      (List.map (fun column -> column ^ "=" ^ cell file columns row column) key)
  in
  Windtrap.cases ~name file rows (fun row -> check (cell file columns row))
