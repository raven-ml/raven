let text file =
  let contents = In_channel.with_open_bin file In_channel.input_all in
  let last = String.length contents - 1 in
  match String.index_opt contents '\n' with
  | Some eol
    when String.starts_with ~prefix:"# tinygrad " contents
         && last > eol
         && contents.[last] = '\n' ->
      String.sub contents (eol + 1) (last - eol - 1)
  | _ ->
      failwith
        (file
       ^ ": not a golden: it is not a '# tinygrad <commit>' line, a body and a \
          newline")

(* Tables *)

type row = { file : string; columns : string array; cells : string array }

let table file =
  match String.split_on_char '\n' (text file) with
  | [] | [ _ ] -> failwith (file ^ ": the table has no row")
  | header :: lines ->
      let columns = Array.of_list (String.split_on_char '\t' header) in
      let row i line =
        let cells = Array.of_list (String.split_on_char '\t' line) in
        if Array.length cells <> Array.length columns then
          failwith
            (Printf.sprintf "%s, line %d: %d cells for %d columns" file (i + 3)
               (Array.length cells) (Array.length columns));
        { file; columns; cells }
      in
      List.mapi row lines

let cell row column =
  match Array.find_index (String.equal column) row.columns with
  | Some i -> row.cells.(i)
  | None ->
      invalid_arg
        (Printf.sprintf "%s: no column %S; the columns are %s" row.file column
           (String.concat ", " (Array.to_list row.columns)))

let key columns row =
  String.concat " "
    (List.map (fun column -> column ^ "=" ^ cell row column) columns)
