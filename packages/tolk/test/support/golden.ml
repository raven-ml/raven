(* dune copies a suite's goldens beside its executable, which is where the suite
   runs under dune test but not under dune exec. *)
let path file =
  if Filename.is_relative file then
    Filename.concat (Filename.dirname Sys.executable_name) file
  else file

let read file = In_channel.with_open_bin (path file) In_channel.input_all
let exists file = Sys.file_exists (path file)

(* A golden's file is its body and a final newline. *)
let golden_body file =
  let contents = read file in
  let last = String.length contents - 1 in
  if last < 0 || contents.[last] <> '\n' then
    failwith (file ^ ": not a golden");
  String.sub contents 0 last

(* Tolk's output

   [x.tolk] sits beside [x.golden] where tolk differs from upstream on purpose.
   Its first line names the divergences it belongs to and the hash of the body
   of [x.golden] it was recorded against:

     # tolk D81 D96 golden=<hash>

   and the rest is the golden's body format. A [.tolk] with no [.golden] is an
   output with no upstream answer, and carries no hash. *)

type header = { entries : string list; hash : string option }

let tolk_name file = Filename.remove_extension file ^ ".tolk"

(* A hash of a golden's body: stable across runs and machines, and available
   without a dependency. *)
let digest body = Digest.(to_hex (string body))

let is_entry token =
  String.length token > 1
  && token.[0] = 'D'
  && String.for_all
       (function '0' .. '9' -> true | _ -> false)
       (String.sub token 1 (String.length token - 1))

let hash_prefix = "golden="

(* The header of [tolk], from its first line. *)
let header tolk line =
  let hash_of token =
    let n = String.length hash_prefix in
    if String.starts_with ~prefix:hash_prefix token then
      Some (String.sub token n (String.length token - n))
    else None
  in
  let classify (entries, hash) token =
    match hash_of token with
    | Some _ when Option.is_some hash ->
        failwith (tolk ^ ": the header has two golden= hashes")
    | Some h -> (entries, Some h)
    | None when is_entry token -> (token :: entries, hash)
    | None ->
        failwith
          (Printf.sprintf
             "%s: the header's %S is neither a D-number nor golden=<hash>" tolk
             token)
  in
  match String.split_on_char ' ' line with
  | "#" :: "tolk" :: tokens ->
      let entries, hash = List.fold_left classify ([], None) tokens in
      { entries = List.rev entries; hash }
  | _ -> failwith (tolk ^ ": not a golden")

let render_header { entries; hash } =
  let hash =
    match hash with None -> [] | Some h -> [ hash_prefix ^ h ]
  in
  String.concat " " ([ "#"; "tolk" ] @ entries @ hash)

(* The first line of a tolk file, which starts with [# tolk], and its body: the
   rest, without the final newline. *)
let tolk_parts tolk =
  let contents = read tolk in
  let last = String.length contents - 1 in
  match String.index_opt contents '\n' with
  | Some eol
    when String.starts_with ~prefix:"# tolk" contents
         && last > eol
         && contents.[last] = '\n' ->
      let line = String.sub contents 0 eol in
      (header tolk line, String.sub contents (eol + 1) (last - eol - 1))
  | _ -> failwith (tolk ^ ": not a golden")

(* The checks of [tolk] against [file]'s golden, on its [header] and [body]. *)
let check_tolk file tolk { entries; hash } body =
  if entries = [] then
    failwith
      (tolk
     ^ ": it names no entry: add the DIVERGENCES.md number of this difference \
        to its header, as # tolk D81 golden=<hash>");
  match (exists file, hash) with
  | false, None -> ()
  | false, Some h ->
      failwith
        (Printf.sprintf
           "%s: its header carries golden=%s but %s does not exist: remove the \
            hash if tolk owns this output, or restore the golden"
           tolk h file)
  | true, None ->
      failwith
        (Printf.sprintf
           "%s: its header has no golden= hash but %s exists: rerun with -u \
            for a text or graph golden, or add golden=%s to the header"
           tolk file
           (digest (golden_body file)))
  | true, Some h ->
      let golden = golden_body file in
      let now = digest golden in
      if h <> now then
        failwith
          (Printf.sprintf
             "%s: %s moved beneath %s: the header says golden=%s, it now \
              hashes to %s. Check that the difference from the new golden is \
              still intended, then rerun with -u for a text or graph golden \
              or set the new hash in the header of a table or an input, or \
              delete %s if the golden now matches tolk."
             tolk file
             (String.concat " " entries)
             h now tolk);
      if String.equal golden body then
        failwith
          (Printf.sprintf
             "%s: its body is the same as %s: delete %s, tolk no longer differs"
             tolk file tolk)

(* The body a test compares against, and, for a [.tolk], the file and the
   entries it names, to label a failure's diff. *)
let expected file =
  let tolk = tolk_name file in
  if not (exists tolk) then (golden_body file, None)
  else begin
    let header, body = tolk_parts tolk in
    check_tolk file tolk header body;
    (body, Some (tolk ^ " (" ^ String.concat " " header.entries ^ ")"))
  end

let body file = fst (expected file)

(* [-u] is windtrap's flag, read from the program's arguments: only then does
   a golden test write its [.tolk], so a plain run keeps a golden's own diff. *)
let updating = Array.exists (fun a -> a = "-u" || a = "--update") Sys.argv

(* The absolute path windtrap maps back to the source file. *)
let absolute file =
  let p = path file in
  if Filename.is_relative p then Filename.concat (Sys.getcwd ()) p else p

(* The test, under [-u], that writes [actual] as [file]'s [.tolk]. It writes
   nothing when [actual] is the golden's body and no [.tolk] exists. *)
let update file actual =
  let tolk = tolk_name file in
  let golden = if exists file then Some (golden_body file) else None in
  match golden with
  | Some g when String.equal g actual ->
      if exists tolk then
        Windtrap.fail
          (tolk ^ ": tolk's output is " ^ file ^ " again: delete " ^ tolk)
  | _ ->
      let entries =
        if exists tolk then (fst (tolk_parts tolk)).entries else []
      in
      let contents =
        render_header { entries; hash = Option.map digest golden }
        ^ "\n" ^ actual ^ "\n"
      in
      Windtrap.expect_file ~file:(absolute tolk) contents

let compare_text file actual =
  if updating then update file actual
  else
    match expected file with
    | exception Failure message -> Windtrap.fail message
    | body, msg -> Windtrap.equal ?msg Windtrap.text body actual

let text file actual =
  Windtrap.test file (fun () -> compare_text file (actual ()))

(* Graphs *)

let graph file sink =
  Windtrap.test file (fun () ->
      compare_text file (Tolk.Graph.to_string (sink ())))

let sink file =
  try Tolk.Graph.of_string (body file)
  with Failure e -> failwith (file ^ ": " ^ e)

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
            (Printf.sprintf "%s, line %d: %d cells for %d columns" file (i + 2)
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

(* A table is read when the suite is declared: a stale [.tolk] fails one test,
   and the rest of the suite runs. *)
let cases ?key file check =
  match table file with
  | exception Failure message ->
      Windtrap.test file (fun () -> Windtrap.fail message)
  | columns, rows ->
      let key = Option.value key ~default:[ List.hd columns ] in
      let name row =
        String.concat " "
          (List.map
             (fun column -> column ^ "=" ^ cell file columns row column)
             key)
      in
      Windtrap.cases file ~id:name rows (fun row ->
          check (cell file columns row))
