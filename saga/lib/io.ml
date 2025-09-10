let read_lines ?(buffer_size = 65536) filename =
  let ic = open_in filename in
  Fun.protect ~finally:(fun () -> close_in ic) @@ fun () ->
  (* Pre-allocate buffer for efficiency *)
  let buffer = Bytes.create buffer_size in
  let lines = ref [] in
  let current_line = Buffer.create 256 in

  let rec read_buffer () =
    let bytes_read = input ic buffer 0 buffer_size in
    if bytes_read = 0 then (
      if
        (* End of file - add remaining line if exists *)
        Buffer.length current_line > 0
      then lines := Buffer.contents current_line :: !lines)
    else
      (* Process the buffer *)
      let rec process_bytes pos =
        if pos < bytes_read then
          let c = Bytes.get buffer pos in
          if c = '\n' then (
            (* Complete line found *)
            lines := Buffer.contents current_line :: !lines;
            Buffer.clear current_line;
            process_bytes (pos + 1))
          else if c <> '\r' then (
            (* Add character to current line (skip \r for Windows
               compatibility) *)
            Buffer.add_char current_line c;
            process_bytes (pos + 1))
          else process_bytes (pos + 1)
      in
      process_bytes 0;
      read_buffer ()
  in

  read_buffer ();
  List.rev !lines

let read_lines_lazy ?(buffer_size = 65536) filename =
  let ic = open_in filename in
  let buffer = Bytes.create buffer_size in
  let buffer_pos = ref 0 in
  let bytes_in_buffer = ref 0 in
  let current_line = Buffer.create 256 in
  let eof_reached = ref false in

  let next_line () =
    if !eof_reached then (
      close_in ic;
      None)
    else
      let rec find_line () =
        (* Check if we need to read more data *)
        if !buffer_pos >= !bytes_in_buffer then (
          bytes_in_buffer := input ic buffer 0 buffer_size;
          buffer_pos := 0;
          if !bytes_in_buffer = 0 then (
            (* EOF reached *)
            eof_reached := true;
            if Buffer.length current_line > 0 then (
              let line = Buffer.contents current_line in
              Buffer.clear current_line;
              Some line)
            else (
              close_in ic;
              None))
          else find_line ())
        else
          (* Process current buffer *)
          let c = Bytes.get buffer !buffer_pos in
          incr buffer_pos;
          if c = '\n' then (
            let line = Buffer.contents current_line in
            Buffer.clear current_line;
            Some line)
          else if c <> '\r' then (
            Buffer.add_char current_line c;
            find_line ())
          else find_line ()
      in
      find_line ()
  in
  Seq.unfold (fun () -> Option.map (fun line -> (line, ())) (next_line ())) ()

let write_lines ?(append = false) filename lines =
  let oc =
    if append then
      open_out_gen
        [ Open_wronly; Open_append; Open_creat; Open_text ]
        0o644 filename
    else open_out filename
  in
  Fun.protect ~finally:(fun () -> close_out oc) @@ fun () ->
  List.iter
    (fun line ->
      output_string oc line;
      output_char oc '\n')
    lines
