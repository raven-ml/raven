type benchmark = string * (unit -> unit)
type result = { name : string; avg_time : float; avg_minor_words : float }

let create name f = (name, f)

let run_bench ~warmup f min_time =
  for _ = 1 to warmup do
    f ()
  done;
  Gc.compact ();
  let initial_minor_words = (Gc.quick_stat ()).minor_words in
  let start = Unix.gettimeofday () in
  let total_runs = ref 0 in
  let batch_size = ref 1 in
  while Unix.gettimeofday () -. start < min_time do
    for _ = 1 to !batch_size do
      f ()
    done;
    total_runs := !total_runs + !batch_size;
    batch_size := !batch_size * 2
  done;
  let total_time = Unix.gettimeofday () -. start in
  let final_minor_words = (Gc.quick_stat ()).minor_words in
  let total_minor_words = final_minor_words -. initial_minor_words in
  if !total_runs = 0 then (infinity, infinity)
  else
    let avg_time = total_time /. float !total_runs in
    let avg_minor_words = total_minor_words /. float !total_runs in
    (avg_time, avg_minor_words)

let run ?(trials = 3) ?(min_time = 1.0) ?(warmup = 3) benchmarks =
  Printf.printf
    "Running %d benchmarks, %d trials each (min_time=%.2fs, warmup=%d runs)...\n\
     %!"
    (List.length benchmarks) trials min_time warmup;
  List.map
    (fun (name, f) ->
      let prefix = Printf.sprintf "- Running: %s ... " name in
      let max_progress_len =
        String.length (Printf.sprintf "[Trial %d/%d]" trials trials)
      in
      Printf.printf "%s%!" prefix;
      let results =
        List.init trials (fun i ->
            let progress_msg = Printf.sprintf "[Trial %d/%d]" (i + 1) trials in
            Printf.printf "\r%s%s%!" prefix progress_msg;
            let res = run_bench ~warmup f min_time in
            res)
      in
      let final_msg = "Done." in
      let clear_spaces = String.make max_progress_len ' ' in
      Printf.printf "\r%s%s\r%!" prefix clear_spaces;
      Printf.printf "%s%s\n%!" prefix final_msg;
      let valid_results =
        List.filter
          (fun (t, w) -> Float.is_finite t && Float.is_finite w)
          results
      in
      if valid_results = [] then { name; avg_time = nan; avg_minor_words = nan }
      else
        let min_avg_time =
          List.fold_left (fun acc (t, _) -> min acc t) infinity valid_results
        in
        let total_minor_words =
          List.fold_left (fun acc (_, w) -> acc +. w) 0. valid_results
        in
        let avg_minor_words =
          total_minor_words /. float (List.length valid_results)
        in
        { name; avg_time = min_avg_time; avg_minor_words })
    benchmarks

let print_report results =
  let results =
    List.filter
      (fun r -> Float.is_finite r.avg_time && Float.is_finite r.avg_minor_words)
      results
  in
  if results = [] then Printf.printf "No valid benchmark results to report.\n"
  else
    let reset = "\x1b[0m" in
    let bold = "\x1b[1m" in
    let green = "\x1b[32m" in
    let cyan = "\x1b[36m" in
    let colorize code text = code ^ text ^ reset in
    let strip_ansi_codes s =
      let len = String.length s in
      let buf = Buffer.create len in
      let i = ref 0 in
      while !i < len do
        if !i + 1 < len && s.[!i] = '\x1b' && s.[!i + 1] = '[' then (
          try
            let j = String.index_from s (!i + 2) 'm' in
            i := j + 1
          with Not_found ->
            Buffer.add_char buf s.[!i];
            incr i)
        else (
          Buffer.add_char buf s.[!i];
          incr i)
      done;
      Buffer.contents buf
    in
    let insert_underscores s =
      let len = String.length s in
      let rec aux i acc =
        if i <= 0 then acc
        else if i <= 3 then String.sub s 0 i ^ acc
        else aux (i - 3) ("_" ^ String.sub s (i - 3) 3 ^ acc)
      in
      if len <= 3 then s else aux len ""
    in
    let format_time_ns t =
      let ns = t *. 1e9 in
      let s = Printf.sprintf "%.2f" ns in
      match String.split_on_char '.' s with
      | [ int_part; dec_part ] ->
          insert_underscores int_part ^ "." ^ dec_part ^ "ns"
      | _ -> s ^ "ns"
    in
    let format_minor_words w = Printf.sprintf "%.2fw" w in
    let format_percentage p =
      if p = 100.0 then colorize green (Printf.sprintf "100.00%%")
      else if p > 10000.0 then ""
      else if p < 0.01 then "<0.01%%"
      else Printf.sprintf "%.2f%%" p
    in
    let pad_left s width =
      let visible_len = String.length (strip_ansi_codes s) in
      if visible_len >= width then s
      else String.make (width - visible_len) ' ' ^ s
    in
    let pad_right s width =
      let visible_len = String.length (strip_ansi_codes s) in
      if visible_len >= width then s
      else s ^ String.make (width - visible_len) ' '
    in
    let min_overall_time =
      List.fold_left (fun acc r -> min acc r.avg_time) infinity results
    in
    let min_overall_words =
      List.fold_left (fun acc r -> min acc r.avg_minor_words) infinity results
    in
    let rows_data =
      List.map
        (fun r ->
          let percentage_val =
            if min_overall_time > 0. then r.avg_time /. min_overall_time *. 100.
            else 0.
          in
          ( r,
            [
              r.name;
              (if r.avg_time = min_overall_time then colorize green
               else fun x -> x)
                (format_time_ns r.avg_time);
              (if r.avg_minor_words = min_overall_words then colorize cyan
               else fun x -> x)
                (format_minor_words r.avg_minor_words);
              format_percentage percentage_val;
            ] ))
        results
    in
    let sorted_rows_data =
      List.sort
        (fun (r1, _) (r2, _) -> compare r1.avg_time r2.avg_time)
        rows_data
    in
    let headers = [ "Name"; "Time/Run"; "mWd/Run"; "vs Fastest" ] in
    let widths =
      List.fold_left
        (fun acc (_, row_strs_colored) ->
          List.map2
            (fun w s_colored ->
              max w (String.length (strip_ansi_codes s_colored)))
            acc row_strs_colored)
        (List.map String.length headers)
        sorted_rows_data
    in
    let top_left = "\u{250C}" in
    let top_mid = "\u{252C}" in
    let top_right = "\u{2510}" in
    let mid_left = "\u{251C}" in
    let mid_mid = "\u{253C}" in
    let mid_right = "\u{2524}" in
    let bot_left = "\u{2514}" in
    let bot_mid = "\u{2534}" in
    let bot_right = "\u{2518}" in
    let hline = "\u{2500}" in
    let vline = "\u{2502}" in
    let repeat_str s n =
      let buf = Buffer.create (n * String.length s) in
      for _ = 1 to n do
        Buffer.add_string buf s
      done;
      Buffer.contents buf
    in
    let make_border left mid right =
      left
      ^ String.concat mid (List.map (fun w -> repeat_str hline (w + 2)) widths)
      ^ right
    in
    let top_border = make_border top_left top_mid top_right in
    let separator = make_border mid_left mid_mid mid_right in
    let bottom_border = make_border bot_left bot_mid bot_right in
    Printf.printf "\nBenchmark Results:\n";
    Printf.printf "%s\n" top_border;
    let print_header_row headers =
      let padded_headers =
        List.mapi
          (fun i s ->
            let w = List.nth widths i in
            let padded_s = if i = 0 then pad_right s w else pad_left s w in
            colorize bold padded_s)
          headers
      in
      let row_str = String.concat (" " ^ vline ^ " ") padded_headers in
      Printf.printf "%s %s %s\n" vline row_str vline
    in
    print_header_row headers;
    Printf.printf "%s\n" separator;
    let print_data_row (_result_record, row_strings_colored) =
      let padded_colored_row =
        List.mapi
          (fun i s_colored ->
            let w = List.nth widths i in
            if i = 0 then pad_right s_colored w else pad_left s_colored w)
          row_strings_colored
      in
      let row_str = String.concat (" " ^ vline ^ " ") padded_colored_row in
      Printf.printf "%s %s %s\n" vline row_str vline
    in
    List.iter print_data_row sorted_rows_data;
    Printf.printf "%s\n" bottom_border;
    flush stdout
