(** Non-scalar data: media files and tables.

    Demonstrates log_media for files and log_table for structured data. Creates
    synthetic data to keep the example self-contained. *)

open Munin

let write_file path text =
  let oc = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc text)

(* Write a tiny PPM image (3x3 red gradient). *)
let write_ppm path =
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      Printf.fprintf oc "P6\n3 3\n255\n";
      for row = 0 to 2 do
        for _col = 0 to 2 do
          let v = 85 * row in
          output_char oc (Char.chr v);
          output_char oc '\000';
          output_char oc '\000'
        done
      done)

let () =
  let root = "_munin" in
  let session = Session.start ~root ~experiment:"media-demo" ~name:"run-1" () in
  let tmp = Filename.concat root "_tmp" in
  (try Unix.mkdir tmp 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());

  (* Log an image at two different steps. *)
  let img_path = Filename.concat tmp "sample.ppm" in
  write_ppm img_path;
  Session.log_media session ~step:1 ~key:"viz/sample" ~kind:`Image
    ~path:img_path;
  write_ppm img_path;
  Session.log_media session ~step:2 ~key:"viz/sample" ~kind:`Image
    ~path:img_path;

  (* Log a text file. *)
  let notes_path = Filename.concat tmp "notes.txt" in
  write_file notes_path "Observation: signal peaks at 550nm.\n";
  Session.log_media session ~step:1 ~key:"notes" ~kind:`File ~path:notes_path;

  (* Log a structured table — e.g. per-class metrics or a confusion matrix. *)
  Session.log_table session ~step:1 ~key:"results/per_band"
    ~columns:[ "band"; "snr"; "coverage" ]
    ~rows:
      [
        [ `String "blue"; `Float 12.3; `Float 0.95 ];
        [ `String "green"; `Float 18.7; `Float 0.98 ];
        [ `String "red"; `Float 15.1; `Float 0.92 ];
      ];

  Session.finish session ();

  (* Read back media entries. *)
  let run = Session.run session in
  Printf.printf "run: %s\n" (Run.id run);
  Printf.printf "media keys: %s\n" (String.concat ", " (Run.media_keys run));
  let entries = Run.media_history run "viz/sample" in
  Printf.printf "viz/sample: %d entries\n" (List.length entries);
  List.iter
    (fun (e : Run.media_entry) ->
      Printf.printf "  step=%d kind=%s path=%s\n" e.step
        (match e.kind with
        | `Image -> "image"
        | `Audio -> "audio"
        | `Table -> "table"
        | `File -> "file")
        (Filename.basename e.path))
    entries
