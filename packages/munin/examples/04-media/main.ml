(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Files a run logs at a step, such as images of samples and a table, then found
   again through the run's history. *)

open Munin

(* A 3x3 PPM image whose red rises down the rows, by [shade] a row. *)
let write_ppm path ~shade =
  Out_channel.with_open_bin path (fun oc ->
      output_string oc "P6\n3 3\n255\n";
      for row = 0 to 2 do
        for _ = 0 to 2 do
          output_char oc (Char.chr (shade * row));
          output_string oc "\000\000"
        done
      done)

let () =
  let store = Store.open_ () in
  let id =
    Session.within (Session.start ~store ~experiment:"media" ~name:"run-1" ())
    @@ fun s ->
    let image = Filename.temp_file "sample" ".ppm" in
    for step = 1 to 2 do
      write_ppm image ~shade:(40 * step);
      Session.log_file s ~step ~key:"viz/sample" image
    done;
    let table = Filename.temp_file "per_band" ".tsv" in
    Out_channel.with_open_bin table (fun oc ->
        output_string oc
          "band\tsnr\tcoverage\nblue\t12.3\t0.95\ngreen\t18.7\t0.98\n");
    Session.log_file s ~step:2 ~key:"results/per_band" table;
    Session.id s
  in
  Printf.printf "run: %s\n" id;
  let h = History.read (Option.get (Store.run store id)) in
  List.iter
    (fun (f : History.file) ->
      Printf.printf "%s at step %d: %s\n" f.key f.step f.path)
    (History.files h)
