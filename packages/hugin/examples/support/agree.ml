(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [agree.exe committed render] replaces the example's [render] with the bytes
   of the [committed] image when their pixels agree, so that the [cmp] after it
   passes, and otherwise leaves [render] for [cmp] to fail on and [dune
   promote] to take.

   Two images agree when they have one shape and differ by at most [levels] in
   each channel of each pixel. Renders of one figure differ between systems in
   the last bits of their arithmetic: OCaml fuses a multiplication and an
   addition into one rounding on arm64 and not on x86-64, and the C library's
   sin, cos and acos round differently. Where that flips a choice, such as the
   number of chords along an arc, a few pixels move by a few levels:
   16-styles' image differs between macOS on arm64 and Linux on x86-64 in 13
   pixels of one line, by up to 3. A moved edge, a changed colour or text
   moves pixels by more. *)

let levels = 3

let agree a b =
  Nx.shape a = Nx.shape b
  &&
  let a = Nx.to_array a and b = Nx.to_array b in
  let close = ref true in
  Array.iteri (fun i v -> if abs (v - b.(i)) > levels then close := false) a;
  !close

let () =
  match Sys.argv with
  | [| _; committed; render |] ->
      if agree (Nx_io.load_image committed) (Nx_io.load_image render) then
        Out_channel.with_open_bin render (fun oc ->
            output_string oc
              (In_channel.with_open_bin committed In_channel.input_all))
  | _ -> failwith "usage: agree.exe COMMITTED RENDER"
