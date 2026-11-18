(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A save whose data passes 4 GiB writes RF64, which loads back as the samples
   it was given. The samples are a mapped npy file of 2^31 + 1 rows of int16,
   sparse but for three of them; the WAV file holds them all, so it takes 4 GiB
   of disk. The dune file says how to run it. *)

open Windtrap
module Audio = Nx_io.Audio

let env k =
  match Sys.getenv_opt k with
  | Some v -> v
  | None -> failwith (k ^ " is not set: the dune file says how to run this")

let dir = env "NX_IO_RF64_DIR"
let rows = (1 lsl 31) + 1

(* The machine's floor of free disk, past what the run writes. *)
let floor_kib = 50 * 1024 * 1024
let needed_kib = (2 * rows / 1024) + (1024 * 1024)

let free_kib () =
  let ic = Unix.open_process_args_in "df" [| "df"; "-Pk"; dir |] in
  let lines = In_channel.input_lines ic in
  ignore (Unix.close_process_in ic);
  match List.rev lines with
  | last :: _ -> (
      match List.filter (( <> ) "") (String.split_on_char ' ' last) with
      | _ :: _ :: _ :: avail :: _ -> int_of_string avail
      | _ -> failwith last)
  | [] -> failwith "df printed nothing"

(* An npy file of [rows] int16 rows of one column, zeros but for [marks],
   written at their rows; the rest is a hole. *)
let sparse path marks =
  let dict =
    Printf.sprintf
      "{'descr': '<i2', 'fortran_order': False, 'shape': (%d, 1), }" rows
  in
  let header_len = 10 + String.length dict + 1 in
  let pad = (64 - (header_len mod 64)) mod 64 in
  let header =
    "\147NUMPY\001\000"
    ^ String.init 2 (fun i ->
        Char.chr (((String.length dict + pad + 1) lsr (8 * i)) land 0xff))
    ^ dict ^ String.make pad ' ' ^ "\n"
  in
  let fd = Unix.openfile path [ Unix.O_WRONLY; O_CREAT; O_TRUNC ] 0o644 in
  let write_at at s =
    ignore (Unix.lseek fd at Unix.SEEK_SET : int);
    ignore (Unix.write_substring fd s 0 (String.length s) : int)
  in
  write_at 0 header;
  List.iter
    (fun (row, v) ->
      write_at
        (String.length header + (2 * row))
        (String.init 2 (fun i -> Char.chr ((v lsr (8 * i)) land 0xff))))
    marks;
  Unix.ftruncate fd (String.length header + (2 * rows));
  Unix.close fd

let marks = [ (0, 0x0102); (rows / 2, -2); (rows - 1, 0x1234) ]

let rf64 =
  test "a save past 4 GiB writes RF64, which loads back" (fun () ->
      let free = free_kib () in
      if free < floor_kib + needed_kib then
        skip ~reason:(Printf.sprintf "%d KiB free in %s" free dir) ();
      let npy = Filename.concat dir "rf64_source.npy"
      and wav = Filename.concat dir "rf64_save.wav" in
      Fun.protect
        ~finally:(fun () ->
          List.iter
            (fun p -> try Sys.remove p with Sys_error _ -> ())
            [ npy; wav ])
        (fun () ->
          sparse npy marks;
          let x = Nx.unpack Nx.int16 (Nx_io.load_npy npy) in
          Nx_io.save_wav wav ~rate:16000 x;
          let head =
            In_channel.with_open_bin wav (fun ic -> really_input_string ic 12)
          in
          equal string "RF64\255\255\255\255WAVE" head;
          let a = Nx_io.load_audio wav in
          equal int rows (Audio.length a);
          List.iter
            (fun (row, v) ->
              let w = Nx.unpack Nx.int16 (Audio.window a ~row ~rows:1) in
              equal (array int) [| v |] (Nx.to_array w))
            ((1, 0) :: marks)))

let () = exit (run "nx.io RF64 save" [ rf64 ])
