(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

let lab = [ "lab" ]
let remove_if_exists path = try Sys.remove path with Sys_error _ -> ()
let runner_pid = Unix.getpid ()

let write_file path contents =
  let channel = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out channel)
    (fun () -> output_string channel contents)

let temporary suffix =
  let path = Filename.temp_file "nx_io_bench_" suffix in
  at_exit (fun () -> if Unix.getpid () = runner_pid then remove_if_exists path);
  path

let le16 value =
  String.init 2 (fun index -> Char.chr ((value lsr (index * 8)) land 0xff))

let le32 value =
  String.init 4 (fun index ->
      Char.chr
        (Int32.to_int
           (Int32.logand (Int32.shift_right_logical value (index * 8)) 0xffl)))

let crc32 contents =
  let table =
    Array.init 256 (fun value ->
        let crc = ref (Int32.of_int value) in
        for _ = 0 to 7 do
          crc :=
            if Int32.logand !crc 1l = 0l then Int32.shift_right_logical !crc 1
            else Int32.logxor (Int32.shift_right_logical !crc 1) 0xedb88320l
        done;
        !crc)
  in
  let crc = ref 0xffffffffl in
  String.iter
    (fun byte ->
      let index =
        Int32.logxor !crc (Int32.of_int (Char.code byte))
        |> Int32.logand 0xffl |> Int32.to_int
      in
      crc := Int32.logxor table.(index) (Int32.shift_right_logical !crc 8))
    contents;
  Int32.logxor !crc 0xffffffffl

let stored_gzip contents =
  let length = String.length contents in
  let output = Buffer.create (length + (length / 65_535 * 5) + 32) in
  Buffer.add_string output "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff";
  let rec blocks offset =
    let remaining = length - offset in
    let size = min remaining 65_535 in
    let final = size = remaining in
    Buffer.add_char output (if final then '\x01' else '\x00');
    Buffer.add_string output (le16 size);
    Buffer.add_string output (le16 (size lxor 0xffff));
    Buffer.add_substring output contents offset size;
    if not final then blocks (offset + size)
  in
  blocks 0;
  Buffer.add_string output (le32 (crc32 contents));
  Buffer.add_string output (le32 (Int32.of_int length));
  Buffer.contents output

let structured =
  Nx.arange Nx.float32 0 (512 * 512) 1 |> Nx.reshape [| 512; 512 |]

let random =
  let state = Random.State.make [| 0x4e58; 0x494f |] in
  let values = Array.init (1024 * 1024) (fun _ -> Random.State.int state 256) in
  Nx.create Nx.uint8 [| 1024; 1024 |] values

let image =
  let width = 512 in
  let height = 512 in
  let values =
    Array.init
      (width * height * 3)
      (fun index ->
        let pixel = index / 3 in
        let x = pixel mod width in
        let y = pixel / width in
        match index mod 3 with
        | 0 -> (x + (y / 8)) land 0xff
        | 1 -> (y + (x / 4)) land 0xff
        | _ -> x lxor y land 0xff)
  in
  Nx.create Nx.uint8 [| height; width; 3 |] values

let npy_input = temporary ".npy"
let npy_output = temporary ".npy"
let npz_deflate_input = temporary ".npz"
let npz_store_input = temporary ".npz"
let npz_output = temporary ".npz"
let npz_deflate_output = temporary ".npz"
let png_input = temporary ".png"
let png_output = temporary ".png"
let jpeg_input = temporary ".jpg"
let jpeg_output = temporary ".jpg"
let gzip_input = temporary ".gz"
let gzip_output = temporary ".raw"

let () =
  Nx_io.save_npy npy_input structured;
  Nx_io.save_npz npz_deflate_input [ ("structured", Nx_io.P structured) ];
  Nx_io.save_npz npz_store_input [ ("random", Nx_io.P random) ];
  Nx_io.save_image png_input image;
  Nx_io.save_image jpeg_input image;
  write_file gzip_input (stored_gzip (String.make (1024 * 1024) '\x5a'))

let benchmarks =
  [
    Thumper.group "npy"
      [
        Thumper.bench ~tags:lab "Load NPY 1 MiB f32" (fun () ->
            Nx_io.load_npy npy_input);
        Thumper.bench "Save NPY 1 MiB f32" (fun () ->
            Nx_io.save_npy npy_output structured);
      ];
    Thumper.group "npz"
      [
        Thumper.bench ~tags:lab "Load NPZ Deflate 1 MiB f32" (fun () ->
            Nx_io.load_npz npz_deflate_input);
        Thumper.bench "Load NPZ Store 1 MiB u8" (fun () ->
            Nx_io.load_npz npz_store_input);
        Thumper.bench "Save NPZ adaptive 1 MiB u8" (fun () ->
            Nx_io.save_npz npz_output [ ("random", Nx_io.P random) ]);
        Thumper.bench "Save NPZ Deflate 1 MiB f32" (fun () ->
            Nx_io.save_npz npz_deflate_output
              [ ("structured", Nx_io.P structured) ]);
      ];
    Thumper.group "png"
      [
        Thumper.bench ~tags:lab "Load PNG 512x512 RGB" (fun () ->
            Nx_io.load_image png_input);
        Thumper.bench "Save PNG 512x512 RGB" (fun () ->
            Nx_io.save_image png_output image);
      ];
    Thumper.group "jpeg"
      [
        Thumper.bench ~tags:lab "Load JPEG 512x512 RGB" (fun () ->
            Nx_io.load_image jpeg_input);
        Thumper.bench "Save JPEG 512x512 RGB" (fun () ->
            Nx_io.save_image jpeg_output image);
      ];
    Thumper.group "gzip"
      [
        Thumper.bench ~tags:lab "Gunzip stored 1 MiB" (fun () ->
            Nx_io.gunzip ~src:gzip_input ~dst:gzip_output);
      ];
  ]

let () =
  Thumper.run "nx_io"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.10;
        Thumper.Budget.no_more_alloc_than 0.05;
      ]
    benchmarks
