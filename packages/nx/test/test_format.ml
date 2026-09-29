(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* How tensors print. [pp]'s layout is a baseline: its text is reviewed when
   accepted. *)

open Windtrap

let v dt xs = Nx.create dt [| Array.length xs |] xs

let printing =
  group "printing"
    [
      test "a scalar, a vector and a matrix" (fun () ->
          expect (Nx.to_string (Nx.scalar Nx.float32 1.5))
          @@ __POS_OF__ {| 1.5 |};
          expect (Nx.to_string (v Nx.int32 [| 1l; -2l; 3l |]))
          @@ __POS_OF__ {| [1, -2, 3] |};
          expect
            (Nx.to_string (Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1)))
          @@ __POS_OF__
               {|
            int32 [2,3]
            [[0, 1, 2],
             [3, 4, 5]]
            |});
      test "an empty tensor and a tensor of three axes" (fun () ->
          expect (Nx.to_string (Nx.zeros Nx.float64 [| 0; 3 |]))
          @@ __POS_OF__
               {|
            float64 [0,3]
            []
            |};
          expect
            (Nx.to_string (Nx.reshape [| 2; 2; 2 |] (Nx.arange Nx.int32 0 8 1)))
          @@ __POS_OF__
               {|
            int32 [2,2,2]
            [[[0, 1],
              [2, 3]],
             [[4, 5],
              [6, 7]]]
            |});
      test "each kind of element" (fun () ->
          expect (Nx.to_string (v Nx.bool [| true; false |]))
          @@ __POS_OF__ {| [true, false] |};
          expect
            (Nx.to_string
               (v Nx.float64 [| 0.1; Float.nan; Float.infinity; -0. |]))
          @@ __POS_OF__ {| [0.1, nan, inf, -0] |};
          expect
            (Nx.to_string
               (v Nx.complex64
                  Complex.[| { re = 1.; im = -2. }; { re = 0.5; im = 3. } |]))
          @@ __POS_OF__ {| [(1-2i), (0.5+3i)] |};
          expect (Nx.to_string (v Nx.uint8 [| 0; 255 |]))
          @@ __POS_OF__ {| [0, 255] |});
      test "a long tensor is truncated" (fun () ->
          expect (Nx.to_string (Nx.arange Nx.int32 0 1000 1))
          @@ __POS_OF__
               {|
            int32 [1000]
            [0, 1, ..., 998, 999]
            |};
          expect
            (Nx.to_string
               (Nx.reshape [| 100; 100 |] (Nx.arange Nx.int32 0 10000 1)))
          @@ __POS_OF__
               {|
            int32 [100,100]
            [[0, 1, ..., 98, 99],
             [100, 101, ..., 198, 199],
             ...
             [9800, 9801, ..., 9898, 9899],
             [9900, 9901, ..., 9998, 9999]]
            |});
      test "a view prints its elements" (fun () ->
          let m = Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1) in
          equal string
            (Nx.to_string (Nx.contiguous (Nx.transpose m)))
            (Nx.to_string (Nx.transpose m)));
      test "print writes to_string and a newline" (fun () ->
          let t = v Nx.int32 [| 1l; 2l |] in
          Nx.print t;
          equal string (Nx.to_string t ^ "\n") (output ()));
      test "pp_shape brackets the dimensions and pp_dtype names the dtype"
        (fun () ->
          equal string "[2,3,4]"
            (Format.asprintf "%a" Nx.pp_shape [| 2; 3; 4 |]);
          equal string "[]" (Format.asprintf "%a" Nx.pp_shape [||]);
          equal string "float32" (Format.asprintf "%a" Nx.pp_dtype Nx.float32);
          equal string "bfloat16" (Format.asprintf "%a" Nx.pp_dtype Nx.bfloat16));
    ]

let () = exit (run "nx format" [ printing ])
