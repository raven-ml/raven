(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Patches, correlation and filters against their definitions: sums and folds
   over the windows of the reference. *)

open Windtrap
open Nx_test

let near = Ref.witness (close ~rel:1e-12 ~abs:1e-12 ())
let small = Gen.map float_of_int (Gen.int_range (-3) 3)

let floats shape =
  Gen.map
    (fun xs -> Nx.create Nx.float64 shape xs)
    (Gen.array ~size:(Gen.constant (Ref.numel shape)) small)

(* The input value at [pos] of the padded input, zero in the padding. *)
let padded (r : float Ref.t) ~padding pos =
  let spatial = r.shape in
  let src = Array.mapi (fun d p -> p - fst padding.(d)) pos in
  if Array.for_all2 (fun p n -> p >= 0 && p < n) src spatial then Ref.get r src
  else 0.

(* One spatial axis or two, their windows' sizes, steps, dilations and paddings,
   over axes short enough that some hold no window. *)
let geometry =
  let open Gen in
  let* k = int_range 1 2 in
  let axis =
    pair
      (quad (int_range 0 6) (int_range 1 3) (int_range 1 2)
         (pair (int_range 0 2) (int_range 0 2)))
      (int_range 1 2)
  in
  let+ axes = array ~size:(constant k) axis in
  let size = Array.map (fun ((n, _, _, _), _) -> n) axes in
  let kernel = Array.map (fun ((_, w, _, _), _) -> w) axes in
  let stride = Array.map (fun ((_, _, s, _), _) -> s) axes in
  let padding = Array.map (fun ((_, _, _, p), _) -> p) axes in
  let dilation = Array.map snd axes in
  (size, kernel, stride, padding, dilation)

let out_size ~size ~kernel ~stride ~dilation ~padding =
  Array.init (Array.length size) (fun d ->
      let lo, hi = padding.(d) in
      let span = (dilation.(d) * (kernel.(d) - 1)) + 1 in
      let padded = size.(d) + lo + hi in
      if padded < span then 0 else ((padded - span) / stride.(d)) + 1)

let patches =
  group "patches"
    [
      prop "extract_patches gathers each window, offsets first, positions last"
        (Gen.bind geometry (fun (size, kernel, stride, padding, dilation) ->
             Gen.map
               (fun t -> (t, kernel, stride, padding, dilation))
               (floats size)))
        (fun (t, kernel_size, stride, padding, dilation) ->
          let size = Nx.shape t in
          let out =
            out_size ~size ~kernel:kernel_size ~stride ~dilation ~padding
          in
          let r = Ref.of_nx t in
          let expected =
            Ref.init
              [| Ref.numel kernel_size; Ref.numel out |]
              (fun i ->
                let off = Ref.unravel kernel_size i.(0)
                and pos = Ref.unravel out i.(1) in
                padded r ~padding
                  (Array.mapi
                     (fun d p -> (p * stride.(d)) + (off.(d) * dilation.(d)))
                     pos))
          in
          equal near expected
            (Ref.of_nx
               (Nx.extract_patches ~kernel_size ~stride ~dilation ~padding t)));
      prop "combine_patches is the adjoint of extract_patches, summing overlaps"
        (Gen.bind geometry (fun (size, kernel, stride, padding, dilation) ->
             let out = out_size ~size ~kernel ~stride ~dilation ~padding in
             Gen.map
               (fun (x, y) -> (x, y, kernel, stride, padding, dilation))
               (Gen.pair (floats size)
                  (floats [| Ref.numel kernel; Ref.numel out |]))))
        (fun (x, y, kernel_size, stride, padding, dilation) ->
          let dot a b = Nx.item [] (Nx.sum (Nx.mul a b)) in
          equal
            (close ~rel:1e-12 ~abs:1e-12 ())
            (dot
               (Nx.extract_patches ~kernel_size ~stride ~dilation ~padding x)
               y)
            (dot x
               (Nx.combine_patches ~output_size:(Nx.shape x) ~kernel_size
                  ~stride ~dilation ~padding y)));
    ]

let window_refusals =
  let one = [| 1 |] and none = [| (0, 0) |] in
  let x = Nx.zeros Nx.float32 [| 2; 3 |] in
  [
    ( "no kernel axis",
      fun () ->
        Nx.extract_patches ~kernel_size:[||] ~stride:[||] ~dilation:[||]
          ~padding:[||] x );
    ( "a stride per kernel axis",
      fun () ->
        Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1; 1 |] ~dilation:one
          ~padding:none x );
    ( "a zero stride",
      fun () ->
        Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 0 |] ~dilation:one
          ~padding:none x );
    ( "a zero kernel size",
      fun () ->
        Nx.extract_patches ~kernel_size:[| 0 |] ~stride:one ~dilation:one
          ~padding:none x );
    ( "a zero dilation",
      fun () ->
        Nx.extract_patches ~kernel_size:[| 2 |] ~stride:one ~dilation:[| 0 |]
          ~padding:none x );
    ( "a negative padding",
      fun () ->
        Nx.extract_patches ~kernel_size:[| 2 |] ~stride:one ~dilation:one
          ~padding:[| (-1, 0) |]
          x );
    ( "more kernel axes than the tensor's",
      fun () ->
        Nx.extract_patches ~kernel_size:[| 1; 1; 1 |] ~stride:[| 1; 1; 1 |]
          ~dilation:[| 1; 1; 1 |]
          ~padding:[| (0, 0); (0, 0); (0, 0) |]
          x );
    ( "an output_size per kernel axis",
      fun () ->
        Nx.combine_patches ~output_size:[| 3; 3 |] ~kernel_size:[| 2 |]
          ~stride:one ~dilation:one ~padding:none x );
    ( "a negative output_size",
      fun () ->
        Nx.combine_patches ~output_size:[| -1 |] ~kernel_size:[| 2 |]
          ~stride:one ~dilation:one ~padding:none x );
    ( "patches of more windows than the geometry gives",
      fun () ->
        Nx.combine_patches ~output_size:[| 1 |] ~kernel_size:[| 2 |] ~stride:one
          ~dilation:one ~padding:none
          (Nx.zeros Nx.float32 [| 2; 1 |]) );
    ( "patches of fewer windows than the geometry gives",
      fun () ->
        Nx.combine_patches ~output_size:[| 3 |] ~kernel_size:[| 2 |] ~stride:one
          ~dilation:one ~padding:none
          (Nx.zeros Nx.float32 [| 2; 1 |]) );
    ( "patches of the wrong kernel size",
      fun () ->
        Nx.combine_patches ~output_size:[| 3 |] ~kernel_size:[| 2 |] ~stride:one
          ~dilation:one ~padding:none
          (Nx.zeros Nx.float32 [| 3; 2 |]) );
    ( "a vector of patches",
      fun () ->
        Nx.combine_patches ~output_size:[| 1 |] ~kernel_size:[| 1 |] ~stride:one
          ~dilation:one ~padding:none
          (Nx.zeros Nx.float32 [| 1 |]) );
  ]

let empty_windows =
  let one = [| 1 |] and none = [| (0, 0) |] in
  group "windows that do not fit"
    [
      test "combine_patches of no window is zeros" (fun () ->
          equal (tensor float_exact)
            (Nx.zeros Nx.float32 [| 1 |])
            (Nx.combine_patches ~output_size:[| 1 |] ~kernel_size:[| 2 |]
               ~stride:one ~dilation:one ~padding:none
               (Nx.zeros Nx.float32 [| 2; 0 |])));
      test
        "combine_patches of no window keeps the leading axes, over two spatial \
         axes" (fun () ->
          equal (tensor float_exact)
            (Nx.zeros Nx.float64 [| 3; 2; 4 |])
            (Nx.combine_patches ~output_size:[| 2; 4 |] ~kernel_size:[| 3; 2 |]
               ~stride:[| 1; 1 |] ~dilation:[| 1; 2 |]
               ~padding:[| (0, 0); (0, 0) |]
               (Nx.zeros Nx.float64 [| 3; 6; 0 |])));
      test "extract_patches of an axis shorter than its window has no window"
        (fun () ->
          equal (array int) [| 4; 2; 0 |]
            (Nx.shape
               (Nx.extract_patches ~kernel_size:[| 2 |] ~stride:one
                  ~dilation:one ~padding:none
                  (Nx.ones Nx.float32 [| 4; 1 |]))));
      test
        "a window that overhangs the padded axis by less than a stride is no \
         window" (fun () ->
          equal (array int) [| 3; 0 |]
            (Nx.shape
               (Nx.extract_patches ~kernel_size:[| 3 |] ~stride:[| 2 |]
                  ~dilation:one
                  ~padding:[| (1, 0) |]
                  (Nx.ones Nx.float32 [| 1 |]))));
      test "extract_patches of an empty axis has no window" (fun () ->
          equal (array int) [| 2; 0 |]
            (Nx.shape
               (Nx.extract_patches ~kernel_size:[| 2 |] ~stride:one
                  ~dilation:one ~padding:none
                  (Nx.zeros Nx.float32 [| 0 |]))));
      test "padding can make a window of an empty axis" (fun () ->
          equal (tensor float_exact)
            (Nx.zeros Nx.float32 [| 2; 1 |])
            (Nx.extract_patches ~kernel_size:[| 2 |] ~stride:one ~dilation:one
               ~padding:[| (1, 1) |]
               (Nx.zeros Nx.float32 [| 0 |])));
      test "each invalid geometry or patch shape raises Invalid_argument"
        (fun () ->
          List.iter
            (fun (msg, f) ->
              match f () with
              | _ -> failf "%s: no refusal" msg
              | exception Invalid_argument _ -> ())
            window_refusals);
      test "the kernel refuses patches of the wrong shape below the frontend"
        (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Nx.Op.eval
                (Fold
                   {
                     output_size = [| 3 |];
                     kernel_size = [| 2 |];
                     stride = one;
                     dilation = one;
                     padding = none;
                     x = Nx.zeros Nx.float32 [| 2; 1 |];
                   })));
      test
        "the kernels refuse a geometry whose sizes do not fit in 64 bits below \
         the frontend" (fun () ->
          let huge = 1 lsl 61 in
          let fold ?(padding = none) ~output_size kernel_size dilation =
            Nx.Op.eval
              (Fold
                 {
                   output_size;
                   kernel_size;
                   stride = Array.map (fun _ -> 1) kernel_size;
                   dilation;
                   padding;
                   x = Nx.zeros Nx.float32 [| 2; 0 |];
                 })
          in
          List.iter
            (fun (msg, f) ->
              raises_match ~msg
                (Exn.invalid_arg ~substring:"invalid window geometry")
                f)
            [
              ( "a dilated kernel",
                fun () -> fold ~output_size:[| 2 |] [| huge |] [| huge |] );
              ( "a kernel's elements",
                fun () ->
                  fold ~output_size:[| 2; 2 |] [| huge; huge |] [| 1; 1 |] );
              ( "a padded axis",
                fun () ->
                  fold ~output_size:[| 2 |]
                    ~padding:[| (max_int, max_int) |]
                    [| 1 |] [| 1 |] );
              ( "an unfold's dilated kernel",
                fun () ->
                  Nx.Op.eval
                    (Unfold
                       {
                         kernel_size = [| huge |];
                         stride = one;
                         dilation = [| huge |];
                         padding = none;
                         x = Nx.zeros Nx.float32 [| 4 |];
                       }) );
            ]);
    ]

let correlation =
  let signal = Gen.bind (Gen.int_range 1 8) (fun n -> floats [| n |]) in
  let kernel = Gen.bind (Gen.int_range 1 8) (fun n -> floats [| n |]) in
  (* The sum of [x (i + j - lo) k (j)], zero outside [x]. *)
  let correlate_ref x k ~lo ~len =
    let xs = Nx.to_array x and ks = Nx.to_array k in
    Ref.init [| len |] (fun i ->
        let s = ref 0. in
        Array.iteri
          (fun j kv ->
            let p = i.(0) + j - lo in
            if p >= 0 && p < Array.length xs then s := !s +. (xs.(p) *. kv))
          ks;
        !s)
  in
  (* The values from [start] of the full correlation, [len] of them. *)
  let window x k ~start ~len =
    correlate_ref x k ~lo:(Nx.numel k - 1 - start) ~len
  in
  (* Values numpy gives for correlate and convolve of these operands. *)
  let numpy x k =
    ( Nx.create Nx.float64 [| Array.length x |] x,
      Nx.create Nx.float64 [| Array.length k |] k )
  in
  let seven = [| 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
  let numpy_cases =
    [
      ( "an even kernel",
        numpy seven [| 1.; 10.; 100.; 1000. |],
        [| 2100.; 3210.; 4321.; 5432.; 6543.; 7654.; 765. |],
        [| 4321.; 5432.; 6543.; 7654. |],
        [| 12.; 123.; 1234.; 2345.; 3456.; 4567.; 5670. |],
        [| 1234.; 2345.; 3456.; 4567. |] );
      ( "an odd kernel",
        numpy seven [| 1.; 10.; 100. |],
        [| 210.; 321.; 432.; 543.; 654.; 765.; 76. |],
        [| 321.; 432.; 543.; 654.; 765. |],
        [| 12.; 123.; 234.; 345.; 456.; 567.; 670. |],
        [| 123.; 234.; 345.; 456.; 567. |] );
      ( "an even input shorter than the kernel",
        numpy [| 1.; 2.; 3. |] [| 1.; 10.; 100.; 1000. |],
        [| 2100.; 3210.; 321.; 32. |],
        [| 3210.; 321. |],
        [| 12.; 123.; 1230.; 2300. |],
        [| 123.; 1230. |] );
      ( "an odd kernel longer than the input",
        numpy [| 1.; 2. |] [| 1.; 10.; 100. |],
        [| 210.; 21.; 2. |],
        [| 210.; 21. |],
        [| 1.; 12.; 120. |],
        [| 12.; 120. |] );
    ]
  in
  let exactly name expected got =
    equal ~msg:name (array float_exact) expected (Nx.to_array got)
  in
  group "correlation"
    [
      cases
        ~name:(fun (n, _, _, _, _, _) -> n)
        "correlate and convolve keep numpy's values" numpy_cases
        (fun (_, (x, k), c_same, c_valid, v_same, v_valid) ->
          exactly "correlate `Same" c_same (Nx.correlate ~padding:`Same x k);
          exactly "correlate `Valid" c_valid (Nx.correlate ~padding:`Valid x k);
          exactly "convolve `Same" v_same (Nx.convolve ~padding:`Same x k);
          exactly "convolve `Valid" v_valid (Nx.convolve ~padding:`Valid x k));
      prop
        "correlate keeps the full correlation, its valid part and its centred \
         part"
        (Gen.pair signal kernel) (fun (x, k) ->
          let n = Nx.numel x and m = Nx.numel k in
          let short = Int.min n m and long = Int.max n m in
          equal ~msg:"full" near
            (window x k ~start:0 ~len:(n + m - 1))
            (Ref.of_nx (Nx.correlate ~padding:`Full x k));
          equal ~msg:"valid" near
            (window x k ~start:(short - 1) ~len:(long - short + 1))
            (Ref.of_nx (Nx.correlate ~padding:`Valid x k));
          let start = if m <= n then (short - 1) / 2 else short / 2 in
          equal ~msg:"same" near
            (window x k ~start ~len:long)
            (Ref.of_nx (Nx.correlate ~padding:`Same x k)));
      prop
        "correlate `Same correlates the kernel with x from i - k/2 to i + \
         (k-1)/2"
        (Gen.pair signal kernel) (fun (x, k) ->
          assume (Nx.numel k <= Nx.numel x);
          equal near
            (correlate_ref x k ~lo:(Nx.numel k / 2) ~len:(Nx.numel x))
            (Ref.of_nx (Nx.correlate ~padding:`Same x k)));
      prop "a float16 correlate and convolve are float32's, rounded once"
        (Gen.pair signal kernel) (fun (x, k) ->
          (* Elements up to 300, whose products pass float16's largest. *)
          let h t = Nx.cast Nx.float16 (Nx.mul_s t 100.) in
          let x = h x and k = h k in
          let f32 t = Nx.cast Nx.float32 t and once t = Nx.cast Nx.float16 t in
          List.iter
            (fun padding ->
              equal ~msg:"correlate" (tensor float_exact)
                (once (Nx.correlate ~padding (f32 x) (f32 k)))
                (Nx.correlate ~padding x k);
              equal ~msg:"convolve" (tensor float_exact)
                (once (Nx.convolve ~padding (f32 x) (f32 k)))
                (Nx.convolve ~padding x k))
            [ `Full; `Same; `Valid ]);
      test "a float16 correlate whose products overflow float16 is finite"
        (fun () ->
          let v xs = Nx.create Nx.float16 [| 2 |] xs in
          let x = v [| 300.; 300. |] and k = v [| 300.; -300. |] in
          exactly "correlate" [| 0. |] (Nx.correlate x k);
          exactly "convolve" [| 0. |] (Nx.convolve x k));
      prop "full convolve is correlate with the kernel flipped"
        (Gen.pair signal kernel) (fun (x, k) ->
          equal
            (tensor (close ~rel:1e-12 ~abs:1e-12 ()))
            (Nx.correlate ~padding:`Full x (Nx.flip k))
            (Nx.convolve ~padding:`Full x k));
      prop "convolution is commutative in every padding"
        (Gen.pair signal kernel) (fun (x, k) ->
          List.iter
            (fun padding ->
              Law.commutative
                (tensor (close ~rel:1e-12 ~abs:1e-12 ()))
                (Nx.convolve ~padding) (x, k))
            [ `Full; `Same; `Valid ]);
    ]

let filters =
  let geometry =
    let open Gen in
    let* n = int_range 1 8 in
    let* m = int_range 1 8 in
    let+ kh = int_range 1 3
    and+ kw = int_range 1 3
    and+ sh = int_range 1 3
    and+ sw = int_range 1 3
    and+ t = floats [| n; m |] in
    (t, [| kh; kw |], [| sh; sw |])
  in
  let windowed f t kernel stride =
    let r = Ref.of_nx t in
    let out =
      out_size ~size:(Nx.shape t) ~kernel ~stride ~dilation:[| 1; 1 |]
        ~padding:[| (0, 0); (0, 0) |]
    in
    Ref.init out (fun i ->
        f
          (Array.init (Ref.numel kernel) (fun k ->
               let o = Ref.unravel kernel k in
               Ref.get r
                 [|
                   (i.(0) * stride.(0)) + o.(0); (i.(1) * stride.(1)) + o.(1);
                 |])))
  in
  let fold f a = Array.fold_left f a.(0) a in
  let mean a = Array.fold_left ( +. ) 0. a /. Float.of_int (Array.length a) in
  group "filters"
    [
      prop
        "maximum, minimum and uniform filters fold each window, stepped by the \
         kernel by default"
        geometry (fun (t, kernel_size, stride) ->
          assume (Array.for_all2 ( <= ) kernel_size (Nx.shape t));
          equal ~msg:"maximum" near
            (windowed (fold Float.max) t kernel_size stride)
            (Ref.of_nx (Nx.maximum_filter ~kernel_size ~stride t));
          equal ~msg:"minimum" near
            (windowed (fold Float.min) t kernel_size stride)
            (Ref.of_nx (Nx.minimum_filter ~kernel_size ~stride t));
          equal ~msg:"uniform" near
            (windowed mean t kernel_size stride)
            (Ref.of_nx (Nx.uniform_filter ~kernel_size ~stride t));
          equal ~msg:"default stride" near
            (windowed (fold Float.max) t kernel_size kernel_size)
            (Ref.of_nx (Nx.maximum_filter ~kernel_size t)));
    ]

let () =
  exit (run "nx windows" [ patches; empty_windows; correlation; filters ])
