(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Classes and windows: one_hot, patches and correlation, each
   against a loop over the host elements that follows its doc, index by
   index. Sums are of small integers held in float64, exact in any order, so
   the loops compare bit for bit. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let same_float a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let same (type v s) (dt : (v, s) D.t) (a : v) (b : v) =
  match D.kind dt with
  | D.Float -> same_float a b
  | D.Complex -> same_float a.Complex.re b.Complex.re && same_float a.im b.im
  | D.Signed | D.Unsigned | D.Boolean -> a = b

let elements dt =
  Testable.make ~pp:(Format.pp_print_list (D.pp_value dt)) ~equal:(fun a b ->
      List.length a = List.length b && List.for_all2 (same dt) a b)

let equal_elements dt a b =
  equal (elements dt) (Array.to_list a) (Array.to_list b)

(* Indices *)

let numel s = Array.fold_left ( * ) 1 s

(* The index of element [j] in C order of shape [s]. *)
let index s j =
  let r = Array.length s in
  let i = Array.make r 0 and j = ref j in
  for a = r - 1 downto 0 do
    i.(a) <- !j mod s.(a);
    j := !j / s.(a)
  done;
  i

let offset s i =
  let o = ref 0 in
  Array.iteri (fun a d -> o := (!o * d) + i.(a)) s;
  !o

(* Values *)

(* A host value of [dt] and shape [s] over drawn bytes: NaNs, signed zeros,
   infinities and extremes among its elements. *)
let drawn (type v s) (dt : (v, s) D.t) s : (v, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt (numel s))) char in
  let data =
    match dt with
    | D.Bool -> String.map (fun c -> Char.chr (Char.code c land 1)) data
    | _ -> data
  in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous s)
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

(* Small integers in float64: their sums are exact in any order. *)
let small s =
  Gen.map
    (fun vs -> Nx.create Nx.float64 s (Array.map Float.of_int vs))
    (Gen.array ~size:(Gen.constant (numel s)) (Gen.int_range (-4) 4))

let of_dtypes dts = Gen.of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) dts

(* One hot *)

let integers =
  List.filter (fun (D.Any dt) -> D.is D.Signed dt || D.is D.Unsigned dt) D.all

type ids = Ids : ('v, 's) D.t * ('v, 's, Nx.host) Nx.t * int -> ids

let ids =
  Gen.with_pp
    (fun ppf (Ids (dt, x, n)) ->
      Format.fprintf ppf "%a %a, %d classes" D.pp dt pp_ints (Nx.shape x) n)
    (let open Gen in
     let* (D.Any dt) = of_dtypes integers in
     let* s = array ~size:(int_range 0 2) (int_range 0 3) in
     let* n = int_range 0 5 in
     let* small = bool in
     let+ x =
       if small then
         map
           (fun vs -> Nx.cast dt (Nx.create Nx.int64 s (Array.map Int64.of_int vs)))
           (array ~size:(constant (numel s)) (int_range (-1) 5))
       else drawn dt s
     in
     Ids (dt, x, n))

(* Element [i] of the result's row [c] is one iff the element, as an int64, is
   [c]. *)
let law_one_hot (Ids (_, x, n)) =
  let vs = Nx.to_array (Nx.cast Nx.int64 x) in
  let expected =
    Array.init (Array.length vs * n) (fun j ->
        if Int64.equal vs.(j / n) (Int64.of_int (j mod n)) then 1 else 0)
  in
  cover "a class named" (Array.exists (fun v -> v = 1) expected);
  cover "an element out of range"
    (Array.exists (fun v -> Int64.compare v 0L < 0 || Int64.compare v (Int64.of_int n) >= 0) vs);
  let y = Nx.one_hot ~num_classes:n x in
  equal (array int) (Array.append (Nx.shape x) [| n |]) (Nx.shape y);
  equal (array int) expected (Nx.to_array y)

let one_hots =
  group "one_hot"
    [
      prop "one where the element names the class" ids law_one_hot;
      test "a float or a negative count raises" (fun () ->
          invalid ~by:"Nx.one_hot" (fun () ->
              Nx.one_hot ~num_classes:2 (Nx.zeros Nx.float32 [| 2 |]));
          invalid ~by:"Nx.one_hot" (fun () ->
              Nx.one_hot ~num_classes:(-1) (Nx.zeros Nx.int32 [| 2 |])));
    ]

(* Window geometry *)

type geometry = {
  lead : int array;
  spatial : int array;
  kernel : int array;
  stride : int array;
  dilation : int array;
  padding : (int * int) array;
}

let pp_geometry ppf g =
  Format.fprintf ppf
    "lead %a, spatial %a, kernel %a, stride %a, dilation %a, padding [%s]"
    pp_ints g.lead pp_ints g.spatial pp_ints g.kernel pp_ints g.stride pp_ints
    g.dilation
    (String.concat "; "
       (Array.to_list
          (Array.map (fun (a, b) -> Printf.sprintf "(%d, %d)" a b) g.padding)))

let geometry =
  Gen.with_pp pp_geometry
    (let open Gen in
     let* lead = array ~size:(int_range 0 1) (int_range 0 2) in
     let* k = int_range 1 2 in
     let axis g = array ~size:(constant k) g in
     let* spatial = axis (int_range 0 5) in
     let* kernel = axis (int_range 1 3) in
     let* stride = axis (int_range 1 3) in
     let* dilation = axis (int_range 1 2) in
     let+ padding = axis (pair (int_range 0 2) (int_range 0 2)) in
     { lead; spatial; kernel; stride; dilation; padding })

let counts g =
  Array.init (Array.length g.kernel) (fun i ->
      let lo, hi = g.padding.(i) in
      let e = g.spatial.(i) + lo + hi in
      let span = (g.dilation.(i) * (g.kernel.(i) - 1)) + 1 in
      if e < span then 0 else ((e - span) / g.stride.(i)) + 1)

(* Each element of the patches, [(lead, p, l)], with the flat position in
   [x] it reads, [None] in the padding. *)
let taps g =
  let cs = counts g in
  let p = numel g.kernel and l = numel cs in
  let s = Array.append g.lead g.spatial in
  List.init (numel g.lead * p * l) (fun e ->
      let lead = index g.lead (e / (p * l)) in
      let j = index g.kernel (e / l mod p) and w = index cs (e mod l) in
      let pos =
        Array.mapi
          (fun i n ->
            let at = (w.(i) * g.stride.(i)) + (j.(i) * g.dilation.(i)) - fst g.padding.(i) in
            if at >= 0 && at < n then Some at else None)
          g.spatial
      in
      if Array.exists Option.is_none pos then (e, None)
      else (e, Some (offset s (Array.append lead (Array.map Option.get pos)))))

let extract g x =
  Nx.extract_patches ~kernel_size:g.kernel ~stride:g.stride ~dilation:g.dilation
    ~padding:g.padding x

let combine g p =
  Nx.combine_patches ~output_size:g.spatial ~kernel_size:g.kernel
    ~stride:g.stride ~dilation:g.dilation ~padding:g.padding p

let patches_shape g =
  Array.append g.lead [| numel g.kernel; numel (counts g) |]

type drawn_patches = Patches : geometry * ('v, 's) D.t * ('v, 's, Nx.host) Nx.t -> drawn_patches

let with_values =
  Gen.with_pp
    (fun ppf (Patches (g, dt, _)) -> Format.fprintf ppf "%a, %a" D.pp dt pp_geometry g)
    (let open Gen in
     let* g = geometry in
     let* (D.Any dt) = of_dtypes D.all in
     let+ x = drawn dt (Array.append g.lead g.spatial) in
     Patches (g, dt, x))

let law_extract (Patches (g, dt, x)) =
  let cs = counts g in
  cover "a window off the axis" (Array.exists (( = ) 0) cs);
  cover "padding read" (List.exists (fun (_, o) -> o = None) (taps g));
  cover "a dilated stride" (Array.exists (( < ) 1) g.dilation && Array.exists (( < ) 1) g.stride);
  let xs = Nx.to_array x in
  let expected =
    Array.of_list
      (List.map (fun (_, o) -> match o with None -> D.zero dt | Some o -> xs.(o)) (taps g))
  in
  let y = extract g x in
  equal (array int) (patches_shape g) (Nx.shape y);
  equal_elements dt expected (Nx.to_array y)

let law_combine g =
  let open Gen in
  let+ p = small (patches_shape g) in
  (g, p)

let combined =
  Gen.with_pp
    (fun ppf (g, _) -> pp_geometry ppf g)
    (Gen.bind geometry law_combine)

(* Each element of [p] added where {!extract_patches} reads it from. *)
let test_combine (g, p) =
  let out = Array.make (numel g.lead * numel g.spatial) 0. in
  let ps = Nx.to_array p in
  List.iter (fun (e, o) -> Option.iter (fun o -> out.(o) <- out.(o) +. ps.(e)) o) (taps g);
  let y = combine g p in
  equal (array int) (Array.append g.lead g.spatial) (Nx.shape y);
  equal (array float_exact) (Array.map (fun v -> v +. 0.) out) (Nx.to_array y)

let dot a b = Array.fold_left ( +. ) 0. (Array.map2 ( *. ) a b)

let adjoint =
  Gen.with_pp
    (fun ppf (g, _, _) -> pp_geometry ppf g)
    (let open Gen in
     let* g = geometry in
     let* x = small (Array.append g.lead g.spatial) in
     let+ y = small (patches_shape g) in
     (g, x, y))

let law_adjoint (g, x, y) =
  equal float_exact
    (dot (Nx.to_array (extract g x)) (Nx.to_array y))
    (dot (Nx.to_array x) (Nx.to_array (combine g y)))

(* Three taps reach element 2: 2048 then 1 and 1 at float16 is 2048, but
   2050 at float32, which float16 holds. *)
let test_combine_narrow () =
  let p = Array.make 9 0. in
  p.(0 * 3 + 2) <- 2048.;
  p.(1 * 3 + 1) <- 1.;
  p.(2 * 3 + 0) <- 1.;
  let y =
    Nx.combine_patches ~output_size:[| 5 |] ~kernel_size:[| 3 |] ~stride:[| 1 |]
      ~dilation:[| 1 |] ~padding:[| (0, 0) |]
      (Nx.create Nx.float16 [| 3; 3 |] p)
  in
  equal float_exact 2050. (Nx.item [ 2 ] y)

let test_patch_errors () =
  let x = Nx.zeros Nx.float32 [| 4; 4 |] in
  let ex ?(kernel = [| 2 |]) ?(stride = [| 1 |]) ?(dilation = [| 1 |])
      ?(padding = [| (0, 0) |]) x =
    Nx.extract_patches ~kernel_size:kernel ~stride ~dilation ~padding x
  in
  let by = "Nx.extract_patches" in
  invalid ~by (fun () -> ex ~kernel:[||] ~stride:[||] ~dilation:[||] ~padding:[||] x);
  invalid ~by (fun () -> ex ~stride:[| 1; 1 |] x);
  invalid ~by (fun () -> ex ~kernel:[| 0 |] x);
  invalid ~by (fun () -> ex ~stride:[| 0 |] x);
  invalid ~by (fun () -> ex ~dilation:[| 0 |] x);
  invalid ~by (fun () -> ex ~padding:[| (-1, 0) |] x);
  invalid ~by (fun () -> ex ~kernel:[| 1; 1; 1 |] ~stride:[| 1; 1; 1 |]
                  ~dilation:[| 1; 1; 1 |] ~padding:[| (0, 0); (0, 0); (0, 0) |] x);
  let by = "Nx.combine_patches" in
  let co ?(out = [| 4 |]) p =
    Nx.combine_patches ~output_size:out ~kernel_size:[| 2 |] ~stride:[| 1 |]
      ~dilation:[| 1 |] ~padding:[| (0, 0) |] p
  in
  invalid ~by (fun () -> co ~out:[| 4; 4 |] (Nx.zeros Nx.float32 [| 2; 3 |]));
  invalid ~by (fun () -> co ~out:[| -1 |] (Nx.zeros Nx.float32 [| 2; 3 |]));
  invalid ~by (fun () -> co (Nx.zeros Nx.float32 [| 3 |]));
  invalid ~by (fun () -> co (Nx.zeros Nx.float32 [| 2; 4 |]));
  invalid ~by (fun () -> co (Nx.zeros Nx.bool [| 2; 3 |]))

let patches =
  group "patches"
    [
      prop "extract_patches reads each window's taps, zeros in the padding"
        with_values law_extract;
      prop "combine_patches adds each element where extract_patches reads it"
        combined test_combine;
      prop "combine_patches is extract_patches' adjoint" adjoint law_adjoint;
      test "narrow floats combine at float32" test_combine_narrow;
      test "a misused geometry raises" test_patch_errors;
    ]

(* Correlation *)

type correlation = {
  x_lead : int array;
  xs : int array;
  ks : int array;
  mode : [ `Full | `Same | `Valid ];
}

let pp_mode ppf = function
  | `Full -> Format.pp_print_string ppf "`Full"
  | `Same -> Format.pp_print_string ppf "`Same"
  | `Valid -> Format.pp_print_string ppf "`Valid"

let correlation =
  Gen.with_pp
    (fun ppf (c, _, _) ->
      Format.fprintf ppf "x %a %a, kernel %a, %a" pp_ints c.x_lead pp_ints c.xs
        pp_ints c.ks pp_mode c.mode)
    (let open Gen in
     let* x_lead = array ~size:(int_range 0 1) (int_range 0 2) in
     let* k = int_range 0 2 in
     let* xs = array ~size:(constant k) (int_range 0 5) in
     let* ks = array ~size:(constant k) (int_range 1 4) in
     let* mode = of_list ~pp:pp_mode [ `Full; `Same; `Valid ] in
     let c = { x_lead; xs; ks; mode } in
     let* x = small (Array.append x_lead xs) in
     let+ kernel = small ks in
     (c, x, kernel))

(* The run of the full correlation each mode keeps, as the doc states it. *)
let kept_run mode n k =
  let m = min n k and longest = max n k in
  match mode with
  | `Full -> (0, n + k - 1)
  | `Valid -> (m - 1, longest - m + 1)
  | `Same -> ((if k <= n then (m - 1) / 2 else m / 2), longest)

let law_correlate (c, x, kernel) =
  let runs = Array.mapi (fun i n -> kept_run c.mode n c.ks.(i)) c.xs in
  let out = Array.map snd runs in
  cover "a kernel longer than x" (Array.exists2 (fun n k -> k > n) c.xs c.ks);
  cover "an empty axis of x" (Array.exists (( = ) 0) c.xs);
  cover "batch axes" (c.x_lead <> [||]);
  cover "an even kernel, Same" (c.mode = `Same && Array.exists (fun k -> k mod 2 = 0) c.ks);
  let s = Array.append c.x_lead c.xs and xv = Nx.to_array x in
  let kv = Nx.to_array kernel in
  let shape = Array.append c.x_lead out in
  let expected =
    Array.init (numel shape) (fun e ->
        let i = index shape e in
        let nl = Array.length c.x_lead in
        let lead = Array.sub i 0 nl in
        let total = ref 0. in
        for t = 0 to numel c.ks - 1 do
          let j = index c.ks t in
          let pos =
            Array.mapi
              (fun a n ->
                let at = fst runs.(a) + i.(nl + a) + j.(a) - (c.ks.(a) - 1) in
                if at >= 0 && at < n then Some at else None)
              c.xs
          in
          if Array.for_all Option.is_some pos then
            total :=
              !total
              +. xv.(offset s (Array.append lead (Array.map Option.get pos)))
                 *. kv.(t)
        done;
        !total +. 0.)
  in
  let y = Nx.correlate ~padding:c.mode x kernel in
  equal (array int) shape (Nx.shape y);
  equal (array float_exact) expected (Nx.to_array y)

let f64 = Nx.create Nx.float64

let correlations =
  group "correlate"
    [
      prop "each kept value sums the kernel's products with x, zeros outside"
        correlation law_correlate;
      test "the doc's Same" (fun () ->
          let x = f64 [| 7 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
          let k = f64 [| 4 |] [| 1.; 10.; 100.; 1000. |] in
          equal (array float_exact)
            [| 2100.; 3210.; 4321.; 5432.; 6543.; 7654.; 765. |]
            (Nx.to_array (Nx.correlate ~padding:`Same x k)));
      test "an empty x lies within a kernel of k at k + 1 positions" (fun () ->
          let k = f64 [| 3 |] [| 1.; 2.; 3. |] in
          equal (array float_exact) [| 0.; 0.; 0.; 0. |]
            (Nx.to_array (Nx.correlate ~padding:`Valid (f64 [| 0 |] [||]) k)));
      test "narrow floats multiply and sum at float32" (fun () ->
          let x = Nx.create Nx.float16 [| 3 |] [| 2048.; 1.; 1. |] in
          let k = Nx.create Nx.float16 [| 3 |] [| 1.; 1.; 1. |] in
          equal (array float_exact) [| 2050. |] (Nx.to_array (Nx.correlate x k)));
      test "integers wrap" (fun () ->
          let x = Nx.create Nx.int8 [| 2 |] [| 100; 100 |] in
          equal (array int) [| -56 |]
            (Nx.to_array (Nx.correlate x (Nx.create Nx.int8 [| 2 |] [| 1; 1 |]))));
      test "misuse raises" (fun () ->
          let by = "Nx.correlate" in
          invalid ~by (fun () ->
              Nx.correlate (Nx.zeros Nx.float32 [| 3 |]) (Nx.zeros Nx.float32 [| 2; 2 |]));
          invalid ~by (fun () ->
              Nx.correlate (Nx.zeros Nx.float32 [| 3 |]) (Nx.zeros Nx.float32 [| 0 |]));
          invalid ~by (fun () ->
              Nx.correlate (Nx.zeros Nx.bool [| 3 |]) (Nx.zeros Nx.bool [| 2 |])));
    ]

let () =
  exit (run "nx windows" [ one_hots; patches; correlations ])
