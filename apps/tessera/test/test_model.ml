(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

(* Small models of each form. *)
let arch pre_ln =
  { Model.pre_ln; width = 16; heads = 2; layers = 2; ffn = 24; dims = 32 }

let post_ln = Model.init ~key:(Nx.Rng.key 1) (arch false)
let pre_ln = Model.init ~key:(Nx.Rng.key 2) (arch true)
let model pre = if pre then pre_ln else post_ln

(* The embedding of each pixel of [px] with [counts], batches of [shape]: pixels
   without tokens are absent. *)
let embed m shape px counts =
  let n = Pixels.length px in
  let rows = Array.make n [||] in
  let rec go from =
    let b, next = Batch.pack shape px counts ~from in
    let e = Nx.to_array (Model.apply m b) in
    let dims = Model.dims m in
    Array.iteri
      (fun i p ->
        let p = Int64.to_int p in
        if p >= 0 then rows.(p) <- Array.sub e (i * dims) dims)
      (Nx.to_array b.pixel);
    if next < n then go next
  in
  go 0;
  rows

let shape ?(pixels = 4) ?(rows = 3) ?(slots = 3) ?(width = 256) () =
  { Batch.pixels; s2_rows = rows; s1_rows = rows; slots; width }

(* Float32 throughout, unit-variance outputs: the reference meets 1e-5 against
   itself over devices and batch sizes. *)
let tol = 1e-5
let same a b = equal (array (array (float tol))) a b
let cases_gen = Gen.(pair bool (Synth.spec ~long:true ()))

(* Each observation repeated as many times as it counts, each copy counting
   once: the computation the reference makes. *)
let copies px counts =
  let px', _ =
    Synth.rebuild px counts (fun _ cs ->
        Array.concat (Array.to_list (Array.mapi (fun i m -> Array.make m i) cs)))
  in
  let ones r = Nx.ones Nx.int32 [| Nx.dim 0 (Nx_ragged.values r) |] in
  (px', { Batch.s2 = ones px'.Pixels.s2; s1 = ones px'.Pixels.s1 })

let identity =
  prop "a token of count m is m copies of it" cases_gen (fun (pre, spec) ->
      let m = model pre in
      let px = Synth.pixels spec in
      let counts = Batch.v2_counts px in
      let px', ones = copies px counts in
      cover "the teacher's form" pre;
      cover "an observation of count > 1"
        (List.exists (fun n -> n > 0 && n < 256) spec.s2);
      same (embed m (shape ()) px counts) (embed m (shape ()) px' ones))

let permutation =
  prop "an embedding does not depend on its observations' order" cases_gen
    (fun (pre, spec) ->
      let m = model pre in
      let px = Synth.pixels spec in
      let counts = Batch.v2_counts px in
      let rng = Random.State.make [| spec.seed |] in
      let shuffle _ cs =
        let a = Array.init (Array.length cs) Fun.id in
        for i = Array.length a - 1 downto 1 do
          let j = Random.State.int rng (i + 1) in
          let t = a.(i) in
          a.(i) <- a.(j);
          a.(j) <- t
        done;
        a
      in
      let px', counts' = Synth.rebuild px counts shuffle in
      cover "a row of several observations"
        (List.exists (fun n -> n > 1) spec.s2);
      same (embed m (shape ()) px counts) (embed m (shape ()) px' counts'))

let layout =
  prop "an embedding does not depend on the batch's shape" cases_gen
    (fun (pre, spec) ->
      let m = model pre in
      let px = Synth.pixels spec in
      let counts = Batch.v2_counts px in
      let alone = shape ~pixels:1 ~rows:1 ~slots:1 () in
      let packed = shape ~pixels:7 ~rows:2 ~slots:5 ~width:300 () in
      cover "pixels sharing a row" (List.length spec.s2 > 1);
      same (embed m alone px counts) (embed m packed px counts))

let normalised =
  prop "an embedding has mean 0 and deviation 1" cases_gen (fun (pre, spec) ->
      let m = model pre in
      let px = Synth.pixels spec in
      Array.iter
        (fun e ->
          if e <> [||] then begin
            let n = Float.of_int (Array.length e) in
            let mean = Array.fold_left ( +. ) 0. e /. n in
            let var =
              Array.fold_left (fun a x -> a +. ((x -. mean) ** 2.)) 0. e /. n
            in
            equal ~msg:"mean" (float tol) 0. mean;
            (* The layer norm's eps, 1e-5, lowers the variance by that much. *)
            equal ~msg:"variance" (float 1e-3) 1. var
          end)
        (embed m (shape ()) px (Batch.v2_counts px)))

let visits pre =
  let m = model pre in
  Nx.Ptree.visits (Nx.Ptree.instantiate (module Model)) m
  |> List.map (Format.asprintf "%a" Nx.Ptree.pp_visit)

let structure =
  group "walk"
    [
      test "reports the form and the heads" (fun () ->
          let v = visits false in
          mem string "net: case \"post_ln\"" v;
          mem string "heads: int 2" v;
          mem string "net.s2.layers: length 2" v;
          mem string "net.s2.layers.0.qk: None" v;
          mem string "norm.s1_mean: a leaf" v);
      test "reports the teacher's form" (fun () ->
          let v = visits true in
          mem string "net: case \"pre_ln\"" v;
          mem string "net.fusion: length 2" v;
          mem string "net.s2.layers.0.qk: Some" v);
      test "keeps the prefixes" (fun () ->
          equal (list int) [ 16; 32 ] post_ln.prefixes;
          equal (list int) [] pre_ln.prefixes);
    ]

let empty_places =
  prop "an empty place holds a finite vector" cases_gen (fun (pre, spec) ->
      let m = model pre in
      let px = Synth.pixels spec in
      let b, _ =
        Batch.pack
          (shape ~pixels:12 ~rows:12 ())
          px (Batch.v2_counts px) ~from:0
      in
      cover "the teacher's form" pre;
      equal (array bool)
        (Array.make (12 * Model.dims m) true)
        (Array.map Float.is_finite (Nx.to_array (Model.apply m b))))

(* The fold *)

(* [px] as TESSERA's preprocessing writes it: each negative band raised. *)
let folded (px : Nx.host Pixels.t) =
  let v = Nx_ragged.values px.s2 in
  let bands = Nx.slice [ Nx.A; Nx.R (0, 10) ] v in
  let bands = Nx.where (Nx.less_s bands 0) (Nx.add_s bands 1000) bands in
  let days = Nx.slice [ Nx.A; Nx.R (10, 11) ] v in
  let v = Nx.concatenate ~axis:1 [ bands; days ] in
  { px with s2 = Nx_ragged.of_lengths (Nx_ragged.lengths px.s2) v }

let fold_law =
  prop "a model reads a tile as it reads TESSERA's file of it" cases_gen
    (fun (pre, spec) ->
      let m = model pre in
      let px = Synth.pixels spec in
      let counts = Batch.v2_counts px in
      let px' = folded px in
      let v = Nx_ragged.values px.s2 in
      cover "a negative band" (Nx.numel v > 0 && Nx.item [] (Nx.min v) < 0);
      equal
        (array (array float_exact))
        (embed m (shape ()) px counts)
        (embed m (shape ()) px' counts))

(* Loading *)

(* The parameters' values against TESSERA's are test_parity's; these are what
   its embeddings do not show. *)

module Pt = Nx_io.Pytorch

let student_pt = "upstream/tiny/student.pt"

let arrays m =
  Nx.Ptree.fold
    (Nx.Ptree.instantiate (module Model))
    (fun p x acc ->
      (Nx.Ptree.Path.to_string p, Nx.to_array (Nx.cast Nx.float64 x)) :: acc)
    m []

let loading =
  group "load"
    [
      test "lays a student's weights out in C order, as init does" (fun () ->
          let m = Model.load student_pt in
          let net = Nx.Ptree.instantiate (module Model.Net) in
          let weights =
            Nx.Ptree.fold net
              (fun p x acc ->
                let p = Nx.Ptree.Path.to_string p in
                if not (String.ends_with ~suffix:".w" p) then acc
                else
                  let l =
                    Nx_array.layout (Option.get (Nx.Repr.array ~by:"test" x))
                  in
                  let c =
                    Nx_array.Layout.contiguous (Nx_array.Layout.shape l)
                  in
                  let strides l = Iarray.to_array (Nx_array.Layout.strides l) in
                  equal ~msg:p (array int) (strides c) (strides l);
                  equal ~msg:p int 0 (Nx_array.Layout.offset l);
                  acc + 1)
              m.net 0
          in
          at_least int ~than:10 weights);
      test "loads a saved model back, of either form" (fun () ->
          List.iter
            (fun m ->
              let file = Filename.temp_file "tessera" ".safetensors" in
              Fun.protect ~finally:(fun () -> Sys.remove file) @@ fun () ->
              Model.save file m;
              let back = Model.load file in
              equal (list int) m.prefixes back.prefixes;
              equal int m.heads back.heads;
              equal
                (list (pair string (array float_exact)))
                (arrays m) (arrays back))
            [ post_ln; pre_ln ]);
      test "refuses an unused tensor" (fun () ->
          let x = Pt.Tensor (Nx.P (Nx.zeros Nx.float32 [| 2 |])) in
          let v =
            match Nx_io.load_pytorch student_pt with
            | Pt.Dict [ (k, Pt.Dict state); args ] ->
                let state = state @ [ (Pt.String "extra.weight", x) ] in
                Pt.Dict [ (k, Pt.Dict state); args ]
            | _ -> assert false
          in
          raises_match
            (Exn.failure
               ~sub:
                 "student: Nx_io.Archive.import: extra.weight: an entry in \
                  the archive, never read (1 of 1)")
            (fun () -> Model.of_pytorch ~name:"student" v));
    ]

let () =
  exit
    (run "tessera.model"
       [
         group "apply"
           [ identity; permutation; layout; normalised; empty_places; fold_law ];
         structure;
         loading;
       ])
