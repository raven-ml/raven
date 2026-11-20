(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* generate_parity.exe MAIN: main.exe's generate against Roar.sample on real
   Multimodal Universe rows, with AION-1's weights, under the alias parity.

   MAIN is main.exe. It reads AION_PARITY's mmu, the rows mmu_parity.exe reads,
   and writes main.exe's files to a temporary directory. Both read aion-base's
   weights and codecs from AION_PARITY's huggingface, as RAVEN_CACHE_ROOT.

   For each case the program runs MAIN generate on DIR's files, then draws the
   same tokens from the library: Survey.read and Codecs.encode of the case's
   rows, in batches of the case's size as main.exe batches them, each batch's
   objects with a present token, and Roar.sample with the key
   [fold_in (fold_in (key seed) r) j] for draw j of row r, at the counts
   written out below. The tokens must be equal, and each entry main.exe decoded
   must equal Codecs.decode of them within 1e-6 of its largest magnitude. The
   cases are:

   - [z]: redshift from the tutorial file's Legacy Survey image and g, r, i, z
   fluxes (AION-1 §7.1.1), rows 0:6 in batches of 4, three draws;

   - [desi]: a DESI spectrum from Gaia's XP coefficients (§7.1.2), rows 0:2 in
   batches of 1, in 12 steps of upstream's linear counts;

   - [image]: a Legacy Survey image from a DESI spectrum, row 0, in 4 steps.

   It runs outside runtest: the weights are 1.27 GB with the codecs, and MMU's
   rows are not committed. *)

open Aion

(* The inputs, under AION_PARITY as the README's Tests section lays them out. *)
let parity =
  match Sys.getenv_opt "AION_PARITY" with
  | Some d when d <> "" -> d
  | _ ->
      prerr_endline
        "generate_parity: AION_PARITY is unset: set it to the directory of the \
         README's Tests section";
      exit 1

let () = Unix.putenv "RAVEN_CACHE_ROOT" parity
let main = Sys.argv.(1)
let dir = Filename.concat parity "mmu"
let out = Filename.temp_dir "aion-generate" ""

type case = {
  name : string;
  survey : Survey.survey;
  source : string;  (** main.exe's argument: a file, or FILE:GROUP. *)
  inputs : Modality.t list;
  target : Modality.t;
  rows : int * int;
  batch : int;
  reveal : int list;
  draws : int;
  seed : int;
}

let phot =
  Modality.[ Image; Scalar Flux_g; Scalar Flux_r; Scalar Flux_i; Scalar Flux_z ]

let cases =
  [
    {
      name = "z";
      survey = Legacy_survey;
      source = "tutorial.h5:/legacysurvey";
      inputs = phot;
      target = Scalar Z;
      rows = (0, 6);
      batch = 4;
      reveal = [ 1 ];
      draws = 3;
      seed = 11;
    };
    {
      name = "desi";
      survey = Gaia;
      source = "gaia.h5";
      inputs = Modality.[ Scalar Xp_bp; Scalar Xp_rp ];
      target = Spectrum Desi;
      rows = (0, 2);
      batch = 1;
      reveal = [ 23; 23; 23; 23; 23; 23; 23; 23; 23; 22; 22; 22 ];
      draws = 1;
      seed = 5;
    };
    {
      name = "image";
      survey = Desi;
      source = "desi.h5";
      inputs = [ Spectrum Desi ];
      target = Image;
      rows = (0, 1);
      batch = 1;
      reveal = [ 144; 144; 144; 144 ];
      draws = 1;
      seed = 3;
    };
  ]

let source c =
  match String.index_opt c.source ':' with
  | Some k ->
      Survey.load
        ~group:(String.sub c.source (k + 1) (String.length c.source - k - 1))
        c.survey
        (Filename.concat dir (String.sub c.source 0 k))
  | None -> Survey.load c.survey (Filename.concat dir c.source)

let run c file =
  let a, b = c.rows in
  let args =
    [
      main;
      "generate";
      "--" ^ Survey.name c.survey;
      Filename.concat dir c.source;
      "--target";
      Modality.key c.target;
      "--inputs";
      String.concat "," (List.map Modality.key c.inputs);
      "--rows";
      Printf.sprintf "%d:%d" a b;
      "--batch";
      string_of_int c.batch;
      "--steps";
      string_of_int (List.length c.reveal);
      "--draws";
      string_of_int c.draws;
      "--seed";
      string_of_int c.seed;
      "--out";
      file;
    ]
  in
  let cmd = String.concat " " (List.map Filename.quote args) in
  if Sys.command cmd <> 0 then failwith (cmd ^ ": failed")

(* The draws of [c] by the library, [[|n; draws; k|]], and each entry main.exe
   decodes, stacked likewise. *)
let replay t codecs c =
  let src = source c in
  let a, b = c.rows in
  let starts =
    List.init ((b - a + c.batch - 1) / c.batch) (fun i -> a + (i * c.batch))
  in
  let batch lo =
    let rows = Array.init (min c.batch (b - lo)) (fun k -> lo + k) in
    let ins =
      List.map (fun m -> Codecs.encode codecs (Survey.read src rows m)) c.inputs
    in
    let present (i : _ Tokens.t) = Nx.to_array (Nx.any ~axes:[ 1 ] i.present) in
    let any = List.map present ins in
    let kept =
      List.filter
        (fun k -> List.exists (fun p -> p.(k)) any)
        (List.init (Array.length rows) Fun.id)
      |> Array.of_list
    in
    if kept = [||] then None
    else
      let idx =
        Nx.create Nx.int64 [| Array.length kept |] (Array.map Int64.of_int kept)
      in
      let take x = Nx.take ~axis:0 idx x in
      let ins =
        List.map
          (fun (i : _ Tokens.t) ->
            {
              i with
              tokens = take i.tokens;
              present = take i.present;
            })
          ins
      in
      let keys =
        Nx.Rng.fold_in_tensor (Nx.Rng.key c.seed)
          (Nx.create Nx.int64 [| Array.length kept |]
             (Array.map (fun k -> Int64.of_int rows.(k)) kept))
      in
      let draws =
        List.init c.draws (fun j ->
            Roar.sample ~keys:(Nx.Rng.fold_in keys j) ~reveal:c.reveal t ins
              c.target)
      in
      (* A scalar token past its codebook's edges, redshift's class 1024,
         decodes to NaN, as main.exe states. *)
      let decoded x : (string * Nx.host Nx.float32_t) list =
        match c.target with
        | Scalar s ->
            let m = Nx.dim 1 (Scalar.edges (Codecs.scalar codecs s)) in
            let none = Nx.greater_equal_s x (Int64.of_int m) in
            let v =
              match
                Codecs.decode codecs c.target
                  (Nx.where none (Nx.zeros_like x) x)
              with
              | Scalar { values; _ } -> values
              | _ -> assert false
            in
            [ ("values", Nx.where none (Nx.full_like v Float.nan) v) ]
        | _ -> (
            match Codecs.decode codecs c.target x with
            | Spectrum { spectrum; _ } -> [ ("flux", spectrum.flux) ]
            | Image { flux; _ } -> [ ("flux", flux) ]
            | Scalar _ -> assert false)
      in
      let x =
        if Modality.tokens c.target = 1 then Nx.reshape [| -1 |] else Fun.id
      in
      Some
        ( Nx.stack ~axis:1 draws,
          List.map
            (fun (name, _) ->
              ( name,
                Nx.stack ~axis:1
                  (List.map (fun d -> List.assoc name (decoded (x d))) draws) ))
            (decoded (x (List.hd draws))) )
  in
  let parts = List.filter_map batch starts in
  ( Nx.concatenate ~axis:0 (List.map fst parts),
    List.map
      (fun (name, _) ->
        ( name,
          Nx.concatenate ~axis:0
            (List.map (fun (_, e) -> List.assoc name e) parts) ))
      (snd (List.hd parts)) )

let largest x = Nx.item [] (Nx.max (Nx.abs (Nx.cast Nx.float64 x)))

let () =
  let t = Transformer.of_archive (Hub.weights Base) in
  let bad = ref 0 in
  List.iter
    (fun c ->
      let file = Filename.concat out (c.name ^ ".npz") in
      run c file;
      let a = Nx_io.load_npz file in
      let got =
        Nx.unpack Nx.int64 (Option.get (Nx_io.Archive.find "tokens" a))
      in
      let codecs = Codecs.load (c.target :: c.inputs) in
      let want, entries = replay t codecs c in
      let same =
        Nx.shape got = Nx.shape want && Nx.item [] (Nx.array_equal got want)
      in
      if not same then incr bad;
      Printf.printf "%-6s tokens [%s] %s\n%!" c.name
        (String.concat "; "
           (List.map string_of_int (Array.to_list (Nx.shape got))))
        (if same then "equal" else "differ");
      List.iter
        (fun (name, w) ->
          let g =
            Nx.unpack Nx.float32 (Option.get (Nx_io.Archive.find name a))
          in
          (* NaN, redshift's class 1024, is equal to NaN here. *)
          let nan x = Nx.isnan x in
          let d =
            Nx.where
              (Nx.logical_and (nan g) (nan w))
              (Nx.zeros Nx.float64 (Nx.shape g))
              (Nx.abs (Nx.sub (Nx.cast Nx.float64 g) (Nx.cast Nx.float64 w)))
          in
          let d = largest d and m = largest w in
          let ok = d <= 1e-6 *. m in
          if not ok then incr bad;
          Printf.printf "%-6s %s %s, largest difference %.3g of %.3g\n%!" c.name
            name
            (if ok then "equal" else "differ")
            d m)
        entries)
    cases;
  Array.iter (fun f -> Sys.remove (Filename.concat out f)) (Sys.readdir out);
  Sys.rmdir out;
  exit (if !bad = 0 then 0 else 1)
