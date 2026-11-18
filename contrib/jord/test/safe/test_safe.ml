(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let read file = In_channel.with_open_bin file In_channel.input_all

(* Baseline 03.00, Esri's processing: no offsets. *)
let n0300 = read "mtd_n0300.xml"

(* Baseline 05.10, Microsoft's processing: -1000 for every band. *)
let n0510 = read "mtd_n0510.xml"

(* Level-1C of the acquisition of n0510, ESA's processing at 05.10, cut. *)
let l1c = read "mtd_l1c_n0510.xml"

(* The values that decode DNs: the metadata without its wavelengths. *)
let decoding (m : Sentinel2.metadata) = { m with wavelengths = [] }

let metadata =
  Testable.make
    ~pp:(fun ppf (m : Sentinel2.metadata) ->
      Format.fprintf ppf "{%s Q %g [%s] nodata %d saturated %d}" m.baseline
        m.quantification
        (String.concat "; "
           (List.map
              (fun (b, o) -> Printf.sprintf "%s %d" (Sentinel2.name b) o)
              m.offsets))
        m.nodata m.saturated)
    ~equal:( = )

let stated ?(offsets = []) baseline =
  {
    Sentinel2.baseline;
    quantification = 10000.;
    offsets;
    nodata = 0;
    saturated = 65535;
    wavelengths = [];
  }

let all = List.map (fun b -> (b, -1000)) Sentinel2.bands

(* [s] with its one occurrence of [a] replaced by [b]. *)
let edit a b s =
  let n = String.length a in
  let rec find i =
    if i + n > String.length s then failf "%S is not in the fixture" a
    else if String.sub s i n = a then i
    else find (i + 1)
  in
  let i = find 0 in
  String.sub s 0 i ^ b ^ String.sub s (i + n) (String.length s - i - n)

let products =
  test "a product's baseline, quantification, special values and offsets"
    (fun () ->
      let read f xml = Result.map decoding (f xml) in
      equal (result metadata string)
        (Ok (stated "03.00"))
        (read Jord_safe.l2a n0300);
      (* B10's offset among them, though Level-2A holds no B10 image. *)
      equal (result metadata string)
        (Ok (stated ~offsets:all "05.10"))
        (read Jord_safe.l2a n0510);
      equal (result metadata string)
        (Ok (stated ~offsets:all "05.10"))
        (read Jord_safe.l1c l1c))

(* MIN, MAX and CENTRAL of each band's Spectral_Information in n0510, an S2A
   product. *)
let s2a =
  Sentinel2.
    [
      (B01, 412., 456., 442.7);
      (B02, 456., 533., 492.7);
      (B03, 538., 583., 559.8);
      (B04, 646., 684., 664.6);
      (B05, 695., 714., 704.1);
      (B06, 731., 749., 740.5);
      (B07, 769., 797., 782.8);
      (B08, 760., 907., 832.8);
      (B8A, 837., 881., 864.7);
      (B09, 932., 958., 945.1);
      (B10, 1337., 1412., 1373.5);
      (B11, 1539., 1682., 1613.7);
      (B12, 2078., 2320., 2202.4);
    ]

let wavelengths =
  test "each band's wavelengths are its satellite's, as its product states"
    (fun () ->
      let stated =
        List.map
          (fun (b, min, max, central) -> (b, { Sentinel2.central; min; max }))
          s2a
      in
      let w f xml = (require_ok (f xml)).Sentinel2.wavelengths in
      is_true (w Jord_safe.l2a n0510 = stated);
      is_true (w Jord_safe.l1c l1c = stated);
      (* S2B's B04, in n0300, is centred elsewhere. *)
      is_true
        (List.assoc Sentinel2.B04 (w Jord_safe.l2a n0300)
        = { Sentinel2.central = 665.; min = 646.; max = 685. });
      let m = require_ok (Jord_safe.l2a n0510) in
      let b = require_ok (Sentinel2.reflectance m Sentinel2.B04) in
      equal (option float_exact) (Some (664.6 /. 1000.)) b.center_wavelength;
      equal (option float_exact) (Some 0.038) b.full_width_half_max)

let levels =
  test "each reader refuses the other level's file, naming its root"
    (fun () ->
      contains ~sub:"Level-1C_User_Product"
        (require_error (Jord_safe.l2a l1c));
      contains ~sub:"Level-2A_User_Product"
        (require_error (Jord_safe.l1c n0510)))

(* The two levels of one datastrip hold one acquisition. *)
let keys =
  test "a product's two levels have equal keys" (fun () ->
      let datastrip xml =
        let a = {|datastripIdentifier="|} in
        let rec find i =
          if String.sub xml i (String.length a) = a then i + String.length a
          else find (i + 1)
        in
        let i = find 0 in
        String.sub xml i (String.index_from xml i '"' - i)
      in
      let key xml = require_ok (Sentinel2.key ~tile:"33UUU" (datastrip xml)) in
      equal string
        "S2A_OPER_MSI_L1C_DS_2APS_20240629T135511_S20240629T102556_N05.10"
        (datastrip l1c);
      equal bool true (key l1c = key n0510))

let reflectance =
  test "the metadata decodes each product's DNs by ESA's rule" (fun () ->
      let dn = Nx.create Nx.uint16 [| 4 |] [| 0; 1000; 2234; 65535 |] in
      let decode xml =
        let m = require_ok (Jord_safe.l2a xml) in
        let b = require_ok (Sentinel2.reflectance m Sentinel2.B04) in
        let v, ok = Band.decode b Nx.float32 dn in
        (Nx.to_array (Nx.cast Nx.float64 v), Nx.to_array ok)
      in
      let single x = Int32.float_of_bits (Int32.bits_of_float x) in
      let mask = [| false; true; true; false |] in
      equal (pair (array float_exact) (array bool))
        ([| 0.; single 0.1; single 0.2234; 0. |], mask)
        (decode n0300);
      equal (pair (array float_exact) (array bool))
        ([| 0.; 0.; single 0.1234; 0. |], mask)
        (decode n0510);
      let m = require_ok (Jord_safe.l1c l1c) in
      let b = require_ok (Sentinel2.reflectance m Sentinel2.B10) in
      equal float_exact 1e-4 b.scale;
      equal float_exact (-0.1) b.offset)

let left_out =
  test "an offset \"Not applicable\" is left out" (fun () ->
      let xml =
        edit {|<BOA_ADD_OFFSET band_id="3">-1000|}
          {|<BOA_ADD_OFFSET band_id="3">Not applicable|} n0510
      in
      let offsets = List.filter (fun (b, _) -> b <> Sentinel2.B04) all in
      equal (result metadata string)
        (Ok (stated ~offsets "05.10"))
        (Result.map decoding (Jord_safe.l2a xml)))

(* The 05.10 file with BOA_QUANTIFICATION_VALUE [q]. *)
let quantification q =
  edit {|<BOA_QUANTIFICATION_VALUE unit="none">10000|}
    ({|<BOA_QUANTIFICATION_VALUE unit="none">|} ^ q)
    n0510

let quantified q =
  ("a quantification of " ^ q, "BOA_QUANTIFICATION_VALUE", quantification q)

let decimals =
  cases "a quantification in plain decimal form reads" ~id:fst
    [ ("10000.0", 10000.); ("1.0E4", 10000.); ("1e+4", 10000.); ("2.5", 2.5) ]
    (fun (q, v) ->
      let m = require_ok (Jord_safe.l2a (quantification q)) in
      equal float_exact v m.Sentinel2.quantification)

(* Each row breaks the 05.10 file in one place, and the error names it. *)
let refusals =
  cases "a file that breaks the format is an error" ~id:(fun (n, _, _) -> n)
    [
      ("not XML", "line", String.sub n0510 0 1000);
      ("text after the root", "follows", n0510 ^ "<x/>");
      ( "another product level",
        "Level-1C_User_Product",
        edit "<n1:Level-2A_User_Product" "<n1:Level-1C_User_Product"
          (edit "</n1:Level-2A_User_Product>" "</n1:Level-1C_User_Product>"
             n0510) );
      ( "no baseline",
        "PROCESSING_BASELINE",
        edit "<PROCESSING_BASELINE>05.10</PROCESSING_BASELINE>" "" n0510 );
      ( "no quantification",
        "BOA_QUANTIFICATION_VALUE",
        edit
          {|<BOA_QUANTIFICATION_VALUE unit="none">10000</BOA_QUANTIFICATION_VALUE>|}
          "" n0510 );
      quantified "0";
      quantified "1_0000";
      quantified "0x2710";
      quantified "1e-320";
      quantified "1e400";
      quantified "-10000";
      quantified ".5";
      quantified "1e";
      quantified "1.0E+";
      quantified "inf";
      ( "NODATA twice",
        "NODATA",
        edit "<SPECIAL_VALUE_TEXT>SATURATED" "<SPECIAL_VALUE_TEXT>NODATA" n0510
      );
      ( "a special value not an integer",
        "SPECIAL_VALUE_INDEX",
        edit "<SPECIAL_VALUE_INDEX>65535" "<SPECIAL_VALUE_INDEX>0x10" n0510 );
      ( "an offset twice",
        "twice for B01",
        edit {|<BOA_ADD_OFFSET band_id="1">|} {|<BOA_ADD_OFFSET band_id="0">|}
          n0510 );
      ( "an offset of no band",
        "band_id \"13\"",
        edit {|<BOA_ADD_OFFSET band_id="12">|}
          {|<BOA_ADD_OFFSET band_id="13">|} n0510 );
      ( "an offset not an integer",
        "-1e3",
        edit {|<BOA_ADD_OFFSET band_id="2">-1000|}
          {|<BOA_ADD_OFFSET band_id="2">-1e3|} n0510 );
      ( "an offset with no band_id",
        "band_id",
        edit {|<BOA_ADD_OFFSET band_id="2">|} {|<BOA_ADD_OFFSET>|} n0510 );
      ( "a band not Sentinel-2's",
        "B13",
        edit {|physicalBand="B8A"|} {|physicalBand="B13"|} n0510 );
      ( "a CENTRAL outside MIN and MAX",
        "CENTRAL 700 is not from MIN 646 to MAX 684",
        edit {|<CENTRAL unit="nm">664.6</CENTRAL>|}
          {|<CENTRAL unit="nm">700</CENTRAL>|} n0510 );
      ( "a wavelength in micrometres",
        "MIN is not a positive number of nanometres",
        edit {|<MIN unit="nm">646</MIN>|} {|<MIN unit="um">0.646</MIN>|} n0510
      );
      ( "a wavelength not a number",
        "MAX: \"684x\"",
        edit {|<MAX unit="nm">684</MAX>|} {|<MAX unit="nm">684x</MAX>|} n0510 );
      ( "a band numbered as another",
        "bandId \"10\" is not B11's",
        edit {|bandId="10" physicalBand="B10"|}
          {|bandId="10" physicalBand="B11"|} n0510 );
    ]
    (fun (_, sub, xml) -> contains ~sub (require_error (Jord_safe.l2a xml)))

(* Level-1C's names, each broken once. *)
let l1c_refusals =
  cases "a Level-1C file that breaks the format is an error"
    ~id:(fun (n, _, _) -> n)
    [
      ( "no quantification",
        "QUANTIFICATION_VALUE",
        edit
          {|<QUANTIFICATION_VALUE unit="none">10000</QUANTIFICATION_VALUE>|}
          "" l1c );
      ( "a quantification of 0",
        "QUANTIFICATION_VALUE \"0\"",
        edit {|<QUANTIFICATION_VALUE unit="none">10000|}
          {|<QUANTIFICATION_VALUE unit="none">0|} l1c );
      ( "an offset twice",
        "RADIO_ADD_OFFSET is stated twice for B01",
        edit {|<RADIO_ADD_OFFSET band_id="1">|}
          {|<RADIO_ADD_OFFSET band_id="0">|} l1c );
      ( "an offset of no band",
        "RADIO_ADD_OFFSET's band_id \"13\"",
        edit {|<RADIO_ADD_OFFSET band_id="12">|}
          {|<RADIO_ADD_OFFSET band_id="13">|} l1c );
    ]
    (fun (_, sub, xml) -> contains ~sub (require_error (Jord_safe.l1c xml)))

(* Tiles *)

(* The tile of n0510, its Sun grid and four of its view grids. *)
let tl = read "mtd_tl_n0510.xml"

let instant s = require_ok (Time.of_rfc3339 s)
let deg = Float.pi /. 180.

(* The difference of two azimuths, in (-180, 180]. *)
let turn a b =
  let d = Float.rem (a -. b) 360. in
  if d > 180. then d -. 360. else if d <= -180. then d +. 360. else d

let grid =
  Testable.make
    ~pp:(fun ppf (g : Grid.t) ->
      let x, y = g.origin in
      Format.fprintf ppf "{%d x %d from (%g, %g)}" g.rows g.cols x y)
    ~equal:Grid.equal

let nodes =
  test "the nodes are cell centres from the tile's corner, 5 km apart"
    (fun () ->
      let t = require_ok (Jord_safe.tile tl) in
      let want =
        Grid.v
          (Crs.utm ~zone:33 ~north:true)
          ~rows:23 ~cols:23 ~origin:(297500., 5902540.) ~row_step:(0., -5000.)
          ~col_step:(5000., 0.)
      in
      equal grid want t.nodes;
      equal float_exact (instant "2024-06-29T10:26:13.410745Z") t.sensing_time)

(* Sun's angles at instant [time] on the nodes moved by [shift] metres east
   and south. *)
let sun_at ?(shift = 0.) (t : Jord_safe.tile) time =
  let xy = Grid.centres ~at:Nx.Host.on t.nodes in
  let xy = Nx.add xy (Nx.create Nx.float64 [| 2 |] [| shift; -.shift |]) in
  let p = Crs.transform ~src:t.nodes.crs ~dst:Crs.lonlat xy in
  Sun.angles (Nx.scalar Nx.float64 time) p

(* The largest zenith difference, and azimuth difference as an arc, between
   [sun_at] and the tile's Sun grid. *)
let sun_errors ?shift (t : Jord_safe.tile) time =
  let z, a = sun_at ?shift t time in
  let ez = ref 0. and ea = ref 0. in
  for i = 0 to 22 do
    for j = 0 to 22 do
      let at x = Nx.item [ i; j ] x in
      ez := Float.max !ez (Float.abs (at z -. at (fst t.sun)));
      ea :=
        Float.max !ea
          (Float.abs (turn (at a) (at (snd t.sun)) *. Float.sin (at z *. deg)))
    done
  done;
  (!ez, !ea)

(* Sun agrees with ESA's grid at SENSING_TIME on the nodes. One instant
   serves the whole tile, so the azimuth's error runs along the rows, from
   0.03° of arc to -0.04°, as a tile sensed over some seconds would give: a
   node is held to 0.02° and 0.05°. On nodes moved half a step, or at the
   item's datetime, the datatake's start, it does not agree. *)
let sun =
  test "Sun's angles on the nodes at SENSING_TIME are the tile's Sun grid"
    (fun () ->
      let t = require_ok (Jord_safe.tile tl) in
      let ez, ea = sun_errors t t.sensing_time in
      is_true ~msg:(Printf.sprintf "zenith %g" ez) (ez <= 0.02);
      is_true ~msg:(Printf.sprintf "azimuth %g" ea) (ea <= 0.05);
      let ez, _ = sun_errors ~shift:2500. t t.sensing_time in
      is_true ~msg:(Printf.sprintf "zenith %g, moved" ez) (ez > 0.02);
      let ez, _ = sun_errors t (instant "2024-06-29T10:20:21.024Z") in
      is_true ~msg:(Printf.sprintf "zenith %g, at datetime" ez) (ez > 0.3);
      (* s2:mean_solar_zenith and s2:mean_solar_azimuth of the item
         (stac/mpc_sentinel2_l2a.json), the file's Mean_Sun_Angle, are the
         grid's means, and Sun's to its bounds: 0.014°, and the azimuth's arc
         0.014° plus 0.6 arcseconds over the cosine of the latitude. *)
      let mean x = Nx.item [] (Nx.mean x) in
      equal (float 1e-6) 30.8216197888911 (mean (fst t.sun));
      equal (float 1e-4) 159.071270075336 (mean (snd t.sun));
      let z, a = sun_at t t.sensing_time in
      let mz = mean z in
      equal (float 0.014) 30.8216197888911 mz;
      equal
        (float (0.014 +. (0.6 /. 3600. /. Float.cos (52. *. deg))))
        0.
        (turn (mean a) 159.071270075336 *. Float.sin (mz *. deg)))

let finite x = Int64.to_int (Nx.item [] (Nx.count (Nx.isfinite x)))

(* The file's four view grids, B04's and B10's at detectors 7 and 8: their
   finite values, counted in the file, the first of them in each, and
   nothing else. *)
let view =
  test "each view grid lies at its band's index and its detector less one"
    (fun () ->
      let t = require_ok (Jord_safe.tile tl) in
      let vz, va = t.view in
      equal (array int) [| 23; 23; 13; 12 |] (Nx.shape vz);
      equal (array int) [| 23; 23; 13; 12 |] (Nx.shape va);
      let at x b d = Nx.slice Nx.[ A; A; I (Sentinel2.index b); I (d - 1) ] x in
      List.iter
        (fun (b, d, n, (i, j), z, a) ->
          let msg = Printf.sprintf "%s, detector %d" (Sentinel2.name b) d in
          equal ~msg int n (finite (at vz b d));
          equal ~msg int n (finite (at va b d));
          equal ~msg float_exact z (Nx.item [ i; j ] (at vz b d));
          equal ~msg float_exact a (Nx.item [ i; j ] (at va b d)))
        Sentinel2.
          [
            (B04, 7, 125, (0, 8), 1.20872, 203.587);
            (B04, 8, 126, (0, 13), 2.28216, 312.46);
            (B10, 7, 124, (0, 8), 1.14302, 203.997);
            (B10, 8, 122, (0, 13), 2.24662, 310.488);
          ];
      equal int (125 + 126 + 124 + 122) (finite vz);
      (* A node two detectors see has an angle from each. *)
      equal (pair float_exact float_exact) (2.27308, 2.16959)
        (Nx.item [ 2; 12 ] (at vz B04 7), Nx.item [ 2; 12 ] (at vz B04 8)))

(* The fixture's four view grids, each a pair of [[23; 23]] float arrays,
   zenith then azimuth, NaN where the file states NaN. *)
let grids (t : Jord_safe.tile) =
  let vz, va = t.view in
  let rows x b d =
    let g = Nx.slice Nx.[ A; A; I (Sentinel2.index b); I (d - 1) ] x in
    Array.init 23 (fun i -> Array.init 23 (fun j -> Nx.item [ i; j ] g))
  in
  List.map
    (fun (b, d) -> (rows vz b d, rows va b d))
    Sentinel2.[ (B04, 7); (B04, 8); (B10, 7); (B10, 8) ]

(* The view's azimuth points from the ground to the sensor, so the zenith
   falls along it, toward the satellite's track: at each interior node, the
   zenith's gradient (east, north), by central differences, points against
   the azimuth's direction (sin φ, cos φ). *)
let toward_sensor =
  test "the view zenith falls along the view azimuth" (fun () ->
      let t = require_ok (Jord_safe.tile tl) in
      let nodes = ref 0 in
      List.iter
        (fun (z, a) ->
          for i = 1 to 21 do
            for j = 1 to 21 do
              let q =
                [ z.(i).(j); z.(i).(j - 1); z.(i).(j + 1); z.(i - 1).(j);
                  z.(i + 1).(j); a.(i).(j) ]
              in
              if List.for_all Float.is_finite q then (
                incr nodes;
                let ge = (z.(i).(j + 1) -. z.(i).(j - 1)) /. 2. in
                let gn = (z.(i - 1).(j) -. z.(i + 1).(j)) /. 2. in
                let phi = a.(i).(j) *. deg in
                let c =
                  ((ge *. Float.sin phi) +. (gn *. Float.cos phi))
                  /. Float.hypot ge gn
                in
                is_true ~msg:(Printf.sprintf "node (%d, %d): %g" i j c)
                  (c < -0.1))
            done
          done)
        (grids t);
      equal int 286 !nodes)

(* The interface's interpolation rule, held out: each grid kept at every other
   node along the rows, 10 km apart, and read by Warp.bilinear at the nodes
   left out between two finite ones. The view's direction, tan θ · (sin φ,
   cos φ), is affine on the ground, so it gives each left-out node to the
   file's printing; the angles, the azimuth by its sine and cosine, miss by
   degrees near nadir. *)
let held_out =
  test "the view's direction interpolates where its angles do not" (fun () ->
      let t = require_ok (Jord_safe.tile tl) in
      let x0, y0 = t.nodes.origin in
      let coarse =
        Grid.v t.nodes.crs ~rows:23 ~cols:12
          ~origin:(x0 -. 2500., y0) ~row_step:(0., -5000.)
          ~col_step:(10000., 0.)
      in
      let p = Warp.pixel coarse (Grid.centres ~at:Nx.Host.on t.nodes) in
      (* [f] of each node's zenith and azimuth, at every other node, read
         at every node. *)
      let read z a f =
        let k = Array.length (f 0. 0.) in
        let at i c = f (z.(i).(2 * c) *. deg) (a.(i).(2 * c) *. deg) in
        let x =
          Nx.init Nx.float64 [| 23; 12; k |] (fun ix ->
              (at ix.(0) ix.(1)).(ix.(2)))
        in
        let valid =
          Nx.init Nx.bool [| 23; 12 |] (fun ix ->
              Float.is_finite z.(ix.(0)).(2 * ix.(1)))
        in
        fst (Warp.bilinear ~valid coarse p x)
      in
      let vector th ph =
        [| Float.tan th *. Float.sin ph; Float.tan th *. Float.cos ph |]
      in
      let angles th ph = [| th; Float.sin ph; Float.cos ph |] in
      let worst = Array.make 4 0. in
      let keep k e = worst.(k) <- Float.max worst.(k) e in
      List.iter
        (fun (z, a) ->
          let v = read z a vector and w = read z a angles in
          for i = 0 to 22 do
            for j = 1 to 21 do
              if j mod 2 = 1
                 && List.for_all Float.is_finite
                      [ z.(i).(j - 1); z.(i).(j); z.(i).(j + 1) ]
              then (
                let e = Nx.item [ i; j; 0 ] v and n = Nx.item [ i; j; 1 ] v in
                let zv = Float.atan (Float.hypot e n) /. deg in
                let av = Float.atan2 e n /. deg in
                let zw = Nx.item [ i; j; 0 ] w /. deg in
                let aw =
                  Float.atan2 (Nx.item [ i; j; 1 ] w) (Nx.item [ i; j; 2 ] w)
                  /. deg
                in
                keep 0 (Float.abs (zv -. z.(i).(j)));
                keep 1 (Float.abs (turn av a.(i).(j)));
                keep 2 (Float.abs (zw -. z.(i).(j)));
                keep 3 (Float.abs (turn aw a.(i).(j))))
            done
          done)
        (grids t);
      (* Measured: the direction 0.0005° in zenith and 0.012° in azimuth,
         the angles 0.058° and 2.0°. *)
      let msg =
        Printf.sprintf "%g %g %g %g" worst.(0) worst.(1) worst.(2) worst.(3)
      in
      is_true ~msg (worst.(0) <= 0.001 && worst.(1) <= 0.02);
      is_true ~msg (worst.(2) > 0.02 && worst.(3) > 1.))

let level =
  test "a Level-1C tile reads as a Level-2A one" (fun () ->
      let l1c =
        edit "<n1:Level-2A_Tile_ID" "<n1:Level-1C_Tile_ID"
          (edit "</n1:Level-2A_Tile_ID>" "</n1:Level-1C_Tile_ID>" tl)
      in
      let a = require_ok (Jord_safe.tile tl) in
      let b = require_ok (Jord_safe.tile l1c) in
      equal grid a.nodes b.nodes;
      let values x = Nx.to_array (fst x.Jord_safe.view) in
      equal int 0 (compare (values a) (values b)))

let grid_of b d = Printf.sprintf {|bandId="%d" detectorId="%d"|} b d

(* Each row breaks the tile in one place, and the error names it. *)
let tile_refusals =
  cases "a tile that breaks the format is an error" ~id:(fun (n, _, _) -> n)
    [
      ( "another root",
        "Level-2A_User_Product",
        edit "<n1:Level-2A_Tile_ID" "<n1:Level-2A_User_Product"
          (edit "</n1:Level-2A_Tile_ID>" "</n1:Level-2A_User_Product>" tl) );
      ( "no SENSING_TIME",
        "SENSING_TIME",
        edit {|<SENSING_TIME metadataLevel="Standard">|}
          {|<SENSING_TIMES metadataLevel="Standard">|}
          (edit "</SENSING_TIME>" "</SENSING_TIMES>" tl) );
      ( "a SENSING_TIME not RFC 3339's",
        "SENSING_TIME: \"2024-06-29 10\"",
        edit "2024-06-29T10:26:13.410745Z" "2024-06-29 10" tl );
      ( "a system not an EPSG code",
        "EPSG:3263x",
        edit "<HORIZONTAL_CS_CODE>EPSG:32633" "<HORIZONTAL_CS_CODE>EPSG:3263x"
          tl );
      ( "a corner not a number",
        "ULX",
        edit "<ULX>300000</ULX>" "<ULX>3e5x</ULX>" tl );
      ( "two corners",
        "corners",
        edit "<ULY>5900040</ULY>" "<ULY>5900050</ULY>" tl );
      ( "a value neither a decimal nor NaN",
        "\"nan\"",
        edit {|<VALUES>31.4559 |} {|<VALUES>nan |} tl );
      ( "a short row",
        "steps or size",
        edit {|<VALUES>NaN NaN NaN NaN 1.2009 |} {|<VALUES>NaN NaN NaN 1.2009 |}
          tl );
      ( "a step in kilometres",
        "metres",
        edit {|<COL_STEP unit="m">5000</COL_STEP>|}
          {|<COL_STEP unit="km">5</COL_STEP>|} tl );
      ( "a view grid of other steps",
        "steps or size",
        let step = {|>
        <Zenith>
          <COL_STEP unit="m">|} in
        edit
          (grid_of 3 7 ^ step ^ "5000")
          (grid_of 3 7 ^ step ^ "2500")
          tl );
      ( "a band of no index",
        "bandId \"13\"",
        edit (grid_of 3 7) (grid_of 13 7) tl );
      ( "detector 0",
        "detectorId \"0\"",
        edit (grid_of 3 7) (grid_of 3 0) tl );
      ( "detector 13",
        "detectorId \"13\"",
        edit (grid_of 3 7) (grid_of 3 13) tl );
      ("a grid twice", "twice", edit (grid_of 3 7) (grid_of 3 8) tl);
    ]
    (fun (_, sub, xml) -> contains ~sub (require_error (Jord_safe.tile xml)))

let () =
  exit
    (run "jord.safe"
       [
         group "products"
           [
             products;
             levels;
             wavelengths;
             keys;
             reflectance;
             left_out;
             decimals;
             refusals;
             l1c_refusals;
           ];
         group "tile"
           [ nodes; sun; view; toward_sensor; held_out; level; tile_refusals ];
       ])
