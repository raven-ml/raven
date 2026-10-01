(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f

(* [Color.pp] rounds to 8 bits, so the witnesses print every digit. *)
let pp_color ppf c =
  Format.fprintf ppf "(v ~alpha:%.17g %.17g %.17g %.17g)" (Color.alpha c)
    (Color.r c) (Color.g c) (Color.b c)

let color = Testable.make ~pp:pp_color ~equal:Color.equal
let colors = array color
let scheme = Testable.make ~pp:Scheme.pp ~equal:Scheme.equal
let hex = Testable.make ~pp:Format.pp_print_string ~equal:String.equal

(* [of_hex s] is the colours of the hexadecimal digits [s], six per colour, as
   the published sources write them. *)
let of_hex s =
  Array.init
    (String.length s / 6)
    (fun i ->
      match Color.of_hex ("#" ^ String.sub s (6 * i) 6) with
      | Ok c -> c
      | Error e -> failwith e)

let to_hex cs = Array.map (fun c -> String.sub (Color.to_hex c) 1 6) cs

let lightness c =
  let l, _, _ = Color.to_oklab c in
  l

let rev cs =
  let n = Array.length cs in
  Array.init n (fun i -> cs.(n - 1 - i))

(* The published tables. The Brewer tables list the designed tables from three
   classes up, the qualitative palettes their colours, as d3-scale-chromatic
   writes them. *)
let brewer_sequential =
  [
    ( "blues",
      Scheme.blues,
      [
        "deebf79ecae13182bd";
        "eff3ffbdd7e76baed62171b5";
        "eff3ffbdd7e76baed63182bd08519c";
        "eff3ffc6dbef9ecae16baed63182bd08519c";
        "eff3ffc6dbef9ecae16baed64292c62171b5084594";
        "f7fbffdeebf7c6dbef9ecae16baed64292c62171b5084594";
        "f7fbffdeebf7c6dbef9ecae16baed64292c62171b508519c08306b";
      ] );
    ( "greens",
      Scheme.greens,
      [
        "e5f5e0a1d99b31a354";
        "edf8e9bae4b374c476238b45";
        "edf8e9bae4b374c47631a354006d2c";
        "edf8e9c7e9c0a1d99b74c47631a354006d2c";
        "edf8e9c7e9c0a1d99b74c47641ab5d238b45005a32";
        "f7fcf5e5f5e0c7e9c0a1d99b74c47641ab5d238b45005a32";
        "f7fcf5e5f5e0c7e9c0a1d99b74c47641ab5d238b45006d2c00441b";
      ] );
    ( "greys",
      Scheme.greys,
      [
        "f0f0f0bdbdbd636363";
        "f7f7f7cccccc969696525252";
        "f7f7f7cccccc969696636363252525";
        "f7f7f7d9d9d9bdbdbd969696636363252525";
        "f7f7f7d9d9d9bdbdbd969696737373525252252525";
        "fffffff0f0f0d9d9d9bdbdbd969696737373525252252525";
        "fffffff0f0f0d9d9d9bdbdbd969696737373525252252525000000";
      ] );
    ( "oranges",
      Scheme.oranges,
      [
        "fee6cefdae6be6550d";
        "feeddefdbe85fd8d3cd94701";
        "feeddefdbe85fd8d3ce6550da63603";
        "feeddefdd0a2fdae6bfd8d3ce6550da63603";
        "feeddefdd0a2fdae6bfd8d3cf16913d948018c2d04";
        "fff5ebfee6cefdd0a2fdae6bfd8d3cf16913d948018c2d04";
        "fff5ebfee6cefdd0a2fdae6bfd8d3cf16913d94801a636037f2704";
      ] );
    ( "purples",
      Scheme.purples,
      [
        "efedf5bcbddc756bb1";
        "f2f0f7cbc9e29e9ac86a51a3";
        "f2f0f7cbc9e29e9ac8756bb154278f";
        "f2f0f7dadaebbcbddc9e9ac8756bb154278f";
        "f2f0f7dadaebbcbddc9e9ac8807dba6a51a34a1486";
        "fcfbfdefedf5dadaebbcbddc9e9ac8807dba6a51a34a1486";
        "fcfbfdefedf5dadaebbcbddc9e9ac8807dba6a51a354278f3f007d";
      ] );
    ( "reds",
      Scheme.reds,
      [
        "fee0d2fc9272de2d26";
        "fee5d9fcae91fb6a4acb181d";
        "fee5d9fcae91fb6a4ade2d26a50f15";
        "fee5d9fcbba1fc9272fb6a4ade2d26a50f15";
        "fee5d9fcbba1fc9272fb6a4aef3b2ccb181d99000d";
        "fff5f0fee0d2fcbba1fc9272fb6a4aef3b2ccb181d99000d";
        "fff5f0fee0d2fcbba1fc9272fb6a4aef3b2ccb181da50f1567000d";
      ] );
    ( "bugn",
      Scheme.bugn,
      [
        "e5f5f999d8c92ca25f";
        "edf8fbb2e2e266c2a4238b45";
        "edf8fbb2e2e266c2a42ca25f006d2c";
        "edf8fbccece699d8c966c2a42ca25f006d2c";
        "edf8fbccece699d8c966c2a441ae76238b45005824";
        "f7fcfde5f5f9ccece699d8c966c2a441ae76238b45005824";
        "f7fcfde5f5f9ccece699d8c966c2a441ae76238b45006d2c00441b";
      ] );
    ( "bupu",
      Scheme.bupu,
      [
        "e0ecf49ebcda8856a7";
        "edf8fbb3cde38c96c688419d";
        "edf8fbb3cde38c96c68856a7810f7c";
        "edf8fbbfd3e69ebcda8c96c68856a7810f7c";
        "edf8fbbfd3e69ebcda8c96c68c6bb188419d6e016b";
        "f7fcfde0ecf4bfd3e69ebcda8c96c68c6bb188419d6e016b";
        "f7fcfde0ecf4bfd3e69ebcda8c96c68c6bb188419d810f7c4d004b";
      ] );
    ( "gnbu",
      Scheme.gnbu,
      [
        "e0f3dba8ddb543a2ca";
        "f0f9e8bae4bc7bccc42b8cbe";
        "f0f9e8bae4bc7bccc443a2ca0868ac";
        "f0f9e8ccebc5a8ddb57bccc443a2ca0868ac";
        "f0f9e8ccebc5a8ddb57bccc44eb3d32b8cbe08589e";
        "f7fcf0e0f3dbccebc5a8ddb57bccc44eb3d32b8cbe08589e";
        "f7fcf0e0f3dbccebc5a8ddb57bccc44eb3d32b8cbe0868ac084081";
      ] );
    ( "orrd",
      Scheme.orrd,
      [
        "fee8c8fdbb84e34a33";
        "fef0d9fdcc8afc8d59d7301f";
        "fef0d9fdcc8afc8d59e34a33b30000";
        "fef0d9fdd49efdbb84fc8d59e34a33b30000";
        "fef0d9fdd49efdbb84fc8d59ef6548d7301f990000";
        "fff7ecfee8c8fdd49efdbb84fc8d59ef6548d7301f990000";
        "fff7ecfee8c8fdd49efdbb84fc8d59ef6548d7301fb300007f0000";
      ] );
    ( "pubu",
      Scheme.pubu,
      [
        "ece7f2a6bddb2b8cbe";
        "f1eef6bdc9e174a9cf0570b0";
        "f1eef6bdc9e174a9cf2b8cbe045a8d";
        "f1eef6d0d1e6a6bddb74a9cf2b8cbe045a8d";
        "f1eef6d0d1e6a6bddb74a9cf3690c00570b0034e7b";
        "fff7fbece7f2d0d1e6a6bddb74a9cf3690c00570b0034e7b";
        "fff7fbece7f2d0d1e6a6bddb74a9cf3690c00570b0045a8d023858";
      ] );
    ( "pubugn",
      Scheme.pubugn,
      [
        "ece2f0a6bddb1c9099";
        "f6eff7bdc9e167a9cf02818a";
        "f6eff7bdc9e167a9cf1c9099016c59";
        "f6eff7d0d1e6a6bddb67a9cf1c9099016c59";
        "f6eff7d0d1e6a6bddb67a9cf3690c002818a016450";
        "fff7fbece2f0d0d1e6a6bddb67a9cf3690c002818a016450";
        "fff7fbece2f0d0d1e6a6bddb67a9cf3690c002818a016c59014636";
      ] );
    ( "purd",
      Scheme.purd,
      [
        "e7e1efc994c7dd1c77";
        "f1eef6d7b5d8df65b0ce1256";
        "f1eef6d7b5d8df65b0dd1c77980043";
        "f1eef6d4b9dac994c7df65b0dd1c77980043";
        "f1eef6d4b9dac994c7df65b0e7298ace125691003f";
        "f7f4f9e7e1efd4b9dac994c7df65b0e7298ace125691003f";
        "f7f4f9e7e1efd4b9dac994c7df65b0e7298ace125698004367001f";
      ] );
    ( "rdpu",
      Scheme.rdpu,
      [
        "fde0ddfa9fb5c51b8a";
        "feebe2fbb4b9f768a1ae017e";
        "feebe2fbb4b9f768a1c51b8a7a0177";
        "feebe2fcc5c0fa9fb5f768a1c51b8a7a0177";
        "feebe2fcc5c0fa9fb5f768a1dd3497ae017e7a0177";
        "fff7f3fde0ddfcc5c0fa9fb5f768a1dd3497ae017e7a0177";
        "fff7f3fde0ddfcc5c0fa9fb5f768a1dd3497ae017e7a017749006a";
      ] );
    ( "ylgn",
      Scheme.ylgn,
      [
        "f7fcb9addd8e31a354";
        "ffffccc2e69978c679238443";
        "ffffccc2e69978c67931a354006837";
        "ffffccd9f0a3addd8e78c67931a354006837";
        "ffffccd9f0a3addd8e78c67941ab5d238443005a32";
        "ffffe5f7fcb9d9f0a3addd8e78c67941ab5d238443005a32";
        "ffffe5f7fcb9d9f0a3addd8e78c67941ab5d238443006837004529";
      ] );
    ( "ylgnbu",
      Scheme.ylgnbu,
      [
        "edf8b17fcdbb2c7fb8";
        "ffffcca1dab441b6c4225ea8";
        "ffffcca1dab441b6c42c7fb8253494";
        "ffffccc7e9b47fcdbb41b6c42c7fb8253494";
        "ffffccc7e9b47fcdbb41b6c41d91c0225ea80c2c84";
        "ffffd9edf8b1c7e9b47fcdbb41b6c41d91c0225ea80c2c84";
        "ffffd9edf8b1c7e9b47fcdbb41b6c41d91c0225ea8253494081d58";
      ] );
    ( "ylorbr",
      Scheme.ylorbr,
      [
        "fff7bcfec44fd95f0e";
        "ffffd4fed98efe9929cc4c02";
        "ffffd4fed98efe9929d95f0e993404";
        "ffffd4fee391fec44ffe9929d95f0e993404";
        "ffffd4fee391fec44ffe9929ec7014cc4c028c2d04";
        "ffffe5fff7bcfee391fec44ffe9929ec7014cc4c028c2d04";
        "ffffe5fff7bcfee391fec44ffe9929ec7014cc4c02993404662506";
      ] );
    ( "ylorrd",
      Scheme.ylorrd,
      [
        "ffeda0feb24cf03b20";
        "ffffb2fecc5cfd8d3ce31a1c";
        "ffffb2fecc5cfd8d3cf03b20bd0026";
        "ffffb2fed976feb24cfd8d3cf03b20bd0026";
        "ffffb2fed976feb24cfd8d3cfc4e2ae31a1cb10026";
        "ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cb10026";
        "ffffccffeda0fed976feb24cfd8d3cfc4e2ae31a1cbd0026800026";
      ] );
  ]

let brewer_diverging =
  [
    ( "brbg",
      Scheme.brbg,
      [
        "d8b365f5f5f55ab4ac";
        "a6611adfc27d80cdc1018571";
        "a6611adfc27df5f5f580cdc1018571";
        "8c510ad8b365f6e8c3c7eae55ab4ac01665e";
        "8c510ad8b365f6e8c3f5f5f5c7eae55ab4ac01665e";
        "8c510abf812ddfc27df6e8c3c7eae580cdc135978f01665e";
        "8c510abf812ddfc27df6e8c3f5f5f5c7eae580cdc135978f01665e";
        "5430058c510abf812ddfc27df6e8c3c7eae580cdc135978f01665e003c30";
        "5430058c510abf812ddfc27df6e8c3f5f5f5c7eae580cdc135978f01665e003c30";
      ] );
    ( "piyg",
      Scheme.piyg,
      [
        "e9a3c9f7f7f7a1d76a";
        "d01c8bf1b6dab8e1864dac26";
        "d01c8bf1b6daf7f7f7b8e1864dac26";
        "c51b7de9a3c9fde0efe6f5d0a1d76a4d9221";
        "c51b7de9a3c9fde0eff7f7f7e6f5d0a1d76a4d9221";
        "c51b7dde77aef1b6dafde0efe6f5d0b8e1867fbc414d9221";
        "c51b7dde77aef1b6dafde0eff7f7f7e6f5d0b8e1867fbc414d9221";
        "8e0152c51b7dde77aef1b6dafde0efe6f5d0b8e1867fbc414d9221276419";
        "8e0152c51b7dde77aef1b6dafde0eff7f7f7e6f5d0b8e1867fbc414d9221276419";
      ] );
    ( "prgn",
      Scheme.prgn,
      [
        "af8dc3f7f7f77fbf7b";
        "7b3294c2a5cfa6dba0008837";
        "7b3294c2a5cff7f7f7a6dba0008837";
        "762a83af8dc3e7d4e8d9f0d37fbf7b1b7837";
        "762a83af8dc3e7d4e8f7f7f7d9f0d37fbf7b1b7837";
        "762a839970abc2a5cfe7d4e8d9f0d3a6dba05aae611b7837";
        "762a839970abc2a5cfe7d4e8f7f7f7d9f0d3a6dba05aae611b7837";
        "40004b762a839970abc2a5cfe7d4e8d9f0d3a6dba05aae611b783700441b";
        "40004b762a839970abc2a5cfe7d4e8f7f7f7d9f0d3a6dba05aae611b783700441b";
      ] );
    ( "puor",
      Scheme.puor,
      [
        "998ec3f7f7f7f1a340";
        "5e3c99b2abd2fdb863e66101";
        "5e3c99b2abd2f7f7f7fdb863e66101";
        "542788998ec3d8daebfee0b6f1a340b35806";
        "542788998ec3d8daebf7f7f7fee0b6f1a340b35806";
        "5427888073acb2abd2d8daebfee0b6fdb863e08214b35806";
        "5427888073acb2abd2d8daebf7f7f7fee0b6fdb863e08214b35806";
        "2d004b5427888073acb2abd2d8daebfee0b6fdb863e08214b358067f3b08";
        "2d004b5427888073acb2abd2d8daebf7f7f7fee0b6fdb863e08214b358067f3b08";
      ] );
    ( "rdbu",
      Scheme.rdbu,
      [
        "ef8a62f7f7f767a9cf";
        "ca0020f4a58292c5de0571b0";
        "ca0020f4a582f7f7f792c5de0571b0";
        "b2182bef8a62fddbc7d1e5f067a9cf2166ac";
        "b2182bef8a62fddbc7f7f7f7d1e5f067a9cf2166ac";
        "b2182bd6604df4a582fddbc7d1e5f092c5de4393c32166ac";
        "b2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac";
        "67001fb2182bd6604df4a582fddbc7d1e5f092c5de4393c32166ac053061";
        "67001fb2182bd6604df4a582fddbc7f7f7f7d1e5f092c5de4393c32166ac053061";
      ] );
    ( "rdgy",
      Scheme.rdgy,
      [
        "ef8a62ffffff999999";
        "ca0020f4a582bababa404040";
        "ca0020f4a582ffffffbababa404040";
        "b2182bef8a62fddbc7e0e0e09999994d4d4d";
        "b2182bef8a62fddbc7ffffffe0e0e09999994d4d4d";
        "b2182bd6604df4a582fddbc7e0e0e0bababa8787874d4d4d";
        "b2182bd6604df4a582fddbc7ffffffe0e0e0bababa8787874d4d4d";
        "67001fb2182bd6604df4a582fddbc7e0e0e0bababa8787874d4d4d1a1a1a";
        "67001fb2182bd6604df4a582fddbc7ffffffe0e0e0bababa8787874d4d4d1a1a1a";
      ] );
    ( "rdylbu",
      Scheme.rdylbu,
      [
        "fc8d59ffffbf91bfdb";
        "d7191cfdae61abd9e92c7bb6";
        "d7191cfdae61ffffbfabd9e92c7bb6";
        "d73027fc8d59fee090e0f3f891bfdb4575b4";
        "d73027fc8d59fee090ffffbfe0f3f891bfdb4575b4";
        "d73027f46d43fdae61fee090e0f3f8abd9e974add14575b4";
        "d73027f46d43fdae61fee090ffffbfe0f3f8abd9e974add14575b4";
        "a50026d73027f46d43fdae61fee090e0f3f8abd9e974add14575b4313695";
        "a50026d73027f46d43fdae61fee090ffffbfe0f3f8abd9e974add14575b4313695";
      ] );
    ( "rdylgn",
      Scheme.rdylgn,
      [
        "fc8d59ffffbf91cf60";
        "d7191cfdae61a6d96a1a9641";
        "d7191cfdae61ffffbfa6d96a1a9641";
        "d73027fc8d59fee08bd9ef8b91cf601a9850";
        "d73027fc8d59fee08bffffbfd9ef8b91cf601a9850";
        "d73027f46d43fdae61fee08bd9ef8ba6d96a66bd631a9850";
        "d73027f46d43fdae61fee08bffffbfd9ef8ba6d96a66bd631a9850";
        "a50026d73027f46d43fdae61fee08bd9ef8ba6d96a66bd631a9850006837";
        "a50026d73027f46d43fdae61fee08bffffbfd9ef8ba6d96a66bd631a9850006837";
      ] );
    ( "spectral",
      Scheme.spectral,
      [
        "fc8d59ffffbf99d594";
        "d7191cfdae61abdda42b83ba";
        "d7191cfdae61ffffbfabdda42b83ba";
        "d53e4ffc8d59fee08be6f59899d5943288bd";
        "d53e4ffc8d59fee08bffffbfe6f59899d5943288bd";
        "d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd";
        "d53e4ff46d43fdae61fee08bffffbfe6f598abdda466c2a53288bd";
        "9e0142d53e4ff46d43fdae61fee08be6f598abdda466c2a53288bd5e4fa2";
        "9e0142d53e4ff46d43fdae61fee08bffffbfe6f598abdda466c2a53288bd5e4fa2";
      ] );
  ]

let brewer_qualitative =
  [
    ("accent", Scheme.accent, "7fc97fbeaed4fdc086ffff99386cb0f0027fbf5b17666666");
    ("dark2", Scheme.dark2, "1b9e77d95f027570b3e7298a66a61ee6ab02a6761d666666");
    ( "paired",
      Scheme.paired,
      "a6cee31f78b4b2df8a33a02cfb9a99e31a1cfdbf6fff7f00cab2d66a3d9affff99b15928"
    );
    ( "pastel1",
      Scheme.pastel1,
      "fbb4aeb3cde3ccebc5decbe4fed9a6ffffcce5d8bdfddaecf2f2f2" );
    ( "pastel2",
      Scheme.pastel2,
      "b3e2cdfdcdaccbd5e8f4cae4e6f5c9fff2aef1e2cccccccc" );
    ( "set1",
      Scheme.set1,
      "e41a1c377eb84daf4a984ea3ff7f00ffff33a65628f781bf999999" );
    ("set2", Scheme.set2, "66c2a5fc8d628da0cbe78ac3a6d854ffd92fe5c494b3b3b3");
    ( "set3",
      Scheme.set3,
      "8dd3c7ffffb3bebadafb807280b1d3fdb462b3de69fccde5d9d9d9bc80bdccebc5ffed6f"
    );
  ]

let d3_viridis =
  "44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725"

let d3_magma =
  "00000401000501010601010802010902020b02020d03030f03031204041405041606051806051a07061c08071e0907200a08220b09240c09260d0a290e0b2b100b2d110c2f120d31130d34140e36150e38160f3b180f3d19103f1a10421c10441d11471e114920114b21114e22115024125325125527125829115a2a115c2c115f2d11612f116331116533106734106936106b38106c390f6e3b0f703d0f713f0f72400f74420f75440f764510774710784910784a10794c117a4e117b4f127b51127c52137c54137d56147d57157e59157e5a167e5c167f5d177f5f187f601880621980641a80651a80671b80681c816a1c816b1d816d1d816e1e81701f81721f817320817521817621817822817922827b23827c23827e24828025828125818326818426818627818827818928818b29818c29818e2a81902a81912b81932b80942c80962c80982d80992d809b2e7f9c2e7f9e2f7fa02f7fa1307ea3307ea5317ea6317da8327daa337dab337cad347cae347bb0357bb2357bb3367ab5367ab73779b83779ba3878bc3978bd3977bf3a77c03a76c23b75c43c75c53c74c73d73c83e73ca3e72cc3f71cd4071cf4070d0416fd2426fd3436ed5446dd6456cd8456cd9466bdb476adc4869de4968df4a68e04c67e24d66e34e65e44f64e55064e75263e85362e95462ea5661eb5760ec5860ed5a5fee5b5eef5d5ef05f5ef1605df2625df2645cf3655cf4675cf4695cf56b5cf66c5cf66e5cf7705cf7725cf8745cf8765cf9785df9795df97b5dfa7d5efa7f5efa815ffb835ffb8560fb8761fc8961fc8a62fc8c63fc8e64fc9065fd9266fd9467fd9668fd9869fd9a6afd9b6bfe9d6cfe9f6dfea16efea36ffea571fea772fea973feaa74feac76feae77feb078feb27afeb47bfeb67cfeb77efeb97ffebb81febd82febf84fec185fec287fec488fec68afec88cfeca8dfecc8ffecd90fecf92fed194fed395fed597fed799fed89afdda9cfddc9efddea0fde0a1fde2a3fde3a5fde5a7fde7a9fde9aafdebacfcecaefceeb0fcf0b2fcf2b4fcf4b6fcf6b8fcf7b9fcf9bbfcfbbdfcfdbf"

let d3_inferno =
  "00000401000501010601010802010a02020c02020e03021004031204031405041706041907051b08051d09061f0a07220b07240c08260d08290e092b10092d110a30120a32140b34150b37160b39180c3c190c3e1b0c411c0c431e0c451f0c48210c4a230c4c240c4f260c51280b53290b552b0b572d0b592f0a5b310a5c320a5e340a5f3609613809623909633b09643d09653e0966400a67420a68440a68450a69470b6a490b6a4a0c6b4c0c6b4d0d6c4f0d6c510e6c520e6d540f6d550f6d57106e59106e5a116e5c126e5d126e5f136e61136e62146e64156e65156e67166e69166e6a176e6c186e6d186e6f196e71196e721a6e741a6e751b6e771c6d781c6d7a1d6d7c1d6d7d1e6d7f1e6c801f6c82206c84206b85216b87216b88226a8a226a8c23698d23698f24699025689225689326679526679727669827669a28659b29649d29649f2a63a02a63a22b62a32c61a52c60a62d60a82e5fa92e5eab2f5ead305dae305cb0315bb1325ab3325ab43359b63458b73557b93556ba3655bc3754bd3853bf3952c03a51c13a50c33b4fc43c4ec63d4dc73e4cc83f4bca404acb4149cc4248ce4347cf4446d04545d24644d34743d44842d54a41d74b3fd84c3ed94d3dda4e3cdb503bdd513ade5238df5337e05536e15635e25734e35933e45a31e55c30e65d2fe75e2ee8602de9612bea632aeb6429eb6628ec6726ed6925ee6a24ef6c23ef6e21f06f20f1711ff1731df2741cf3761bf37819f47918f57b17f57d15f67e14f68013f78212f78410f8850ff8870ef8890cf98b0bf98c0af98e09fa9008fa9207fa9407fb9606fb9706fb9906fb9b06fb9d07fc9f07fca108fca309fca50afca60cfca80dfcaa0ffcac11fcae12fcb014fcb216fcb418fbb61afbb81dfbba1ffbbc21fbbe23fac026fac228fac42afac62df9c72ff9c932f9cb35f8cd37f8cf3af7d13df7d340f6d543f6d746f5d949f5db4cf4dd4ff4df53f4e156f3e35af3e55df2e661f2e865f2ea69f1ec6df1ed71f1ef75f1f179f2f27df2f482f3f586f3f68af4f88ef5f992f6fa96f8fb9af9fc9dfafda1fcffa4"

let d3_plasma =
  "0d088710078813078916078a19068c1b068d1d068e20068f2206902406912605912805922a05932c05942e05952f059631059733059735049837049938049a3a049a3c049b3e049c3f049c41049d43039e44039e46039f48039f4903a04b03a14c02a14e02a25002a25102a35302a35502a45601a45801a45901a55b01a55c01a65e01a66001a66100a76300a76400a76600a76700a86900a86a00a86c00a86e00a86f00a87100a87201a87401a87501a87701a87801a87a02a87b02a87d03a87e03a88004a88104a78305a78405a78606a68707a68808a68a09a58b0aa58d0ba58e0ca48f0da4910ea3920fa39410a29511a19613a19814a099159f9a169f9c179e9d189d9e199da01a9ca11b9ba21d9aa31e9aa51f99a62098a72197a82296aa2395ab2494ac2694ad2793ae2892b02991b12a90b22b8fb32c8eb42e8db52f8cb6308bb7318ab83289ba3388bb3488bc3587bd3786be3885bf3984c03a83c13b82c23c81c33d80c43e7fc5407ec6417dc7427cc8437bc9447aca457acb4679cc4778cc4977cd4a76ce4b75cf4c74d04d73d14e72d24f71d35171d45270d5536fd5546ed6556dd7566cd8576bd9586ada5a6ada5b69db5c68dc5d67dd5e66de5f65de6164df6263e06363e16462e26561e26660e3685fe4695ee56a5de56b5de66c5ce76e5be76f5ae87059e97158e97257ea7457eb7556eb7655ec7754ed7953ed7a52ee7b51ef7c51ef7e50f07f4ff0804ef1814df1834cf2844bf3854bf3874af48849f48948f58b47f58c46f68d45f68f44f79044f79143f79342f89441f89540f9973ff9983ef99a3efa9b3dfa9c3cfa9e3bfb9f3afba139fba238fca338fca537fca636fca835fca934fdab33fdac33fdae32fdaf31fdb130fdb22ffdb42ffdb52efeb72dfeb82cfeba2cfebb2bfebd2afebe2afec029fdc229fdc328fdc527fdc627fdc827fdca26fdcb26fccd25fcce25fcd025fcd225fbd324fbd524fbd724fad824fada24f9dc24f9dd25f8df25f8e125f7e225f7e425f6e626f6e826f5e926f5eb27f4ed27f3ee27f3f027f2f227f1f426f1f525f0f724f0f921"

let okabe_ito = "000000e69f0056b4e9009e73f0e4420072b2d55e00cc79a7"
let tableau10 = "4e79a7f28e2ce1575976b7b259a14fedc949af7aa1ff9da79c755fbab0ab"

let named =
  [
    Scheme.viridis;
    Scheme.magma;
    Scheme.inferno;
    Scheme.plasma;
    Scheme.cividis;
    Scheme.turbo;
    Scheme.twilight;
    Scheme.okabe_ito;
    Scheme.tableau10;
  ]
  @ List.map (fun (_, s, _) -> s) brewer_sequential
  @ List.map (fun (_, s, _) -> s) brewer_diverging
  @ List.map (fun (_, s, _) -> s) brewer_qualitative

let brewer = List.map (fun (_, s, _) -> s) (brewer_sequential @ brewer_diverging)

(* Generators *)

let unit =
  Gen.frequency [ (3, Gen.float_range 0. 1.); (1, Gen.of_list [ 0.; 0.5; 1. ]) ]

let gen_color =
  Gen.with_pp pp_color
    (Gen.map
       (fun ((r, g, b), alpha) -> Color.v ~alpha r g b)
       (Gen.pair
          (Gen.triple unit unit unit)
          (Gen.frequency [ (3, Gen.constant 1.); (1, unit) ])))

let gen_colors = Gen.array ~size:(Gen.int_range 1 12) gen_color
let gen_named = Gen.of_list ~pp:Scheme.pp named
let gen_brewer = Gen.of_list ~pp:Scheme.pp brewer

let gen_scheme =
  let base =
    Gen.one_of
      [
        gen_named;
        Gen.map Scheme.ramp gen_colors;
        Gen.map Scheme.palette gen_colors;
      ]
  in
  Gen.with_pp Scheme.pp
    (Gen.map
       (fun (s, reversed) -> if reversed then Scheme.reverse s else s)
       (Gen.pair base Gen.bool))

(* A scheme with a value drawn to reach every case of the index: bin boundaries
   and their neighbours, the ends, values beyond them, infinities and [nan]. *)
let gen_reading =
  let pp ppf (s, u) = Format.fprintf ppf "(%a, %h)" Scheme.pp s u in
  Gen.with_pp pp
    (Gen.bind gen_scheme (fun s ->
         let n = Array.length (Scheme.table s) in
         let boundary =
           Gen.map
             (fun (i, nudge) ->
               let u = Float.of_int i /. Float.of_int n in
               match nudge with 0 -> Float.pred u | 1 -> u | _ -> Float.succ u)
             (Gen.pair (Gen.int_range 0 n) (Gen.int_range 0 2))
         in
         Gen.map
           (fun u -> (s, u))
           (Gen.frequency
              [
                (4, boundary);
                (2, Gen.float_range (-0.5) 1.5);
                (1, Gen.any_float);
              ])))

(* The colour [s] paints [u] with, by the index the continuous reading states:
   the floor of [N *. u], clamped. *)
let gathered s u =
  let t = Scheme.table s in
  let n = Array.length t in
  let i = Float.floor (Float.of_int n *. u) in
  if i < 0. then t.(0)
  else if i > Float.of_int (n - 1) then t.(n - 1)
  else t.(int_of_float i)

(* Making schemes *)

let rbw = [| Color.blue; Color.white; Color.red |]

let ramp_stops cs =
  let m = Array.length cs in
  let t = Scheme.table (Scheme.ramp cs) in
  equal int 256 (Array.length t);
  for i = 0 to 255 do
    if i * (m - 1) mod 255 = 0 then
      equal
        ~msg:(Printf.sprintf "entry %d" i)
        color
        cs.(i * (m - 1) / 255)
        t.(i)
  done

let ramp_copies () =
  let cs = [| Color.black; Color.white |] in
  let s = Scheme.ramp cs in
  cs.(0) <- Color.red;
  equal colors [| Color.black; Color.white |] (Scheme.colors 2 s)

let palette_copies () =
  let cs = [| Color.black; Color.white |] in
  let s = Scheme.palette cs in
  cs.(0) <- Color.red;
  equal colors [| Color.black; Color.white |] (Scheme.table s)

let making =
  group "making"
    [
      test "ramp raises on no colours" (fun () ->
          invalid (fun () -> Scheme.ramp [||]));
      test "palette raises on no colours" (fun () ->
          invalid (fun () -> Scheme.palette [||]));
      prop "a ramp's table holds its colours where an entry falls on them"
        gen_colors ramp_stops;
      prop "a ramp's table is its reading in 256 classes" gen_colors (fun cs ->
          let s = Scheme.ramp cs in
          equal colors (Scheme.colors 256 s) (Scheme.table s));
      prop "a ramp through 256 colours has them as its table"
        (Gen.array ~size:(Gen.constant 256) gen_color)
        (fun cs -> equal colors cs (Scheme.table (Scheme.ramp cs)));
      test "a ramp copies its colours" ramp_copies;
      prop "a palette's table is its colours" gen_colors (fun cs ->
          equal colors cs (Scheme.table (Scheme.palette cs)));
      test "a palette copies its colours" palette_copies;
      prop "reverse is an involution" gen_scheme
        (Law.involutive scheme Scheme.reverse);
      prop "reverse reverses the table" gen_scheme (fun s ->
          equal colors (rev (Scheme.table s)) (Scheme.table (Scheme.reverse s)));
    ]

(* Continuous reading *)

let four = [| Color.black; Color.red; Color.green; Color.blue |]

let index_cases =
  [
    (0., 0);
    (Float.pred 0.25, 0);
    (0.25, 1);
    (Float.succ 0.25, 1);
    (0.5, 2);
    (Float.pred 0.75, 2);
    (0.75, 3);
    (Float.pred 1., 3);
    (1., 3);
    (-0., 0);
    (-0.5, 0);
    (neg_infinity, 0);
    (1.5, 3);
    (infinity, 3);
    (Float.min_float, 0);
    (Float.max_float, 3);
    (-.Float.max_float, 0);
  ]

let index_case (u, i) =
  equal color four.(i) (Scheme.color (Scheme.palette four) u)

let viridis_bins () =
  let t = Scheme.table Scheme.viridis in
  equal color t.(128) (Scheme.color Scheme.viridis 0.5);
  equal color t.(127) (Scheme.color Scheme.viridis (Float.pred 0.5));
  equal color t.(255) (Scheme.color Scheme.viridis (Float.pred 1.));
  equal color t.(0) (Scheme.color Scheme.viridis (Float.pred (1. /. 256.)));
  equal color t.(1) (Scheme.color Scheme.viridis (1. /. 256.))

let reversed_reading () =
  let t = Scheme.table Scheme.viridis in
  let r = Scheme.reverse Scheme.viridis in
  equal color t.(255) (Scheme.color r 0.);
  equal color t.(127) (Scheme.color r 0.5);
  equal color t.(0) (Scheme.color r 1.)

let table_is_fresh () =
  let t = Scheme.table Scheme.viridis in
  let first = t.(0) in
  t.(0) <- Color.red;
  equal color first (Scheme.table Scheme.viridis).(0)

let continuous =
  group "continuous reading"
    [
      cases "palette of four bins"
        ~name:(fun (u, i) -> Printf.sprintf "%h is in bin %d" u i)
        index_cases index_case;
      test "viridis's bins split at multiples of 1/256" viridis_bins;
      test "nan paints transparent by default" (fun () ->
          equal color Color.transparent (Scheme.color Scheme.viridis nan));
      test "nan paints the unknown colour" (fun () ->
          equal color Color.red
            (Scheme.color ~unknown:Color.red Scheme.viridis nan));
      prop "color agrees with the table gathered by the stated index"
        gen_reading (fun (s, u) ->
          let expected =
            if Float.is_nan u then Color.transparent else gathered s u
          in
          equal color expected (Scheme.color s u));
      test "a reversed scheme reads its table backwards" reversed_reading;
      test "table is a fresh array" table_is_fresh;
      cases "table lengths"
        ~name:(fun (s, n) -> Format.asprintf "%a has %d" Scheme.pp s n)
        [
          (Scheme.viridis, 256);
          (Scheme.turbo, 256);
          (Scheme.blues, 256);
          (Scheme.rdbu, 256);
          (Scheme.twilight, 510);
          (Scheme.okabe_ito, 8);
          (Scheme.paired, 12);
          (Scheme.ramp [| Color.red |], 256);
          (Scheme.palette [| Color.red |], 1);
        ]
        (fun (s, n) -> equal int n (Array.length (Scheme.table s)));
    ]

(* Discrete reading *)

let colors_is_fresh () =
  let cs = Scheme.colors 3 Scheme.blues in
  let first = cs.(0) in
  cs.(0) <- Color.red;
  equal color first (Scheme.colors 3 Scheme.blues).(0)

let palette_cycles () =
  let p = of_hex okabe_ito in
  equal colors
    (Array.append p [| p.(0); p.(1) |])
    (Scheme.colors 10 Scheme.okabe_ito)

(* A reversed palette is read from its reversed table: a class keeps its colour
   whatever the number of classes. *)
let reversed_palette_classes (p, (n, m)) =
  let r = Scheme.reverse p in
  let a = Scheme.colors n r and b = Scheme.colors m r in
  for i = 0 to Int.min n m - 1 do
    equal ~msg:(Printf.sprintf "class %d" i) color a.(i) b.(i)
  done;
  let t = rev (Scheme.table p) in
  Array.iteri
    (fun i c ->
      equal ~msg:(Printf.sprintf "class %d" i) color t.(i mod Array.length t) c)
    a

let reversed_okabe_ito () =
  let r = Scheme.reverse Scheme.okabe_ito in
  let purple = (of_hex "cc79a7").(0) in
  equal color purple (Scheme.colors 3 r).(0);
  equal color purple (Scheme.colors 8 r).(0);
  equal color purple (Scheme.colors 10 r).(0)

let gen_palette =
  Gen.one_of
    [
      Gen.of_list ~pp:Scheme.pp
        (Scheme.okabe_ito :: Scheme.tableau10
        :: List.map (fun (_, s, _) -> s) brewer_qualitative);
      Gen.with_pp Scheme.pp (Gen.map Scheme.palette gen_colors);
    ]

let brewer_designed (name, s, specs) =
  List.iteri
    (fun k spec ->
      equal
        ~msg:(Printf.sprintf "%s, %d classes" name (k + 3))
        (array hex)
        (to_hex (of_hex spec))
        (to_hex (Scheme.colors (k + 3) s)))
    specs

let brewer_below_three ~diverging (name, s, specs) =
  let t3 = of_hex (List.hd specs) in
  equal ~msg:(name ^ ", one class") colors [| t3.(1) |] (Scheme.colors 1 s);
  let two = if diverging then [| t3.(0); t3.(2) |] else [| t3.(1); t3.(2) |] in
  equal ~msg:(name ^ ", two classes") colors two (Scheme.colors 2 s)

(* Past its largest table a Brewer scheme reads as the ramp through that table:
   with twice as many classes less one, the even classes are its colours. *)
let brewer_past_largest (name, s, specs) =
  let largest = of_hex (List.nth specs (List.length specs - 1)) in
  let k = Array.length largest in
  let ramp = Scheme.ramp largest in
  equal ~msg:name colors (Scheme.colors (k + 1) ramp) (Scheme.colors (k + 1) s);
  let wide = Scheme.colors ((2 * k) - 1) s in
  Array.iteri
    (fun i c ->
      equal
        ~msg:(Printf.sprintf "%s, class %d" name (2 * i))
        color c
        wide.(2 * i))
    largest

let brewer_reversed (s, n) =
  equal colors (rev (Scheme.colors n s)) (Scheme.colors n (Scheme.reverse s))

let brewer_continuous (name, s, specs) =
  let largest = of_hex (List.nth specs (List.length specs - 1)) in
  equal ~msg:name colors (Scheme.table (Scheme.ramp largest)) (Scheme.table s)

let ramp_classes () =
  equal colors rbw (Scheme.colors 3 (Scheme.ramp rbw));
  equal colors [| Color.white |] (Scheme.colors 1 (Scheme.ramp rbw));
  equal colors
    [| Color.mix 0.5 Color.black Color.white |]
    (Scheme.colors 1 (Scheme.ramp [| Color.black; Color.white |]));
  equal colors
    [| Color.black; Color.mix 0.5 Color.black Color.white; Color.white |]
    (Scheme.colors 3 (Scheme.ramp [| Color.black; Color.white |]));
  equal colors [| Color.red; Color.red |]
    (Scheme.colors 2 (Scheme.ramp [| Color.red |]))

let gen_ramp_count = Gen.pair gen_colors (Gen.int_range 0 40)

let listed_classes () =
  let t = Scheme.table Scheme.viridis in
  equal colors [| t.(128) |] (Scheme.colors 1 Scheme.viridis);
  equal colors [| t.(0); t.(255) |] (Scheme.colors 2 Scheme.viridis);
  equal colors [| t.(0); t.(128); t.(255) |] (Scheme.colors 3 Scheme.viridis);
  equal colors
    [| t.(255); t.(127); t.(0) |]
    (Scheme.colors 3 (Scheme.reverse Scheme.viridis))

let cyclic_classes () =
  let t = Scheme.table Scheme.twilight in
  equal colors [| t.(0); t.(255) |] (Scheme.colors 2 Scheme.twilight);
  equal colors
    [| t.(0); t.(127); t.(255); t.(382) |]
    (Scheme.colors 4 Scheme.twilight);
  equal colors
    [| t.(509); t.(382); t.(254); t.(127) |]
    (Scheme.colors 4 (Scheme.reverse Scheme.twilight))

let discrete =
  group "discrete reading"
    [
      test "colors raises on a negative count" (fun () ->
          invalid (fun () -> Scheme.colors (-1) Scheme.viridis));
      prop "colors 0 is empty" gen_scheme (fun s ->
          equal colors [||] (Scheme.colors 0 s));
      prop "colors n has n colours"
        (Gen.pair gen_scheme (Gen.int_range 0 40))
        (fun (s, n) -> equal int n (Array.length (Scheme.colors n s)));
      test "colors is a fresh array" colors_is_fresh;
      test "a palette starts again after its last colour" palette_cycles;
      prop "a reversed palette keeps each class's colour"
        (Gen.pair gen_palette
           (Gen.pair (Gen.int_range 0 30) (Gen.int_range 0 30)))
        reversed_palette_classes;
      test "class 0 of reversed okabe_ito is reddish purple" reversed_okabe_ito;
      cases "Brewer sequential designed tables"
        ~name:(fun (n, _, _) -> n)
        brewer_sequential brewer_designed;
      cases "Brewer diverging designed tables"
        ~name:(fun (n, _, _) -> n)
        brewer_diverging brewer_designed;
      cases "Brewer sequential below three classes"
        ~name:(fun (n, _, _) -> n)
        brewer_sequential
        (brewer_below_three ~diverging:false);
      cases "Brewer diverging below three classes"
        ~name:(fun (n, _, _) -> n)
        brewer_diverging
        (brewer_below_three ~diverging:true);
      cases "Brewer past the largest table"
        ~name:(fun (n, _, _) -> n)
        (brewer_sequential @ brewer_diverging)
        brewer_past_largest;
      cases "Brewer continuous reading is the ramp through the largest table"
        ~name:(fun (n, _, _) -> n)
        (brewer_sequential @ brewer_diverging)
        brewer_continuous;
      prop "a reversed Brewer scheme reverses its classes"
        (Gen.pair gen_brewer (Gen.int_range 0 30))
        brewer_reversed;
      test "a ramp's classes fall on its colours" ramp_classes;
      prop "a ramp through m colours gives them in m classes" gen_colors
        (fun cs ->
          equal colors cs (Scheme.colors (Array.length cs) (Scheme.ramp cs)));
      prop "a reversed ramp reverses its classes" gen_ramp_count (fun (cs, n) ->
          let s = Scheme.ramp cs in
          equal colors
            (rev (Scheme.colors n s))
            (Scheme.colors n (Scheme.reverse s)));
      test "a table scheme samples its ends and middle" listed_classes;
      test "a cyclic scheme samples at multiples of 1/n" cyclic_classes;
    ]

(* The catalogue *)

let d3_table (name, s, spec) =
  equal ~msg:name (array hex)
    (Array.init (String.length spec / 6) (fun i -> String.sub spec (6 * i) 6))
    (to_hex (Scheme.table s))

let published_ends (name, s, n, first, last) =
  let t = Scheme.table s in
  let v (r, g, b) = Color.v r g b in
  equal ~msg:name int n (Array.length t);
  equal ~msg:(name ^ ", first") color (v first) t.(0);
  equal ~msg:(name ^ ", last") color (v last) t.(n - 1)

let monotone ~msg ~rising cs =
  let ls = Array.map lightness cs in
  for i = 0 to Array.length ls - 2 do
    let msg = Printf.sprintf "%s, %d to %d" msg i (i + 1) in
    if rising then less ~msg float_exact ~than:ls.(i + 1) ls.(i)
    else greater ~msg float_exact ~than:ls.(i + 1) ls.(i)
  done

let brewer_sequential_lightness (name, s, specs) =
  monotone ~msg:name ~rising:false (Scheme.table s);
  List.iteri
    (fun k _ ->
      monotone
        ~msg:(Printf.sprintf "%s, %d classes" name (k + 3))
        ~rising:false
        (Scheme.colors (k + 3) s))
    specs

(* The designed diverging tables of an odd number of classes are lightest at
   their middle class. *)
let brewer_diverging_lightness (name, s, specs) =
  List.iteri
    (fun k _ ->
      let n = k + 3 in
      if n mod 2 = 1 then begin
        let cs = Scheme.colors n s in
        let msg = Printf.sprintf "%s, %d classes" name n in
        monotone ~msg ~rising:true (Array.sub cs 0 ((n / 2) + 1));
        monotone ~msg ~rising:false (Array.sub cs (n / 2) ((n / 2) + 1))
      end)
    specs

let catalogue =
  group "catalogue"
    [
      cases "perceptually uniform tables match d3's"
        ~name:(fun (n, _, _) -> n)
        [
          ("viridis", Scheme.viridis, d3_viridis);
          ("magma", Scheme.magma, d3_magma);
          ("inferno", Scheme.inferno, d3_inferno);
          ("plasma", Scheme.plasma, d3_plasma);
        ]
        d3_table;
      cases "tables end on their published colours"
        ~name:(fun (n, _, _, _, _) -> n)
        [
          ( "cividis",
            Scheme.cividis,
            256,
            (0., 0.135112, 0.304751),
            (0.995737, 0.909344, 0.217772) );
          ( "turbo",
            Scheme.turbo,
            256,
            (0.18995, 0.07176, 0.23217),
            (0.4796, 0.01583, 0.01055) );
          ( "twilight",
            Scheme.twilight,
            510,
            (0.8857501584075443, 0.8500092494306783, 0.8879736506427196),
            (0.8857115512284565, 0.8500218611585632, 0.8857253899008712) );
        ]
        published_ends;
      cases "qualitative palettes"
        ~name:(fun (n, _, _) -> n)
        ([
           ("okabe_ito", Scheme.okabe_ito, okabe_ito);
           ("tableau10", Scheme.tableau10, tableau10);
         ]
        @ brewer_qualitative)
        d3_table;
      cases "perceptually uniform lightness rises"
        ~name:(Format.asprintf "%a" Scheme.pp)
        [
          Scheme.viridis;
          Scheme.magma;
          Scheme.inferno;
          Scheme.plasma;
          Scheme.cividis;
        ] (fun s -> monotone ~msg:"" ~rising:true (Scheme.table s));
      cases "Brewer sequential lightness falls"
        ~name:(fun (n, _, _) -> n)
        brewer_sequential brewer_sequential_lightness;
      cases "Brewer diverging tables are lightest at their middle"
        ~name:(fun (n, _, _) -> n)
        brewer_diverging brewer_diverging_lightness;
    ]

(* Colour vision deficiency *)

(* [simulate] at severities 1, 0.5 and 0.25 on chosen colours, computed with
   numpy from the published matrices and colour-science's sRGB transfer
   functions. Severity 0.25 takes the mean of the matrices at 0.2 and 0.3:
   colour-science's own interpolation extrapolates from the tabulated severity
   above. *)
let reference =
  Scheme.
    [
      (Protan, 1.0, "#e69f00", (0.7256835180656197, 0.6368090996896602, 0.0));
      ( Protan,
        1.0,
        "#56b4e9",
        (0.6077318809138621, 0.7016688606180559, 0.9235861750161118) );
      ( Protan,
        1.0,
        "#cc79a7",
        (0.5000020800082335, 0.5447448128199099, 0.6614045396012909) );
      (Protan, 1.0, "#ff0000", (0.4266084717107862, 0.37265427742344537, 0.0));
      (Protan, 1.0, "#00ff00", (0.9999999999999999, 0.8994280662235058, 0.0));
      (Protan, 1.0, "#0000ff", (0.0, 0.34786682616620246, 0.9999999999999999));
      ( Protan,
        1.0,
        "#808080",
        (0.5019610163806513, 0.5019607843137255, 0.5019607843137255) );
      (Protan, 0.5, "#e69f00", (0.79659823222834, 0.6396799613360813, 0.0));
      ( Protan,
        0.5,
        "#56b4e9",
        (0.5277557921362921, 0.6975637407178212, 0.9193659884132122) );
      ( Protan,
        0.5,
        "#cc79a7",
        (0.6289100761422508, 0.5285019985885628, 0.6561649211883508) );
      (Protan, 0.5, "#ff0000", (0.7070293407072645, 0.3367733269637425, 0.0));
      (Protan, 0.5, "#00ff00", (0.8431568211671276, 0.9291403717967618, 0.0));
      (Protan, 0.5, "#0000ff", (0.0, 0.27373726744684695, 0.9999999999999999));
      ( Protan,
        0.5,
        "#808080",
        (0.5019607843137255, 0.5019607843137255, 0.5019607843137255) );
      (Protan, 0.25, "#e69f00", (0.8427649506988457, 0.635462194261293, 0.0));
      ( Protan,
        0.25,
        "#56b4e9",
        (0.4598333923290971, 0.6992510295256622, 0.9167867301230859) );
      ( Protan,
        0.25,
        "#cc79a7",
        (0.7060637390927305, 0.5095394398628927, 0.6548586994800066) );
      (Protan, 0.25, "#ff0000", (0.844788346999637, 0.2728550935712972, 0.0));
      (Protan, 0.25, "#00ff00", (0.6653774731059253, 0.9568444879829533, 0.0));
      (Protan, 0.25, "#0000ff", (0.0, 0.20557358725853903, 0.9999999999999999));
      ( Protan,
        0.25,
        "#808080",
        (0.5019609003472053, 0.5019609003472053, 0.5019607843137255) );
      ( Deutan,
        1.0,
        "#e69f00",
        (0.7912188809936646, 0.7047548394660739, 0.06602160686294209) );
      ( Deutan,
        1.0,
        "#56b4e9",
        (0.5283559905891689, 0.6434489626297191, 0.9103195548911502) );
      ( Deutan,
        1.0,
        "#cc79a7",
        (0.582287239095612, 0.5978370960093383, 0.6464481345362898) );
      (Deutan, 1.0, "#ff0000", (0.6400595551963536, 0.5658069412154982, 0.0));
      ( Deutan,
        1.0,
        "#00ff00",
        (0.936051045605102, 0.8392477353639614, 0.22919186560921978) );
      (Deutan, 1.0, "#0000ff", (0.0, 0.2411713587834845, 0.9861943658413105));
      ( Deutan,
        1.0,
        "#808080",
        (0.5019607843137255, 0.5019605522466644, 0.5019610163806513) );
      ( Deutan,
        0.5,
        "#e69f00",
        (0.8232259120717721, 0.676174538422451, 0.015748725190612094) );
      ( Deutan,
        0.5,
        "#56b4e9",
        (0.4865780190531073, 0.6678036525513844, 0.9126042421423254) );
      ( Deutan,
        0.5,
        "#cc79a7",
        (0.6551862110440684, 0.5595063327226766, 0.6490604491592873) );
      (Deutan, 0.5, "#ff0000", (0.7658123104948851, 0.46337300695061817, 0.0));
      ( Deutan,
        0.5,
        "#00ff00",
        (0.802318763520275, 0.8971284931841, 0.18022732834488495) );
      (Deutan, 0.5, "#0000ff", (0.0, 0.21078761035829408, 0.9925500758224352));
      ( Deutan,
        0.5,
        "#808080",
        (0.5019607843137255, 0.5019607843137255, 0.5019610163806513) );
      (Deutan, 0.25, "#e69f00", (0.8536204850284747, 0.6549818006273773, 0.0));
      ( Deutan,
        0.25,
        "#56b4e9",
        (0.4386815483725052, 0.684062148071515, 0.9133923867467735) );
      ( Deutan,
        0.25,
        "#cc79a7",
        (0.7145860965049227, 0.5274484339658774, 0.6513432462495894) );
      (Deutan, 0.25, "#ff0000", (0.864017362825895, 0.36225759881676206, 0.0));
      ( Deutan,
        0.25,
        "#00ff00",
        (0.6472157480000154, 0.9398261166647122, 0.13309867115553453) );
      (Deutan, 0.25, "#0000ff", (0.0, 0.16618605904124847, 0.9960509913262744));
      ( Deutan,
        0.25,
        "#808080",
        (0.5019607843137255, 0.5019607843137255, 0.5019607843137255) );
      ( Tritan,
        1.0,
        "#e69f00",
        (0.9853012239022497, 0.5474988563048454, 0.5305798426305184) );
      (Tritan, 1.0, "#56b4e9", (0.0, 0.7597299680769783, 0.7758004826455593));
      ( Tritan,
        1.0,
        "#cc79a7",
        (0.8402780719645178, 0.4704045797114998, 0.5395432992601948) );
      (Tritan, 1.0, "#ff0000", (0.9999999999999999, 0.0, 0.058385974375206574));
      (Tritan, 1.0, "#00ff00", (0.0, 0.9689475205967898, 0.849616272166806));
      (Tritan, 1.0, "#0000ff", (0.0, 0.4203799863961778, 0.5872787961247297));
      ( Tritan,
        1.0,
        "#808080",
        (0.5019607843137255, 0.5019607843137255, 0.5019607843137255) );
      ( Tritan,
        0.5,
        "#e69f00",
        (0.913474641518838, 0.6075846043308872, 0.3341053818141741) );
      ( Tritan,
        0.5,
        "#56b4e9",
        (0.29523237773195277, 0.7191233882121, 0.86562558031111) );
      ( Tritan,
        0.5,
        "#cc79a7",
        (0.7991020723679276, 0.4822505827406824, 0.6174219368223378) );
      (Tritan, 0.5, "#ff0000", (0.9999999999999999, 0.0, 0.07340030160382305));
      ( Tritan,
        0.5,
        "#00ff00",
        (0.17934100273321993, 0.9815220464654156, 0.5358218174619684) );
      (Tritan, 0.5, "#0000ff", (0.0, 0.24174578899873472, 0.8781750247471142));
      ( Tritan,
        0.5,
        "#808080",
        (0.5019607843137255, 0.5019607843137255, 0.5019607843137255) );
      ( Tritan,
        0.25,
        "#e69f00",
        (0.8850041743030294, 0.6258434809867117, 0.2584345593281348) );
      ( Tritan,
        0.25,
        "#56b4e9",
        (0.3778436459992354, 0.70574081173152, 0.8859848768467945) );
      ( Tritan,
        0.25,
        "#cc79a7",
        (0.7715793038553951, 0.49390825328732596, 0.6379139392867238) );
      ( Tritan,
        0.25,
        "#ff0000",
        (0.9550589307629441, 0.18431534082824902, 0.1189448997814305) );
      ( Tritan,
        0.25,
        "#00ff00",
        (0.3966902452076125, 0.9746622301209543, 0.39078521207903566) );
      (Tritan, 0.25, "#0000ff", (0.0, 0.18368804638242772, 0.9358754170253042));
      ( Tritan,
        0.25,
        "#808080",
        (0.5019607843137255, 0.5019607843137255, 0.5019607843137255) );
    ]

let reference_case (d, severity, h, (r, g, b)) =
  let c = Scheme.simulate ~severity d (of_hex (String.sub h 1 6)).(0) in
  let near = float 1e-12 in
  equal ~msg:"red" near r (Color.r c);
  equal ~msg:"green" near g (Color.g c);
  equal ~msg:"blue" near b (Color.b c)

let name_deficiency = function
  | Scheme.Protan -> "protan"
  | Scheme.Deutan -> "deutan"
  | Scheme.Tritan -> "tritan"

let gen_deficiency = Gen.of_list [ Scheme.Protan; Scheme.Deutan; Scheme.Tritan ]

let oklab_distance c c' =
  let l, a, b = Color.to_oklab c and l', a', b' = Color.to_oklab c' in
  Float.sqrt (((l -. l') ** 2.) +. ((a -. a') ** 2.) +. ((b -. b') ** 2.))

let okabe_ito_apart d =
  let cs = Array.map (Scheme.simulate d) (Scheme.colors 8 Scheme.okabe_ito) in
  for i = 0 to 7 do
    for j = i + 1 to 7 do
      greater
        ~msg:(Printf.sprintf "colours %d and %d" i j)
        float_exact ~than:0.05
        (oklab_distance cs.(i) cs.(j))
    done
  done

let identity_at_zero (d, c) =
  let c' = Scheme.simulate ~severity:0. d c in
  let near = float 1e-12 in
  equal ~msg:"red" near (Color.r c) (Color.r c');
  equal ~msg:"green" near (Color.g c) (Color.g c');
  equal ~msg:"blue" near (Color.b c) (Color.b c');
  equal ~msg:"alpha" float_exact (Color.alpha c) (Color.alpha c')

let cvd =
  group "colour vision deficiency"
    [
      cases "matches the published model"
        ~name:(fun (d, s, h, _) ->
          Printf.sprintf "%s %g %s" (name_deficiency d) s h)
        reference reference_case;
      prop "severity 0 is the identity up to rounding"
        (Gen.pair gen_deficiency gen_color)
        identity_at_zero;
      prop "severity defaults to 1" (Gen.pair gen_deficiency gen_color)
        (fun (d, c) ->
          equal color (Scheme.simulate ~severity:1. d c) (Scheme.simulate d c));
      prop "alpha is kept"
        (Gen.triple gen_deficiency gen_color (Gen.float_range 0. 1.))
        (fun (d, c, severity) ->
          equal float_exact (Color.alpha c)
            (Color.alpha (Scheme.simulate ~severity d c)));
      cases "raises on a severity outside [0, 1]" ~name:(Printf.sprintf "%h")
        [ -0.1; Float.pred 0.; Float.succ 1.; nan; infinity; neg_infinity ]
        (fun severity ->
          invalid (fun () -> Scheme.simulate ~severity Scheme.Deutan Color.red));
      cases "Okabe and Ito's colours stay apart" ~name:name_deficiency
        [ Scheme.Protan; Scheme.Deutan; Scheme.Tritan ]
        okabe_ito_apart;
    ]

(* Comparing and formatting *)

let gen_pair = Gen.pair gen_scheme gen_scheme

let unequal_cases =
  [
    ("two named schemes", Scheme.viridis, Scheme.magma);
    ("a scheme and its reverse", Scheme.rdbu, Scheme.reverse Scheme.rdbu);
    ( "a ramp through viridis's table and viridis",
      Scheme.ramp (Scheme.table Scheme.viridis),
      Scheme.viridis );
    ( "a ramp and a palette of the same colours",
      Scheme.ramp rbw,
      Scheme.palette rbw );
    ( "ramps through different colours",
      Scheme.ramp rbw,
      Scheme.ramp [| Color.blue; Color.red |] );
    ( "palettes of different colours",
      Scheme.palette rbw,
      Scheme.palette (rev rbw) );
    ( "a palette of a named palette's colours and that palette",
      Scheme.palette (Scheme.table Scheme.okabe_ito),
      Scheme.okabe_ito );
  ]

let printing () =
  let print s = Format.asprintf "%a" Scheme.pp s in
  expect
    (String.concat "\n"
       (List.map print
          [
            Scheme.viridis;
            Scheme.reverse Scheme.rdbu;
            Scheme.ramp [| Color.black; Color.white |];
            Scheme.reverse (Scheme.palette [| Color.red; Color.blue |]);
            Scheme.okabe_ito;
            Scheme.twilight;
          ]))
  @@ __POS_OF__
       {|
    viridis
    reverse(rdbu)
    ramp(#000000 #ffffff)
    reverse(palette(#ff0000 #0000ff))
    okabe_ito
    twilight
    |}

let comparing =
  group "comparing and formatting"
    [
      prop "equal is an equivalence" gen_pair (Law.equivalence scheme);
      test "equal schemes made alike" (fun () ->
          equal scheme (Scheme.ramp rbw) (Scheme.ramp (Array.copy rbw));
          equal scheme (Scheme.palette rbw) (Scheme.palette (Array.copy rbw));
          equal scheme Scheme.viridis Scheme.viridis);
      cases "unequal schemes"
        ~name:(fun (n, _, _) -> n)
        unequal_cases
        (fun (_, s, s') -> not_equal scheme s s');
      test "pp names named schemes and shows made ones" printing;
    ]

let () =
  exit
    (run "Scheme" [ making; continuous; discrete; catalogue; cvd; comparing ])
