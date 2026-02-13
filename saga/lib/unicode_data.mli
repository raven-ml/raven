(* Do not edit. Generated from uucp 17.0.0 via saga/gen/gen_unicode_data.ml. *)

[@@@ocamlformat "disable"]

type general_category =
  [ `Cc | `Cf | `Cn | `Co | `Cs
  | `Ll | `Lm | `Lo | `Lt | `Lu
  | `Mc | `Me | `Mn | `Nd | `Nl
  | `No | `Pc | `Pd | `Pe | `Pf
  | `Pi | `Po | `Ps | `Sc | `Sk
  | `Sm | `So | `Zl | `Zp | `Zs ]

type script =
  [ `Adlm | `Aghb | `Ahom | `Arab | `Armi | `Armn | `Avst | `Bali
  | `Bamu | `Bass | `Batk | `Beng | `Berf | `Bhks | `Bopo | `Brah
  | `Brai | `Bugi | `Buhd | `Cakm | `Cans | `Cari | `Cham | `Cher
  | `Chrs | `Copt | `Cpmn | `Cprt | `Cyrl | `Deva | `Diak | `Dogr
  | `Dsrt | `Dupl | `Egyp | `Elba | `Elym | `Ethi | `Gara | `Geor
  | `Glag | `Gong | `Gonm | `Goth | `Gran | `Grek | `Gujr | `Gukh
  | `Guru | `Hang | `Hani | `Hano | `Hatr | `Hebr | `Hira | `Hluw
  | `Hmng | `Hmnp | `Hung | `Ital | `Java | `Kali | `Kana | `Kawi
  | `Khar | `Khmr | `Khoj | `Kits | `Knda | `Krai | `Kthi | `Lana
  | `Laoo | `Latn | `Lepc | `Limb | `Lina | `Linb | `Lisu | `Lyci
  | `Lydi | `Mahj | `Maka | `Mand | `Mani | `Marc | `Medf | `Mend
  | `Merc | `Mero | `Mlym | `Modi | `Mong | `Mroo | `Mtei | `Mult
  | `Mymr | `Nagm | `Nand | `Narb | `Nbat | `Newa | `Nkoo | `Nshu
  | `Ogam | `Olck | `Onao | `Orkh | `Orya | `Osge | `Osma | `Ougr
  | `Palm | `Pauc | `Perm | `Phag | `Phli | `Phlp | `Phnx | `Plrd
  | `Prti | `Rjng | `Rohg | `Runr | `Samr | `Sarb | `Saur | `Sgnw
  | `Shaw | `Shrd | `Sidd | `Sidt | `Sind | `Sinh | `Sogd | `Sogo
  | `Sora | `Soyo | `Sund | `Sunu | `Sylo | `Syrc | `Tagb | `Takr
  | `Tale | `Talu | `Taml | `Tang | `Tavt | `Tayo | `Telu | `Tfng
  | `Tglg | `Thaa | `Thai | `Tibt | `Tirh | `Tnsa | `Todr | `Tols
  | `Toto | `Tutg | `Ugar | `Vaii | `Vith | `Wara | `Wcho | `Xpeo
  | `Xsux | `Yezi | `Yiii | `Zanb | `Zinh | `Zyyy | `Zzzz ]

val general_category : int -> general_category
val script : int -> script
val is_white_space : int -> bool
val is_alphabetic : int -> bool
val is_numeric : int -> bool
val case_fold : int -> int list
val grapheme_cluster : int -> int
