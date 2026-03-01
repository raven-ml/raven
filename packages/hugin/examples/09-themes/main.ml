(* Themes and context scaling.

   Themes control the entire visual appearance: colors, fonts, line widths.
   Context functions like Theme.talk scale everything up for presentations. *)

open Hugin

let () =
  let x = Nx.linspace Nx.float32 0. 10. 80 in
  let base =
    layers
      [
        line ~x ~y:(Nx.sin x) ~label:"sin" ();
        line ~x ~y:(Nx.cos x) ~label:"cos" ();
      ]
    |> legend
  in
  grid
    [
      [
        base |> with_theme Theme.default |> title "Default";
        base |> with_theme Theme.dark |> title "Dark";
      ];
      [
        base |> with_theme Theme.minimal |> title "Minimal";
        base |> with_theme (Theme.talk Theme.default) |> title "Talk";
      ];
    ]
  |> render_png "themes.png"
