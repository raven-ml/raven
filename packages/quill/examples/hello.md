<!-- quill:cell id="c_21swpr9nmlzh" -->
# Hello Quill

A sample notebook to test the TUI.


<!-- quill:cell id="c_nt02ejs0jlrx" -->
```ocaml
let greeting = "Hello from Quill!"
let () = print_endline greeting
```

<!-- quill:cell id="c_vwh224nbgzms" -->
Some markdown text between cells. Try pressing **e** to edit the code above,
then **Escape** to exit, or **Ctrl-Enter** to run.


<!-- quill:cell id="c_1l7dzrlbekw8" -->
```ocaml
let square x = x * x

let () =
  List.iter
    (fun n -> Printf.printf "%d^2 = %d\n" n (square n))
    [1; 2; 3; 4; 5]
```

<!-- quill:cell id="c_q446fmuel0h4" -->
```ocaml
let rec fib n =
  if n <= 1 then n
  else fib (n - 1) + fib (n - 2)

let () =
  Printf.printf "fib(10) = %d\n" (fib 10)
```

<!-- quill:cell id="c_m4thd3m0t3xt" -->
## Math Equations

Text cells support LaTeX math. Inline math uses single dollars: the quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.

Display math uses double dollars:

$$\int_0^\infty e^{-x^2}\, dx = \frac{\sqrt{\pi}}{2}$$

$$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$$

<!-- quill:cell id="c_m4thc0d3c3ll" -->
```ocaml
(* The Euler identity: e^(i*pi) + 1 = 0 *)
let () =
  let open Complex in
  let e_i_pi = exp { re = 0.; im = Float.pi } in
  Printf.printf "e^(iπ) = %.4f + %.4fi\n" e_i_pi.re e_i_pi.im
```
