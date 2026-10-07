# `01-units`

A unit is an exact value with one canonical text, and a quantity is a tensor in
a unit. This example builds units, converts between them, and computes with
quantities.

```bash
dune exec contrib/ymir/examples/01-units/main.exe
```

## What You'll Learn

- Building units from SI units, prefixes and the algebra `*`, `/`, `**`
- Reading a unit's canonical text, and why two spellings of one unit are equal
- Rounding an exact conversion factor once with `Unit.ratio`
- Making quantities with `Quantity.v` and reading them with `Quantity.value`
- Arithmetic that computes the unit: `Quantity.mul`, `Quantity.add`

## Key Functions

| Function                       | Purpose                                             |
| ------------------------------ | --------------------------------------------------- |
| `Unit.(kilo metre / second)`   | A unit from the SI's units and prefixes             |
| `Unit.to_string`               | The unit's canonical text                           |
| `Unit.ratio dtype u w`         | The factor from `u` to `w`, rounded once to `dtype` |
| `Quantity.v u x`               | The tensor `x` in unit `u`                          |
| `Quantity.value u q`           | The payload of `q` in unit `u`                      |
| `Quantity.mul`, `Quantity.add` | Arithmetic on quantities                            |

## Next Steps

Continue to [02-constants-and-names](../02-constants-and-names/).
