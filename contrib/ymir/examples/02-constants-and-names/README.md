# `02-constants-and-names`

The SI's defining constants are exact units. Measured constants come from a
CODATA release the program names. Vocabularies give units their symbols.

```bash
dune exec contrib/ymir/examples/02-constants-and-names/main.exe
```

## What You'll Learn

- Exact constants as units: `Unit.planck`, `Unit.speed_of_light`
- Measured constants from `Codata.v2022`, rounded once to a dtype
- Declaring a constant with `Constant.v` in the published notation
- Spelling units with `Vocabulary.si` and a vocabulary of your own
- Reading prefixed symbols with `Vocabulary.lookup`

## Key Functions

| Function                           | Purpose                                       |
| ---------------------------------- | --------------------------------------------- |
| `Codata.newtonian_gravitation`     | G from a CODATA release                       |
| `Constant.quantity dtype k`        | A constant's value as a scalar quantity       |
| `Constant.uncertainty dtype k`     | Its standard uncertainty                      |
| `Vocabulary.v`, `Vocabulary.union` | Build a vocabulary                            |
| `Vocabulary.pp voc`                | Format a unit with a vocabulary's symbols     |
| `Vocabulary.lookup voc s`          | The unit a symbol names, SI prefixes included |

## Next Steps

Continue to [03-frames-and-directions](../03-frames-and-directions/).
