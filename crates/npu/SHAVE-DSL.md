# The SHAVE kernel language

A subset of C for ACT-SHAVE kernels of the NPU (`NPU.md` §3), compiled by
`npu::shave::dsl`. A kernel is one C function over the SHAVE's vector types. The compiler packs its operations
into bundles, keeps every latency and hardware rule, fills branch delay
slots and assigns the machine's registers.

```
cargo build --release -p npu --example shave
shave build KERNEL.sk OUT.elf [-DNAME=VALUE ...] [--asm OUT.s]
```

`OUT.elf` is the linked kernel (`NPU.md` §3.9). `--asm` also writes the
assembly the compiler produced, and the tool prints the cycles per round of
every run-time loop. Sources end in `.sk`, and the kernels in `shave/` are the
models, namely `softmax_lse.sk`, `softmax_q8.sk` (u8 output, measured, not in a
shipped graph), `gnorm.sk` (f32 path, stack arrays, strides),
`trinv.sk` (run-time trip counts, one body compiled for two widths).

**What is C and what is not.** The syntax, the preprocessor directives, the
types' meaning, the operators, pointer arithmetic, `restrict`, the control
statements and their semantics are C's. It is not a C compiler. Every
operator and intrinsic is exactly one instruction (or a fixed pair), there
is one function, no structs, and a kernel that needs more registers than
the processor has does not compile. §10 lists what C has and this does not.

## 1. A kernel

```c
/* y[i] = exp(x[i] - 1) for N f16 values, 8 per vector */
#ifndef N
#define N 1024                  /* shave build ... -DN=4096 */
#endif

void controlled_act(const u32 *params) {
    const half8 *restrict x = (const half8 *)params[0];   /* input tensor */
    half8 *restrict y = (half8 *)params[9];               /* output tensor */
    const half8 one = 1.0;
    for (u32 i = 0; i < N / 8; i++) {
        *y = exp(*x - one);
        x++;
        y++;
    }
}
```

A file is preprocessor directives, constants, and one function `void
NAME(TYPE *PARAM)`: the kernel's entry `controlled_act`, whatever its name.
It returns at the end of its block. `/* */` and `//` are comments.

## 2. Types

| Type | Is | Register |
|---|---|---|
| `u32`, `unsigned` | 32-bit unsigned integer | `i` |
| `int` | 32-bit signed integer | `i` |
| `half` | one f16, in the low half of the register (the high half is undefined) | `i` |
| `float` | one f32, as its bits | `i` |
| `half8` | 8 x f16 | `v` |
| `float4` | 4 x f32 | `v` |
| `int4` | 4 x i32 | `v` |
| `T *` | pointer to any of these, and to `u8` / `void` (1 byte, for address arithmetic) | `i` |

- A variable is a register (`i0`-`i17`, `v0`-`v23`), never memory. The
  compiler puts variables that are not alive at the same time into the same
  machine register. At most 18 `i` and 24 `v` values can be alive at once
  (§6, "Registers").
- `const` on a `u32` / `int` with a constant value makes a compile-time
  constant (no register), as in `const u32 NVEC = N / 8;`, also in front of the
  function. In front of the function `const float` / `const half` with a
  constant value names a floating constant (§3). Elsewhere `const` is
  accepted and means nothing.
- Integers convert between `u32` and `int` freely. Everything else needs
  the same type on both sides. A pointer converts only by a cast.
- **Casts** change the type, never the bits. They work between pointers and
  integers, and between vector types (`(int4)x` is the same 128 bits, as in
  C's vector extension). There is no cast between a float type and an
  integer. The conversions are intrinsics (§5).
- Reading a variable that is not set on every path to that point is a
  compile error (with `if (c) a = 1; b = a;`, set `a` in both branches).

**Arrays.**

| Declaration | Is |
|---|---|
| `register half8 m[8];` | 8 registers. The index must be a compile-time constant (also a loop variable of an unrolled loop, §4). With an array of `restrict` pointers (`register half8 *restrict p[4];`), every element points to memory of its own |
| `float ssv[64];` | 64 floats **on the stack** (7 KB in all, rounded up to 16 bytes each). `ssv[k]` with a constant index loads / stores, and `ssv` is its address, a `float *` |

## 3. Constants and the preprocessor

| Directive | |
|---|---|
| `#define NAME value` | an integer constant (a constant expression, and macros cannot take arguments). `-DNAME=VALUE` on the command line defines one before the file is read |
| `#ifdef` `#ifndef` `#if expr` `#elif expr` `#else` `#endif` | as in C, and `#if` takes a constant expression (no `defined()`) |
| `#error text` | compilation fails with the text |
| `#pragma unroll`, `#pragma nounroll` | in front of a `for` (§4) |

So a parameter with a default is `#ifndef N` / `#define N 4096` / `#endif`,
as in C. A `-D` whose name the file never uses is an error.
`_Static_assert(condition, "text");` checks a condition on the constants,
in front of the function or inside it.

Constant expressions use C's operators on integers, `+ - * / % << >> & | ^
~ !`, the comparisons, `&& ||`. Integer constants are 64-bit while the
compiler computes and must fit 32 bits where they reach an instruction.
Floating constants (`1.5`, `-65504.0`, `2e-3`, `0.5f`) are allowed where a
`half`, `float`, `half8` or `float4` is expected, and a vector gets the value
in every lane. Expressions of them are computed by the compiler too, in
double precision, with `+ - * /`, unary `-`, the comparisons and `&& || !`
(which give an integer, so `_Static_assert(SCALE * D <= 1.0, ...)` works).
An integer beside a float converts as in C (`1.0 / D`, but `1 / D` is the
integer division). The value is rounded once, to the type it is used as.
`% << >> & | ^ ~` and casts to an integer do not take a float. A name for
one is `const float INVD = 1.0 / D;` in front of the function (`const half`
is the same, because the name has the value, not the type, and each use takes the
type expected there).

## 4. Statements

| Statement | |
|---|---|
| `T a, *restrict p = ..., m;` | declarations, anywhere in a block |
| `a = e;` `a += e;` (`-= *= &= \|= ^= <<= >>=`) `a++;` `a--;` `++a;` | assignment, where `a` is a variable, `p[k]` or `*p` |
| `if (c) S else S` | |
| `for (init; c; step) S` | |
| `while (c) S` | |
| `do S while (c);` | the cheapest loop, with one compare and branch per round |
| `{ ... }` | block, with its own scope |
| `barrier();` | everything before it is scheduled before everything after it (§6) |

Assignments are statements, not expressions (`a = b = 0`, `x[i++]` are not
there). In `for`, init and step may be several assignments separated by
commas.

**Conditions** compare two integers or pointers with `== != < <= > >=`, or
are a single integer (`if (n)` means `n != 0`). As in C the comparison is
unsigned if either side is `u32` or a pointer, else signed. `&&`, `||`, `!`
work on constants only, so nest the `if`s.

**Compile time or run time.** The compiler decides as an optimizing C
compiler would:

- With a constant condition, `if` compiles only the chosen branch. The
  other one is not even checked for registers or indices.
- `for` whose variable starts at a constant runs **at run time when the body
  allows it, else unrolled.** The body allows it when it does not need the
  variable as a constant (as a register array's index, a pointer's constant
  index, a lane number, a shift count, in a constant `if`). Unrolled, the
  loop leaves no code of its own and its variable is a constant in each
  copy of the body. `for (int u = 0; u < 8; u++) m[u] = max(m[u], p[u]);` is
  eight `max` in one block. The meaning is the same either way, but the cost is
  not, and the build's loop report shows which loops are at run time.
  `#pragma unroll` / `#pragma nounroll` in front of the `for` decide it by
  hand. An unrolled loop has a constant condition and a step that changes
  only its variable, and at most 4096 rounds.
- A run-time `for (T i = a; i < b; i++)` whose body does not use `i` (and
  does not assign `b`) counts a register down. There is one `if (a < b)` in front
  unless both are constants, then one compare and branch per round (one
  round known at compile time is just the body, with no loop). Any
  other run-time `for`, and `while`, is `if (c) do { body; step; } while
  (c);` with the condition evaluated in every round.

A run-time branch costs a compare, a branch and six delay slots, and ends
the block the scheduler works on (§6). Unrolling inside a run-time loop is
how a kernel gets its throughput.

## 5. Expressions

Every operator on registers is one instruction, and "cycles" is its latency
(`NPU.md` §3.2), which the compiler takes care of. Operands of one
operation have the same type. A constant operand takes the other's type
(and costs a register and 2-3 instructions each time, so keep it in a
variable inside loops).

**Operators**

| Types | Operators | Instruction | Cycles |
|---|---|---|---|
| `half8` | `+ - *` | `VAU.ADD.F16` `SUB` `MUL` | 3 |
| `float4` | `+ - *` | `VAU.ADD.F32` `SUB` `MUL` | 3 |
| `int4` | `+ -` | `VAU.ADD.I32` `SUB` | 2 |
| `int4` | `>> constant` (arithmetic) | `VAU.SHR.I32` | 3 |
| any vector | `& \| ^` | `VAU.AND` `OR` `XOR` | 1 |
| `u32`, `int` | `+ -` | `IAU.ADD` `SUB`, and a constant from 0 to 255 sits in the instruction | 1 |
| `u32`, `int` | `*` | `IAU.MUL` (low 32 bits) | 2 |
| `u32`, `int` | `& \| ^` | `IAU.AND` `OR` `XOR` | 1 |
| `u32`, `int` | `<< constant`, `u32 >> constant` (logical) | `IAU.SHL`, `IAU.SHR.U32` | 1 |
| pointer | `p + n`, `p - n`, `p += n`, `p++` | `IAU.ADD` / `SUB`, in elements as in C, and a run-time `n` costs a shift too | 1-2 |
| `half8 v` | `v[k]` | `CMU.CP.16` gives lane `k` as a `half` | 1 |
| `float4 v`, `int4 v` | `v[k]` | `CMU.CP.32` gives lane `k` as a `float` / `int` | 1 |

`/` and `%`, unary `-`, `~` and `!` work on constants only (`0 - x` for a
negation). There is no `int >>` (the SHAVE's arithmetic shift is not in the
instruction table yet), so use `u32`. `half8 *` multiplies treat f16 subnormal
inputs as zero. A lane cannot be assigned.

**Memory.** `p[k]` and `*p` with `k` a constant are a load as a value and a store
as the target of an assignment. For a run-time index move the pointer, as in `(p
+ i)[0]`. Offsets must stay within -16384 to 16383 bytes.

| Pointer to | Load | Store | Cycles (load) |
|---|---|---|---|
| `u32` `int` `float` | `LSU.LDO.32` | `LSU.STO.32` | 7 |
| `half8` `float4` `int4` | `LSU.LD.128` at offset 0, which holds its LSU for two cycles, else two `LSU.LDO.64` | `LSU.ST.128` / two `LSU.STO.64` | 8 / 7 |
| `half` | not on its own, use `load_splat(p)` | `LSU.STO.16` | |
| `u8` | no | of an `int4`, its low 8 bytes, `LSU.STO.64` (what `f16_to_u8f` gives) | |

A vector costs two LSU slots either way, so there is one vector load or store per
cycle.

**`restrict`** is what makes memory code fast. Without it the compiler must
assume that any store may change what any later load reads, so it keeps
them in order, 8 cycles apart. A `restrict` pointer promises, as in C, that
the memory accessed through it (and through pointers computed from it) is
accessed in no other way. Accesses through two different `restrict`
pointers are then scheduled freely.

A pointer initialized from a `restrict` pointer is based on it
(`const half8 *p = x;`, `p += 8`). Later it may only be assigned addresses
based on the same pointer. Its first value may be any address, which is how
the elements of a `restrict` array get theirs.

Stack arrays and the parameter block are separate from everything else by
themselves. Declare the input and output tensors' pointers `restrict`. The
same tensor passed as input and output, element by element, is still fine.

**Intrinsics**

| Call | Result | Instruction | Cycles |
|---|---|---|---|
| `exp(half8)` `sigm` `tanh` `log` `sqrt` | half8 | `VAU.EXP` `SIGM` `TANH` `LOG` `SQT` (within 6e-4) | 5 |
| `max(a, b)` `min(a, b)` | half8 / float4 | `CMU.MAX.F16` ... `CMU.MIN.F32` | 2 |
| `rot(v, n)` | v's type | `CMU.ALIGNVEC v v n` rotates down by `n` bytes | 1 |
| `align(a, b, n)` | a's type | `CMU.ALIGNVEC` takes bytes n to 15 of `a`, then bytes 0 to n-1 of `b` | 1 |
| `splat(half)` `splat(float)` `splat(u32)` | half8 / float4 / int4 | `CMU.CP.128.16.R`, `CMU.CP.128.32.R` put the value in every lane | 1 |
| `rcp(half)` `rsqrt(half)` `sqrt(half)` | half | `SAU.RCP` `RQT` `SQT` | 5 |
| `sum(float4)` | float | `SAU.SUMX.F32` gives the sum of the 4 lanes | 5 |
| `f16_to_f32(half8, h)` | float4 | `CMU.CP.128.F16.F32` converts lanes 4h to 4h+3 (h = 0, 1) | 1 |
| `i32_to_f32(int4)` | float4 | `CMU.CP.128.I32.F32` | 2 |
| `f32_to_i32(float4)` | int4 | `CMU.CP.128.F32.I32S` (truncates, saturates) | 2 |
| `f32_to_f16(float4, k)` | half | `CMU.CP.16.F32.F16` gives lane `k` as f16 | 1 |
| `f16_to_u8f(half8)` | int4 | `CMU.CP.64.F16.U8FS` puts the 8 values as u8 fractions (x * 255, to nearest, saturating) in the low 8 bytes, and the other 8 are undefined | 2 |
| `load_splat(half *)` | half8 | `LSU.LD.128.16.R` loads one f16 into all 8 lanes | 7 |
| `load_f32(half *)` | float4 | `LSU.LD.128.F16.F32` loads 4 x f16 as 4 x f32 | 7 |

`max` / `min` rewrite the condition flags, so the compiler keeps them away
from a block's compare (§6). SAU operations do not share a bundle with VAU
ones.

This compiler's output was run on the NPU and checked against numpy for integer `+ - * ^ << >> &`, `half8 + - *`,
`float4 + - *`, `int4 -` and `>>`, loads and stores of `u32` / `float` /
`half` / `half8` / `float4` at offset 0 and others, pointer arithmetic,
`v[k]` on `half8`, `exp sigm max align rot splat rcp sum f16_to_f32
f32_to_f16 load_splat`, vector casts, constant vectors, signed and unsigned
conditions, `if` / `else`, `do`, `while`, run-time `for` (counted, with
zero rounds, and general), unrolled `for`, stack arrays. The other operations are
instructions `NPU.md` §3.7 lists as executed on the device with these
latencies, not yet run from this compiler. Check a kernel that uses one
(§8).

To add an operation, add a row in `INTRINSICS` or `OPERATORS`
(`src/shave/dsl/lower.rs`), its latency in `mach::latency` and its effect
in `mach::Machine::bundle` (`src/shave/mach.rs`), and what it cannot be
issued with in `conflicts.tbl`, measured on the device (`scripts/shavedev.py
time` with the instruction beside each form in use; `NPU.md` §2
"Provenance"). Its encoding is in `isa.tbl`, which holds every instruction
of the processor. The latency must be measured on the device
(`scripts/shavedev.py lat`).

## 6. What the compiler does

**Blocks.** Run-time control (`if`, the loops, `barrier()`) cuts the kernel
into straight-line blocks. Each block is scheduled on its own and ends only
when all its results have arrived, so nothing overlaps between two rounds
of a loop. Put several independent chains into one round (an unrolled
`for`) to keep the units busy while results are on their way.

**Values.** Inside a block every assignment makes a new value with a
register of its own (SSA form, where only the value a later block reads stays
in the variable's register). So a variable
used again for something else costs nothing. `t = a * k; y[0] = t; t = b *
k; y[1] = t;` is scheduled as two independent chains.

**Schedule.** Within a block the operations are reordered freely as long as
the result is the same as executing them in source order. Every block is
scheduled twice, cycle by cycle. Once the operation with the longest chain
of latencies behind it goes first, once the one furthest along its chain
(a chain that was begun is carried on before another is started, so of 16
unrolled chains three or four are in progress at a time, not all 16 with
an intermediate value each). The shorter schedule that fits the registers
counts, of two equally long ones the one with fewer values alive. An
operation is placed in a cycle when

- every value it reads has arrived, and a register it overwrites is not read
  later by an operation that precedes it in the source,
- a unit that can run it is free (loads, stores and immediates use either LSU, and a
  copy uses either LSU or the CMU),
- no SAU operation shares the bundle with a VAU one (shared datapath),
- **no other result of the same unit for the same register file arrives in
  the same cycle** (`i` or `v`), because the processor loses one of them,
- **the LSU is not in the second cycle of a 128-bit access**,
- **it shares no port of the processor with an operation of the same or a
  nearby cycle** (`src/shave/conflicts.tbl`), meaning,
  of what the compiler emits, a compare of two registers (`CMU.CMII`) not
  beside a load or store on LSU1, and `rot` / `align` not beside `sum`,
- memory accesses that may overlap (`restrict`, or the same pointer value with
  disjoint offsets, never does) keep their order, so a load comes at least 8 cycles
  after a store, a store after a load has arrived.

Operations are then moved to the last cycle that delays nothing. A loaded
value holds its register from shortly before its first reader, not from
the block's start. Loads and constants always move, and any other operation as far
as its operands are alive anyway (an address computed from a pointer that
is used again later). A block also stays open for the second cycle of a
128-bit store in its last cycle.

**Branches.** A block that ends in a branch issues it seven cycles before
its end, so the six delay slots hold the block's last operations. A short
block is padded with `NOP`. The compare sits in the cycle before the
branch, and no `max` / `min` (they rewrite the flags) is on its way then.
Operations that share the branch's bundle stay unconditional
(`PEU.PCCX`). The kernel's return is placed the same way.

**Registers.** A value holds a machine register from the cycle its first
result arrives until its last read, through every block in between. Which
blocks those are follows from the branches (a value read in a later round
of a loop holds its register for the whole loop). Between its last read and
its next assignment a variable holds none. A result still on its way does
not occupy its register (compiled code relies on the same). `i0`-`i17` and
`v0`-`v23`, which a kernel may use without saving. There is no spilling, so
when a kernel needs more, the error names the values in the way (after the
compiler tried the schedules with the fewest values alive). `barrier();`
between two phases keeps the second from starting, and taking registers,
before the first is done.

**Report.** `shave build` prints the cycles per round of each run-time loop
(an outer loop counts its inner loops once). Memory stalls are not in it, since
on the device a bundle with loads or stores costs 1.2-2 cycles (`NPU.md`
§3.5).

## 7. The kernel's interface

The parameter points to packed `MemRefData` records, 0x24 bytes (9 words)
each, holding the inputs and then the outputs (`NPU.md` §3.9).

| Word | Field |
|---|---|
| 0 | data address |
| 2 | number of dimensions |
| 3 | address of the dimensions (u32 each, innermost first) |
| 4 | address of the strides (i64 each, **in bits**, innermost first) |

So the first input's address is `params[0]`, the first output's (of a
one-input op) `params[9]`. Each invocation sees only its own tile of every
tensor, so read the dimensions rather than assuming them.

A kernel must not write outside its output and its stack arrays, because nothing
protects memory, and a wild store or a loop that does not end loses the
device (`ze_result 0x70000001`) until the next graph is loaded. A loop
bound read from the tensor's data is such a loop.

## 8. Testing a kernel

1. **On the host.** `mach::Machine` runs the compiled program cycle by
   cycle with the processor's timing. A consumer scheduled too early reads
   the old value, a branch takes effect after its delay slots. Its
   statistics count every violation of the rules of §6 (and reads of
   registers nothing wrote), and a test asserts they are zero. See
   `tests/shave_dsl.rs`, which maps the tensors and the parameter block, runs, and
   compares with a reference in Rust. The emulator knows the instructions of
   §5. Transcendentals are exact in it and within 6e-4 on the device.
2. **On the NPU.** The kernel runs as the carrier op's ELF in a one-op graph
   (compiler patch 0008, `NPU_RS_KERNEL_DIR`). `scripts/sktest.py` does it
   for the three kernels' carriers (`--kernel softmax|gnorm|trinv`), comparing the output
   against numpy and against another build, and the kernel's time from the
   profile.

The emulator is only as good as its model. Four of its rules could only be
found on the device, where a kernel that passed in the emulator lost the
device. They are the result-port rule, the two-cycle 128-bit access, `CMU.MAX`
rewriting the flags, and the read port `CMU.CMII` shares with LSU1. Such shared
resources are a table (`conflicts.tbl`) the scheduler, the assembler
(`shave build` refuses such a bundle) and the emulator
(`Stats::conflicts`) use. A new kind of kernel is done when step 2
passes.

## 9. Writing a fast kernel

- **Unroll.** One VAU operation per cycle is the limit for float work, and a
  round with U independent chains reaches it once U x (VAU operations per
  chain) covers the chain's latency. `softmax_lse.sk` uses U = 8.
- **But keep the hot code small.** Code that is not in the instruction
  cache costs ~15 cycles per bundle on the device (`NPU.md` §3.5), which
  the emulator and the loop report do not show. Unroll within a loop that
  runs many rounds, not a long computation into one piece of straight-line
  code (`trinv.sk`'s header has the numbers).
- **Use a block's idle start.** A block's first cycles wait for its loads.
  Work on values from the round before fits there. `gnorm.sk` sums a head's
  squares in the next head's round.
- **`restrict`** on the tensors' pointers, or every load waits for every
  earlier store.
- **Read the report.** A round much longer than (operations on the busiest
  unit) + (the longest chain's latency) has a dependency you did not mean.
  A loop you meant to be unrolled and that shows up in the report is at run
  time. Use `#pragma unroll`.
- **`max` / `min` in a loop** delay the loop's compare until the last of
  them has arrived, so the round gets up to 8 cycles longer than its
  operations need.
- **Registers.** U accumulators + U temporaries + constants must stay
  under 24 vectors. What counts is the values alive at once, not the
  variables declared.
- **Pointers.** `p[u]` over one pointer moved once per round (`p += U`)
  costs one IAU operation per round, not per vector.
- **Constants** in variables outside the loop.
- `do { } while ()` where the first round always runs, because `for` and `while`
  pay one more compare and branch in front.

## 10. Not in the language

- More than one function, macros with arguments, `#include`, `typedef`,
  structs, unions, `switch`, `goto`, `break`, `continue`, `return`.
- Assignments and `++` inside expressions, and the comma and `?:` operators.
- `&&`, `||`, `!` and comparisons as run-time values.
- Run-time indices, pointers to pointers, arrays of more than one
  dimension, and initializers of arrays.
- Division, `int >>`, unary minus on registers, and arithmetic on `half` and
  `float` scalars (they come from and go to vectors and memory).
- Value-converting casts and the usual arithmetic conversions between
  integers and floats.
- Spilling. `i18`-`i31` and `v24`-`v31` are not used.
- Multiply-accumulate (`VAU.MACP*`), operand swizzle, hardware loops,
  per-lane predication and library calls.

## 11. Grammar

```
file      = { directive | constant | assert } function
function  = [ "static" ] [ "inline" ] "void" NAME "(" type [ "*" ] NAME ")" block
constant  = "const" type NAME "=" cexpr ";"      (u32 / int; float / half: a floating cexpr)
assert    = "_Static_assert" "(" cexpr [ "," STRING ] ")" ";"
block     = "{" { stmt } "}"
stmt      = block | decl ";" | simple ";" | assert | ";"
          | "if" "(" expr ")" stmt [ "else" stmt ]
          | "for" "(" [ decl | simples ] ";" [ expr ] ";" [ simples ] ")" stmt
          | "while" "(" expr ")" stmt
          | "do" stmt "while" "(" expr ")" ";"
decl      = [ "const" ] [ "register" ] type declarator { "," declarator }
declarator= [ "*" ] [ "const" ] [ "restrict" ] NAME [ "[" cexpr "]" ] [ "=" expr ]
type      = "void" | "u8" | "u32" | "unsigned" | "int" | "half" | "float"
          | "half8" | "float4" | "int4"
simples   = simple { "," simple }
simple    = lvalue ( "=" | "+=" | "-=" | "*=" | "&=" | "|=" | "^=" | "<<=" | ">>=" ) expr
          | lvalue "++" | lvalue "--" | "++" lvalue | "--" lvalue
          | "barrier" "(" ")"
lvalue    = NAME | expr "[" expr "]" | "*" expr
expr      = C's expression grammar over: constants, NAME, NAME "(" args ")",
            expr "[" expr "]", "(" type [ "*" ] ")" expr, "(" expr ")",
            unary - ! ~ * &, and by rising precedence:
            ||  &&  |  ^  &  == !=  < <= > >=  << >>  + -  * / %
cexpr     = an expr of constants
directive = "#define" NAME cexpr | "#if" cexpr | "#ifdef" NAME | "#ifndef" NAME
          | "#elif" cexpr | "#else" | "#endif" | "#error" text
          | "#pragma" ( "unroll" | "nounroll" )
```
