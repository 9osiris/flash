flash - flash attention from scratch, no libraries [educational project]

disclaimer:
this is a toy, not a real inference kernel. it is float32 on one cpu
thread, there is no gpu, no simd, no bf16 here. i built it to
understand how flash attention actually works, not to serve
anything. dont use it for real inference.

why i made this:
every model i care about spends most of its time in attention, and
every explanation of flash attention says "tiling plus online
softmax" and then shows a diagram. i wanted to feel the trick in
code. the naive version builds the whole n by n score matrix, which
is the thing that kills you at long context. the flash version never
builds it, it walks k and v in blocks and keeps a running max and a
running sum per row, rescaling the partial output every time the max
moves. same math, same answer, way less memory traffic. i wrote the
naive one first on purpose so the tests can diff them against each
other, that is the whole point of the repo. c++ because the real
kernels are c++ and because kvblocks and tokkit already live in this
lane, this is the piece that sits between them: tokkit turns text
into tokens, kvblocks caches the keys and values, flash is what
actually mixes them.

the guide:
build it:
  make

run the demo:
  ./flash demo

check flash against naive:
  ./flash verify 512 64

time both:
  ./flash bench 512 64

run the tests:
  make test

verify prints the max abs diff between flash and naive, it should be
around 1e-7. bench prints naive ms, flash ms, and the speedup. on
this box flash runs about 2.1x faster at n=512 and 2.4x at n=2048,
and that is just from not writing the score matrix to memory. the
attention code is in src/attn.h and src/attn.cpp, the cli is
src/main.cpp.

what it does:
- naive attention, full n by n scores, stable softmax, causal mask
- flash attention, tiled over k/v, online softmax with running max
  and running sum, same output as naive to float precision
- causal masking in both paths, scale defaults to 1/sqrt(d)
- multi-head wrapper, each head independent, flash or naive per call
- block sizes are knobs, try 8, 16, 32, 64 and watch the diff stay
  tiny and the timing move
- verify and bench commands with a seeded rng so runs are repeatable

numbers from my box:
- verify 512x64 causal: max abs diff 0.00000013
- verify 1024x64 no mask: max abs diff 0.00000012
- bench 512x64: naive 35.76 ms, flash 16.73 ms, 2.14x
- bench 2048x64: naive 600.63 ms, flash 250.59 ms, 2.40x

tests:
26 checks, all passing. flash vs naive across shapes (2x2 up to
129x32), block sizes from 2 to 64, d=1 to d=128, causal and not.
causal rows provably ignore later keys. outputs stay inside the v
range because they are convex combos. huge scores stay finite
because of the running max. deterministic across runs, heads are
independent, explicit scale is honored, n=0 is a no-op.
