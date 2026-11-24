flash - flash attention from scratch, no libraries [educational project]

disclaimer:
this is a toy, not a real inference kernel. it is float32 on one cpu
thread, there is no gpu, no bf16 here. simd is in now, avx2 dot and
axpy helpers behind make simd, but it is still a toy. i built it to
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

build it with avx2:
  make simd

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
this box flash runs about 1.7x faster at n=512 and 1.8x at n=2048
in the plain build, and that is just from not writing the score
matrix to memory. make simd builds the same code with -mavx2 -mfma
and the dot and pv loops run 8 floats at a time: flash drops to
5.7 ms at n=512 and 90 ms at n=2048, about 3x faster than the
scalar flash. the attention code is in src/attn.h and src/attn.cpp,
the simd helpers are in src/simd.h, the cli is src/main.cpp.

what it does:
- naive attention, full n by n scores, stable softmax, causal mask
- flash attention, tiled over k/v, online softmax with running max
  and running sum, same output as naive to float precision
- causal masking in both paths, scale defaults to 1/sqrt(d)
- multi-head wrapper, each head independent, flash or naive per call
- avx2 dot and axpy helpers in src/simd.h, used by the score and pv
  loops in both paths, scalar fallback when built without -mavx2
- block sizes are knobs, try 8, 16, 32, 64 and watch the diff stay
  tiny and the timing move
- verify and bench commands with a seeded rng so runs are repeatable

numbers from my box:
- verify 512x64 causal: max abs diff 0.00000013
- verify 1024x64 no mask: max abs diff 0.00000012
- bench 512x64 scalar: naive 30.60 ms, flash 18.33 ms, 1.67x
- bench 2048x64 scalar: naive 483.60 ms, flash 270.82 ms, 1.79x
- bench 512x64 avx2: naive 21.74 ms, flash 5.72 ms, 3.80x
- bench 2048x64 avx2: naive 357.36 ms, flash 90.35 ms, 3.96x

tests:
104 checks, all passing. flash vs naive across shapes (2x2 up to
512x64), block sizes from 2 to 64, d=1 to d=128, causal and not.
the simd dot and axpy helpers match plain scalar loops at 15
lengths from 1 to 128, including the non-multiple-of-8 tails.
causal rows provably ignore later keys. outputs stay inside the v
range because they are convex combos. huge scores stay finite
because of the running max. deterministic across runs, heads are
independent, explicit scale is honored, n=0 is a no-op.
