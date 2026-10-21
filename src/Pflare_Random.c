/*
  The random numbers used by PFLARE (the PMISR measure, the DDC tie-breaking,
  the CR initial error and the GMRES polynomial start vector) all come from
  pflare_random_fill, seeded the same way on every call.

  PFLARE used to draw them from the Fortran intrinsic random_number after
  random_seed(put=seed), with seed(i) = comm_rank + 1 + i. That sequence is
  defined by the Fortran runtime, so it differed between compilers. This file
  reproduces exactly what gfortran's runtime (libgfortran/intrinsics/random.c,
  GCC 7 onwards, checked against GCC 13.3) does for that seed followed by
  random_number on a real(PetscReal) array, so the numbers are bitwise the
  same as previous gfortran builds and no longer depend on the compiler:

  - The generator is xoshiro256** (Blackman and Vigna,
    https://prng.di.unimi.it/), with a 256-bit state of four uint64_t.
  - random_seed(size=) gives 8, the state size in default (4-byte) integers.
    random_seed(put=seed) copies seed(8 - i) into the i-th 4-byte word
    (i = 0..7, in memory order) of the state, and then XORs each uint64_t of
    the state with a fixed key, so poor seeds (like ours, small consecutive
    integers) do not give a poor state. The copy is done with memcpy, as
    gfortran does, so it also follows the byte order of the machine.
  - random_seed(put=) sets the master state, which the calling thread's state
    is copied from (the master state is then jumped ahead by 2^128 draws for
    the next thread, which never matters here as we reseed before every fill),
    so the first draw comes from the scrambled seed itself.
  - random_number on a real(8) array takes one 64-bit output per entry, keeps
    the top 53 bits and multiplies by 2^-64. On a real(4) array it takes the
    top 32 bits of each 64-bit output, keeps the top 24 of those and multiplies
    by 2^-32. Both conversions are exact and give 0 <= u < 1. Real(16)
    (__float128) takes two outputs per entry: all 64 bits of the first, the
    top 49 of the second.
*/

#include <stdint.h>
#include <string.h>
#include "Pflare_Random.h"

// The number of uint64_t in the xoshiro256** state
#define PFLARE_RANDOM_STATE_SIZE 4
// The number of default (4-byte) integers in the seed, what random_seed(size=) returns
#define PFLARE_RANDOM_SEED_SIZE 8

// Keys for scrambling the seed in order to avoid poor seeds (gfortran's xor_keys)
static const uint64_t pflare_random_xor_keys[PFLARE_RANDOM_STATE_SIZE] = {0xbd0c5b6e50c2df49ULL, 0xd46061cd46e1df38ULL, 0xbb4f4d4ed6103544ULL, 0x114a583d0756ad39ULL};

// -------------------------------------------------------------------------------------------------------------------------------

static inline uint64_t pflare_random_rotl(const uint64_t x, const int k)
{
   return (x << k) | (x >> (64 - k));
}

// -------------------------------------------------------------------------------------------------------------------------------

static inline uint64_t pflare_random_next(uint64_t *s)
{
   // One step of xoshiro256**, returns the next 64-bit output

   const uint64_t result = pflare_random_rotl(s[1] * 5, 7) * 9;
   const uint64_t t      = s[1] << 17;

   s[2] ^= s[0];
   s[3] ^= s[1];
   s[1] ^= s[2];
   s[0] ^= s[3];

   s[2] ^= t;

   s[3] = pflare_random_rotl(s[3], 45);

   return result;
}

// -------------------------------------------------------------------------------------------------------------------------------

PETSC_INTERN void pflare_random_fill(int comm_rank, PetscInt n, PetscReal *out)
{
   // Seeds the random number generator with a seed that depends on
   // comm_rank (so each rank gets different random numbers, but repeated runs
   // get the same ones) and then fills out[0..n-1] with uniform random numbers
   // in 0 <= u < 1

   // ~~~~~~
   uint32_t seed[PFLARE_RANDOM_SEED_SIZE], seed_words[PFLARE_RANDOM_SEED_SIZE];
   uint64_t state[PFLARE_RANDOM_STATE_SIZE];
   PetscInt i_loc;
   // ~~~~~~

   // Ensure we seed the same so subsequent runs get the same random
   // Seed different on each process so we don't have the same random numbers on each processor
   // This is seed(i_loc) = comm_rank + 1 + i_loc for the 1-based i_loc of the Fortran,
   // evaluated in unsigned arithmetic to give the bits of the 4-byte default integer
   for (i_loc = 0; i_loc < PFLARE_RANDOM_SEED_SIZE; i_loc++) {
      seed[i_loc] = (uint32_t)comm_rank + 2u + (uint32_t)i_loc;
   }
   // random_seed(put=seed): the seed goes into the state in reverse order...
   for (i_loc = 0; i_loc < PFLARE_RANDOM_SEED_SIZE; i_loc++) {
      seed_words[i_loc] = seed[PFLARE_RANDOM_SEED_SIZE - 1 - i_loc];
   }
   memcpy(state, seed_words, sizeof(state));
   // ...and is then scrambled
   for (i_loc = 0; i_loc < PFLARE_RANDOM_STATE_SIZE; i_loc++) {
      state[i_loc] ^= pflare_random_xor_keys[i_loc];
   }

   // Fill with random numbers
   for (i_loc = 0; i_loc < n; i_loc++) {
#if defined(PETSC_USE_REAL_SINGLE)
      // Take the higher bits, then keep the 24 bits a real(4) can hold
      const uint32_t high = (uint32_t)(pflare_random_next(state) >> 32) & (~(uint32_t)0 << 8);
      out[i_loc]          = (PetscReal)high * (PetscReal)(1.0 / 4294967296.0);
#elif defined(PETSC_USE_REAL_DOUBLE)
      // Keep the 53 bits a real(8) can hold
      const uint64_t bits = pflare_random_next(state) & (~(uint64_t)0 << 11);
      out[i_loc]          = (PetscReal)bits * (PetscReal)(1.0 / 18446744073709551616.0);
#elif defined(PETSC_USE_REAL___FLOAT128)
      // All 64 bits of the first output and the top 49 bits of the second (113 bits)
      const uint64_t bits_1 = pflare_random_next(state);
      const uint64_t bits_2 = pflare_random_next(state) & (~(uint64_t)0 << 15);
      const PetscReal two_m64 = (PetscReal)1.0 / (PetscReal)18446744073709551616.0;
      out[i_loc]              = (PetscReal)bits_1 * two_m64 + (PetscReal)bits_2 * (two_m64 * two_m64);
#else
   #error "pflare_random_fill supports single, double and __float128 PetscReal only"
#endif
   }
}

// -------------------------------------------------------------------------------------------------------------------------------
