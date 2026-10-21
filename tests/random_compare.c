/*
  TEMPORARY (stage 9 of docs/dev/fortran_removal_plan.md): checks that the C
  generator in src/Pflare_Random.c fills exactly the same bits as the retired
  Fortran generator (random_seed(put=) + random_number, here renamed
  pflare_random_fill_fortran in src/Pflare_Random_Fortran.F90), for many ranks
  and lengths. The C object is linked in directly, as its symbol is internal to
  libpflare. Exits with an error on the first difference. Deleted, with the
  Fortran, once the check has passed
*/

#include <petscsys.h>
#include <stdint.h>
#include <string.h>
#include "../src/Pflare_Random.h"

// PETSC_EXTERN gives C linkage when PETSc compiles C as C++
PETSC_EXTERN void pflare_random_fill_fortran(int comm_rank, PetscInt n, PetscReal *out);

int main(int argc, char **argv)
{
   const PetscInt n_list[]     = {1, 2, 3, 7, 100, 1000, 10001, 65536};
   const int      extra_ranks[] = {100, 1023, 4096, 123456, 2147483000};
   const PetscInt n_n = (PetscInt)(sizeof(n_list) / sizeof(n_list[0]));
   const int      n_extra = (int)(sizeof(extra_ranks) / sizeof(extra_ranks[0]));
   PetscReal     *out_f, *out_c, *out_f2;
   PetscInt       i_n, i, n, n_max = 65536, n_entries = 0, n_cases = 0;
   int            i_rank, rank;
   uint64_t       checksum = 0;

   PetscFunctionBeginUser;
   PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
   PetscCall(PetscMalloc3(n_max, &out_f, n_max, &out_c, n_max, &out_f2));

   for (i_rank = 0; i_rank < 64 + n_extra; i_rank++) {
      rank = i_rank < 64 ? i_rank : extra_ranks[i_rank - 64];
      for (i_n = 0; i_n < n_n; i_n++) {
         n = n_list[i_n];
         // Poison the outputs so a short fill cannot pass
         memset(out_f, 0xff, (size_t)n_max * sizeof(PetscReal));
         memset(out_c, 0xee, (size_t)n_max * sizeof(PetscReal));
         memset(out_f2, 0xdd, (size_t)n_max * sizeof(PetscReal));
         // Interleave the calls: each reseeds, so the order must not matter
         pflare_random_fill_fortran(rank, n, out_f);
         pflare_random_fill(rank, n, out_c);
         pflare_random_fill_fortran(rank, n, out_f2);
         if (memcmp(out_f, out_f2, (size_t)n * sizeof(PetscReal)) != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Fortran not repeatable, rank %d n %" PetscInt_FMT, rank, n);
         for (i = 0; i < n; i++) {
            if (memcmp(&out_f[i], &out_c[i], sizeof(PetscReal)) != 0)
               SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Bit difference at rank %d n %" PetscInt_FMT " entry %" PetscInt_FMT ": Fortran %.17g C %.17g", rank, n, i, (double)out_f[i], (double)out_c[i]);
            if (!(out_c[i] >= 0 && out_c[i] < 1)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Out of range at rank %d n %" PetscInt_FMT " entry %" PetscInt_FMT, rank, n, i);
         }
         // The entries past n must be untouched
         if (n < n_max && memcmp(&out_c[n], &out_c[n_max - 1], sizeof(PetscReal)) != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "C wrote past n");
         // A checksum of the bits, to show the runs are not trivially equal
         for (i = 0; i < n; i++) {
            uint64_t bits = 0;
            memcpy(&bits, &out_c[i], sizeof(PetscReal));
            checksum = checksum * 1099511628211ULL ^ bits;
         }
         n_entries += n;
         n_cases++;
      }
   }
   // The zero-length fill must not write anything
   memset(out_c, 0xee, sizeof(PetscReal));
   pflare_random_fill(0, 0, out_c);
   memset(out_f, 0xee, sizeof(PetscReal));
   if (memcmp(out_c, out_f, sizeof(PetscReal)) != 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "C wrote with n = 0");

   pflare_random_fill(0, 3, out_c);
   PetscCall(PetscPrintf(PETSC_COMM_SELF, "sizeof(PetscReal) %d; rank 0 first three: %.17g %.17g %.17g\n", (int)sizeof(PetscReal), (double)out_c[0], (double)out_c[1], (double)out_c[2]));
   PetscCall(PetscPrintf(PETSC_COMM_SELF, "random_compare: %" PetscInt_FMT " cases, %" PetscInt_FMT " entries, all bitwise identical (bit checksum %016llx)\n", n_cases, n_entries, (unsigned long long)checksum));

   PetscCall(PetscFree3(out_f, out_c, out_f2));
   PetscCall(PetscFinalize());
   return 0;
}
