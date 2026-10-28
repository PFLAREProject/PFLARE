/*
  C port of Sorting.F90 (module sorting)

  The Fortran built a sorted set in a binary tree (Binary_Tree.F90), inserting
  the entries in the order of a Knuth shuffle to balance the tree. Here that is
  sorted_unique, on PETSc's sort routines; create_knuth_shuffle and
  create_knuth_shuffle_tree_array had no other use and are not ported.
  The array arguments are passed with their sizes, and the indices returned by
  intersect_pre_sorted_indices_only are 0-based
*/

#include <petscsys.h>
#include "pflareimpl.h"
#include "Sorting.h"

// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------
// Some routines to do with sorting arrays
// -------------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------------------------------------

// Sort the n entries of array and remove the duplicates, in place
// On return n is the number of unique entries, which are in array[0..n-1] in
// ascending order
// This replaces inserting the entries into a binary tree (in the order of a
// Knuth shuffle, to balance the tree) and reading the tree back out in
// ascending order, which gives the same sorted set as the tree ignores duplicates

PETSC_INTERN PetscErrorCode sorted_unique(PetscInt array[], PetscInt *n)
{
   PetscFunctionBegin;

   PetscCall(PetscSortInt(*n, array));
   PetscCall(PetscSortedRemoveDupsInt(n, array));

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

// Intersection of two arrays - Assume they are sorted and only return the matching
// indices for both arrays

PETSC_INTERN PetscErrorCode intersect_pre_sorted_indices_only(const PetscInt array1[], PetscInt size_array1, const PetscInt array2[], PetscInt size_array2, int indices_array1[], int indices_array2[], int *intersect_count)
{
   int index1, index2;

   PetscFunctionBegin;

   index1 = 0;
   index2 = 0;
   *intersect_count = 0;

   // Now loop through and do the intersection
   while (index1 < size_array1 && index2 < size_array2) {
      if (array1[index1] == array2[index2]) {
         indices_array1[*intersect_count] = index1;
         indices_array2[*intersect_count] = index2;
         *intersect_count = *intersect_count + 1;
         index1 = index1 + 1;
         index2 = index2 + 1;
      } else if (array1[index1] < array2[index2]) {
         index1 = index1 + 1;
      } else {
         index2 = index2 + 1;
      }
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------

// Combine and sort the contents of two pre-sorted arrays
// sorted_array must have room for size_array1 + size_array2 entries

PETSC_INTERN PetscErrorCode merge_pre_sorted(const PetscInt array1[], PetscInt size_array1, const PetscInt array2[], PetscInt size_array2, PetscInt sorted_array[])
{
   PetscInt index1, index2, counter;

   PetscFunctionBegin;

   index1 = 0;
   index2 = 0;
   counter = 0;

   // Now loop through
   while (index1 < size_array1 && index2 < size_array2) {
      if (array1[index1] <= array2[index2]) {
         sorted_array[counter] = array1[index1];
         index1 = index1 + 1;
      } else {
         sorted_array[counter] = array2[index2];
         index2 = index2 + 1;
      }
      counter = counter + 1;
   }

   // If we've gotten here we need to check if we hit the end of both arrays
   if (index1 < size_array1) {
      PetscCall(PetscArraycpy(&sorted_array[counter], &array1[index1], size_array1 - index1));
   }
   if (index2 < size_array2) {
      PetscCall(PetscArraycpy(&sorted_array[counter], &array2[index2], size_array2 - index2));
   }

   PetscFunctionReturn(PETSC_SUCCESS);
}

// -------------------------------------------------------------------------------------------------------------------------------
