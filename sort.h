

#ifndef _SORT_H_
#define _SORT_H_


///////////////////////////////////////////////////////////////////////////////
// Quick sort functions (implementation is in sort.tcc)

/**
 * Sort 'tmp' array in increasing order.
 */
template <typename T> 
void quick_sort_oO( T *tmp, long begin, long end );


/**
 * Sort tmp array in increasing order. The associated 'index' array
 * will be modified in the same manner as 'tmp' array. 
 */
template <typename T, typename I> 
void quick_sort_oO( T *tmp, I *index, long begin, long end );



/**
 * Sort 'tmp' array in decreasing order.
 */
template <typename T> 
void quick_sort_Oo( T *tmp, long begin, long end );


/**
 * Sort tmp array in decreasing order. The associated 'index' array
 * will be modified in the same manner as 'tmp' array. 
 */
template <typename T, typename I> 
void quick_sort_Oo( T *tmp, I *index, long begin, long end );

#include "sort.tcc"


#endif // _SORT_H_

