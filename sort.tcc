


///////////////////////////////////////////////////////////////////
// Classic QuickSort in increasing order
template <typename T>
long partition_oO( T *tmp, long begin, long end ) {
    long i = begin - 1;
    long j = end;
    T pivot = tmp[begin];

    T tmp_tmp;
    while( 1 ) {
	do{
	    j -= 1;
	} while( (j >= begin) && (tmp[j] >= pivot) );

	do{
	    i += 1;
	} while( (i < end) && (tmp[i] < pivot) );

	if( i < j ) {
	    tmp_tmp = tmp[i];
	    tmp[i] = tmp[j];
	    tmp[j] = tmp_tmp;

	} else {
	    if( j < begin ) {
		return( begin+1 );
	    }
	    return( j+1 );
	}
    }
    return( -1 );
}

template <typename T>
void quick_sort_oO( T *tmp, long begin, long end ) {
    if( begin < end ) {
	long p = partition_oO( tmp, begin, end );
	if( (begin < p) && (p < end) ) {
	    quick_sort_oO( tmp, begin, p );
	    quick_sort_oO( tmp, p, end );
	}
    }
}

///////////////////////////////////////////////////////////////////
// QuickSort in increasing order with index array
template <typename T, typename I>
long partition_oO( T *tmp, I *index, long begin, long end ) {
    long i = begin - 1;
    long j = end;
    T pivot = tmp[begin];

    T tmp_tmp;
    I index_tmp=(I)0;
    while( 1 ) {
	do{
	    j -= 1;
	} while( (j >= begin) && (tmp[j] >= pivot) );

	do{
	    i += 1;
	} while( (i < end) && (tmp[i] < pivot) );

	if( i < j ) {
	    tmp_tmp = tmp[i];
	    tmp[i] = tmp[j];
	    tmp[j] = tmp_tmp;

	    index_tmp = index[i];
	    index[i] = index[j];
	    index[j] = index_tmp;

	} else {
	    if( j < begin ) {
		return( begin+1 );
	    }
	    return( j+1 );
	}
    }
    return( -1 );
}

template <typename T, typename I>
void quick_sort_oO( T *tmp, I *index, long begin, long end ) {
    if( begin < end ) {
	long p = partition_oO( tmp, index, begin, end );
	if( (begin < p) && (p < end) ) {
	    quick_sort_oO( tmp, index, begin, p );
	    quick_sort_oO( tmp, index, p, end );
	}
    }
}

///////////////////////////////////////////////////////////////////
// Classic QuickSort in decreasing order
template <typename T>
long partition_Oo( T *tmp, long begin, long end ) {
    long i = begin - 1;
    long j = end;
    T pivot = tmp[begin];

    T tmp_tmp;
    while( true ) {
	do{
	    j -= 1;
	} while( (j >= begin) && (tmp[j] <= pivot) );

	do{
	    i += 1;
	} while( (i < end) && (tmp[i] > pivot) );

	if( i < j ) {
	    tmp_tmp = tmp[i];
	    tmp[i] = tmp[j];
	    tmp[j] = tmp_tmp;
	} else {
	    if( j < begin ) {
		return( begin+1 );
	    }
	    return( j+1 );
	}
    }
    return( -1 );
}

template <typename T>
void quick_sort_Oo( T *tmp, long begin, long end ) {
    if( begin < end ) {
	long p = partition_Oo( tmp, begin, end );
	if( (begin < p) && (p < end) ) {
	    quick_sort_Oo( tmp, begin, p );
	    quick_sort_Oo( tmp, p, end );
	}
    }
}

///////////////////////////////////////////////////////////////////
// QuickSort in decreasing order with index array
template <typename T, typename I>
long partition_Oo( T *tmp, I *index, long begin, long end ) {
    long i = begin - 1;
    long j = end;
    T pivot = tmp[begin];

    T tmp_tmp;
    I index_tmp=(I)0;

    while( 1 ) {
	do{
	    j -= 1;
	} while( (j >= begin) && (tmp[j] <= pivot) );
	do{
	    i += 1;
	} while( (i < end) && (tmp[i] > pivot) );
	if( i < j ) {
	    tmp_tmp = tmp[i];
	    tmp[i] = tmp[j];
	    tmp[j] = tmp_tmp;

	    index_tmp = index[i];
	    index[i] = index[j];
	    index[j] = index_tmp;
	} else {
	    if( j < begin ) {
		return( begin+1 );
	    }
	    return( j+1 );
	}
    }
    return( -1 );
}

template <typename T, typename I>
void quick_sort_Oo( T *tmp, I *index, long begin, long end ) {
    if( begin < end ) {
	long p = partition_Oo( tmp, index, begin, end );
	if( (begin < p) && (p < end) ) {
	    quick_sort_Oo( tmp, index, begin, p );
	    quick_sort_Oo( tmp, index, p, end );
	}
    }
}







