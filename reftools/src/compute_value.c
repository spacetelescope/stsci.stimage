# include <stdio.h>
# include <string.h>
# include <stdlib.h>    /* for strtol function */
# include <ctype.h>
# include <math.h>

#include "imphttab.h"

typedef struct {
  int    ndim;   /* number of dimensions for each position */
  double *index; /* array index along each axis, 
                  used to determine which axis is being interpolated */
  double *pos;   /* value of each axis at position given by index */
  double value;  /* value at position */
} BoundingPoint;

/* Perform interpolation, if necessary, to derive the final
 output value for this obsmode from the appropriate row. 
 */
double ComputeValue(PhtRow *tabrow, PhotPar *obs) {
  /* Parameters:
   PhtRow  *tabrow:    values read from matching row of table
   char   *obsmode:    full obsmode string from header 
   
   obsmode string needs to contain values of parameterized values 
   appropriate for this observation.
   */
  
  double value;
  int parnum;
  int n,p, nx, ndim;
  double *out;
  double *obsindx; /* index of each obsmode par value in tabrow->parvals */ 
  double *obsvals; /* value for each obsmode par in the same order as tabrow */
  
  int *ndpos;
  int **bounds; /* [ndim,2] array for bounds around obsvals values */
  double resvals[2]; /* Values from tabrow->results for interpolation */
  double resindx[2]; /* 1-D indices into tabrow->results for bounding positions*/
  int posindx;      /* N dimensional array of indices for a position */
  int indx,pdim,ppos,xdim,xpos;
  int tabparlen;
  /* 
   intermediate products used in iterating over dims 
   */
  int iter, x;
  int dimpow,iterpow;
  double *ndposd;
  int b0,b1,pindx;
  int deltadim;            /* index of varying dimension */
  double bindx[2],bvals[2]; /* indices into results array for bounding values */
  double rinterp;          /* placeholder for interpolated result */
  BoundingPoint **points;   /* array of bounding points defining the area of interpolation */
  
  /* Define functions called in this functions */ 
  double linterp(double *, int, double *, double);
  void byteconvert(int, int *, int);
  int computedeltadim(BoundingPoint *, BoundingPoint *);  
  long computeindex(int *, double *, int);  
  void computebounds(double *, int, double, int *, int*);  
  int strneq_ic(char *, char*, int);
  BoundingPoint **InitBoundingPointArray(int, int);
  void FreeBoundingPointArray(BoundingPoint **, int);
  
  /* Initialize variables and allocate memory */  
  value = 0.0;
  ndim = tabrow->parnum;
  if (ndim == 0) {
    /* No parameterized values, so simply return value
     stored in 1-element array results */
    return(tabrow->results[0]);
  } 
  dimpow = pow(2,ndim);
  
  obsindx = (double *)calloc(ndim, sizeof(double));
  obsvals = (double *)calloc(ndim, sizeof(double)); /* Final answer */
  ndpos = (int *)calloc(ndim, sizeof(int));
  ndposd = (double *)calloc(ndim, sizeof(double));
  bounds = (int **) calloc(ndim, sizeof(int *));
  for (p=0;p<ndim;p++) bounds[p] = (int *) calloc(2, sizeof(int));    
  
  /* We have parameterized values, so linear interpolation 
   will be needed in all dimensions to get correct value.
   Start by getting the floating-point indices for each 
   parameterized value
   */
  /* 
   Start by matching up the obsmode parameters with those in the table row 
   These are the values along each dimension of the interpolation
   */
  for (p=0;p<ndim;p++){
    tabparlen = strlen(tabrow->parnames[p]);
    for(n=0;n<obs->npar;n++){
      if (strneq_ic(tabrow->parnames[p],obs->parnames[n],tabparlen)){
        obsvals[p] = obs->parvalues[n];
        break;
      }
    }
    
    if (obsvals[p] == 0.0) {
      printf("ERROR: No obsmode value found for %s\n",tabrow->parnames[p]);
      
      free(obsindx);
      free(obsvals);
      free(ndpos);
      free(ndposd);
      for (p=0;p<ndim;p++) free(bounds[p]);
      free(bounds);
      
      return ('\0');
    }
    
    /* check whether we're going beyond the data in the table (extrapolation) */
    /* if we are, return -9999 */
    nx = tabrow->nelem[p+1];
    
    if ((obsvals[p] < tabrow->parvals[p][0]) ||
        (obsvals[p] > tabrow->parvals[p][nx-1])) {
      printf("WARNING: Parameter value %s%f is outside table data bounds.\n",
             tabrow->parnames[p],obsvals[p]);
      
      free(obsindx);
      free(obsvals);
      free(ndpos);
      free(ndposd);
      for (p=0;p<ndim;p++) free(bounds[p]);
      free(bounds);
      
      return -9999.0;
    }
  }
  
  /* Set up array of BoundingPoint objects to keep track of information needed
   for the interpolation 
   */
  points = InitBoundingPointArray(dimpow,ndim);
  
  /* Now find the index of the obsmode parameterized value
   into each parameterized value array
   Equivalent to getting positions in each dimension (x,y,...).
   */
  for (p=0;p<ndim;p++){    
    nx = tabrow->nelem[p+1];
    
    out = (double *) calloc(nx, sizeof(double));
    
    for (n=0; n<nx;n++) out[n] = n;
    
    value = linterp(tabrow->parvals[p], nx, out, obsvals[p]);    
    if (value == -99) {
      free(obsindx);
      free(obsvals);
      free(ndpos);
      free(ndposd);
      for (p=0;p<ndim;p++) free(bounds[p]);
      free(bounds);
      free(out);
      return('\0');
    }
    
    obsindx[p] = value;  /* Index into dimension p */
    computebounds(out, nx, (double)floor(value), &b0, &b1);
    
    bounds[p][0] = b0;
    bounds[p][1] = b1;
    /* Free memory so we can use this array for the next variable*/
    free(out);
  } /* End loop over each parameterized value */
  
  /* 
   Loop over each axis and perform interpolation to find final result
   
   For each axis, interpolate between all pairs of positions along the same axis
   An example with 3 parameters/dimensions for a point with array index (2.2, 4.7, 1.3):
   Iteration 1: for all z and y positions , interpolate between pairs in x
   (2,4,1)vs(3,4,1), (2,5,1)vs(3,5,1), (2,4,2)vs(3,4,2), and (2,5,2)vs(3,5,2)
   Iteration 2: for all z positions, interpolate between pairs from iteration 1 in y
   (2.2, 4,1)vs(2.2, 5, 1) and (2.2, 4, 2)vs(2.2, 5, 2)
   Iteration 3: interpolate between pairs from iteration 2 in z
   (2.2, 4.7, 1) vs (2.2, 4.7, 2) ==> final answer
   */
  for (iter=ndim; iter >0; iter--) {
    iterpow = pow(2,iter);
    for (p=0;p < iterpow;p++){
      pdim = floor(p/2);
      ppos = p%2;
      if (iter == ndim) {
        /* Initialize all intermediate products and perform first 
         set of interpolations over the first dimension 
         */
        /* Create a bitmask for each dimension for each position */
        byteconvert(p,ndpos,ndim);
        for (n=0;n<ndim;n++) {
          pindx = bounds[n][ndpos[n]];
          points[pdim][ppos].index[n] = (double)pindx;
          points[pdim][ppos].pos[n] = tabrow->parvals[n][pindx];
        }
        
        /* Determine values from tables which correspond to 
         bounding positions to be interpolated */
        indx = computeindex(tabrow->nelem, points[pdim][ppos].index, ndim);
        points[pdim][ppos].value = tabrow->results[indx];
        
      } /* End if(iter==ndim) */ 
      
      if (ppos == 1) {
        /* Determine which axis is varying, so we know which 
         input value from the obsmode string 
         we need to use for the interpolation */
        deltadim = computedeltadim(&points[pdim][0],&points[pdim][1]);
        if (deltadim < 0 || deltadim >= ndim) {
          printf("ERROR: Deltadim out of range: %i\n",deltadim);
          free(obsindx);
          free(obsvals);
          free (ndpos);
          free (ndposd);
          for (p=0;p<ndim;p++)free(bounds[p]);
          free(bounds);
          
          return('\0');
        }
        bindx[0] = points[pdim][0].pos[deltadim];
        bindx[1] = points[pdim][1].pos[deltadim];
        bvals[0] = points[pdim][0].value;
        bvals[1] = points[pdim][1].value;
        
        /*Perform interpolation now and record the results */
        rinterp = linterp(bindx, 2, bvals,obsvals[deltadim]);
        
        /* 
         Update intermediate arrays with results in 
         preparation for the next iteration 
         */
        if (rinterp == -99) return('\0');
        /* Determine where the result of this interpolation should go */
        x = floor((p-1)/2);
        xdim = floor(x/2);
        xpos = x%2;
        /* update bpos and bindx for iteration over next dimension
         */
        points[xdim][xpos].value = rinterp;
        for (n=0;n<ndim;n++) {
          points[xdim][xpos].index[n] = points[pdim][0].index[n];
          points[xdim][xpos].pos[n] = points[pdim][0].pos[n];
        }
        points[xdim][xpos].index[deltadim] = obsindx[deltadim];
        points[xdim][xpos].pos[deltadim] = obsvals[deltadim];
        
      } /* Finished with this pair of positions (end if(ppos==1)) */
      
    } /* End loop over p, data stored for interpolation in changing dimension */
    
  } /* End loop over axes(iterations), iter, for interpolation */
  
  /* Record result */
  value = points[0][0].value;
  
  /* clean up memory allocated within this function */
  free(obsindx);
  free(obsvals);
  free (ndpos);
  free (ndposd);
  for (p=0;p<tabrow->parnum;p++)free(bounds[p]);
  free(bounds);
  
  FreeBoundingPointArray(points,dimpow);
  return (value);
}


/* This routine implements 1-D linear interpolation
 It returns the interpolated value from f(x) that corresponds
 to the input position xpos, where f(x) is sampled at positions x.
 */
double linterp(double *x, int nx, double *fx, double xpos) {
  
  int i0, i1;  /* x values that straddle xpos */
  
  int n;
  double value;
  
  void computebounds (double *, int, double , int *, int *);
  
  /* interpolation calculated as: 
   yi + (xpos - xi)*(yi1 - yi)/(xi1-xi)
   */
  /* Start by finding which elements in x array straddle xpos value */
  computebounds(x, nx, xpos, &i0, &i1);
  if ((x[i1] - x[i0]) == 0){
    printf("==>ERROR: Linear interpolation reached singularity...\n");
    return(-99);
  }
  /* Now, compute interpolated value */
  value = fx[i0] + (xpos - x[i0])*(fx[i1] - fx[i0])/(x[i1]-x[i0]);
  return(value);
} 

/* Compute index into x array of val and returns 
 the indices for the bounding values, taking into account 
 boundary conditions. 
 */
void computebounds (double *x, int nx, double val, int *i0, int *i1) {
  int n;
  
  /* first test for whether we've got an end case here */
  if (x[nx-1] == val) {
    *i0 = nx-2;
    *i1 = nx-1;
  } else {
    for(n=0;n < nx; n++){
      if (x[n] <= val) {
        *i0 = n;
      } else {
        if (n > 0 && n < nx -1 ){
          *i1 = n;
        } else if (n == 0) {
          *i0 = 0;
          *i1 = 1;
        } else {
          *i0 = nx-2;
          *i1 = nx-1;
        }
        break;
      }
    }
  }
}

/* Compute the 1-D index of a n-D (ndim) position given by the array 
 of values in pos[] 
 */
long computeindex(int *nelem, double *pos, int ndim) {
  int n, szaxis;    
  long indx;
  
  indx = 0;
  szaxis = 1;
  for (n=0;n<ndim;n++) {
    indx += szaxis*pos[n];
    szaxis *= nelem[n+1];
  }
  return(indx); 
  
}

/* Convert an int value into an array of 0,1 values to represent
 which bytes are 0 or 1 in the integer.  The result array must
 already be allocated for the number of bytes to be checked in the
 integer value.
 */
void byteconvert(int val, int *result, int ndim) {
  int i,bval;
  
  bval = 1;
  for (i=0;i<ndim;i++){
    if ((val & bval) > 0){
      result[i] = 1;
    } else {
      result[i] = 0;
    }
    bval = bval << 1;
  }
}

/*
 Given 2 N dimensional sets of array indices, determine which dimension 
 changes from one set to the other.  
 
 NOTE: 
 This assumes that the positions only change in 1 dimension at a time. 
 */
int computedeltadim(BoundingPoint *pos1, BoundingPoint *pos2){
  int p;
  int xdim;
  double diff;
  
  for (p=0;p<pos1->ndim;p++){
    diff = pos2->index[p] - pos1->index[p];
    if ( diff != 0) {
      xdim = p;
      break;
    }
  }
  return(xdim);
}

/* This routine frees memory allocated to a row's entries,
 so that the next call to 'ReadPhotTab' will have an empty 
 structure to use for the storing the rows values. 
 */
void ClosePhotRow (PhtRow *tabrow) {
  
  int i;
  
  for (i=0; i<tabrow->parnum; i++){
    free(tabrow->parvals[i]);
    free(tabrow->parnames[i]);
  }
  free(tabrow->parvals);
  free(tabrow->parnames);
  
  free(tabrow->nelem);
  free(tabrow->results);
}

/* Initialize the array of BoundingPoint objects */
BoundingPoint **InitBoundingPointArray(int npoints, int ndim){
  int i,j;
  int pdim;
  void InitBoundingPoint(BoundingPoint *, int);
  BoundingPoint **points; 
  
  pdim = npoints/2;
  points = (BoundingPoint **)calloc(pdim,sizeof(BoundingPoint *));
  for (i=0;i<pdim;i++) {
    points[i] = (BoundingPoint *)calloc(2,sizeof(BoundingPoint));
    InitBoundingPoint(&points[i][0],ndim);
    InitBoundingPoint(&points[i][1],ndim);
  }
  return(points);   
}
void InitBoundingPoint(BoundingPoint *point, int ndim){
  
  point->index = (double *)calloc(ndim, sizeof(double));
  point->pos = (double *)calloc(ndim, sizeof(double));
  point->ndim = ndim;
  point->value=0.0;
  
}
/* Free the memory allocated to an array of BoundingPoint objects */
void FreeBoundingPointArray(BoundingPoint **points, int npoints){
  int i;
  int pdim;
  void FreeBoundingPoint(BoundingPoint *);
  pdim = npoints/2;
  
  for (i=0;i<pdim;i++) {
    FreeBoundingPoint(&points[i][0]);
    FreeBoundingPoint(&points[i][1]);
    free(points[i]);
  }
  free(points);
}

void FreeBoundingPoint(BoundingPoint *point){
  free(point->index);
  free(point->pos);
}


void InitPhotPar(PhotPar *obs, char *name, char *pedigree) {
  /* Initializes the PhotPar structure for use in this routine.
   
   This routine should be called by the user's code, and is not 
   explicitly called within this library.
   The parameter's 'name' and 'pedigree' should come from RefTab.
   */
  obs->name[0] = '\0';
  obs->pedigree[0] = '\0';
  /* Start by copying in required values from input table RefTab */
	strcpy(obs->name, name);
	strcpy(obs->pedigree, pedigree);
  
  /* Initialize remainder of fields to be used as output in this code */
	obs->descrip2[0] = '\0';
	obs->goodPedigree = PEDIGREE_UNKNOWN;
  
  obs->obsmode[0] = '\0';     /* obsmode of science data */
  obs->photmode[0] = '\0'; /* obsmode used for comparison with IMPHTTAB */
  
  /* parsed out value of any parameterized values */
  /* tab->obspars=NULL; */
  obs->npar = 0;
  
  /* Output values derived from table */
  obs->photflam = 0;
  obs->photplam = 0;
  obs->photbw = 0;
  obs->photzpt = 0;
  
}

int AllocPhotPar(PhotPar *obs, int npar){
  int status = 0;
  
  int i;
  
  obs->npar = npar;
  
  obs->parnames = (char **) malloc(npar * sizeof(char *));
  /*printf("Allocated %d parnames\n",npar);*/
  for (i=0;i<npar;i++) {
    obs->parnames[i] = (char *) malloc(SZ_FITS_REC * sizeof(char));
    obs->parnames[i][0] = '\0';
  }
  obs->parvalues = (double *) malloc(npar * sizeof(double));
  if (obs->parnames == NULL || obs->parvalues == NULL) {
    return(status=OUT_OF_MEMORY);
  }
  
  return(status);
  
}
void FreePhotPar(PhotPar *obs){
  int n;
  
  for (n=0;n<obs->npar;n++){
    free(obs->parnames[n]);
  }
  free(obs->parnames);
  free(obs->parvalues);
}


/* This function compares two strings without regard to case, returning
 one if the strings are equal.
 
 Phil Hodge, 1997 Dec 12:
 Function created.
 */

int streq_ic_IMPHTTAB (char *s1, char *s2) {
  
	int c1, c2;
	int i;
  
	c1 = 1;
	for (i = 0;  c1 != 0;  i++) {
    
    c1 = s1[i];
    c2 = s2[i];
    if (isupper(c1))
      c1 = tolower (c1);
    if (isupper(c2))
      c2 = tolower (c2);
    if (c1 != c2)
      return (0);
	}
	return (1);
}

int strneq_ic (char *s1, char *s2, int n) {
  
	int c1, c2;
	int i;
  
  if (n == 0)
    return 0;
  
	c1 = 1;
	for (i = 0;  i < n;  i++) {
    
    c1 = s1[i];
    c2 = s2[i];
    if (isupper(c1))
      c1 = tolower (c1);
    if (isupper(c2))
      c2 = tolower (c2);
    if (c1 != c2)
      return (0);
	}
	return (1);
}

