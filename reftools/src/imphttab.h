/* Constants and Definitions for use with IMPHTTAB library */
/* Definitions based on those defined in acs.h */

#define YES         1
#define NO          0 

# define SZ_LINE      255
# define SZ_FITS_REC   82
# define SZ_FNAME      63

/* from xtables.h */
# define SZ_COLNAME 79

/* Codes for goodPedigree in RefImage and RefTab to specify whether
 pedigree is good, dummy, or not specified.
 */
# define GOOD_PEDIGREE      1
# define DUMMY_PEDIGREE     0
# define PEDIGREE_UNKNOWN (-1)

/* Error codes for IMPHTTAB code. [based on acserr.h] */

# define PHOT_OK                 0

/* set in stxtools/errxit.x */
# define ERROR_RETURN           2

# define OUT_OF_MEMORY          111

# define MAXPARS 3          /* Max number of parameterizations supported + 1 */

/* The following definition needs to be kept updated with MAXPARS */
static char colnames[9][12] = {"OBSMODE", "DATACOL", "RESULT", "NELEM1", "NELEM2", "PAR1VALUES", "PAR2VALUES", "RESULT1", "RESULT2"};

static char *photnames[4] = {"PHOTZPT","PHOTFLAM", "PHOTPLAM", "PHOTBW"};

typedef struct {
	char obsmode[SZ_FITS_REC];  /* obsmode string read from table row */
	char datacol[SZ_FITS_REC];
  char **parnames; /* record the par. variable names for comparison with obsmode string */
  int parnum;      /* number of parameterized variables */
	double *results;  /* results[telem] or results[nelem1*nelem2*...] */
  int telem;     /* total number of parameterized variable values */
  int *nelem;    /* multiple paramerized variables will each N values */
	double **parvals; /* need to support multiple parameterized variables */
} PhtRow;

typedef struct {
  char name[SZ_LINE+1];            /* name of table */
  int goodPedigree;               /* DUMMY_PEDIGREE if dummy */
  char pedigree[SZ_FITS_REC];    /* value of pedigree (header or row) */
  char descrip[SZ_FITS_REC];     /* value of descrip from header */
  char descrip2[SZ_FITS_REC];    /* value of descrip from row */
  int exists;                     /* does reference table exist? */
  
  char obsmode[SZ_FITS_REC];	/* obsmode of science data */
  char photmode[SZ_FITS_REC]; /* obsmode used for comparison with IMPHTTAB */
  
  /* parsed out value of any parameterized values */
  double *parvalues;
  char **parnames;
  int npar;
  
  
  /* Output values derived from table */
  double photflam;
  double photplam;
  double photbw;
  double photzpt;
  
} PhotPar;

void InitPhotPar(PhotPar *obs, char *name, char *pedigree);
int AllocPhotPar(PhotPar *obs, int npar);
void FreePhotPar(PhotPar *obs);
void ClosePhotRow(PhtRow *tabrow);
double ComputeValue(PhtRow *tabrow, PhotPar *obs);
