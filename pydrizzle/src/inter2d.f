      real function nrievl (x, y, datain, nxpix, nypix, lendan, intere,
     *  scale)

C NRIEVAL -- Procedure to evaluate the 2D interpolant at a given value
C of x and y. NRIEVAL allows the interpolation of a few interpolated
C points without the computing time and storage required for the
C sequential version. The routine assumes that 1 <= x <= nxpix and
C 1 <= y <= nypix.
C
C This is a modified version to allow for a stretch factor (the scale
C parameter) in the SINC case.
C
C Modifications: Richard Hook, ST-ECF@STScI, December 2003
C
C This version supports most options, with the exception of SPLINE3. 
C
C Richard Hook modified it, September 2002
C Warren Hack, June 2004
C
      IMPLICIT NONE

      integer nxpix, nypix
      integer lendan
      integer intere
      real scale
      real x, y
      real datain(lendan,*)
      integer nx, ny
      real hold21, hold12, hold22
      real sx, sy, tx, ty
      real xval, yval
      real value
      real coeff(16 +3,16 +3)
      integer nterms
      integer rowleh
      integer xindex, yindex, firstw, lastrw
      integer i, j
C  Used for SPLINE3 option
C      integer kx, ky
C      real tmp(19*19)

C Define common terms
         nx = int(x)
         ny = int(y)
   

C Nearest neighbour
      nrievl = 0.
      IF(INTERE.EQ.1) THEN
         nrievl = (datain(int (x+0.5), int (y+0.5)))

C Bilinear
      ELSE IF(INTERE.EQ.2) THEN
         sx = x - nx
         tx = 1. - sx
         sy = y - ny
         ty = 1. - sy
         if (nx .ge. nxpix) THEN      
            hold21 = 2. * datain(nx,ny) - datain(nx-1,ny)
         ELSE     
            hold21 = datain(nx+1,ny)
         ENDIF    

         if (ny .ge. nypix) THEN     
            hold12 = 2. * datain(nx,ny) - datain(nx,ny-1)
         ELSE    
            hold12 = datain(nx,ny+1)
         ENDIF    

         if (nx .ge. nxpix .and. ny .ge. nypix) THEN
            hold22 = 2. * hold21 - (2. * datain(nx,ny-1) - 
     *         datain(nx-1,ny-1))
         ELSE IF (nx .ge. nxpix) THEN
            hold22 = 2. * hold12 - datain(nx-1,ny+1)
         ELSE IF (ny .ge. nypix) THEN     
               hold22 = 2. * hold21 - datain(nx+1,ny-1)
         ELSE     
               hold22 = datain(nx+1,ny+1)
         ENDIF    

         value = tx * ty * datain(nx,ny) + sx * ty * hold21 + sy * tx
     *    * hold12 + sx * sy * hold22

         nrievl = value
         
C Bicubic polynomial: 'poly3'
      ELSE IF(INTERE.EQ.3) THEN
         rowleh = 16 + 3
         nterms = 4
         yindex = 1

C	    major problem is that near the edge the interior polynomial
C	    must be defined
C
C	    use boundary projection to extend the data rows
         DO j = ny-1, ny+2

C		check that the data row is defined
            IF (j .ge.1 .and. j .le. nypix ) THEN

C		         extend the rows
               xindex = 1
               DO i=nx-1, nx+2 
                  IF (i .lt. 1) THEN
                     coeff(xindex,yindex) = 2. * datain(1,j) - 
     *                     datain(2-i,j)
                  ELSE IF (i .gt. nxpix) THEN
                     coeff(xindex,yindex) = 2. * datain(nxpix,j) - 
     *                     datain(2*nxpix-i,j)
                  ELSE
                     coeff(xindex,yindex) = datain(i,j)
                  ENDIF
                  xindex = xindex + 1
               ENDDO

            ELSE IF (j .eq. (ny + 2)) THEN
C		         extend the rows
               xindex = 1
               DO i=nx-1, nx+2
                  IF (i .lt. 1) THEN
                     coeff(xindex,yindex) = 2. * datain(1,nypix-2) - 
     *                   datain(2-i,nypix-2)
                  ELSE if (i .gt. nxpix) THEN
                     coeff(xindex,yindex) = 2. * datain(nxpix,nypix-2)
     *                   - datain(2*nxpix-i,nypix-2)
                  ELSE
                     coeff(xindex,yindex) = datain(i,nypix-2)
                  ENDIF
                  xindex = xindex + 1
               ENDDO
            ENDIF
            yindex = yindex + 1
         ENDDO
C End of loop over rows (j)
C	 project columns
C
C  awsur -- Weighted sum of 2 real vectors.
C           awsur(a,b,c,npix,k1,k2) : c(i) = k1*a(i) + k2*b(i)

         firstw = max (1, 3 - ny)
         IF (firstw .gt. 1) THEN
            DO j= 1, firstw 
               call WSUMR (coeff(1, firstw), coeff(1, 2*firstw-j), 
     *            coeff(1,j), nterms, 2., -1.)
            ENDDO
         ENDIF

         lastrw = min (nterms, nypix - ny + 2)
         IF (lastrw .lt. nterms) THEN
            DO j= lastrw + 1, nterms - 1
               call WSUMR (coeff(1,lastrw), coeff(1,2*lastrw-j), 
     *            coeff(1,j), nterms, 2., -1.)
            ENDDO
         ELSE if (lastrw .eq. 2) THEN
            call WSUMR (coeff(1,lastrw), coeff(1,4), coeff(1,4), 
     *         nterms, 2., -1.)

         ELSE
            call WSUMR (coeff(1,lastrw), coeff(1,2*lastrw-4), 
     *         coeff(1,4), nterms, 2., -1.)
         ENDIF

C  center the x value and call evaluation routine
         xval = 2 + (x - nx)
         yval = 2 + (y - ny)
         call iibip3 (coeff, 0, rowleh, xval, yval, value, 1)

         nrievl = (value)

C BIPOLY5 case: 'poly5'      
      ELSE IF (INTERE.EQ.4) THEN
         rowleh = 16 + 3
         nterms = 6

C	     major problem is to define interior polynomial near the edge
C
C	     loop over the rows of data
         yindex = 1
         DO j = ny - 2, ny+3
C		select the  rows containing data
            if (j .ge. 1 .and. j .le. nypix) THEN
            
C		         extend the rows
               xindex = 1
               DO i = nx - 2, nx+3
                  if (i .lt. 1) THEN
                     coeff(xindex,yindex) = 2. * datain(1,j) - 
     *                     datain(2-i,j)
                  ELSE if (i .gt. nxpix) THEN
                     coeff(xindex,yindex) = 2. * datain(nxpix,j) - 
     *                     datain(2*nxpix-i,j)
                  ELSE
                     coeff(xindex,yindex) = datain(i,j)
                  ENDIF
                  xindex = xindex + 1
               ENDDO

            ELSE if (j .eq. (ny + 3)) THEN
C		         extend the rows
               xindex = 1
               DO i = nx - 2, nx + 3
                  IF (i .lt. 1) THEN
                     coeff(xindex,yindex) = 2. * datain(1,nypix-3) - 
     *                     datain(2-i,nypix-3)
                  ELSE if (i .gt. nxpix) THEN
                     coeff(xindex,yindex) = 2. * datain(nxpix,nypix-3) 
     *                   - datain(2*nxpix-i,nypix-3)
                  ELSE
                     coeff(xindex,yindex) = datain(i,nypix-3)
                  ENDIF

                  xindex = xindex + 1
               ENDDO
            ENDIF

C     End of loop of 'j'
            yindex = yindex + 1
         ENDDO
C	   project columns

         firstw = max (1, 4 - ny)
         if (firstw .gt. 1) THEN
            DO j = 1, firstw
               call WSUMR (coeff(1,firstw), coeff(1,2*firstw-j), 
     *            coeff(1,j), nterms, 2., -1.)
            ENDDO
         ENDIF

         lastrw = min (nterms, nypix - ny + 3)
         if (lastrw .lt. nterms) THEN
            DO j = lastrw + 1, nterms - 1 
               call WSUMR (coeff(1,lastrw), coeff(1,2*lastrw-j),
     *            coeff(1,j), nterms, 2., -1.)
            ENDDO
         ELSE if (lastrw .eq. 3) THEN
            call WSUMR (coeff(1,lastrw), coeff(1,6), coeff(1,6), 
     *            nterms, 2., -1.)
         ELSE
            call WSUMR (coeff(1,lastrw), coeff(1,2*lastrw-6), 
     *            coeff(1,6), nterms, 2., -1.)
         ENDIF
C	     call evaluation routine
         xval = 3 + (x - nx)
         yval = 3 + (y - ny)
         call iibip5 (coeff, 0, rowleh, xval, yval, value, 1)

         nrievl = (value)
C End of 'poly5' case
C
C  Spline case not implemented due to requirement of 'iispld',
C  derived from ii_spline2d.x, to dynamically allocate a working
C  array for the computation.
C
C      ELSE IF (INTERE.EQ.5) THEN
C         rowleh = 16 + 3
C         ky = 0
C         DO 510 j = ny - 16 /2 + 1, ny + 16 /2
C            if (.not.(j .lt. 1 .or. j .gt. nypix)) THEN
C               ky = ky + 1
C                if (ky .eq. 1) THEN
C                   yindex = ny - j + 1
C                ENDIF
C 
C                kx = 0
C                DO 540 i = nx - 16 /2 + 1, nx + 16 /2
C                   if (.not.(i .lt. 1 .or. i .gt. nxpix)) THEN
C                      kx = kx + 1
C                      if (kx .eq. 1) THEN
C                         xindex = nx - i + 1
C                      ENDIF
C                      coeff(kx+1,ky+1) = datain(i,j)
C                   ENDIF
C 
C 540            ENDDO
C                coeff(1,ky+1) = 0.
C                coeff(kx+2,ky+1) = 0.
C                coeff(kx+3,ky+1) = 0.
C             ENDIF
C 510      ENDDO
C             
C          call SET1R (0., coeff(1,1), kx+3)
C          call SET1R (0., coeff(1,ky+2), kx+3)
C          call SET1R (0., coeff(1,ky+3),kx+3)
C          call iispld (coeff, tmp, kx, ky+2, rowleh, rowleh)
C          call iispld (tmp, coeff, ky, kx+2, rowleh, rowleh)
C          xval = xindex + 1 + (x - nx)
C          yval = yindex + 1 + (y - ny)
C         call iibis3 (coeff, 0, rowleh, xval, yval, value, 1)
C         nrievl = (value)
      ELSE IF (INTERE.EQ.6 .OR. INTERE.EQ.7) THEN
         call iinisc (datain, 0, lendan, nypix, x, y, value, 1, 15, 0
     *      .001, 0.001, scale)
         nrievl = (value)
                  
      ENDIF

      end
