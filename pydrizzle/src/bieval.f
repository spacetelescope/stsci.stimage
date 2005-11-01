C ii_bieval:  Evaluation routines for interpolation kernels.
C             Called by 'mrievl' from INTER2D.F
C
C All routines converted from SPP to Fortran using 'xc -f' then hand
C  edited to include all comments and to use F77 constructs.
C  Warren Hack, 15 June 2004
C
C
C II_BINEAREST -- Procedure to evaluate the nearest neighbour interpolant.
C The real array coeff contains the coefficients of the 2D interpolant.
C The procedure assumes that 1 <= x <= nxpix and 1 <= y <= nypix and that
C coeff(1+first_point) = datain(1,1).
C
      subroutine iibint (coeff, firstt, lencof, x, y, zfit, npts)
C
C real  coeff[ARB]   # 1D coefficient array
C int   firstt       # offset of first data point
C int   lencof       # row length of coeff
C real  x[npts]      # array of x values
C real  y[npts]      # array of y values
C real  zfit[npts]	# array of interpolated values
C int	  npts		   # number of points to be evaluated
C
      integer firstt
      integer lencof
      integer npts
      real coeff(*)
      real x(npts), y(npts), zfit(npts)
      integer nx, ny
      integer index, i
      save

      do i = 1, npts 
         nx = int(x(i) + 0.5)
         ny = int(y(i) + 0.5)

C	      define pointer to data[nx,ny]
         index = firstt + (ny - 1) * lencof + nx
         zfit(i) = coeff(index)
      ENDDO

      return
      end
      
C II_BILINEAR -- Procedure to evaluate the bilinear interpolant.
C The real array coeff contains the coefficients of the 2D interpolant.
C The procedure assumes that 1 <= x <= nxpix and 1 <= y <= nypix
C and that coeff(1+first_point) = datain(1,1).
C
      subroutine iibilr (coeff, firstt, lencof, x, y, zfit, npts)

C real   coeff[ARB]     # 1D array of coefficients
C int    firstt         # offset of first data point
C int    lencof         # row length of coeff
C real   x[npts]        # array of x values
C real   y[npts]        # array of y values
C real   zfit[npts]	   # array of interpolated values
C int	   npts		      # number of data points

      integer firstt
      integer lencof
      integer npts
      real coeff(*)
      real x(npts), y(npts), zfit(npts) 
      integer nx, ny 
      integer index, i
      real sx, sy, tx, ty

      save
      do i = 1, npts 
         nx = int(x(i))
         ny = int(y(i))
         sx = x(i) - nx
         tx = 1. - sx
         sy = y(i) - ny
         ty = 1. - sy
         index = firstt + (ny - 1) * lencof + nx
         zfit(i) = tx * ty * coeff(index) + sx * ty * 
     *      coeff(index + 1) + sy * tx * coeff(index+lencof) +  
     *      sx * sy * coeff(index+lencof+1)
      ENDDO

      return
      end


C II_BIPOLY3 -- Procedure to evaluate the bicubic polynomial interpolant.
C The real array coeff contains the coefficients of the 2D interpolant.
C The procedure assumes that 1 <= x <= nxpix and  1 <= y <= nypix
C and that coeff[1+first_point] = datain[1,1]. The interpolant is
C evaluated using Everett's central difference formula.

      subroutine iibip3 (coeff, firstt, lencof, x, y, zfit, npts)

C real   coeff[ARB]     # 1D array of coefficients
C int    firstt         # offset of first data point
C int    lencof         # row length of coeff
C real   x[npts]        # array of x values
C real   y[npts]        # array of y values
C real   zfit[npts]	   # array of interpolated values
C int	   npts		      # number of data points
      integer firstt
      integer lencof
      integer npts
      real coeff(*)
      real x(npts), y(npts), zfit(npts)  
      integer nxold, nyold
      integer nx, ny
      integer firstw
      integer index
      integer i, j
      real sx, sy, tx, ty

      real sx2m1, tx2m1
      real cd20(4), cd21(4)
      real ztemp(4)
      real cd20y, cd21y
      save

      nxold = -1
      nyold = -1
      do i = 1, npts 
         nx = int(x(i))
         sx = x(i) - nx
         tx = 1. - sx
         sx2m1 = sx * sx - 1.
         tx2m1 = tx * tx - 1.

         ny = int(y(i))
         sy = y(i) - ny
         ty = 1. - sy

C	         define pointer to datain[nx,ny-1]
         firstw = firstt + (ny - 2) * lencof + nx

C	         loop over the 4 surrounding rows of data
C	         calculate the  central differences at each value of y
C
C	         if new data point caculate the central differnences in x
C	         for each y

         index = firstw
         if (nx .ne. nxold .or. ny .ne. nyold) THEN
            do j = 1, 4 
               cd20(j) = 1./6. * (coeff(index+1) -  
     *            2. * coeff(index) + coeff(index-1))
               cd21(j) = 1./6. * (coeff(index+2) -  
     *            2. * coeff(index+1) + coeff(index))
               index = index + lencof
            ENDDO
         ENDIF

C	         interpolate in x at each value of y
         index = firstw
         do j = 1, 4 
            ztemp(j) = sx * (coeff(index+1) + sx2m1 * cd21(j)) + 
     *          tx * (coeff(index) + tx2m1 * cd20(j))
            index = index + lencof
         ENDDO

C  	      calculate y central differences
         cd20y = 1./6. * (ztemp(3) - 2. * ztemp(2) + ztemp(1))
         cd21y = 1./6. * (ztemp(4) - 2. * ztemp(3) + ztemp(2))

C	         interpolate in y
         zfit(i) = sy * (ztemp(3) + (sy * sy - 1.) * cd21y) +  
     *      ty * ( ztemp(2) + (ty * ty - 1.) * cd20y)

         nxold = nx
         nyold = ny
      ENDDO

      return
      end

C II_BIPOLY5 -- Procedure to evaluate a biquintic polynomial.
C The real array coeff contains the coefficents of the 2D interpolant.
C The routine assumes that 1 <= x <= nxpix and 1 <= y <= nypix
C and that coeff[1+first_point] = datain[1,1]. The interpolant is evaluated
C using Everett's central difference formula.
C
      subroutine iibip5 (coeff, firstt, lencof, x, y, zfit, npts)
C real   coeff[ARB]     # 1D array of coefficients
C int    firstt         # offset of first data point
C int    lencof         # row length of coeff
C real   x[npts]        # array of x values
C real   y[npts]        # array of y values
C real   zfit[npts]	   # array of interpolated values
C int	   npts		      # number of data points

      integer firstt
      integer lencof
      integer npts
      real coeff(*)
      real x(npts),y(npts), zfit(npts) 
      integer nxold, nyold
      integer nx, ny
      integer firstw
      integer index
      integer i, j
      real sx, sx2, tx, tx2, sy, sy2, ty, ty2
      real sx2m1, sx2m4, tx2m1, tx2m4
      real cd20(6), cd21(6)
      real cd40(6), cd41(6)
      real ztemp(6)
      real cd20y, cd21y
      real cd40y, cd41y
      save
      
      nxold = -1
      nyold = -1
      do 110 i = 1, npts 
         nx = int(x(i))
         sx = x(i) - nx
         sx2 = sx * sx
         sx2m1 = sx2 - 1.
         sx2m4 = sx2 - 4.
         tx = 1. - sx
         tx2 = tx * tx
         tx2m1 = tx2 - 1.
         tx2m4 = tx2 - 4.

         ny = int(y(i))
         sy = y(i) - ny
         sy2 = sy * sy
         ty = 1. - sy
         ty2 = ty * ty
         
C	      calculate value of pointer to data[nx,ny-2]
         
         firstw = firstt + (ny - 3) * lencof + nx

C	      calculate the central differences in x at each value of y
         index = firstw
         if (nx .ne. nxold .or. ny .ne. nyold) THEN
            do 200 j = 1, 6 
               cd20(j) = 1./6. * (coeff(index+1) -  
     *            2. * coeff(index) + coeff(index-1))
               cd21(j) = 1./6. * (coeff(index+2) - 
     *            2. * coeff(index+1) + coeff(index))
               cd40(j) = 1./120. * (coeff(index-2) - 
     *            4. * coeff(index-1) + 6. * coeff(index) - 
     *            4. * coeff(index+1) + coeff(index+2))
               cd41(j) = 1./120. * (coeff(index-1) - 
     *            4. * coeff(index)   + 6. * coeff(index+1) - 
     *            4. * coeff(index+2) + coeff(index+3))
               index = index + lencof
200         ENDDO
         ENDIF

C	      interpolate in x at each value of y
         index = firstw
         do 300 j = 1, 6 
            ztemp(j) = sx * (coeff(index+1) + sx2m1 * (cd21(j) + 
     *         sx2m4 * cd41(j))) + tx * (coeff(index) + 
     *         tx2m1 * (cd20(j) + tx2m4 * cd40(j)))
            index = index + lencof
300      ENDDO
C	      central differences in y
         cd20y = 1./6. * (ztemp(4) - 2. * ztemp(3) + ztemp(2))
         cd21y = 1./6. * (ztemp(5) - 2. * ztemp(4) + ztemp(3))
         cd40y = 1./120. * (ztemp(1) - 4. * ztemp(2) +  
     *      6. * ztemp(3) - 4. * ztemp(4) + ztemp(5))
         cd41y = 1./120. * (ztemp(2) - 4. * ztemp(3) +  
     *      6. * ztemp(4) - 4. * ztemp(5) + ztemp(6))


C	      interpolate in y
         zfit(i) = sy * (ztemp(4) + (sy2 - 1.) * (cd21y + (sy2 - 4.) 
     *      * cd41y)) + ty * (ztemp(3) + (ty2 - 1.) * (cd20y + 
     *      (ty2 - 4.) * cd40y))
         nxold = nx
         nyold = ny
110      ENDDO
      
      return
      end

C II_BISPLINE3 -- Procedure to evaluate a bicubic spline.
C The real array coeff contains the B-spline coefficients.
C The procedure assumes that 1 <= x <= nxpix and 1 <= y <= nypix
C and that coeff[1+first_point] = B-spline[2].
C
      subroutine iibis3 (coeff, firstt, lencof, x, y, zfit, npts)
C real   coeff[ARB]     # 1D array of coefficients
C int    firstt         # offset of first data point
C int    lencof         # row length of coeff
C real   x[npts]        # array of x values
C real   y[npts]        # array of y values
C real   zfit[npts]	   # array of interpolated values
C int	   npts		      # number of data points

      integer firstt
      integer lencof
      integer npts
      real coeff(*)
      real x(npts), y(npts)
      real zfit(npts)
      integer nx, ny
      integer firstw
      integer index
      integer i, j
      real sx, tx, sy, ty
      real bx(4), by(4)
      real accum
      real sum
      save
      do i = 1, npts 
         nx = int(x(i))
         sx = x(i) - nx
         tx = 1. - sx

         ny = int(y(i))
         sy = y(i) - ny
         ty = 1. - sy

C        calculate the x B-splines
         bx(1) = tx ** 3
         bx(2) = 1. + tx * (3. + tx * (3. - 3. * tx))
         bx(3) = 1. + sx * (3. + sx * (3. - 3. * sx))
         bx(4) = sx ** 3

C        calculate the y B-splines
         by(1) = ty ** 3
         by(2) = 1. + ty * (3. + ty * (3. - 3. * ty))
         by(3) = 1. + sy * (3. + sy * (3. - 3. * sy))
         by(4) = sy ** 3

C	      calculate the pointer to data[nx,ny-1]
         firstw = firstt + (ny - 2) * lencof + nx
         accum = 0.
         index = firstw
         do j = 1, 4 
            sum = coeff(index-1) * bx(1) + coeff(index) * bx(2) + 
     *         coeff(index+1) * bx(3) + coeff(index+2) * bx(4)
            accum = accum + sum * by(j)
            index = index + lencof
         ENDDO
         zfit(i) = accum
      ENDDO
      return
      end
      
      
C II_NISINC -- Procedure to evaluate the 2D sinc function.  The real array
C coeff contains the data. The procedure assumes that 1 <= x <= nxpix and
C 1 <= y <= nypix and that coeff[1+first_point] = datain[1,1]. The since
C truncation length is nsinc. The taper is a cosbell function which is
C valid for 0 <= x <= PI / 2 (Abramowitz and Stegun, 1972, Dover Publications,
C p 76). If the point to be interpolated is less than mindx and mindy from
C a data point no interpolation is done and the data point is returned. This
C routine does not use precomputed arrays.
C
C This version, renamed II_NISINC allows for a scaling of the
C interpolations array.
C 
C Interface modified to accept pre-initialized arrays for taper, ac, ar.
C   Original version dynamically allocated these arrays in the code.
C  WJH, 15 June 2004
C
      subroutine iinisc (coeff, firstt, lencof, lenary, x, y, zfit, 
     *npts, nconv, taper, ac, ar, mindx, mindy, scale)

C real   coeff[ARB]     # 1D array of coefficients
C int    first_point    # offset to first data point
C int    len_coeff      # row length of coeff
C int    len_array      # column length of coeff
C real   x[npts]        # array of x values
C real   y[npts]        # array of y values
C real   zfit[npts]     # array of interpolated values
C real   scale          # stretch factor
C int    npts           # the number of input points.
C int    nconv          # number of points in interpolation arrays
C          where nconv = 2 * nsinc + 1 and nsinc = sinc truncation length
C real   taper[NCONV]   # taper array
C real   ac[NCONV]      # ac array
C real   ar[NCONV]      # ar array
C real   mindx          # interpolation minimum in x
C real   mindy		      # interpolation minimum in y
C
      integer firstt
      integer lencof
      integer lenary
      integer npts
      integer nsinc
      real mindx, mindy
      real scale
      real coeff(*)
      real x(npts), y(npts)
      real zfit(npts)
      integer i, j, k
      integer nx, ny
      integer index
      integer mink, maxk, offk
      integer minj, maxj, offj
      integer lastpt
      integer nconv
      real taper(nconv), ac(nconv), ar(nconv) 

      real sconst
      real a2, a4
      real sdx, dx, dy, dxn, dyn
      real ax, ay, px, py
      real sum, sumx, sumy
      real dx2
      real halfpi

      save
C     Compute original sinc truncation length
      nsinc = (nconv - 1) / 2
      
C     Compute the constants for the cosine bell taper.
      halfpi = 1.5707963267948966192
      sconst = ( halfpi / nsinc) ** 2
      a2 = -0.49670
      a4 = 0.03705

C	   Precompute the taper array. Incorporate the sign change portion
C	   of the sinc interpolator into the taper array.

      if (mod (nsinc, 2) .eq. 0) THEN
         sdx = 1.0
      ELSE
         sdx = -1.0
         do j = -nsinc, nsinc 
            dx2 = sconst * j * j
            taper(j+nsinc+1) = sdx * (1.0 + a2 * dx2 + 
     *      a4 * dx2 * dx2) ** 2
            sdx = -sdx
         ENDDO
      ENDIF
   
      do i = 1, npts 
C     	   define the fractional pixel interpolation.
         nx = nint (x(i))
         ny = nint (y(i))
         if (nx .lt. 1 .or. nx .gt. lencof .or. ny .lt. 1 .or. 
     *      ny .gt. lenary) THEN
            zfit(i) = 0.0
C              go onto next point
            GOTO 130
         ENDIF

C ### Scale the offset
         dx = (x(i) - nx)*scale
         dy = (y(i) - ny)*scale

C	         define pointer to data[nx,ny]
         if (abs (dx) .lt. mindx .and. abs (dy) .lt. mindy) THEN 
            index = firstt + (ny - 1) * lencof + nx
            zfit(i) = coeff(index)
            goto 130
         ENDIF

C 	         Compute the x and y sinc arrays using a cosbell taper.
         dxn = 1 + nsinc + dx
         dyn = 1 + nsinc + dy
         sumx = 0.0
         sumy = 0.0
         do j = 1, nconv 
            ax = dxn - j
            ay = dyn - j
            if (ax .eq. 0.0) THEN
               px = 1.0
            ELSE if (dx .eq. 0.0) THEN
               px = 0.0
            ELSE
               px = taper(j-1) / ax
            ENDIF

            if (ay .eq. 0.0) THEN
               py = 1.0
            ELSE if (dy .eq. 0.0) THEN
               py = 0.0
            ELSE
               py = taper(j-1) / ay
            ENDIF

            ac(j-1) = px
            ar(j-1) = py
            sumx = sumx + px
            sumy = sumy + py
         ENDDO

C	         Compute the limits of the convolution.
         minj = max (1, ny - nsinc)
         maxj = min (lenary, ny + nsinc)
C           Get the index into ar
         offj = nsinc - ny

         mink = max (1, nx - nsinc)
         maxk = min (lencof, nx + nsinc)
C           Get the index into ac
         offk = nsinc - nx

C	         Do the convolution.
         zfit(i) = 0.0
         do 210 j = ny - nsinc, minj - 1 
            sum = 0.0
            do 220 k = nx - nsinc, mink - 1
               sum = sum + ac(k+offk) * coeff(firstt+1)
220            continue
            do 230 k = mink, maxk
               sum = sum + ac(k+offk) * coeff(firstt+k)
230            continue
            do 240 k = maxk + 1, nx + nsinc
               sum = sum + ac(k+offk) * coeff(firstt+lencof)
240            continue
            zfit(i) = zfit(i) + ar(j+offj) * sum
210         continue

         do 250 j = minj, maxj 
            index = firstt + (j - 1) * lencof
            sum = 0.0
            do 260 k = nx - nsinc, mink - 1
               sum = sum + ac(k+offk) * coeff(index+1)
260            continue

            do 270 k = mink, maxk
               sum = sum + ac(k+offk) * coeff(index+k)
270            continue
            do 280 k = maxk + 1, nx + nsinc
               sum = sum + ac(k+offk) * coeff(index+lencof)
280            continue
            zfit(i) = zfit(i) + ar(j+offj) * sum
250         continue

         do 290 j = maxj + 1, ny + nsinc 
            lastpt = firstt + (lenary - 1) * lencof
            sum = 0.0
            do 300 k = nx - nsinc, mink - 1
               sum = sum + ac(k+offk) * coeff(lastpt+1)
300            continue
            do 310 k = mink, maxk
               sum = sum + ac(k+offk) * coeff(lastpt+k)
310            continue
            do 320 k = maxk + 1, nx + nsinc
               sum = sum + ac(k+offk) * coeff(lastpt+lencof)
320            continue
            zfit(i) = zfit(i) + ar(j+offj) * sum
290         continue

         zfit(i) = zfit(i) / sumx / sumy

130   ENDDO
131   continue

      return
      end


C II_NILSINC -- Procedure to evaluate the 2D sinc function.  The real array
C coeff contains the data. The procedure assumes that 1 <= x <= nxpix and
C 1 <= y <= nypix and that coeff[1+first_point] = datain[1,1]. The since
C truncation length is nsinc. The taper is a cosbell function which is
C valid for 0 <= x <= PI / 2 (Abramowitz and Stegun, 1972, Dover Publications,
C p 76). If the point to be interpolated is less than mindx and mindy from
C a data point no interpolation is done and the data point is returned. This
C routine does use precomputed arrays.
C
C Renamed II_NILSINC when scale was added,
C  Richard Hook, ST-ECF/STScI, December 2003
      subroutine iinilc (coeff, firstt, lencof, lenary, x, y, zfit, npts
     *, ltable, nconv, nxincr, nyincr, mindx, mindy, scale)

C real   coeff[ARB]           # 1D array of coefficients
C int    first_point          # offset to first data point
C int    len_coeff            # row length of coeff
C int    len_array            # column length of coeff
C real   x[npts]              # array of x values
C real   y[npts]              # array of y values
C real   zfit[npts]           # array of interpolated values
C real   scale                # scale for offset
C int    npts                 # the number of input points.
C real   ltable[nconv,nconv,nxincr,nyincr]  # the pre-computed look-up table
C int    nconv                # the sinc truncation full width
C int	   nxincr				   # the interpolation resolution in x
C int	   nyincr				   # the interpolation resolution in y
C real	mindx				      # interpolation mininmum in x
C real	mindy				      # interpolation mininmum in y
      integer firstt
      integer lencof
      integer lenary
      integer npts
      integer nconv, nxincr, nyincr
      real mindx, mindy
      real scale
      real coeff(*)
      real x(npts), y(npts)
      real zfit(npts)
      real ltable(nconv,nconv,nxincr,nyincr)
      integer i, j, k
      integer nsinc
      integer xc, yc
      integer lutx, luty
      integer minj, maxj, offj
      integer mink, maxk, offk
      integer index
      integer lastpt
      real dx, dy, sum
      save

      nsinc = (nconv - 1) / 2
      do 110 i = 1, npts 

C	      Return zero outside of data.
         xc = nint (x(i))
         yc = nint (y(i))
         if (xc .lt. 1 .or. xc .gt. lencof .or. yc .lt. 1 .or. 
     *      yc .gt. lenary) THEN
            zfit(i) = 0.0
            goto 110
         ENDIF

         dx = (x(i) - xc)*scale
         dy = (y(i) - yc)*scale
         if (abs(dx) .lt. mindx .and. abs(dy) .lt. mindy) THEN
            index = firstt + (yc - 1) * lencof + xc
            zfit(i) = coeff(index)
         ENDIF

         if (nxincr .eq. 1) THEN
            lutx = 1
         ELSE
            lutx = nint ((-dx + 0.5) * (nxincr - 1)) + 1
         ENDIF

         if (nyincr .eq. 1) THEN
            luty = 1
         ELSE
            luty = nint ((-dy + 0.5) * (nyincr - 1)) + 1
         ENDIF

C	         Compute the convolution limits.
         minj = max (1, yc - nsinc)
         maxj = min (lenary, yc + nsinc)
         offj = 1 - yc + nsinc
         mink = max (1, xc - nsinc)
         maxk = min (lencof, xc + nsinc)
         offk = 1 - xc + nsinc

C	         Initialize
         zfit(i) = 0.0

C	         Do the convolution.
         do 160 j = yc - nsinc, minj - 1 
            sum = 0.0
            do 170 k = xc - nsinc, mink - 1
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(firstt+1)
170            continue
            do 180 k = mink, maxk
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(firstt+k)
180            continue
            do 190 k = maxk + 1, xc + nsinc
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(firstt+lencof)
190            continue
            zfit(i) = zfit(i) + sum
160         continue

         do 200 j = minj, maxj 
            index = firstt + (j - 1) * lencof
            sum = 0.0
            do 210 k = xc - nsinc, mink - 1
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(index+1)
210            continue
            do 220 k = mink, maxk
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(index+k)
220            continue
            do 230 k = maxk + 1, xc + nsinc
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(index+lencof)
230            continue
            zfit(i) = zfit(i) + sum
200         continue

         do 240 j = maxj + 1, yc + nsinc 
            lastpt = firstt + (lenary - 1) * lencof
            sum = 0.0
            do 250 k = xc - nsinc, mink - 1
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(lastpt+1)
250            continue
            do 260 k = mink, maxk
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(lastpt+k)
260            continue
            do 270 k = maxk + 1, xc + nsinc
               sum = sum + ltable(k+offk,j+offj,lutx,luty) * 
     *            coeff(lastpt+lencof)
270            continue
            zfit(i) = zfit(i) + sum
240         continue

110      continue
111      continue
      return
      end
c     firstw  first_row
c     iinisc  ii_nisinc
c     iibidz  ii_bidriz
c     dhxfrc  dhxfrac
c     iibis3  ii_bispline3
c     dhyfrc  dhyfrac
c     iibid0  ii_bidriz0
c     boundy  boundary
c     iibid1  ii_bidriz1
c     lastpt  last_point
c     lenary  len_array
c     iibint  ii_binearest
c     iibip3  ii_bipoly3
c     iibip5  ii_bipoly5
c     iibilr  ii_bilinear
c     iinilc  ii_nilsinc
c     lencof  len_coeff
c     firstt  first_point
