set term postscript enhanced color
set pm3d map
set palette rgbformulae 22,13,-31
set size square
set xrange[-180:180]
set yrange[-180:180]
set output '2dprob.ps'
splot '2D_test.file.2dprob'

