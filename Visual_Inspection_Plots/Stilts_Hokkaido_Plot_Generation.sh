#!/bin/bash
set -eu

stilts plot2cube \
   xpix=1898 ypix=729 \
   zlog=true xflip=true xlabel=MunicipalityCategory ylabel='TotalFloorArea (m^{2})' zlabel='TradePrice (Yen)' texttype=latex fontsize=23 \
   xmin=0.94 xmax=4.06 ymin=10 ymax=2040 zmin=8494 zmax=5.698E9 phi=-178.01 theta=25.69 psi=157.26 \
   title='北海道' legend=false \
   auxmap=rainbow2 auxflip=true auxmin=1 auxmax=120 \
   auxvisible=true auxlabel=AvgTimeToNearestStation_max \
   layer=Mark \
      in=../Data_processing/Hokkaido_cleaned_test_buildings.csv ifmt=CSV \
      x=MunicipalityCategory y=TotalFloorArea z='TradePrice ' weight=AvgTimeToNearestStation \
      shading=weighted size=4 combine=max \
   omode=out out=Hokkaido.png
