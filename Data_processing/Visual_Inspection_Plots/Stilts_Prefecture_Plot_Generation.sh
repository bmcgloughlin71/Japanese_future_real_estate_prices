#!/bin/bash
set -eu

cleaned_data_dir="../Cleaned_Data_Sets/"

#Cube Plots
for file in $(ls -1 ${cleaned_data_dir}/*test*buildings.csv); do 

prefecture=$(basename "$file" | awk -F '_' '{print $1}')

stilts plot2cube \
   xpix=1898 ypix=729 \
   zlog=true xflip=true xlabel=MunicipalityCategory ylabel='TotalFloorArea (m^{2})' zlabel='TradePrice (Yen)' texttype=latex fontsize=23 \
   xmin=0.94 xmax=4.06 ymin=10 ymax=2040 zmin=8494 zmax=5.698E9 phi=-178.01 theta=25.69 psi=157.26 \
   title="${prefecture}" legend=false \
   auxmap=rainbow2 auxflip=true auxmin=1 auxmax=120 \
   auxvisible=true auxlabel=AvgTimeToNearestStation_max \
   layer=Mark \
      in=../Cleaned_Data_Sets//${prefecture}_cleaned_test_buildings.csv ifmt=CSV \
      x=MunicipalityCategory y=TotalFloorArea z=TotalTransactionValue weight=AverageTimeToStation \
      shading=weighted size=4 combine=max \
   omode=out out=$prefecture.png

done

#Map of Japan
stilts plot2sky \
   xpix=1898 ypix=606 \
   reflectlon=false \
   clon=138.57 clat=36.14 radius=11.234 \
   legend=false \
   auxmap=rainbow2 auxflip=true auxfunc=log auxmin=1000000 auxmax=230000000 \
   auxvisible=true auxlabel=TotalTransactionValue \
   layer=Mark \
      in=${cleaned_data_dir}/All_prefectures_buildings_with_migration_coords_pop.csv ifmt=CSV \
      lon=longitude lat=latitude weight=TotalTransactionValue \
      shading=weighted size=2 \
      omode=out out=Map_of_Japan.png
