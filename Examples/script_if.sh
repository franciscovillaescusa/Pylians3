#!/bin/bash

dims=512
do_RSD=1
axis=0

#name of the folders containing the snapshots
snapshot_folder=('/scratch/villa/SAM/CDM/'       \
                 '/scratch2/villa/SAM/NU0.3/'    \
                 '/scratch2/villa/SAM/NU0.3s8/'  \
                 '/scratch2/villa/SAM/NU0.6/'    \
                 '/scratch2/villa/SAM/NU0.6s8/'  \
                 '/scratch/villa/SAM/CDM/'       \
                 '/scratch/villa/SAM/CDM/')

#root of the files containing the galaxy catalogues
root_catalogue=('LG_NCDM_' \
                'LG_NU03_' \
                'LG_N3s8_' \
                'LG_NU06_' \
                'LG_N6s8_' \
                'LG_DC_N_' \
                'LG_SA_N_')

#name of the folders containing the 2PCF files
root_f_out=('0.0/2PCF_RS_gal_0.0_z='      \
            '0.3/2PCF_RS_gal_0.3_z='      \
            '0.3s8/2PCF_RS_gal_0.3s8_z='  \
            '0.6/2PCF_RS_gal_0.6_z='      \
            '0.6s8/2PCF_RS_gal_0.6s8_z='  \
            '0.0_DC/2PCF_RS_gal_0.0_z='   \
            '0.0_SA/2PCF_RS_gal_0.0_z=')

#redshifts
z=('3.06.dat' '2.07.dat' '0.99.dat' '0.51.dat' '0.00.dat')

#snapshots number corresponding to the above redshifts
suffix_snapshot_CDM=('snap_026' 'snap_031' 'snap_040' 'snap_047' 'snap_062')
suffix_snapshot=('snapdir_026/snap_026' \
                 'snapdir_031/snap_031' \
                 'snapdir_040/snap_040' \
                 'snapdir_047/snap_047' \
                 'snapdir_062/snap_062')


#do a loop over the different cosmologies
for i in ${!snapshot_folder[*]}
do

    #do a loop over the different redshifts for the same cosmology
    for j in ${!z[*]}
    do
	
	if [ "$i" == "0" -o "$i" -ge "5" ]; then
	    snapshot_fname=${snapshot_folder[$i]}${suffix_snapshot_CDM[$j]}
	else
	    snapshot_fname=${snapshot_folder[$i]}${suffix_snapshot[$j]}
	fi

	f_cat=${root_catalogue[$i]}${z[$j]}
	f_out=${root_f_out[$i]}${z[$j]}

	echo $snapshot_fname
	echo $f_cat
	echo $f_out

	mpirun -np 8 python correlation_function.py $snapshot_fname $f_cat $f_out $do_RSD $axis

    done
    echo ' '
done

