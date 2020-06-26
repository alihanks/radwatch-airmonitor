ls ~/Dropbox/UCB\ Air\ Monitor/Data/Roof/PAVLOVSKY/*/*.CNF > ./log_files/cnf_tmp.dat
comm -3 ./log_files/converted_spectras.dat ./log_files/cnf_tmp.dat > ./log_files/spectras_not_converted.dat
cat ./log_files/spectras_not_converted.dat | tr -d '\t' > ./log_files/tmp.dat
mv ./log_files/tmp.dat ./log_files/spectras_not_converted.dat

#python convert_cnf_to_spe.py ./log_files/spectras_not_converted.dat ./log_files/spectras_converted.dat
