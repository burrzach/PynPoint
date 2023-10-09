#! /usr/bin/bash

#for performing subtraction of many different ifu files in series
#pass dir names as arguments and this script will perform psf subtraction in products folder of each dir

echo "Performing batch subtraction for the following directories:" >> /home/zburr/batch_subtract.log
echo "" >> /home/zburr/batch_subtract.log
for DIR in $*; do
	echo "$DIR" >> /home/zburr/batch_subtract.log
	RUN="python /home/zburr/psf_subtraction.py $DIR/products/"
	if [ -e $DIR/products ]; then
		if ! $RUN; then
			echo "     -ERROR (check log)" >> /home/zburr/batch_subtract.log
		else
			echo "     -success!" >> /home/zburr/batch_subtract.log
			cd $DIR/../..
			mv $DIR ./processed/
		fi
	else
		echo "     -ERROR (dir not found)" >> /home/zburr/batch_reduct.log
	fi	
done

echo "Done" >> /home/zburr/batch_subtract.log
echo "" >> /home/zburr/batch_subtract.log

