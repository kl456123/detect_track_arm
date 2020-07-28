#!/bin/bash

PROG=./build/test_detector
MODEL=/home/indemind/Models/cleaner_machine.mnn
INPUT_DIR=//data/test_images/tmp88/
USE_VIDEO=1
OUTPUT_DIR=./test_images

if [ ! -d ${OUTPUT_DIR} ]
then
    mkdir ${OUTPUT_DIR}
fi

rm ${OUTPUT_DIR}/*

${PROG} ${MODEL} ${INPUT_DIR} ${USE_VIDEO} ${OUTPUT_DIR}
