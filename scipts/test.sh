#!/bin/bash

PROG=./build/detector
MODEL=/home/indemind/Models/cleaner_machine.mnn

USE_VIDEO=0

INPUT_DIR=/data/test_images/tmp40/*
OUTPUT_DIR=./test_images

if [ ! -d ${OUTPUT_DIR} ]
then
    mkdir ${OUTPUT_DIR}
fi

rm ${OUTPUT_DIR}/*



for f in ${INPUT_DIR};do
    FN=$(basename "${f}")
    EXT="${FN##*.}"
    if [ "${EXT}" == "jpg" ] || [ "${EXT}" == "png" ]
    then
        echo ${EXT}
        OUTPUT_FN=${OUTPUT_DIR}/${FN}
        ${PROG} ${MODEL} ${f} ${USE_VIDEO} ${OUTPUT_FN}
    # else
        # echo ${EXT}
    fi
done
