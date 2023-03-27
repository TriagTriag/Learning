#!/bin/bash

# go into each subfolder
# cd ~/Banded

for dir in $(find . -maxdepth 1 -type d); do
    # char1="~/Banded/"
    # char="'"
    # echo $PWD
    dir_str="$dir"
    cd $dir_str
    # echo $dir_str
    # run pdfcrop on all PDFs in current folder
    # for file 
    for file in *.pdf; do
        # echo $file
        pdfcrop $file $file
    done
    # check if dir is not equal to .
    if [ $dir_str != "." ]; then  
        # go back to parent folder
        cd ..
    fi
done

# echo all the pdfs in a folder



