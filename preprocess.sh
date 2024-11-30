#!/usr/bin/env bash

rm -rf preprocessed
mkdir -p preprocessed

find original -type f -name "*.jpg" -not -path "*/_*" | \
    while read file;
    do
        basename=${file#original/}
        output=preprocessed/$(echo $basename | sed 's:/:_:g');
        echo $output
        magick $file -resize 10% -gravity center -crop 300x300+0+0 +repage $output &&\
            exiftool -overwrite_original -all= $output;
    done
