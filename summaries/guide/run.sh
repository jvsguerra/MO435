#!/usr/bin/env bash

file=$1
wd=$2

if [ -z "$1" ]; then

  if [ -z "$1" ]; then
    echo '> Missing tex file'
  fi

  echo 'Usage:'
  echo -e "\t"'./run .tex [wd]'

else

  if [ ! -f "${wd}/${file}" ]; then
    exit 1
  fi


  docker run --rm -it -v "$(pwd)/${wd}":/home adnrv/texlive pdflatex "${file}" >/dev/null
  sudo chown jvsguerra ${wd}/*
  sudo chgrp jvsguerra ${wd}/*


fi
