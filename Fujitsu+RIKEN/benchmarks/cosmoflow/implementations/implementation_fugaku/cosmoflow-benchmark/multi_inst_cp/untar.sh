#!/bin/bash

SrcDir="$1" && shift
DstDir="$1" && shift
CompType="$1" && shift
Num="$1" && shift
EndFlag="$1" && shift

Exec(){
    echo "$@"
    "$@"
    if [ $? -ne 0 ]; then
	echo Error at "$@"
	exit 2
    fi
}

case $CompType in
    gz | gzip) CompOpt="-I pigz"
               CompExt=".gz" ;;
         lz4 ) CompOpt="-I lz4"
               CompExt=".lz4" ;;
          xz ) CompOpt="-I pixz -p"
               CompExt=".xz" ;;
        none ) ;;
            *) echo "Unsupported compression type $Compress"
               exit 1;;
esac


Count=0
while true
do
    TarFile=`find $SrcDir -maxdepth 1 -name "*.tar${CompExt}" | head -n 1`

    if [ "$TarFile" != "" ] ; then
	BaseName=$(basename $TarFile)
	Type=${BaseName#*_}
	Type=${Type%_*}

	Exec tar ${CompOpt} -xf $TarFile -C $DstDir/$Type
	Exec rm $TarFile

	Count=$(( $Count + 1 ))

	if [ $Count -eq $Num ] ; then
	    touch $EndFlag
	    echo "Done!"
	    break
	fi
    else
	sleep 0.5
    fi
done
