echo Converting the audio files to FLAC ...
FOLDER=$PWD
COUNTER=$(find -name *.wav|wc -l)
for f in $PWD/**/**/*.wav; do
    COUNTER=$((COUNTER - 1))
    echo -ne "\rConverting ($COUNTER) : $f..."
    ffmpeg -y -loglevel fatal -i $f -ac 1 -ar 16000 ${f/\.wav/.flac}
done
