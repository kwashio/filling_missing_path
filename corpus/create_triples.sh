#!/usr/bin/env bash

wiki_dump_file=$1

# Parse wikipedia. Splitting to 20 files and running in parallel.
echo 'Parsing wikipedia...'
split -nl/20 $wiki_dump_file $wiki_dump_file"_";

for x in {a..t}
do
( python parse_wikipedia.py $wiki_dump_file"_a"$x $wiki_dump_file"_a"$x"_parsed\
" ) &
done
wait

python term_to_id_make.py $wiki_dump_file